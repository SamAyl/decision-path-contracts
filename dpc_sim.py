"""Decision-path simulation (Section 6 of the manuscript).

A seeded per-second simulation of a feature pipeline (stateful stream stage) feeding a fast
decision path. Decisions are made at event arrival and read features whose lag equals the
horizontal deviation between cumulative arrival and departure curves of the feature stage.

Compared: five capacity controllers x two decision policies (naive vs. decision-path contract),
under two workload families with 30 seeds each; plus duplicate-effect and fraud-velocity outcomes.
Run: python dpc_sim.py  -> prints tables, writes sim_results.csv, sim_validation.csv
"""
import csv, math
import numpy as np

MU = 1000.0
N_MIN, N_MAX = 10, 40
U = 0.70                    # target utilization (paper's value)
ACT_MEAN, ACT_SIG = 30.0, 0.3
CTRL_DT = 10
COOLDOWN = 120
H_CLEAR = 30.0              # backlog clearance horizon (s) in the paper's control rule
F = 5.0                     # feature freshness bound (s)
DELTA = 30                  # checkpoint interval (s)
R_OUT = 30                  # outage duration (s)
A_LAT = 0.25                # application latency (s): in-flight effects at crash become ambiguous
K_VEL, R_ATK, ATK_DUR, WIN = 10, 2.0, 120, 60   # velocity rule and attack parameters
T = 4 * 3600
SEEDS = 30

def workload(rng, kind):
    t = np.arange(T)
    base = 6000 * (1 + 0.25 * np.sin(2 * np.pi * t / 3600))
    eps = np.zeros(T); z = 0.0
    for i in range(0, T, 60):
        z = 0.7 * z + rng.normal(0, 0.05); eps[i:i + 60] = z
    lam = base * np.exp(eps)
    sched, events = [], []
    def add_burst(s, dur, amp):
        ramp = np.clip(np.minimum(np.arange(dur), dur - np.arange(dur)) / 20.0, 0, 1)
        lam[s:s + dur] *= 1 + (amp - 1) * ramp
    starts = np.sort(rng.choice(np.arange(900, T - 900, 600), size=3 if kind == "scheduled" else 5, replace=False))
    for j, s in enumerate(starts):
        dur = int(rng.uniform(120, 300)); amp = rng.uniform(2.3, 3.0)
        add_burst(s, dur, amp)
        scheduled = kind == "scheduled" or j % 2 == 0     # mixed: alternate scheduled / unscheduled
        if scheduled:
            sched.append((s, dur, amp * rng.uniform(0.9, 1.1)))   # forecast with +-10% amplitude error
        events.append((s, dur))
    arr = rng.poisson(lam).astype(float)
    crash = int(rng.uniform(1800, T - 1800)) if kind == "mixed" else None
    attacks = [int(s + rng.uniform(0, max(1, d - 30))) for s, d in events]
    return arr, sched, crash, attacks

def forecast(sched, t, base_now):
    for s, dur, amp in sched:
        if s <= t < s + dur:
            return base_now * amp
    return base_now

def run(arr, sched, crash, attacks, ctrl, rng, u=U):
    n = 10; pending = []; last_scale = -10**9
    A = np.concatenate([[0.0], np.cumsum(arr)])     # A[t] = arrivals before second t
    D = np.zeros(T + 1); q = 0.0; d = 0.0
    cost = 0.0; served_hist = []
    ckpt_D = 0.0; down_until = -1; dup = amb = None; replay = 0.0
    lam_obs = arr[:CTRL_DT].mean()
    for t in range(T):
        for p in [p for p in pending if p[0] <= t]:
            n = min(N_MAX, n + p[1]); pending.remove(p)
        if t % DELTA == 0 and t >= down_until:
            ckpt_D = d
        if crash is not None and t == crash:
            dup = d - ckpt_D                       # effects applied since checkpoint, re-emitted on replay
            amb = rng.poisson(arr[t] * A_LAT)      # in-flight effects with unknown outcome
            replay = dup; down_until = t + R_OUT
        down = t < down_until
        if t == down_until and replay:
            q += replay; d -= replay; replay = 0.0  # replayed events re-enter the queue
        cap = 0.0 if down else n * MU
        served = min(q + arr[t], cap)
        q = q + arr[t] - served; d += served
        D[t + 1] = d
        served_hist.append(served / max(cap, 1.0))
        cost += n + sum(k for _, k in pending)
        if t % CTRL_DT or t == 0 or down or ctrl == "fixed_min":
            continue
        lam_obs = arr[t - CTRL_DT:t].mean()
        util = min(1.0, float(np.mean(served_hist[-CTRL_DT:])))
        if ctrl.startswith("utilization"):
            need = n * util / u
        elif ctrl == "lag":
            need = (lam_obs + q / H_CLEAR) / (MU * u)
        elif ctrl == "forecast_lag":
            # use the scheduled forecast for the burst that will be active once new capacity is ready
            lam_hat = max(lam_obs, forecast(sched, t + int(ACT_MEAN) + CTRL_DT, 6000.0))
            need = (lam_hat + q / H_CLEAR) / (MU * u)
        tgt = int(min(N_MAX, max(N_MIN, math.ceil(need - 1e-9))))
        cur = n + sum(k for _, k in pending)
        if tgt > cur:
            pending.append((t + int(ACT_MEAN * math.exp(rng.normal(0, ACT_SIG))), tgt - cur)); last_scale = t
        elif tgt < n and not pending and t - last_scale >= COOLDOWN:
            n = tgt; last_scale = t
    # ---- feature lag: newest processed event's arrival time h(t) with A(h) = D(t)
    ts = np.arange(T + 1, dtype=float)
    Dm = np.maximum.accumulate(np.clip(D, 0, None))
    h = np.interp(Dm, A, ts)
    lag = np.clip(ts - h, 0, None)[:T]
    stale = lag > F
    stale_frac = arr[stale].sum() / arr.sum()
    # ---- fraud velocity: attacker tx at t_a + j/r; approved while observed count in window < k
    fr_naive, fr_contract = [], []
    for ta in attacks:
        times = ta + np.arange(int(ATK_DUR * R_ATK)) / R_ATK
        an = ac = 0
        for j, tj in enumerate(times):
            l = lag[min(T - 1, int(tj))]
            seen = np.sum((times[:j] <= tj - l) & (times[:j] > tj - WIN))
            if seen < K_VEL - 0: an += 1
            seen_c = np.sum((times[:j] > tj - WIN)) if l > F else seen   # fallback: in-path counter
            if seen_c < K_VEL: ac += 1
        fr_naive.append(an); fr_contract.append(ac)
    return dict(stale_pct=100 * stale_frac, max_lag=float(lag.max()), cost_ws=cost,
                fraud_naive=float(np.mean(fr_naive)), fraud_contract=float(np.mean(fr_contract)),
                dup_nonidem=dup if dup is not None else np.nan, dup_idem=0.0 if dup is not None else np.nan,
                ambiguous=amb if amb is not None else np.nan,
                pred_dup=(arr[crash] * ((crash % DELTA) if crash is not None else 0)) if crash is not None else np.nan)

CTRLS = ["fixed_min", "utilization", "utilization_matched", "lag", "forecast_lag"]
NAMES = {"fixed_min": "Fixed minimum", "utilization": "Utilization (HPA-style)",
         "utilization_matched": "Utilization, cost-matched", "lag": "Backlog-aware (Q/H)",
         "forecast_lag": "Forecast + backlog-aware"}
KINDS = ["scheduled", "mixed"]

def ci(x):
    x = np.asarray([v for v in x if not (isinstance(v, float) and np.isnan(v))], float)
    return (x.mean(), 1.96 * x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else (np.nan, np.nan)

def tune(kind):
    tw = [workload(np.random.default_rng(90000 + 100 * KINDS.index(kind) + s), kind) for s in range(6)]
    target = np.mean([run(*w, "forecast_lag", np.random.default_rng(5 + i))["cost_ws"] for i, w in enumerate(tw)])
    best = None
    for u in np.arange(0.30, 0.71, 0.02):
        c = np.mean([run(*w, "utilization", np.random.default_rng(5 + i), u=u)["cost_ws"] for i, w in enumerate(tw)])
        if best is None or abs(c - target) < best[1]:
            best = (round(float(u), 2), abs(c - target))
    return best[0]

if __name__ == "__main__":
    rows = []
    for kind in KINDS:
        um = tune(kind); print(f"{kind}: cost-matched utilization target u = {um}")
        for s in range(SEEDS):
            w = workload(np.random.default_rng(1000 * (KINDS.index(kind) + 1) + s), kind)
            for c in CTRLS:
                r = run(*w, c, np.random.default_rng(11 + s), u=um if c == "utilization_matched" else U)
                r.update(kind=kind, seed=s, ctrl=c, u=um if c == "utilization_matched" else U); rows.append(r)
    with open("sim_results.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    for kind in KINDS:
        print(f"\n=== {kind} ({SEEDS} seeds; mean ± 95% CI) ===")
        for c in CTRLS:
            sub = [r for r in rows if r["kind"] == kind and r["ctrl"] == c]
            g = {k: ci([r[k] for r in sub]) for k in ("stale_pct", "max_lag", "cost_ws", "fraud_naive", "fraud_contract", "dup_nonidem", "ambiguous", "pred_dup")}
            print(f"{NAMES[c]:28s} stale% {g['stale_pct'][0]:5.2f}±{g['stale_pct'][1]:4.2f} | maxlag {g['max_lag'][0]:6.1f} | "
                  f"cost kWs {g['cost_ws'][0]/1e3:6.1f}±{g['cost_ws'][1]/1e3:3.1f} | fraud naive {g['fraud_naive'][0]:5.1f} contract {g['fraud_contract'][0]:5.1f}"
                  + (f" | dup {g['dup_nonidem'][0]:8.0f} (pred {g['pred_dup'][0]:8.0f}) amb {g['ambiguous'][0]:5.0f}" if kind == "mixed" else ""))

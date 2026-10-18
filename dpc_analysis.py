"""Analytical results for the decision-path contract manuscript (Sections 3 and 5).

Reproduces the original burst scenario and adds the paper's new quantities:
feature lag (horizontal deviation between cumulative arrival and departure curves),
stale-decision windows, deadline misses, duplicate effects on replay, and
fraud-velocity exposure.
Run: python dpc_analysis.py
"""
import math
import numpy as np

DT = 0.01
HORIZON = 600.0
MU = 1000.0

def arrivals(t):
    return 18000.0 if 60 <= t < 180 else 6000.0

def workers(policy, t):
    if policy == "fixed_min":
        return 10
    if policy == "fixed_peak":
        return 26
    if policy == "reactive":
        return 26 if 90 <= t < 210 else 10
    if policy == "anticipatory":
        return 26 if 60 <= t < 210 else 10
    raise ValueError(policy)

def simulate(policy, F=5.0, B=2.0, s=0.2):
    """Fluid FIFO queue with exact feature lag l(t) = t - A^{-1}(D(t))."""
    n = int(HORIZON / DT)
    ts = np.arange(n) * DT
    lam = np.array([arrivals(t) for t in ts])
    cap = np.array([workers(policy, t) * MU for t in ts])
    A = np.concatenate([[0.0], np.cumsum(lam * DT)])[:-1]
    Q = np.zeros(n); D = np.zeros(n); q = 0.0; d = 0.0
    for i in range(n):
        Q[i] = q; D[i] = d
        served = min(q + lam[i] * DT, cap[i] * DT)
        q = q + lam[i] * DT - served; d += served
    # lag: newest processed event's arrival time h with A(h) = D(t)
    h = np.interp(D, A, ts)
    lag = ts - h
    # waiting time of an event arriving at t (in-pipeline decision): time until D reaches A(t)
    wait = np.interp(A, D, ts, right=np.nan) - ts
    stale = lag > F + 1e-9
    late = (wait + s) > B + 1e-9
    res = dict(
        peak_q=Q.max(),
        clear_s=(ts[np.where((Q > 1) )[0][-1]] + DT) if (Q > 1).any() else None,
        mean_q_s=Q.sum() * DT / A[-1] if A[-1] else 0.0,
        worker_s=sum(workers(policy, t) for t in np.arange(0, HORIZON, 1.0)),
        max_lag=lag.max(),
        stale_window_s=stale.sum() * DT,
        stale_decisions=(lam * DT)[stale].sum(),
        stale_frac=(lam * DT)[stale].sum() / (lam * DT).sum(),
        late_frac=(lam * DT)[late].sum() / (lam * DT).sum(),
    )
    return res, ts, lam, Q, lag

def stale_window_closed_form(lam0, lamb, C0, Tb, F):
    """No scaling, constant C0 in (lam0, lamb): lag during burst grows as l=(1-C0/lamb)*tau;
    after the burst, lag falls back below F once queue span < F."""
    e = lamb - C0
    t_start = F * lamb / e if e > 0 else math.inf          # lag = Q/lamb = e*tau/lamb
    Qp = e * Tb
    # after burst end: queue contains burst events then baseline events; lag decreases;
    # numerical evaluation is used in the paper; the onset formula is exact.
    return t_start, Qp

if __name__ == "__main__":
    print("Policy         peakQ     clears   meanQ s  worker-s  maxLag  staleWin  staleDec   stale%  late%")
    out = {}
    for p in ("fixed_min", "reactive", "anticipatory", "fixed_peak"):
        r, *_ = simulate(p)
        out[p] = r
        print(f"{p:13s} {r['peak_q']:9.0f} {str(round(r['clear_s'],1)) if r['clear_s'] else '-':>8s} {r['mean_q_s']:8.2f} "
              f"{r['worker_s']:8.0f} {r['max_lag']:7.1f} {r['stale_window_s']:8.1f} {r['stale_decisions']:9.0f} "
              f"{100*r['stale_frac']:6.1f} {100*r['late_frac']:6.1f}")
    # checks against the original manuscript
    assert abs(out["fixed_min"]["peak_q"] - 960000) < 200
    assert abs(out["reactive"]["peak_q"] - 240000) < 200
    assert abs(out["fixed_min"]["mean_q_s"] - 34.29) < 0.02
    assert abs(out["reactive"]["mean_q_s"] - 1.43) < 0.02
    assert out["reactive"]["worker_s"] == 7920 and out["anticipatory"]["worker_s"] == 8400
    print("onset of staleness without scaling (F=5 s):", stale_window_closed_form(6000, 18000, 10000, 120, 5.0)[0], "s after burst onset")
    # replay duplicates
    for lam, delta in ((6000, 10), (6000, 30), (6000, 60)):
        print(f"replay: lambda={lam} delta={delta}s -> expected duplicate effects without idempotency {lam*delta/2:,.0f}; "
              f"ambiguous in-flight at a=0.25 s: {lam*0.25:,.0f}")
    # fraud velocity exposure: attacker rate r, threshold k, lag l
    for l in (0, 5, 30, 120):
        print(f"fraud: lag {l:4d}s -> approved fraudulent tx ~ k-1 + r*l = {9 + 2*l}")

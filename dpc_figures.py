"""Figures 1-3 for the decision-path contract manuscript. Run after dpc_sim.py."""
import csv, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import dpc_analysis as AN

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
DARK, MID, LIGHT, ACC, GATE = "#1f3b57", "#5b7a99", "#e8eef4", "#b5532e", "#fbeee6"

# ---------------- Figure 1: architecture with contract checkpoints
fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=300); ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis("off")
def box(x, y, w, h, t, fc=LIGHT, ec=DARK, fs=7.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.1", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs)
def arr(x1, y1, x2, y2, ls="-", c=DARK):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.1, ls=ls, mutation_scale=10))
box(1, 40, 18, 10, "Enterprise\nproducers")
box(24, 40, 20, 10, "Durable log\noffsets · envelope")
box(49, 40, 22, 10, "Feature pipeline\n(stateful stream stage)")
box(77, 40, 22, 10, "Feature store\neffective + publish time")
box(77, 20, 22, 10, "C1 freshness &\ndeadline check", fc=GATE, ec=ACC)
box(49, 20, 22, 10, "Versioned model\nor approved fallback")
box(24, 20, 20, 10, "C2 idempotent\napplication (decision id)", fc=GATE, ec=ACC)
box(1, 20, 18, 10, "Business\neffect + record")
box(1, 2, 98, 10, "Coordinated control and governance\nfeature lag ℓ(t) · queue · productive capacity · deadline budget · reconciliation", fc="#dde6ef")
arr(19.6, 45, 23.4, 45); arr(44.6, 45, 48.4, 45); arr(71.6, 45, 76.4, 45); arr(88, 39.4, 88, 30.6)
arr(76.4, 25, 71.6, 25); arr(48.4, 25, 44.6, 25); arr(23.4, 25, 19.6, 25)
ax.plot([10, 10, 81], [39.4, 35, 35], color=MID, lw=1.1, ls=(0, (3, 2)))
arr(81, 35, 81, 30.6, ls=(0, (3, 2)), c=MID)
ax.text(30, 35.8, "each event also triggers a decision on the fast path", fontsize=7.2, color=MID, style="italic")
arr(74, 39.4, 74, 12.6, ls=(0, (2, 2)), c=MID); arr(34, 19.4, 34, 12.6, ls=(0, (2, 2)), c=MID)
ax.text(74.8, 15.2, "lag ℓ(t)", fontsize=7.2, color=MID, style="italic"); ax.text(35, 15.2, "ambiguous outcomes", fontsize=7.2, color=MID, style="italic")
fig.tight_layout(pad=0.2); fig.savefig("fig1.png"); plt.close(fig)

# ---------------- Figure 2: analytical feature lag
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8), dpi=300, gridspec_kw={"width_ratios": [1.4, 1]})
styles = {"fixed_min": ("Fixed minimum", "#9aa9b8", "-"), "reactive": ("Reactive (30 s delay)", DARK, "-"),
          "anticipatory": ("Anticipatory", ACC, (0, (4, 2)))}
for p, (lab, c, ls) in styles.items():
    r, ts, lam, Q, lag = AN.simulate(p)
    ax[0].plot(ts, lag, color=c, ls=ls, lw=1.6, label=lab)
    if p == "reactive":
        ax[1].plot(ts, lag, color=c, lw=1.6)
    if p == "anticipatory":
        ax[1].plot(ts, lag, color=c, ls=ls, lw=1.6)
for a in ax:
    a.axhline(5, color=ACC, lw=0.8, ls=":"); a.axvspan(60, 180, color="#eef1f4", zorder=0)
ax[0].text(430, 9, "freshness bound F = 5 s", fontsize=7.5, color=ACC)
ax[0].set_xlim(0, 600); ax[0].set_xlabel("Time (s)"); ax[0].set_ylabel("Feature lag ℓ(t) (s)")
ax[0].legend(frameon=False, fontsize=7.5, loc="upper right"); ax[0].set_title("(a) Constructed burst, all policies", fontsize=9, loc="left")
ax[1].set_xlim(50, 130); ax[1].set_ylim(-0.5, 15); ax[1].set_xlabel("Time (s)")
ax[1].set_title("(b) Reactive vs. anticipatory (zoom)", fontsize=9, loc="left")
ax[1].annotate("stale for 37.5 s", xy=(90, 13.3), xytext=(98, 12.5), fontsize=7.5, arrowprops=dict(arrowstyle="-", color=MID, lw=0.7))
fig.tight_layout(); fig.savefig("fig2.png"); plt.close(fig)

# ---------------- Figure 3: simulation
rows = list(csv.DictReader(open("sim_results.csv")))
ctrls = ["fixed_min", "utilization", "lag", "forecast_lag"]
lab = {"fixed_min": "Fixed\nminimum", "utilization": "Utilization\n(HPA-style)", "lag": "Backlog-\naware", "forecast_lag": "Forecast +\nbacklog-aware"}
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), dpi=300)
x = np.arange(len(ctrls)); w = 0.38
for k, (kind, c) in enumerate((("scheduled", MID), ("mixed", DARK))):
    m, e = [], []
    for ct in ctrls:
        v = np.array([float(r["stale_pct"]) for r in rows if r["kind"] == kind and r["ctrl"] == ct])
        m.append(v.mean()); e.append(1.96 * v.std(ddof=1) / np.sqrt(len(v)))
    ax[0].bar(x + (k - 0.5) * w, m, w, yerr=e, color=c, capsize=2, label="scheduled bursts" if kind == "scheduled" else "mixed + crash", error_kw=dict(lw=0.8))
    for xi, v, ee in zip(x + (k - 0.5) * w, m, e):
        ax[0].text(xi, v + ee + 0.5, f"{v:.1f}", ha="center", fontsize=6.6)
ax[0].set_xticks(x, [lab[c] for c in ctrls], fontsize=7.6); ax[0].set_ylabel("Decisions on stale features (%)")
ax[0].legend(frameon=False, fontsize=7.5); ax[0].set_title("(a) Silent stale decisions without a contract", fontsize=9, loc="left")
for k, (key, c, nm) in enumerate((("fraud_naive", "#9aa9b8", "Without contract"), ("fraud_contract", ACC, "With freshness contract"))):
    m = [np.mean([float(r[key]) for r in rows if r["kind"] == "mixed" and r["ctrl"] == ct]) for ct in ctrls]
    ax[1].bar(x + (k - 0.5) * w, m, w, color=c, label=nm)
    for xi, v in zip(x + (k - 0.5) * w, m):
        ax[1].text(xi, v + 2, f"{v:.0f}", ha="center", fontsize=7)
ax[1].set_xticks(x, [lab[c] for c in ctrls], fontsize=7.6); ax[1].set_ylabel("Fraudulent tx approved per attack")
ax[1].legend(frameon=False, fontsize=7.5); ax[1].set_title("(b) Velocity-attack exposure (mixed)", fontsize=9, loc="left")
fig.tight_layout(); fig.savefig("fig3.png"); plt.close(fig)
print("figures written")

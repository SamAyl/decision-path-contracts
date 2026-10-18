# Decision-Path Contracts for Real-Time AI — reproduction code

Code and data that reproduce every numerical result, table, and figure in:

> S. Ayloo, *Decision-Path Contracts for Real-Time AI: Keeping Streaming Decisions Fresh and Exactly Applied in Cloud-Native Enterprise Systems* (manuscript under submission, 2026).

The paper shows that when a feature pipeline falls behind, a fast AI serving path silently decides on stale
features, and that checkpoint recovery re-applies external effects. It proposes a decision-path contract
(freshness/deadline admissibility + idempotent application) and evaluates it analytically and in simulation.

## Quick start

```bash
pip install -r requirements.txt
python dpc_analysis.py   # Section 5: constructed burst (Table 2), replay (Table 3), fraud exposure
python dpc_sim.py        # Section 6: simulation study, 30 seeds x 2 workloads (~30 s) -> sim_results.csv
python dpc_figures.py    # Figures 1–3
```

All runs are seeded; the committed `sim_results.csv` and `figures/` are the exact outputs reported in the paper.

## Key definitions

* Feature lag: `l(t) = t - A^{-1}(D(t))`, the horizontal deviation between the feature pipeline's cumulative
  arrival curve `A` and departure curve `D`.
* Duplicate effects after recovery without idempotency: throughput x time since the last checkpoint
  (expected `X * Delta / 2`).

## Files

| File | Paper section | What it does |
|---|---|---|
| `dpc_analysis.py` | 5 | Fluid model at 0.01 s resolution; asserts the original scenario values |
| `dpc_sim.py` | 6 | Per-second simulation: 5 capacity controllers x 2 decision policies, bursts, crash, velocity attacks |
| `dpc_figures.py` | 4–6 | Figures 1–3 |
| `sim_results.csv` | Table 4 | One row per (workload, seed, controller) |

The simulator is a controlled model for comparing decision rules, not a measurement of a production system.

## License

MIT

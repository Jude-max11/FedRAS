"""WCNC grid-step sensitivity: DP objective and runtime vs the budget-grid step.

Quantifies the discretization error of the
pseudo-polynomial DP and its runtime trade-off. Reuses the IDENTICAL instance
model and dp_exact as scripts/dp_scheduler_study.py (same rng seed, same draw
order per deadline) so numbers are comparable with Table III.

Writes outputs/wcnc_extra/grid_sensitivity.csv.
"""
import csv, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dp_scheduler_study import instance, values, dp_exact

STEPS = [0.2, 0.1, 0.05, 0.025, 0.01]   # includes the paper's 0.05 MB default


def main():
    rows = []
    rng = np.random.default_rng(2026)
    finest = STEPS[-1]
    for deadline in (1.0, 0.5, 0.25):
        insts = []
        for _ in range(30):
            rate, mob, base = instance(500, 2.0, rng)
            insts.append(values(rate, mob, base, deadline))
        for budget in (300.0, 40.0, 12.0):
            per_val = {s: [] for s in STEPS}
            per_ms = {s: [] for s in STEPS}
            for U in insts:
                for s in STEPS:
                    t0 = time.perf_counter()
                    v = dp_exact(U, 25, budget, step=s)
                    per_ms[s].append((time.perf_counter() - t0) * 1e3)
                    per_val[s].append(v)
            ref = np.array(per_val[finest])
            for s in STEPS:
                v = np.array(per_val[s])
                rel = np.abs(ref - v) / np.maximum(np.abs(ref), 1e-9)
                rows.append({
                    'deadline': deadline, 'budget': budget, 'step': s,
                    'dp_obj': round(float(v.mean()), 6),
                    'rel_gap_vs_finest_mean': round(float(rel.mean()), 8),
                    'rel_gap_vs_finest_max': round(float(rel.max()), 8),
                    'dp_ms': round(float(np.mean(per_ms[s])), 1),
                })
                print(rows[-1], flush=True)
    os.makedirs('outputs/wcnc_extra', exist_ok=True)
    out = 'outputs/wcnc_extra/grid_sensitivity.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print('written', out)


if __name__ == '__main__':
    main()

"""Scheduler study: exact DP vs greedy heuristics for the per-round
client-rank assignment (multiple-choice knapsack with admission cap).

v4 (real-byte era): update sizes are the real serialized B-only uploads
(size(r) = r * 5632 / 1e6 MB); the DP, both greedy heuristics, the cohort
myopic baseline, and the fixed-rank baseline all use the SAME conservative
ceil discretization onto a 0.01 MB grid, so every method optimizes over the
identical feasible region. Budgets are given in real MB.

Instances are drawn from the same synthetic link model as the simulator:
rate ~ LogNormal(2.5,0.7) clipped to [1,40] Mbps, mobility ~ Beta(shape,5),
per-client contribution base ~ U(0.15,0.45).  Delivery probability
P(r)=exp(-mob - 8*size(r)/(rate*deadline)) and useful value
G(r)=base*sqrt(min(r,16)/16).  We report achieved objective, gap to the exact
DP optimum, and solver runtime across deadlines, budgets and client counts.
"""
import argparse, csv, math, time
import numpy as np

RANKS = np.array([2, 4, 8, 16])
RMAX = 16
BYTES_PER_RANK = 5632          # B-only, FP32 (see journal_model_audit.py)
STEP = 0.01                    # MB per budget grid unit (ceil everywhere)
SIZE = BYTES_PER_RANK * RANKS / 1e6   # real MB per rank
CAP_GAIN = np.sqrt(np.minimum(RANKS, RMAX) / RMAX)
COST = np.maximum(1, np.ceil(SIZE / STEP - 1e-9)).astype(int)  # grid units

def instance(n, shape, rng):
    rate = np.clip(rng.lognormal(2.5, 0.7, n), 1, 40)
    mob = rng.beta(shape, 5, n)
    base = rng.uniform(0.15, 0.45, n)
    return rate, mob, base

def values(rate, mob, base, deadline):
    # P: n x |R|, G: n x |R|
    P = np.exp(-mob[:, None] - 8.0 * SIZE[None, :] / (rate[:, None] * deadline))
    G = base[:, None] * CAP_GAIN[None, :]
    return G * P

def dp_exact(U, K, budget, step=STEP):
    n = U.shape[0]
    B = max(1, int(math.ceil(budget / step - 1e-9)))
    sizes = np.maximum(1, np.ceil(SIZE / step - 1e-9)).astype(int)
    NEG = -1e18
    V = np.full((K + 1, B + 1), NEG)
    V[0, 0] = 0.0
    for i in range(n):
        W = np.full_like(V, NEG)
        for j, r in enumerate(RANKS):
            s = sizes[j]
            if s > B:
                continue
            cand = V[:-1, :B - s + 1] + U[i, j]
            np.maximum(W[1:, s:], cand, out=W[1:, s:])  # first max-wins tie
        V = np.maximum(V, W)
    return float(V.max())

def greedy(U, K, budget, per_byte, step=STEP):
    # Same ceil discretization as the DP: identical feasible region.
    n = U.shape[0]
    Bu = max(1, int(math.ceil(budget / step - 1e-9)))
    costs = np.maximum(1, np.ceil(SIZE / step - 1e-9)).astype(int)
    key = U.max(axis=1) if not per_byte else (U / SIZE[None, :]).max(axis=1)
    ridx = np.argmax(U / SIZE[None, :] if per_byte else U, axis=1)
    order = np.argsort(-key)
    tot, used_u, cnt = 0.0, 0, 0
    for i in order:
        c = costs[ridx[i]]
        if cnt >= K or used_u + c > Bu:
            continue
        tot += U[i, ridx[i]]; used_u += c; cnt += 1
    return tot

def cohort_myopic(U, K, budget, rng, step=STEP):
    """Trainer protocol: uniform cohort sample, per-client argmax, admit by value/byte."""
    n = U.shape[0]
    Bu = max(1, int(math.ceil(budget / step - 1e-9)))
    costs = np.maximum(1, np.ceil(SIZE / step - 1e-9)).astype(int)
    cohort = rng.choice(n, min(K, n), replace=False)
    ridx = np.argmax(U, axis=1)
    key = (U / SIZE[None, :])[cohort, ridx[cohort]]
    order = cohort[np.argsort(-key)]
    tot, used_u = 0.0, 0
    for i in order:
        c = costs[ridx[i]]
        if used_u + c > Bu:
            continue
        tot += U[i, ridx[i]]; used_u += c
    return tot

def fixed_rank(U, r_idx, K, budget, step=STEP):
    n = U.shape[0]
    Bu = max(1, int(math.ceil(budget / step - 1e-9)))
    c = max(1, int(math.ceil(SIZE[r_idx] / step - 1e-9)))
    order = np.argsort(-U[:, r_idx])[:K]
    tot = used = 0
    for i in order:
        if used + c > Bu:
            continue
        tot += U[i, r_idx]; used += c
    return tot

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='outputs/dp_scheduler_study_v4.csv')
    p.add_argument('--grid-sweep', action='store_true',
                   help='add the budget-grid sensitivity sweep (steps 0.02-0.005 MB vs a 0.0025 reference)')
    a = p.parse_args()
    rng = np.random.default_rng(2026)
    rows = []
    # main grid: deadlines x budgets (real MB; 1.2 MB binds at ~13 rank-16 of K=25)
    for deadline in (2.0, 1.0, 0.1):
        for budget in (30.0, 4.0, 1.2):
            res = {m: [] for m in ('DP', 'greedy_vb', 'greedy_v', 'myopic', 'r16')}
            t_dp = []
            for _ in range(30):
                rate, mob, base = instance(500, 2.0, rng)
                U = values(rate, mob, base, deadline)
                t0 = time.perf_counter(); v = dp_exact(U, 25, budget); t_dp.append((time.perf_counter() - t0) * 1e3)
                res['DP'].append(v)
                res['greedy_vb'].append(greedy(U, 25, budget, True))
                res['greedy_v'].append(greedy(U, 25, budget, False))
                res['myopic'].append(cohort_myopic(U, 25, budget, rng))
                res['r16'].append(fixed_rank(U, 3, 25, budget))
            row = {'deadline': deadline, 'budget': budget, 'clients': 500}
            for m, v in res.items():
                row[m] = float(np.mean(v))
                row[m + '_gap'] = float(np.mean((np.array(res['DP']) - np.array(v)) / np.maximum(res['DP'], 1e-9)))
            row['dp_ms'] = float(np.mean(t_dp))
            rows.append(row)
            print({k: (round(x, 4) if isinstance(x, float) else x) for k, x in row.items()}, flush=True)
    # scalability of the DP
    for n in (100, 250, 500, 1000):
        ts = []
        for _ in range(10):
            rate, mob, base = instance(n, 2.0, rng)
            U = values(rate, mob, base, 2.0)
            t0 = time.perf_counter(); dp_exact(U, 25, 30.0); ts.append((time.perf_counter() - t0) * 1e3)
        rows.append({'deadline': 2.0, 'budget': 30.0, 'clients': n, 'dp_ms': float(np.mean(ts))})
        print('scale', n, round(float(np.mean(ts)), 1), 'ms', flush=True)
    if a.grid_sweep:
        # budget-grid sensitivity: binding 1.2 MB and slack 30 MB regimes,
        # each step vs the 0.0025 MB reference, 30 instances per cell.
        for budget in (1.2, 30.0):
            ref = {dl: [] for dl in (2.0, 1.0, 0.1)}
            for deadline in (2.0, 1.0, 0.1):
                for _ in range(30):
                    rate, mob, base = instance(500, 2.0, rng)
                    ref[deadline].append(dp_exact(values(rate, mob, base, deadline), 25, budget, step=0.0025))
            for step in (0.02, 0.01, 0.005):
                for deadline in (2.0, 1.0, 0.1):
                    gaps = []
                    for v_ref in ref[deadline]:
                        rate, mob, base = instance(500, 2.0, rng)
                        v = dp_exact(values(rate, mob, base, deadline), 25, budget, step=step)
                        gaps.append((v_ref - v) / max(v_ref, 1e-9))
                    rows.append({'deadline': deadline, 'budget': budget, 'clients': 500,
                                 'grid_step': step, 'objective_gap': float(np.mean(gaps))})
                    print('grid', budget, step, deadline, round(float(np.mean(gaps)), 4), flush=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r})); w.writeheader(); w.writerows(rows)
    print('written', a.out)

if __name__ == '__main__':
    main()

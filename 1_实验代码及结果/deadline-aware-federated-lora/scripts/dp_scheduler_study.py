"""Scheduler study: exact DP vs greedy heuristics for the per-round
client-rank assignment (multiple-choice knapsack with admission cap).

Instances are drawn from the same link model as the trace-driven simulator:
rate ~ LogNormal(2.5,0.7) clipped to [1,40] Mbps, mobility ~ Beta(shape,5),
per-client contribution base ~ U(0.15,0.45).  Update size s(r)=0.02r+0.5 MB,
delivery probability P(r)=exp(-mob - 8*s(r)/(rate*deadline)) and useful value
G(r)=base*sqrt(min(r,16)/16).  We report achieved objective, gap to the exact
DP optimum, and solver runtime across deadlines, budgets and client counts.
"""
import csv, time
import numpy as np

RANKS = np.array([2, 4, 8, 16])
RMAX = 16
SIZE = 0.02 * RANKS + 0.5          # MB
CAP = np.sqrt(np.minimum(RANKS, RMAX) / RMAX)

def instance(n, shape, rng):
    rate = np.clip(rng.lognormal(2.5, 0.7, n), 1, 40)
    mob = rng.beta(shape, 5, n)
    base = rng.uniform(0.15, 0.45, n)
    return rate, mob, base

def values(rate, mob, base, deadline):
    # P: n x |R|, G: n x |R|
    P = np.exp(-mob[:, None] - 8.0 * SIZE[None, :] / (rate[:, None] * deadline))
    G = base[:, None] * CAP[None, :]
    return G * P

def dp_exact(U, K, budget, step=0.05):
    n = U.shape[0]
    B = int(round(budget / step))
    sizes = np.maximum(1, np.round(SIZE / step).astype(int))
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

def greedy(U, K, budget, per_byte):
    n = U.shape[0]
    key = U.max(axis=1) if not per_byte else (U / SIZE[None, :]).max(axis=1)
    ridx = np.argmax(U / SIZE[None, :] if per_byte else U, axis=1)
    order = np.argsort(-key)
    tot, used, cnt = 0.0, 0.0, 0
    for i in order:
        if cnt >= K or used + SIZE[ridx[i]] > budget:
            continue
        tot += U[i, ridx[i]]; used += SIZE[ridx[i]]; cnt += 1
    return tot

def cohort_myopic(U, K, budget, rng):
    """Trainer protocol: uniform cohort sample, per-client argmax, admit by value/byte."""
    n = U.shape[0]
    cohort = rng.choice(n, min(K, n), replace=False)
    ridx = np.argmax(U, axis=1)
    key = (U / SIZE[None, :])[cohort, ridx[cohort]]
    order = cohort[np.argsort(-key)]
    tot, used = 0.0, 0.0
    for i in order:
        if used + SIZE[ridx[i]] > budget:
            continue
        tot += U[i, ridx[i]]; used += SIZE[ridx[i]]
    return tot

def fixed_rank(U, r_idx, K, budget):
    n = U.shape[0]
    order = np.argsort(-U[:, r_idx])[:K]
    tot = used = 0.0
    for i in order:
        if used + SIZE[r_idx] > budget:
            continue
        tot += U[i, r_idx]; used += SIZE[r_idx]
    return tot

def main():
    rows = []
    rng = np.random.default_rng(2026)
    # main grid: deadlines x budgets
    for deadline in (1.0, 0.5, 0.25):
        for budget in (300.0, 40.0, 12.0):
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
            U = values(rate, mob, base, 1.0)
            t0 = time.perf_counter(); dp_exact(U, 25, 300.0); ts.append((time.perf_counter() - t0) * 1e3)
        rows.append({'deadline': 1.0, 'budget': 300.0, 'clients': n, 'dp_ms': float(np.mean(ts))})
        print('scale', n, round(float(np.mean(ts)), 1), 'ms', flush=True)
    with open('outputs/dp_scheduler_study.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print('written outputs/dp_scheduler_study.csv')

if __name__ == '__main__':
    main()

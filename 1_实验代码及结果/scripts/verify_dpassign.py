#!/usr/bin/env python3
"""Independent exhaustive checker for the trainer's dp_assign: random small
instances, enumerate every admissible (admission, rank) profile, compare the
objective with the DP output. Mirrors the DP's own utility/cost formulas."""
import itertools
import math
import sys

sys.path.insert(0, 'scripts')
from train_torch_fedlora import dp_assign, budget_units, size_mb, BUDGET_GRID  # noqa: E402

BYTES_PER_RANK = 5632
RANKS = [2, 4, 8, 16]
RMAX = 16


def brute(mob, rate, deadline, ids, cap, budget):
    Bu_cap = math.floor(budget / BUDGET_GRID + 1e-9)
    best, best_prof = -1.0, None
    for prof in itertools.product([0] + RANKS, repeat=len(ids)):
        adm = [r for r in prof if r > 0]
        if len(adm) > cap:
            continue
        cost = sum(budget_units(size_mb(r, BYTES_PER_RANK)) for r in adm)
        if cost > Bu_cap:
            continue
        obj = 0.0
        for i, r in zip(ids, prof):
            if r == 0:
                continue
            obj += (math.exp(-mob[i]) * math.exp(-8.0 * size_mb(r, BYTES_PER_RANK) / (rate[i] * deadline))
                    * math.sqrt(min(r, RMAX) / RMAX))
        if obj > best:
            best, best_prof = obj, dict(zip(ids, prof))
    return {k: v for k, v in best_prof.items() if v > 0}, best


def dp_obj(assign, mob, rate, deadline):
    return sum(math.exp(-mob[i]) * math.exp(-8.0 * size_mb(r, BYTES_PER_RANK) / (rate[i] * deadline))
               * math.sqrt(min(r, RMAX) / RMAX) for i, r in assign.items())


def main(trials=60, seed=7):
    import numpy as np
    g = np.random.default_rng(seed)
    worst = 0.0
    for t in range(trials):
        n = int(g.integers(4, 8))
        ids = list(range(n))
        mob = g.beta(2, 5, n)
        rate = np.clip(g.lognormal(2.5, .7, n), 1, 40)
        deadline = float(g.uniform(0.05, 2.0))
        cap = int(g.integers(2, n + 1))
        budget = float(g.choice([0.2, 0.4, 0.6, 1.0]))
        a = dp_assign(mob, rate, deadline, RANKS, RMAX, ids, cap, budget, BYTES_PER_RANK)
        b, bobj = brute(mob, rate, deadline, ids, cap, budget)
        aobj = dp_obj(a, mob, rate, deadline)
        gap = bobj - aobj
        worst = max(worst, gap)
        assert gap < 1e-9, f'trial {t}: brute {bobj:.6f} > dp {aobj:.6f}\nbrute={b}\ndp={a}'
        # identical objective AND identical profile (uniqueness holds: costs and
        # utilities are distinct with probability 1 under continuous draws)
        assert a == b, f'trial {t}: profiles differ\nbrute={b}\ndp={a}'
    print(f'dp_assign verification passed: {trials} trials, worst objective gap {worst:.2e}, '
          f'all profiles identical to exhaustive optimum')


if __name__ == '__main__':
    main()

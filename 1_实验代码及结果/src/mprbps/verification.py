from __future__ import annotations

import numpy as np

from .model import ActionEstimate, Policy
from .solvers import brute_force, solve_gcdp, solve_greedy_repair


def verify_gcdp(trials: int = 30, seed: int = 13) -> None:
    rng = np.random.default_rng(seed)
    policies = [Policy("n", 0, 32, secagg=False), Policy("s", 0.5, 8, secagg=True)]
    worst = 0.0
    for trial in range(trials):
        n, k, m = 6, 4, 2
        grid = {}
        for h in [0] + list(range(m, k + 1)):
            grid[h] = {}
            for cid in range(n):
                actions = []
                for policy in policies:
                    value = float(rng.uniform(-0.2, 1.0) - (0.03 * h if policy.secagg else 0))
                    feasible = not policy.secagg or h >= m
                    actions.append(ActionEstimate(cid, policy, h, value, 0.8, 1.0, 0.2, 0.5, 0.3, 0.1, 100, feasible))
                grid[h][cid] = actions
        exact = brute_force(grid, k, m)
        gcdp = solve_gcdp(grid, k, m)
        gap = abs(exact.objective - gcdp.objective)
        worst = max(worst, gap)
        if gap > 1e-10:
            raise AssertionError(f"trial={trial}: exhaustive={exact.objective}, gcdp={gcdp.objective}")
        greedy = solve_greedy_repair(grid, k, m)
        if greedy.objective > gcdp.objective + 1e-10:
            raise AssertionError(f"trial={trial}: greedy={greedy.objective} exceeds GCDP={gcdp.objective}")
    print(f"GCDP verification passed: {trials} trials, maximum objective gap={worst:.3e}")

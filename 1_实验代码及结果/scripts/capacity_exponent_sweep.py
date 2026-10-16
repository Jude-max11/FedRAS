#!/usr/bin/env python3
"""Capacity-exponent sensitivity: how do the DP optimum, the greedy gap, and
the chosen rank distribution change when the capacity exponent deviates from
the assumed 1/2? 500-client instances, binding and slack budgets, 5 seeds.
Writes outputs/journal_stats/capacity_exponent_sweep.csv."""
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, 'scripts')
from train_torch_fedlora import dp_assign, greedy_assign  # noqa: E402

BYTES_PER_RANK, RANKS, RMAX = 5632, [2, 4, 8, 16], 16
OUT = 'outputs/journal_stats'


def objective(assign, mob, rate, deadline, cap_exp):
    tot = 0.0
    for i, r in assign.items():
        tot += (math.exp(-mob[i]) * math.exp(-8.0 * (r * BYTES_PER_RANK / 1e6) / (rate[i] * deadline))
                * (min(r, RMAX) / RMAX) ** cap_exp)
    return tot


def main():
    os.makedirs(OUT, exist_ok=True)
    g = np.random.default_rng(11)
    rows = []
    for cap_exp in (0.25, 0.5, 0.75, 1.0):
        for budget, tag in ((1.2, 'binding'), (30.0, 'slack')):
            for seed in range(5):
                rng = np.random.default_rng(1000 * seed + int(cap_exp * 100))
                n, cap, deadline = 500, 25, 0.1
                mob = rng.beta(2, 5, n)
                rate = np.clip(rng.lognormal(2.5, 0.7, n), 1, 40)
                ids = list(range(n))
                kw = dict(cap_exp=cap_exp)
                dp = dp_assign(mob, rate, deadline, RANKS, RMAX, ids, cap, budget, BYTES_PER_RANK, **kw)
                gv = greedy_assign(mob, rate, deadline, RANKS, RMAX, ids, cap, budget, BYTES_PER_RANK, **kw)
                gp = greedy_assign(mob, rate, deadline, RANKS, RMAX, ids, cap, budget, BYTES_PER_RANK,
                                   per_byte=True, **kw)
                # fixed rank-16 rule, made budget-feasible: admit in utility order
                # while the ceil-grid budget holds (an infeasible all-16 profile
                # would not be a valid comparator under a binding budget)
                Bu_cap = math.floor(budget / 0.01 + 1e-9)
                f16 = {}
                used = 0
                order = sorted(ids, key=lambda j: -math.exp(-mob[j]) * rate[j])
                for j in order:
                    c = math.ceil(16 * BYTES_PER_RANK / 1e6 / 0.01 - 1e-9)
                    if len(f16) >= cap or used + c > Bu_cap:
                        break
                    f16[j] = 16; used += c
                o_dp = objective(dp, mob, rate, deadline, cap_exp)
                rows.append(dict(
                    cap_exp=cap_exp, budget=tag, seed=seed,
                    dp_obj=o_dp,
                    greedy_value_gap=1 - objective(gv, mob, rate, deadline, cap_exp) / o_dp,
                    greedy_vpb_gap=1 - objective(gp, mob, rate, deadline, cap_exp) / o_dp,
                    fixed16_gap=1 - objective(f16, mob, rate, deadline, cap_exp) / o_dp,
                    dp_mean_rank=float(np.mean(list(dp.values()))),
                    dp_admitted=len(dp)))
    with open(f'{OUT}/capacity_exponent_sweep.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    # summary
    print(f'{"exp":>5} {"budget":>8} {"val-greedy gap":>14} {"vpb gap":>9} {"fixed16 gap":>12} {"DP mean rank":>12} {"adm":>5}')
    for cap_exp in (0.25, 0.5, 0.75, 1.0):
        for tag in ('binding', 'slack'):
            rs = [r for r in rows if r['cap_exp'] == cap_exp and r['budget'] == tag]
            m = lambda k: sum(r[k] for r in rs) / len(rs)
            print(f'{cap_exp:>5} {tag:>8} {m("greedy_value_gap")*100:>13.1f}% {m("greedy_vpb_gap")*100:>8.1f}% '
                  f'{m("fixed16_gap")*100:>11.1f}% {m("dp_mean_rank"):>12.2f} {m("dp_admitted"):>5.1f}')


if __name__ == '__main__':
    main()

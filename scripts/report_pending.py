#!/usr/bin/env python3
"""Aggregate the Table-5(d) top-up and the 6-epoch estimation-error arm."""
import csv, glob, os, statistics as st, sys
sys.path.insert(0, '/root/autodl-tmp/FedRAS/scripts')
from collect_v6 import stats  # noqa: E402

R = '/root/autodl-tmp/FedRAS/outputs/real_v6'
L5 = {'FedLoRA_r4': 'FedLoRA-4', 'FedLoRA_r16': 'FedLoRA-16', 'Random_rank': 'Random',
      'Adaptive_rank': 'FedRAS', 'SA_mono': 'SA-FedLoRA', 'Oort_utility': 'Oort',
      'ROBA_style': 'EL'}
BASE = [('FedLoRA_r4', 'FedLoRA-4'), ('FedLoRA_r16', 'FedLoRA-16'),
        ('Random_rank', 'Random'), ('Adaptive_rank', 'FedRAS')]
TOP = [('SA_mono', 'SA-FedLoRA'), ('Oort_utility', 'Oort'), ('ROBA_style', 'EL')]


def acc(tag, method):
    v = []
    for d in sorted(glob.glob(f'{R}/{tag}_seed*')):
        s = stats(d, method)
        if s:
            v.append(s['acc'])
    return (st.mean(v), len(v)) if v else (float('nan'), 0)


print('=== Table 5(d): 4 methods already in the paper, 3 new ones in parentheses ===')
SC = [('基准 2.0s', None), ('截止 10s', 'dl10.0'), ('移动性 0.7', 'mob0.7'),
      ('alpha 0.1', 'alpha0.1'), ('alpha 1.0', 'alpha1.0')]
for ds, dsn in (('cifar10', 'CIFAR-10'), ('cifar100', 'CIFAR-100'), ('mnist', 'MNIST')):
    print(f'-- {dsn}')
    for lab, sc in SC:
        row = {}
        for key, name in BASE:
            pat = f'p2_tab5d_{ds}_{sc}' if sc else f'p1_tab3_{ds}_dl2.0'
            row[name], n = acc(pat, key)
        for key, name in TOP:
            pat = f'p2_tab5df_{ds}_{sc}' if sc else None
            row[name] = acc(pat, key)[0] if pat else float('nan')
        vals = '  '.join(f'{k}={row[k]:.4f}' for k in
                         ['FedLoRA-4', 'FedLoRA-16', 'Random', 'FedRAS', 'SA-FedLoRA', 'Oort', 'EL'])
        best = max(row, key=lambda k: row[k] if row[k] == row[k] else -1)
        print(f'   {lab:10s} {vals}   max={best}')

print()
print('=== estimation-error arm (6 epoch), cifar10 ===')
REF = {'2.0': acc('p1_tab3_cifar10_dl2.0', 'Adaptive_rank')[0],
       '0.1': acc('p1_tab3_cifar10_dl0.1', 'Adaptive_rank')[0]}
def full(tag):
    v = [s for s in (stats(d, 'Adaptive_rank') for d in sorted(glob.glob(f'{R}/{tag}_seed*'))) if s]
    if not v:
        return None
    return (len(v), st.mean([x['acc'] for x in v]), st.mean([x['timely'] for x in v]),
            st.mean([x['mb'] for x in v]), st.mean([x['rank'] for x in v]),
            st.mean([x['kappa'] for x in v]))
print(f'{"arm":22s} {"n":>3s} {"acc":>7s} {"timely":>7s} {"MB":>6s} {"rank":>6s} {"cap":>6s}')
for dl in ('2.0', '0.1'):
    print(f'-- baseline FedRAS @ {dl}s (Table 3)')
    b = full(f'p1_tab3_cifar10_dl{dl}')
    print(f'{"baseline":22s} {b[0]:3d} {b[1]:7.4f} {b[2]:7.3f} {b[3]:6.2f} {b[4]:6.2f} {b[5]:6.3f}')
    for lab in ('mu0.5', 'mu2.0', 'rate-0.3', 'rate-0.15', 'rate0.15', 'rate0.3'):
        f = full(f'p6b_{lab}_dl{dl}')
        if f:
            print(f'{lab:22s} {f[0]:3d} {f[1]:7.4f} {f[2]:7.3f} {f[3]:6.2f} {f[4]:6.2f} {f[5]:6.3f}')

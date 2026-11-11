#!/usr/bin/env python3
"""Emit Table 3 rows in the Chinese manuscript's LaTeX format.

Usage: python scripts/emit_tab3_zh.py [root]
Prints one block per dataset; a cell block is emitted only when all 10 seeds
of that dataset/deadline pair have finished (>= window rounds).
Also prints a diagnostic table with kappa spread to stderr.
"""
import glob, os, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_v6 import stats  # noqa: E402

ORDER = [('FedLoRA_r4', 'FedLoRA-4'), ('FedLoRA_r16', 'FedLoRA-16'),
         ('Random_rank', 'Random'), ('SA_mono', r'\mbox{SA-FedLoRA}'),
         ('Oort_utility', 'Oort'), ('ROBA_style', 'EL'),
         ('Adaptive_rank', 'FedRAS')]
DS = [('cifar10', 'CIFAR-10'), ('cifar100', 'CIFAR-100'), ('mnist', 'MNIST')]
DLS = ['2.0', '0.1', '0.05']


def acc_cell(v):
    a = st.mean([x['acc'] for x in v])
    sd = st.stdev([x['acc'] for x in v]) if len(v) > 1 else 0.0
    return f"{a:.4f}"[1:] + r"$\pm$" + f"{sd:.4f}"[1:]


def cell(v):
    return (acc_cell(v), f"{st.mean([x['timely'] for x in v]):.3f}",
            f"{st.mean([x['mb'] for x in v]):.2f}",
            f"{st.mean([x['rank'] for x in v]):.2f}",
            f"{st.mean([x['kappa'] for x in v]):.3f}")


root = sys.argv[1] if len(sys.argv) > 1 else 'outputs/real_v6'
for ds, lab in DS:
    data, ns = {}, {}
    for m, _ in ORDER:
        row, nr = [], []
        for dl in DLS:
            dirs = sorted(glob.glob(os.path.join(root, f'p1_tab3_{ds}_dl{dl}_seed*')))
            v = [s for s in (stats(d, m) for d in dirs) if s]
            nr.append(len(v))
            row.append(cell(v) if len(v) >= 10 else None)
        data[m], ns[m] = row, nr
    if all(all(r is None for r in data[m]) for m, _ in ORDER):
        print(f"% {lab}: no complete block yet (n={ns[ORDER[0][0]]})")
        continue
    print(rf"\SetCell[c=16]{{l}} \textbf{{{lab}}} & & & & & & & & & & & & & & & \\")
    for m, mlab in ORDER:
        parts = []
        for i in range(3):
            c = data[m][i]
            parts.append(' & '.join(c) if c else r'-- & & & & ')
        print(f"{mlab} & " + ' & '.join(parts) + r"\\")
    print()
    print(f"% n per deadline {ds}: {ns['Adaptive_rank']}", file=sys.stderr)
    print(f"% kappa mean+-sd {ds}", file=sys.stderr)
    for m, mlab in ORDER:
        ks = []
        for dl in DLS:
            dirs = sorted(glob.glob(os.path.join(root, f'p1_tab3_{ds}_dl{dl}_seed*')))
            v = [s for s in (stats(d, m) for d in dirs) if s]
            ks.append(st.stdev([x['kappa'] for x in v]) if len(v) > 1 else float('nan'))
        print(f"%   {mlab:14s} kappa sd " + ' '.join(f"{k:.4f}" for k in ks), file=sys.stderr)

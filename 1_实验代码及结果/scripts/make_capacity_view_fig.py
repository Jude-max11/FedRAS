"""Capacity-view figure: accuracy vs round and accuracy vs cumulative
delivered capacity (dc = aggregated x update_norm, the single currency).

Mirrors make_shift_adapt_fig.py's CLI and data flow but restyles to the
submission COL contract and swaps the cost axis from attempted bytes to
cumulative measured delivered capacity. A policy's curve ending further
left has reached the same accuracy for less delivered-capacity spend; at
stress, curves that stall on the backbone line are the zero-communication
control realized.

Usage: make_capacity_view_fig.py OUTPREFIX OUTPDF [--methods m1,m2]
"""
import csv
import glob
import json
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt

MM = 1 / 25.4
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'pdf.fonttype': 42, 'svg.fonttype': 'none', 'font.size': 7,
    'axes.labelsize': 7, 'axes.spines.right': False, 'axes.spines.top': False,
    'axes.linewidth': 0.8, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    'legend.frameon': False, 'legend.fontsize': 6.2, 'lines.linewidth': 1.4,
})
COL = {'FedAvg_full': '#7f7f7f', 'FedLoRA_r4': '#2ca02c', 'FedLoRA_r16': '#1f77b4',
       'Random_rank': '#9467bd', 'SA_rank': '#ff7f0e', 'Adaptive_rank': '#d62728',
       'Oort_utility': '#e377c2', 'JointAB': '#7f7f7f'}
LBL = {'FedAvg_full': 'FedAvg (full model)', 'FedLoRA_r4': 'fixed $r{=}4$',
       'FedLoRA_r16': 'fixed $r{=}16$', 'Random_rank': 'Random rank',
       'SA_rank': 'SA-anneal', 'Adaptive_rank': 'FedRAS (ours)',
       'Oort_utility': 'Oort-style', 'JointAB': 'Joint A/B'}


def load(prefix, methods):
    by, bb = {}, []
    for d in sorted(glob.glob(prefix + '_seed*')):
        cfg = os.path.join(d, 'config.json')
        if not os.path.exists(cfg):
            continue
        c = json.load(open(cfg))
        bb.append(c.get('backbone_acc'))
        for r in csv.DictReader(open(os.path.join(d, 'metrics.csv'))):
            if r['method'] not in methods:
                continue
            dc = (float(r['aggregated']) * float(r['update_norm'])
                  if r['aggregated'] and r['update_norm'] else 0.0)
            by.setdefault(r['method'], {}).setdefault(int(r['round']), []).append(
                (float(r['accuracy']), dc))
    curves = {}
    for m, rnds in by.items():
        xs = sorted(rnds)
        acc = [sum(a for a, _ in rnds[x]) / len(rnds[x]) for x in xs]
        dc = [sum(d for _, d in rnds[x]) / len(rnds[x]) for x in xs]
        curves[m] = (xs, acc, dc)
    return curves, (sum(bb) / len(bb) if bb else None)


def main():
    prefix, out = sys.argv[1], sys.argv[2]
    methods = (sys.argv[3].split(',') if len(sys.argv) > 3 and sys.argv[3] != '--methods' else
               ['FedAvg_full', 'FedLoRA_r4', 'FedLoRA_r16', 'Adaptive_rank'])
    if '--methods' in sys.argv:
        methods = sys.argv[sys.argv.index('--methods') + 1].split(',')
    curves, backbone = load(prefix, methods)
    fig, (a, b) = plt.subplots(1, 2, figsize=(130 * MM, 48 * MM))
    fig.subplots_adjust(left=0.09, right=0.982, bottom=0.19, top=0.90, wspace=0.34)
    marks = {'FedAvg_full': '.', 'FedLoRA_r4': 's', 'FedLoRA_r16': '^',
             'Random_rank': 'v', 'SA_rank': 'D', 'Adaptive_rank': 'o',
             'Oort_utility': 'P', 'JointAB': 'X'}
    for m, (xs, acc, dc) in sorted(curves.items()):
        cum = [0.0]
        for v in dc:
            cum.append(cum[-1] + v)
        mk = dict(marker=marks.get(m, 'o'), ms=2.6, markevery=6, mew=0.4)
        a.plot(xs, acc, label=LBL.get(m, m), color=COL.get(m), lw=1.3,
               zorder=3 if m == 'Adaptive_rank' else 2, **mk)
        b.plot(cum, [acc[0]] + acc, label=LBL.get(m, m), color=COL.get(m), lw=1.3,
               zorder=3 if m == 'Adaptive_rank' else 2, **mk)
    if backbone is not None:
        for ax in (a, b):
            ax.axhline(backbone, color='k', linestyle=':', linewidth=0.9, zorder=1)
        a.annotate('zero-communication control', (0.02, backbone + 0.006), fontsize=6)
    a.set_xlabel('federated round')
    a.set_ylabel('test accuracy')
    b.set_xlabel('cumulative delivered capacity')
    b.set_ylabel('test accuracy')
    b.legend(loc='lower right', handlelength=1.4, borderaxespad=0.3,
             frameon=True, framealpha=0.9, edgecolor='none')
    for ax, tag in zip((a, b), 'ab'):
        ax.text(-0.28, 1.08, f'({tag})', transform=ax.transAxes,
                fontsize=8, fontweight='bold', ha='left', va='bottom')
    fig.savefig(out)
    print('wrote', out, 'methods:', sorted(curves))


if __name__ == '__main__':
    main()

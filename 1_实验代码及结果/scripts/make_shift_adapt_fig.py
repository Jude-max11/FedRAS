#!/usr/bin/env python3
"""Adaptive-task figure: (a) accuracy vs round with the zero-communication
control, (b) accuracy vs cumulative uplink bytes (bytes-to-target curves).
Reads run dirs of the form <prefix>_seed*/metrics.csv plus each run's
config.json backbone_acc as the zero-comm line.

Usage: make_shift_adapt_fig.py OUTPREFIX OUTPDF [--methods m1,m2]"""
import csv
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PAPER = dict(linewidth=1.4)
plt.rcParams.update({'font.size': 8, 'axes.linewidth': 0.6, 'legend.fontsize': 7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7})
COLORS = {'FedAvg_full': '#888888', 'FedLoRA_r4': '#4c72b0', 'FedLoRA_r16': '#dd8452',
          'Random_rank': '#937860', 'Adaptive_rank': '#c44e52', 'SA_rank': '#8172b3',
          'SA_mono': '#937860', 'Rate_thresh': '#8c8c8c', 'Oort_utility': '#e377c2',
          'JointAB': '#bcbcbc'}
LABELS = {'FedAvg_full': 'Full-model FedAvg', 'FedLoRA_r4': 'Fixed r=4', 'FedLoRA_r16': 'Fixed r=16',
          'Random_rank': 'Random rank', 'Adaptive_rank': 'FedRAS', 'SA_rank': 'Annealed (heat-cool)',
          'SA_mono': 'Annealed (monotone)', 'Rate_thresh': 'Rate threshold', 'Oort_utility': 'Oort-style',
          'JointAB': 'Joint A/B'}


def load(prefix, methods):
    dirs = sorted(glob.glob(prefix + '_seed*'))
    by = {}
    bb = []
    for d in dirs:
        cfg = os.path.join(d, 'config.json')
        if not os.path.exists(cfg):
            continue
        c = json.load(open(cfg))
        bb.append(c.get('backbone_acc'))
        for r in csv.DictReader(open(os.path.join(d, 'metrics.csv'))):
            if r['method'] in methods:
                by.setdefault(r['method'], {}).setdefault(int(r['round']), []).append(
                    (float(r['accuracy']), float(r['bytes_mb'])))
    curves = {}
    for m, rnds in by.items():
        xs = sorted(rnds)
        acc = [sum(a for a, _ in rnds[x]) / len(rnds[x]) for x in xs]
        mb = [sum(b for _, b in rnds[x]) / len(rnds[x]) for x in xs]
        curves[m] = (xs, acc, mb)
    return curves, (sum(bb) / len(bb) if bb else None)


def main():
    prefix, out = sys.argv[1], sys.argv[2]
    methods = (sys.argv[3].split(',') if len(sys.argv) > 3 and sys.argv[3] != '--methods' else
               ['FedAvg_full', 'FedLoRA_r4', 'FedLoRA_r16', 'Adaptive_rank'])
    if '--methods' in sys.argv:
        methods = sys.argv[sys.argv.index('--methods') + 1].split(',')
    curves, backbone = load(prefix, methods)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.5), constrained_layout=True)
    for m, (xs, acc, mb) in sorted(curves.items()):
        a.plot(xs, acc, label=LABELS.get(m, m), color=COLORS.get(m, None), **PAPER)
        b.plot([0] + list(__import__('itertools').accumulate(mb)),
               [acc[0]] + acc, label=LABELS.get(m, m), color=COLORS.get(m, None), **PAPER)
    if backbone is not None:
        a.axhline(backbone, color='k', linestyle=':', linewidth=1.0)
        a.annotate('zero-communication control', (0.02, backbone + 0.004), fontsize=6.5)
        b.axhline(backbone, color='k', linestyle=':', linewidth=1.0)
    a.set_xlabel('federated round'); a.set_ylabel('test accuracy')
    b.set_xlabel('cumulative uplink bytes (MB)'); b.set_ylabel('test accuracy')
    a.legend(frameon=False, loc='lower right'); b.legend(frameon=False, loc='lower right')
    fig.savefig(out)
    print('wrote', out, 'methods:', sorted(curves))


if __name__ == '__main__':
    main()

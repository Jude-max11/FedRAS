"""Delivered-update norm vs rank (Fig. normlaw, single column, 1 panel).

Two series (nominal 2.0 s, stress 0.1 s), ranks {2,4,8,16}, five seeds each,
with a least-squares sqrt fit c*sqrt(r) overlaid. Style contract matches
make_v4_figs_submission.py.
"""
import csv
import glob
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

SKILL_SCRIPTS = os.environ.get(
    'NATURE_FIGURE_SCRIPTS', os.path.expanduser('~/.codex/skills/nature-figure/scripts'))
if os.path.isdir(SKILL_SCRIPTS):
    sys.path.insert(0, SKILL_SCRIPTS)
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(ROOT), '3_ComputerNetworks期刊latex版本',
                   'CN_package_v4', 'recolored_figures')
os.makedirs(OUT, exist_ok=True)
MM = 1 / 25.4
SINGLE = 90 * MM

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'pdf.fonttype': 42, 'svg.fonttype': 'none',
    'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7.5,
    'axes.spines.right': False, 'axes.spines.top': False,
    'axes.linewidth': 0.8, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'legend.frameon': False, 'legend.fontsize': 6.2,
})
RANKS = [2, 4, 8, 16]
SOURCES = {('nominal', 2): 'nrm_r2_seed*', ('nominal', 4): 'cifar10_seed*',
           ('nominal', 8): 'nrm_r8_seed*', ('nominal', 16): 'cifar10_seed*',
           ('stress', 2): 'nrm_r2_dl0.1_seed*', ('stress', 4): 'cifar10_dl0.1_seed*',
           ('stress', 8): 'nrm_r8_dl0.1_seed*', ('stress', 16): 'cifar10_dl0.1_seed*'}


def norm_of(rundir_glob, method):
    vals = []
    for f in sorted(glob.glob(os.path.join(V4, rundir_glob, 'metrics.csv'))):
        rows = [r for r in csv.DictReader(open(f)) if r['method'] == method]
        un = [float(r['update_norm']) for r in rows if r['update_norm'] != '']
        if un:
            vals.append(np.mean(un))
    return vals


V4 = os.path.join(ROOT, 'outputs', 'real_fedlora_v4')
data = {}
for (regime, r), pat in SOURCES.items():
    m = f'FedLoRA_r{r}'
    data[(regime, r)] = norm_of(pat, m)

fig, ax = plt.subplots(1, 1, figsize=(SINGLE, 2.3))
jit = {'nominal': 0.94, 'stress': 1.06}
for regime, color, marker, label in (('stress', '#d62728', 's', 'Stress (0.1 s)'),
                                     ('nominal', '#1f77b4', 'o', 'Nominal (2.0 s)')):
    mu = [np.mean(data[(regime, r)]) for r in RANKS]
    sd = [np.std(data[(regime, r)], ddof=1) for r in RANKS]
    xs = [r * jit[regime] for r in RANKS]
    fc = 'white' if regime == 'stress' else color
    ax.errorbar(xs, mu, yerr=sd, fmt=marker + '-', color=color, ms=4.5,
                mfc=fc, lw=1.2, capsize=2, label=label)
# sqrt fit on pooled nominal points: norm = c * sqrt(r)
x = np.array(RANKS, dtype=float)
y = np.array([np.mean(data[('nominal', r)]) for r in RANKS])
c = np.sum(y * np.sqrt(x)) / np.sum(np.sqrt(x) ** 2)
xx = np.linspace(1.6, 16.6, 100)
ax.plot(xx, c * np.sqrt(xx), color='#7f7f7f', lw=1.0, ls=(0, (4, 2.5)),
        label=r'$\propto\sqrt{r}$ fit')
ax.set_xticks(RANKS)
ax.set_xlabel('Rank $r$')
ax.set_ylabel('Delivered-update norm')
ax.legend(loc='upper left', handlelength=1.4)
fig.tight_layout(pad=.4)
require_matplotlib_panel_alignment(fig, tolerance_pt=1.5, strict=True)
fig.savefig(os.path.join(OUT, 'fig_normlaw.pdf'))
fig.savefig(os.path.join(OUT, 'fig_normlaw.png'), dpi=600)
ratio = np.mean(data[('nominal', 16)]) / np.mean(data[('nominal', 2)])
print(f'saved fig_normlaw  c={c:.4f}  r16/r2 norm ratio={ratio:.2f} (sqrt predicts {np.sqrt(8):.2f})')

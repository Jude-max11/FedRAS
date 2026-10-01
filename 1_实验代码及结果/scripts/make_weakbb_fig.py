"""Weak-backbone capacity-payoff figure (Fig. weakbb, double column, 2 panels).

Style contract matches make_v4_figs_submission.py (90/190 mm, 7 pt, Arial).
Panel (a): weak-backbone nominal accuracy bars + pretrained-backbone line.
Panel (b): weak-backbone stress scatter, timely vs accuracy, annotated with
attempted MB and delivered rank.
"""
import glob
import os
import re
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
DOUBLE = 190 * MM

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'pdf.fonttype': 42, 'svg.fonttype': 'none',
    'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7,
    'axes.spines.right': False, 'axes.spines.top': False,
    'axes.linewidth': 0.8, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'legend.frameon': False, 'legend.fontsize': 6.2, 'lines.linewidth': 1.4,
})

COL = {'FedLoRA_r4': '#2ca02c', 'FedLoRA_r16': '#1f77b4', 'Adaptive_rank': '#d62728'}
LBL = {'FedLoRA_r4': 'LoRA $r{=}4$', 'FedLoRA_r16': 'LoRA $r{=}16$',
       'Adaptive_rank': 'FedRAS (ours)'}
V4 = os.path.join(ROOT, 'outputs', 'real_fedlora_v4')


def weak_stats(tag, method):
    """Per-seed (final acc, timely, bytes, rank) over rounds 30-39."""
    accs, tim, byt, rks = [], [], [], []
    for f in sorted(glob.glob(os.path.join(V4, f'{tag}/metrics.csv'))):
        rows = [r for r in csv_rows(f) if r['method'] == method]
        if not rows or max(int(r['round']) for r in rows) < 39:
            continue
        accs.append(float([r for r in rows if int(r['round']) == 39][0]['accuracy']))
        w = [r for r in rows if 30 <= int(r['round']) < 40]
        tim.append(np.mean([float(r['timely_ratio']) for r in w]))
        byt.append(np.mean([float(r['bytes_mb']) for r in w]))
        rk = [float(r['mean_rank']) for r in w if r['mean_rank'] != '']
        rks.append(np.mean(rk) if rk else np.nan)
    return list(map(np.array, (accs, tim, byt, rks)))


def csv_rows(f):
    import csv
    with open(f) as fh:
        return list(csv.DictReader(fh))


def backbone_acc(tag_glob):
    vals = []
    for lg in sorted(glob.glob(os.path.join(ROOT, 'logs', tag_glob))):
        m = re.search(r'pretrained backbone test acc = ([\d.]+)', open(lg).read())
        if m:
            vals.append(float(m.group(1)))
    return np.mean(vals) if vals else np.nan


# ---- panel data ------------------------------------------------------------
nom = {m: weak_stats(f'weak_seed*', m) for m in COL}
strs = {m: weak_stats(f'weak_dl0.1_seed*', m) for m in COL}
bb = backbone_acc('rr_weak_seed*.log')

# ---- figure ----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.5))

ax = axes[0]
xs = np.arange(3)
means = [nom[m][0].mean() for m in COL]
stds = [nom[m][0].std(ddof=1) for m in COL]
ax.bar(xs, means, .55, yerr=stds, error_kw=dict(lw=.8, capsize=2),
       color=[COL[m] for m in COL])
ax.axhline(bb, color='#7f7f7f', lw=1.0, ls=(0, (4, 2.5)))
ax.text(2.42, bb + .004, 'pretrained\nbackbone', fontsize=5.6, color='#555555',
        ha='right', va='bottom', linespacing=1.2,
        bbox=dict(fc='white', ec='none', alpha=.85, pad=1.2))
ax.set_xticks(xs)
ax.set_xticklabels([LBL[m] for m in COL])
ax.tick_params(axis='x', which='major', pad=3)
ax.set_ylabel('Final accuracy')
ax.set_ylim(.45, .62)
ax.set_title('(a) Nominal deadline (2.0 s)', loc='left', fontsize=7)

ax = axes[1]
for m in COL:
    a, t, b, r = strs[m]
    ax.errorbar(t.mean(), a.mean(), xerr=t.std(ddof=1), yerr=a.std(ddof=1),
                fmt='o', ms=5, color=COL[m], lw=.8, capsize=2, label=LBL[m])
    ax.annotate(f'{b.mean():.2f} MB, rank {r.mean():.0f}',
                (t.mean(), a.mean()), textcoords='offset points',
                # r=16 sits left of FedRAS: label above-right, else the text
                # tail crosses FedRAS's upper error-bar whisker
                xytext=(6, 3) if m == 'FedLoRA_r16' else (6, -11),
                fontsize=5.6, color=COL[m])
ax.set_xlabel('Timely delivery ratio (rounds 30–39)')
ax.set_ylabel('Final accuracy')
ax.set_xlim(.46, .76)
ax.set_ylim(.50, .62)
ax.set_title('(b) Stress deadline (0.1 s)', loc='left', fontsize=7)
ax.legend(loc='lower right', handlelength=1.2, fontsize=5.8)

fig.tight_layout(pad=.4)
require_matplotlib_panel_alignment(
    fig, tolerance_pt=1.5, strict=True)
fig.savefig(os.path.join(OUT, 'fig_weakbb.pdf'))
fig.savefig(os.path.join(OUT, 'fig_weakbb.png'), dpi=600)
print('saved fig_weakbb  backbone=%.4f' % bb)

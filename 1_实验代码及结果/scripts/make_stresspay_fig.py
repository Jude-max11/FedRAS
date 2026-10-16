"""Trainer deadline-stress payoff strip (4 panels, full-width, compact).

Network-side twin of the weak-backbone strip: the headline +22% timely /
-23% bytes result against fixed rank 16 at the 0.1 s deadline, ten seeds.
Panels: (a) timely bars (b) attempted-MB bars (FedAvg clipped, annotated)
(c) per-seed timely gain vs r16 (d) per-seed byte saving vs r16.
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, 'outputs', 'real_fedlora_v4', 'all_metrics.csv')
OUT = os.path.join(os.path.dirname(ROOT), '3_ComputerNetworks期刊latex版本',
                   'CN_package_v4', 'recolored_figures', 'panels')
os.makedirs(OUT, exist_ok=True)

MM = 1 / 25.4
PW, PH = 42.0, 40.0
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'pdf.fonttype': 42, 'svg.fonttype': 'none',
    'font.size': 6, 'axes.linewidth': .6,
})

METH = ['FedAvg_full', 'FedLoRA_r4', 'FedLoRA_r16', 'Random_rank', 'Adaptive_rank']
LBL = {'FedAvg_full': 'FedAvg', 'FedLoRA_r4': 'r4', 'FedLoRA_r16': 'r16',
       'Random_rank': 'rand', 'Adaptive_rank': 'FedRAS'}
COL = {'FedAvg_full': '#555555', 'FedLoRA_r4': '#2ca02c', 'FedLoRA_r16': '#1f77b4',
       'Random_rank': '#bbbbbb', 'Adaptive_rank': '#d62728'}

d = pd.read_csv(CSV)
d = d[(d.dataset == 'cifar10_dl0.1') & (d['round'] >= 30)]
g = d.groupby(['method', 'seed'])[['accuracy', 'timely_ratio', 'bytes_mb']].mean()
tm = {m: g.xs(m)['timely_ratio'] for m in METH}
bm = {m: g.xs(m)['bytes_mb'] for m in METH}

fr, r16 = g.xs('Adaptive_rank'), g.xs('FedLoRA_r16')
gain_t = (fr.timely_ratio / r16.timely_ratio - 1) * 100
save_b = (1 - fr.bytes_mb / r16.bytes_mb) * 100
pt = wilcoxon(fr.timely_ratio, r16.timely_ratio).pvalue
pb = wilcoxon(fr.bytes_mb, r16.bytes_mb).pvalue
print(f"timely gain {gain_t.mean():+.1f}% ({(gain_t>0).sum()}/10, p={pt:.3f}); "
      f"byte saving {save_b.mean():.1f}% ({(save_b>0).sum()}/10, p={pb:.3f})")

def style(ax):
    ax.set_xticks(range(5))
    ax.set_xticklabels([LBL[m] for m in METH], fontsize=5.6, rotation=35, ha='right')
    ax.tick_params(labelsize=5.6, pad=1.5, length=2)

# (a) timely
fig, ax = plt.subplots(figsize=(PW*MM, PH*MM))
ax.bar(range(5), [tm[m].mean() for m in METH], .68, color=[COL[m] for m in METH])
ax.set_ylim(0, .75); style(ax); ax.set_ylabel('Timely delivery ratio', fontsize=6)
fig.subplots_adjust(left=.22, right=.985, top=.96, bottom=.16)
fig.savefig(f'{OUT}/fig_stresspay_a.pdf'); plt.close(fig)

# (b) MB (FedAvg clipped at 1.0, annotated)
fig, ax = plt.subplots(figsize=(PW*MM, PH*MM))
vals = [min(bm[m].mean(), 1.0) for m in METH]
ax.bar(range(5), vals, .68, color=[COL[m] for m in METH])
ax.axhline(1.0, color='#999999', lw=.5, ls=(0, (3, 2)))
ax.annotate('52.9', (0, 1.0), xytext=(0, 2), textcoords='offset points',
            ha='center', fontsize=5.6, color='#555555')
ax.set_ylim(0, 1.12); style(ax); ax.set_ylabel('MB per round', fontsize=6)
fig.subplots_adjust(left=.19, right=.985, top=.96, bottom=.16)
fig.savefig(f'{OUT}/fig_stresspay_b.pdf'); plt.close(fig)

def dots(ax, series, lab, pval, color='#d62728'):
    x = np.linspace(-.22, .22, len(series))
    ax.scatter(x, series, s=8, color=color, zorder=3, linewidths=.4, edgecolors='white')
    ax.hlines(series.mean(), -.3, .3, color='black', lw=1.1, zorder=4)
    ax.annotate(f'{series.mean():+.0f}%, p={pval:.3f}', (0, series.mean()),
                xytext=(0, 4), textcoords='offset points', ha='center', fontsize=5.2)
    ax.axhline(0, color='#999999', lw=.7)
    ax.set_xlim(-.5, .5); ax.set_xticks([])
    ax.tick_params(labelsize=5.6, pad=1.5, length=2)
    ax.set_xlabel(lab, fontsize=5.8, labelpad=1.5)

fig, ax = plt.subplots(figsize=(PW*MM, PH*MM))
dots(ax, gain_t.values, 'Timely gain vs r16 (%, per seed)', pt)
ax.set_ylim(-6, gain_t.max() + 8)
fig.subplots_adjust(left=.2, right=.97, top=.96, bottom=.2)
fig.savefig(f'{OUT}/fig_stresspay_c.pdf'); plt.close(fig)

fig, ax = plt.subplots(figsize=(PW*MM, PH*MM))
dots(ax, save_b.values, 'Byte saving vs r16 (%, per seed)', pb)
ax.set_ylim(min(save_b.min() - 3, 12), save_b.max() + 4)
fig.subplots_adjust(left=.2, right=.97, top=.96, bottom=.2)
fig.savefig(f'{OUT}/fig_stresspay_d.pdf'); plt.close(fig)
print('saved fig_stresspay_a..d')

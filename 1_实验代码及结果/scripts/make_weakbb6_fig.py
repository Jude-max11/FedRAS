"""Six-method weak-backbone superiority strip (4 panels, full-width, compact).

Replaces the 3-method 2-panel fig_weakbb in the KBS version. Reads the
seed-corrected weak_* runs (6 method arms) directly. Panels:
  (a) final accuracy bars, nominal deadline (backbone line)
  (b) final accuracy bars, stress deadline
  (c) paired per-seed accuracy difference FedRAS - SA-anneal (5/5 positive)
  (d) attempted bytes per round bars, stress
Style: Arial, 190 mm strip total, each panel 42 mm wide, compact.
"""
import glob, os, re
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V4 = os.path.join(ROOT, 'outputs', 'real_fedlora_v4')
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

METH = ['FedLoRA_r4', 'SA_rank', 'Oort_utility', 'FedLoRA_r16', 'JointAB', 'Adaptive_rank']
LBL  = {'FedLoRA_r4': 'r4', 'SA_rank': 'SA', 'Oort_utility': 'Oort', 'FedLoRA_r16': 'r16',
        'JointAB': 'A/B', 'Adaptive_rank': 'FedRAS'}
COL  = {'FedLoRA_r4': '#2ca02c', 'SA_rank': '#8c564b', 'Oort_utility': '#ff7f0e',
        'FedLoRA_r16': '#1f77b4', 'JointAB': '#9467bd', 'Adaptive_rank': '#d62728'}

def load(tag):
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(V4, tag, 'metrics.csv')))])
    d = d[d['round'] >= 30]
    g = d.groupby(['method', 'seed'])[['accuracy', 'timely_ratio', 'bytes_mb']].mean()
    return g

nom, strs = load('weak_seed?'), load('weak_dl0.1_seed?')
accs_n = {m: nom.xs(m)['accuracy'] for m in METH}
accs_s = {m: strs.xs(m)['accuracy'] for m in METH}
mbs_s  = {m: strs.xs(m)['bytes_mb'].mean() for m in METH}

bb = []
for s in range(10):
    t = open(f'{ROOT}/logs/rr_weak_seed{s}.log').read()
    bb.append(float(re.search(r'pretrained backbone test acc = ([\d.]+)', t).group(1)))
bbm = float(np.mean(bb))

def bars(ax, accs, ylim=None):
    means = [accs[m].mean() for m in METH]
    errs = [accs[m].std(ddof=1) for m in METH]
    ax.bar(range(6), means, .68, yerr=errs, error_kw=dict(lw=.7, capsize=1.6, capthick=.7),
           color=[COL[m] for m in METH], edgecolor='none')
    ax.axhline(bbm, color='#7f7f7f', lw=.9, ls=(0, (4, 2.5)))
    ax.set_xticks(range(6))
    ax.set_xticklabels([LBL[m] for m in METH], fontsize=5.6)
    top = max(m + e for m, e in zip([accs[m].mean() for m in METH],
                                    [accs[m].std(ddof=1) for m in METH]))
    bot = min(m - e for m, e in zip([accs[m].mean() for m in METH],
                                    [accs[m].std(ddof=1) for m in METH]))
    if ylim is None:
        ylim = (bot - .004, top + .006)
    ax.set_ylim(*ylim)
    ax.set_xticklabels([LBL[m] for m in METH], fontsize=5.6, rotation=35, ha='right')
    ax.tick_params(labelsize=5.6, pad=1.5, length=2)
    ax.set_ylabel('Final accuracy', fontsize=6)

# (a) nominal
fig, ax = plt.subplots(figsize=(PW * MM, PH * MM))
bars(ax, accs_n)
ax.text(.985, bbm + .001, 'backbone', fontsize=5.0, color='#555555',
        ha='right', va='bottom', transform=ax.get_yaxis_transform())
fig.subplots_adjust(left=.20, right=.985, top=.96, bottom=.16)
fig.savefig(f'{OUT}/fig_weakbb6_a.pdf'); plt.close(fig)

# (b) stress
fig, ax = plt.subplots(figsize=(PW * MM, PH * MM))
bars(ax, accs_s)
fig.subplots_adjust(left=.20, right=.985, top=.96, bottom=.16)
fig.savefig(f'{OUT}/fig_weakbb6_b.pdf'); plt.close(fig)

# (c) paired FedRAS - SA per seed
fig, ax = plt.subplots(figsize=(PW * MM, PH * MM))
for i, (accs, lab, mk) in enumerate([(accs_n, 'nominal', 'o'), (accs_s, 'stress', '^')]):
    diffs = (accs['Adaptive_rank'].values - accs['SA_rank'].values) * 100
    x = np.full(len(diffs), i) + np.linspace(-.13, .13, len(diffs))
    ax.scatter(x, diffs, s=9, marker=mk, color=COL['Adaptive_rank'] if i else '#555555',
               zorder=3, linewidths=.4, edgecolors='white')
    ax.hlines(diffs.mean(), i - .26, i + .26, color='black', lw=1.1, zorder=4)
    ax.annotate(f'{diffs.mean():+.2f} pp', (i, diffs.mean()), xytext=(0, 4),
                textcoords='offset points', ha='center', fontsize=5.4)
ax.axhline(0, color='#999999', lw=.7)
ax.set_xticks([0, 1]); ax.set_xticklabels(['nominal', 'stress'], fontsize=5.6)
ax.set_xlim(-.55, 1.55); ax.set_ylim(-.15, 2.1)
ax.tick_params(labelsize=5.6, pad=1.5, length=2)
ax.set_ylabel('Accuracy gain vs SA (pp)', fontsize=6)
fig.subplots_adjust(left=.21, right=.97, top=.96, bottom=.16)
fig.savefig(f'{OUT}/fig_weakbb6_c.pdf'); plt.close(fig)

# (d) attempted bytes at stress
fig, ax = plt.subplots(figsize=(PW * MM, PH * MM))
vals = [mbs_s[m] for m in METH]
ax.bar(range(6), vals, .68, color=[COL[m] for m in METH])
ax.set_xticks(range(6)); ax.set_xticklabels([LBL[m] for m in METH], fontsize=5.6, rotation=35, ha='right')
ax.set_ylim(0, 1.05)
ax.tick_params(labelsize=5.6, pad=1.5, length=2)
ax.set_ylabel('MB per round', fontsize=6)
ax.annotate('-22%', (3.5, .955), fontsize=6, ha='center', color='#333333',
            xytext=(0, 2), textcoords='offset points')
fig.subplots_adjust(left=.185, right=.985, top=.96, bottom=.16)
fig.savefig(f'{OUT}/fig_weakbb6_d.pdf'); plt.close(fig)

print('saved fig_weakbb6_a..d to', OUT)

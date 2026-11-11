#!/usr/bin/env python3
"""Fig. 5(a): relative delivered update norm vs rank, two backbones, v6 runs."""
import csv, glob, os, sys, statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

R = 'outputs/real_v6'
OUT = sys.argv[1] if len(sys.argv) > 1 else 'figs_v6'
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['TeX Gyre Termes', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.2, 'ytick.major.size': 2.2,
    'xtick.labelsize': 6.4, 'ytick.labelsize': 6.4,
    'axes.labelsize': 6.8, 'legend.fontsize': 5.8, 'pdf.fonttype': 42,
})


def norm(tag, method, lo=5):
    vals = []
    for d in sorted(glob.glob(f'{R}/{tag}_seed*')):
        f = os.path.join(d, 'metrics.csv')
        if not os.path.exists(f):
            continue
        for r in csv.DictReader(open(f)):
            if r['method'] != method or int(r['round']) < lo:
                continue
            try:
                vals.append(float(r['update_norm']))
            except ValueError:
                pass
    return st.mean(vals) if vals else float('nan')


curves = []
n16 = norm('p1_tab3_cifar10_dl2.0', 'FedLoRA_r16')
n4 = norm('p1_tab3_cifar10_dl2.0', 'FedLoRA_r4')
n8 = norm('p3_fig4_rmax8', 'Adaptive_rank')
curves.append(('full (0.839)', [4, 8, 16], [n4 / n16, n8 / n16, 1.0], 's'))
d16 = norm('p1_tab4_weakdisj_2.0', 'FedLoRA_r16')
d4 = norm('p1_tab4_weakdisj_2.0', 'FedLoRA_r4')
curves.append(('disjoint (0.311)', [4, 16], [d4 / d16, 1.0], 'o'))

fig, ax = plt.subplots(figsize=(1.95, 1.72))
xs = np.linspace(2, 16, 100)
ax.plot(xs, np.sqrt(xs / 16), '--', color='#c0392b', lw=1.0, label=r'$\sqrt{r/r_{\max}}$')
for lab, x, y, mk in curves:
    ax.plot(x, y, mk + '-', color='#2c2c2c', ms=3.2, lw=.9, mfc='w', mew=.9, label=lab)
ax.set_xlabel('rank $r$')
ax.set_ylabel('relative delivered norm')
ax.set_xlim(2, 16)
ax.set_ylim(.3, 1.05)
ax.legend(frameon=False, loc='lower right', handlelength=1.6, borderpad=.2)
ax.tick_params(pad=1.5)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=.22)
fig.savefig(os.path.join(OUT, 'fig_s6_mech_a.pdf'), bbox_inches='tight', pad_inches=.02)
print('wrote', os.path.join(OUT, 'fig_s6_mech_a.pdf'))

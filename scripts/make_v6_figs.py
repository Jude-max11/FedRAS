#!/usr/bin/env python3
"""Regenerate the paper's figures 2 and 3 from outputs/real_v6 (v6 rerun).

Fig. 2: delivered capacity kappa vs attempted MB per round, 0.1 s, 3 datasets.
Fig. 3: (a)(b) final accuracy vs kappa on the disjoint backbone, (c) ViT final
        accuracy per method, (d) ViT accuracy vs cumulative attempted uplink.
"""
import math, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_v6 import stats  # noqa: E402

ROOT = 'outputs/real_v6'
OUT = sys.argv[1] if len(sys.argv) > 1 else 'figs_v6'
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['TeX Gyre Termes', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.2, 'ytick.major.size': 2.2,
    'xtick.labelsize': 6.2, 'ytick.labelsize': 6.2,
    'axes.labelsize': 6.6, 'legend.fontsize': 6.2,
    'pdf.fonttype': 42,
})

M = [('FedLoRA_r4', 'FedLoRA-4'), ('FedLoRA_r16', 'FedLoRA-16'),
     ('Random_rank', 'Random'), ('SA_mono', 'SA'), ('Oort_utility', 'Oort'),
     ('ROBA_style', 'EL'), ('Adaptive_rank', 'FedRAS')]
COL = {'FedLoRA-4': '#6baed6', 'FedLoRA-16': '#123c5f', 'Random': '#8f8f8f',
       'SA': '#e07b39', 'Oort': '#3d6c9a', 'EL': '#2b8c8c', 'FedRAS': '#c0392b'}
MARK = {'FedLoRA-4': 's', 'FedLoRA-16': '^', 'Random': 'o', 'SA': 'p',
        'Oort': 'P', 'EL': 'd', 'FedRAS': 'o'}


def series(pattern, method):
    import glob
    xs, ys = [], []
    for d in sorted(glob.glob(os.path.join(ROOT, pattern))):
        s = stats(d, method)
        if s:
            xs.append(s['mb'])
            ys.append(s['kappa'])
    return np.array(xs), np.array(ys)


# ---------------- Fig. 2
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.62), sharey=False)
titles = [('cifar10', 'CIFAR-10'), ('cifar100', 'CIFAR-100'), ('mnist', 'MNIST')]
for ax, (ds, title) in zip(axes, titles):
    for m, lab in M:
        x, y = series(f'p1_tab3_{ds}_dl0.1_seed*', m)
        if not len(x):
            continue
        ax.plot(x, y, 'o', ms=1.5, color=COL[lab], alpha=.30, mew=0, zorder=1)
        ax.plot(x.mean(), y.mean(), MARK[lab], ms=6.0, color=COL[lab],
                mec='k', mew=.55, zorder=3, label=lab)
    x16, y16 = series(f'p1_tab3_{ds}_dl0.1_seed*', 'FedLoRA_r16')
    xr, yr = series(f'p1_tab3_{ds}_dl0.1_seed*', 'Adaptive_rank')
    ax.plot([x16.mean(), xr.mean()], [y16.mean(), yr.mean()], '--',
            color='k', lw=.5, alpha=.55, zorder=2)
    ax.set_title(title, fontsize=6.8, pad=3)
    ax.set_xlabel('attempted MB per round')
    ax.set_ylabel(r'delivered capacity $\kappa$')
    ax.tick_params(pad=1.5)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.text(.03, .93, f'({chr(97 + list(axes).index(ax))})', transform=ax.transAxes,
            fontsize=6.4)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=7, frameon=False,
           bbox_to_anchor=(.5, 1.035), handletextpad=.35, columnspacing=1.0)
fig.tight_layout(pad=.25, rect=(0, 0, 1, .90))
fig.savefig(os.path.join(OUT, 'fig_s6_main_3ds.pdf'), bbox_inches='tight', pad_inches=.02)
plt.close(fig)

# ---------------- Fig. 3 legend + (a)(b)(c)(d)
fig, ax = plt.subplots(figsize=(4.6, .22))
for m, lab in M:
    ax.plot([], [], MARK[lab], ms=5.2, color=COL[lab], mec='k', mew=.5, label=lab)
ax.axis('off')
ax.legend(loc='center', ncol=7, frameon=False, handletextpad=.3, columnspacing=1.0)
fig.savefig(os.path.join(OUT, 'fig_s6_weak_legend.pdf'), bbox_inches='tight', pad_inches=.01)
plt.close(fig)


def scatter_acc_kappa(tag, fname):
    fig, ax = plt.subplots(figsize=(1.85, 1.72))
    for m, lab in M:
        xs, ys = [], []
        import glob
        for d in sorted(glob.glob(os.path.join(ROOT, tag + '_seed*'))):
            s = stats(d, m)
            if s:
                xs.append(s['kappa'])
                ys.append(s['acc'] * 100)
        if not len(xs):
            continue
        ax.plot(xs, ys, 'o', ms=1.4, color=COL[lab], alpha=.30, mew=0)
        ax.plot(np.mean(xs), np.mean(ys), MARK[lab], ms=5.6, color=COL[lab],
                mec='k', mew=.55, zorder=3)
    ax.set_xlabel(r'delivered capacity $\kappa$')
    ax.set_ylabel('final accuracy (%)')
    ax.tick_params(pad=1.5)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=.22)
    fig.savefig(os.path.join(OUT, fname), bbox_inches='tight', pad_inches=.02)
    plt.close(fig)


scatter_acc_kappa('p1_tab4_weakdisj_2.0', 'fig_s6_weak_a.pdf')
scatter_acc_kappa('p1_tab4_weakdisj_0.1', 'fig_s6_weak_b.pdf')

# (c) ViT final accuracy bars with per-seed dots
fig, ax = plt.subplots(figsize=(1.85, 1.80))
import glob
xs = np.arange(len(M))
for i, (m, lab) in enumerate(M):
    accs = []
    for d in sorted(glob.glob(os.path.join(ROOT, 'p1_tab4_vit_0.5_seed*'))):
        s = stats(d, m)
        if s:
            accs.append(s['acc'] * 100)
    accs = np.array(accs)
    ax.bar(i, accs.mean(), width=.72, color=COL[lab], edgecolor='k', lw=.45, zorder=1)
    jit = np.linspace(-.20, .20, len(accs))
    ax.plot(i + jit, accs, 'o', ms=1.3, color='k', alpha=.65, mew=0, zorder=2)
ax.set_xticks(xs)
ax.set_xticklabels([l for _, l in M], rotation=38, ha='right', fontsize=5.6)
ax.set_ylabel('final accuracy (%)')
ax.tick_params(pad=1.5)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=.22)
fig.savefig(os.path.join(OUT, 'fig_s6_weak_c.pdf'), bbox_inches='tight', pad_inches=.02)
plt.close(fig)

# (d) ViT accuracy vs cumulative attempted uplink
import csv
fig, ax = plt.subplots(figsize=(1.95, 1.72))
for m, lab in M:
    curves = []
    for d in sorted(glob.glob(os.path.join(ROOT, 'p1_tab4_vit_0.5_seed*'))):
        f = os.path.join(d, 'metrics.csv')
        if not os.path.exists(f):
            continue
        rows = [r for r in csv.DictReader(open(f)) if r['method'] == m]
        if not rows:
            continue
        cum, cur = [], 0.0
        for r in sorted(rows, key=lambda r: int(r['round'])):
            try:
                cur += float(r['bytes_mb'])
                cum.append((cur, float(r['accuracy']) * 100))
            except ValueError:
                pass
        if cum:
            curves.append(cum)
    if not curves:
        continue
    n = min(len(c) for c in curves)
    cum = np.mean([[c[i][0] for i in range(n)] for c in curves], axis=0)
    acc = np.mean([[c[i][1] for i in range(n)] for c in curves], axis=0)
    ax.plot(cum, acc, '-', color=COL[lab], lw=.8, label=lab)
ax.set_xlabel('cumulative attempted uplink (MB)')
ax.set_ylabel('test accuracy (%)')
ax.tick_params(pad=1.5)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=.22)
fig.savefig(os.path.join(OUT, 'fig_s6_weak_d.pdf'), bbox_inches='tight', pad_inches=.02)
plt.close(fig)
print('wrote to', OUT)

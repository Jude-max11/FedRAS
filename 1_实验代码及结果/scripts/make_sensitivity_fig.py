"""Sensitivity figure for the paper: accuracy (and bytes) vs rmax, Dirichlet
alpha, and cohort size K, for Adaptive_rank on CIFAR-10 (five seeds)."""
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = pd.read_csv(os.path.join(ROOT, 'outputs/real_fedlora_v3/all_metrics.csv'))
acc = d.sort_values('round').groupby(['dataset', 'seed', 'method']).tail(1)
net = d[d['round'] >= 30].groupby(['dataset', 'seed', 'method'])[['bytes_norm']].mean().reset_index()

def vals(ds, m='Adaptive_rank'):
    a = acc[(acc.dataset == ds) & (acc.method == m)]
    b = net[(net.dataset == ds) & (net.method == m)]
    return (a.accuracy.mean(), a.accuracy.std(), b.bytes_norm.mean()) if len(a) else (None, None, None)

plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': .3, 'figure.dpi': 200, 'savefig.bbox': 'tight'})
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1))
panels = [
    (['cifar10_rmax8', 'cifar10', 'cifar10_rmax32'], [8, 16, 32], 'provisioned $r_{max}$'),
    (['cifar10_alpha0.1', 'cifar10', 'cifar10_alpha1.0'], [0.1, 0.5, 1.0], 'Dirichlet $\\alpha$'),
    (['cifar10_pr5', 'cifar10', 'cifar10_pr20'], [5, 10, 20], 'cohort size $K$'),
]
for ax, (dss, xs, xlabel) in zip(axes, panels):
    ys, es, by = [], [], []
    for x, ds in zip(xs, dss):
        m, s_, bt = vals(ds)
        ys.append(m); es.append(s_); by.append(bt)
    ax.errorbar(xs, ys, yerr=es, marker='o', color='#27ae60', capsize=3)
    ax2 = ax.twinx()
    ax2.plot(xs, by, 's--', color='#7f8c8d', ms=4)
    ax2.set_ylabel('bytes/round', color='#7f8c8d')
    ax2.tick_params(axis='y', colors='#7f8c8d')
    ax.set_xlabel(xlabel); ax.set_ylabel('accuracy')
axes[0].set_title('(a) adapter width'); axes[1].set_title('(b) data heterogeneity'); axes[2].set_title('(c) participation')
fig.tight_layout(); fig.savefig(os.path.join(ROOT, 'FedLoRA lunwen', 'sensitivity.png')); plt.close(fig)
print('sensitivity.png written')

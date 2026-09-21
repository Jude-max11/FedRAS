"""Three additional publication figures from existing experiment data."""
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, 'FedLoRA lunwen')
plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': .3,
                     'figure.dpi': 200, 'savefig.bbox': 'tight'})
C = {'FedAvg_full': '#c0392b', 'FedAvg': '#c0392b', 'FedLoRA_r4': '#7f8c8d', 'FedLoRA_r16': '#2980b9',
     'Random_rank': '#8e44ad', 'Reliability_only': '#f39c12', 'Adaptive_rank': '#27ae60', 'SA_rank': '#e67e22'}
L = {'FedAvg_full': 'FedAvg (full model)', 'FedAvg': 'FedAvg (full model)', 'FedLoRA_r4': 'FedLoRA r=4',
     'FedLoRA_r16': 'FedLoRA r=16', 'Random_rank': 'Random rank', 'Reliability_only': 'Reliability-only',
     'Adaptive_rank': 'FedRAS (ours)', 'SA_rank': 'SA-annealing'}

d = pd.read_csv(os.path.join(ROOT, 'outputs/real_fedlora_v3/all_metrics.csv'))

# ---- Fig A: delivered rank dynamics under three deadlines + SA schedule ----
fig, ax = plt.subplots(figsize=(6.4, 3.4))
for ds, dl, color in [('cifar10', '1.0', '#27ae60'), ('cifar10_dl0.5', '0.5', '#2980b9'),
                      ('cifar10_dl0.25', '0.25', '#8e44ad')]:
    g = d[(d.dataset == ds) & (d.method == 'Adaptive_rank')]
    z = g.groupby('round')['mean_rank'].agg(['mean', 'std'])
    ax.plot(z.index, z['mean'], label=f'FedRAS, deadline {dl}s', color=color)
    ax.fill_between(z.index, z['mean'] - z['std'], z['mean'] + z['std'], alpha=.15, color=color)
g = d[(d.dataset == 'cifar10_ext') & (d.method == 'SA_rank')]
z = g.groupby('round')['mean_rank'].agg(['mean', 'std'])
ax.plot(z.index, z['mean'], '--', label='SA-annealing (fixed schedule)', color='#e67e22')
ax.set_xlabel('Communication round'); ax.set_ylabel('Delivered mean rank')
ax.set_ylim(0, 20)
ax.legend(fontsize=8, loc='lower right')
fig.tight_layout(); fig.savefig(os.path.join(PAPER, 'rank_dynamics.png')); plt.close(fig)

# ---- Fig B: cross-dataset trajectories (CIFAR-100 / MNIST) ----
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, ds, title in zip(axes, ['cifar100', 'mnist'], ['(a) CIFAR-100', '(b) MNIST']):
    for m in ['FedAvg_full', 'FedLoRA_r4', 'FedLoRA_r16', 'Random_rank', 'Adaptive_rank']:
        g = d[(d.dataset == ds) & (d.method == m)]
        if g.empty: continue
        z = g.groupby('round')['accuracy'].agg(['mean', 'std'])
        ax.plot(z.index, z['mean'], label=L[m], color=C[m])
        ax.fill_between(z.index, z['mean'] - z['std'], z['mean'] + z['std'], alpha=.12, color=C[m])
    ax.set_xlabel('Communication round'); ax.set_ylabel('Test accuracy'); ax.set_title(title)
    ax.legend(fontsize=7, loc='lower right')
fig.tight_layout(); fig.savefig(os.path.join(PAPER, 'cross_dataset.png')); plt.close(fig)

# ---- Fig D: simulator network view vs deadline ----
sw = pd.read_csv(os.path.join(ROOT, 'outputs/paper_sweep/sweep_summary.csv'), skiprows=[1])
sw.columns = ['scenario', 'method', 'acc_m', 'acc_s', 'tim_m', 'tim_s', 'byt_m', 'byt_s', 'rk_m', 'rk_s', 'uv_m', 'uv_s']
DL = {'stress/deadline_025': 0.25, 'stress/deadline_050': 0.5, 'main_cifar10': 1.0, 'stress/deadline_200': 2.0}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for m in ['FedAvg', 'FedLoRA_r4', 'FedLoRA_r16', 'Random_rank', 'Adaptive_rank', 'SA_rank']:
    xs, ty, by = [], [], []
    for scen, x in DL.items():
        r = sw[(sw.scenario == scen) & (sw.method == m)]
        if r.empty: continue
        xs.append(x); ty.append(float(r.tim_m.iloc[0])); by.append(float(r.byt_m.iloc[0]))
    ls = '--' if m == 'SA_rank' else '-'
    axes[0].plot(xs, ty, marker='o', ms=4, ls=ls, color=C[m], label=L.get(m, m))
    axes[1].plot(xs, by, marker='s', ms=4, ls=ls, color=C[m], label=L.get(m, m))
for ax, yl, t in [(axes[0], 'Timely-update ratio', '(a) timely delivery vs deadline'),
                  (axes[1], 'Bytes per round', '(b) communication vs deadline')]:
    ax.set_xlabel('Round deadline (s)'); ax.set_ylabel(yl); ax.set_title(t)
    ax.set_xticks([0.25, 0.5, 1.0, 2.0])
axes[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(PAPER, 'sim_deadline.png')); plt.close(fig)
print('three figures written')

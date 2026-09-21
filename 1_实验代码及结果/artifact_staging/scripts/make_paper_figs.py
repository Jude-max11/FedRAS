"""Generate publication figures for the paper from real experiment outputs."""
import os, shutil
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, 'FedLoRA lunwen')
plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': .3,
                     'figure.dpi': 200, 'savefig.bbox': 'tight'})
COLORS = {'FedAvg': '#c0392b', 'FedLoRA_r4': '#7f8c8d', 'FedLoRA_r16': '#2980b9',
          'Random_rank': '#8e44ad', 'Reliability_only': '#f39c12', 'Adaptive_rank': '#27ae60',
          'SA_rank': '#e67e22',
          'DP': '#27ae60', 'greedy_vb': '#2980b9', 'greedy_v': '#8e44ad', 'r16': '#7f8c8d'}
LABEL = {'FedAvg': 'FedAvg (full model)', 'FedLoRA_r4': 'FedLoRA r=4', 'FedLoRA_r16': 'FedLoRA r=16',
         'Random_rank': 'Random rank', 'Reliability_only': 'Reliability-only', 'Adaptive_rank': 'FedRAS (ours)',
         'SA_rank': 'SA-annealing'}

# 1) simulator: accuracy vs round, nominal + deadline stress
d = pd.read_csv(os.path.join(ROOT, 'outputs/paper_sweep/all_round_metrics.csv'))
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, scen, title in zip(axes, ['main_cifar10', 'stress/deadline_025'],
                           ['(a) nominal deadline', '(b) tight deadline (0.25 s)']):
    g = d[d.scenario == scen]
    for m, h in g.groupby('method'):
        z = h.groupby('round')['accuracy'].agg(['mean', 'std'])
        ax.plot(z.index, z['mean'], label=LABEL.get(m, m), color=COLORS.get(m))
        ax.fill_between(z.index, z['mean'] - z['std'], z['mean'] + z['std'], alpha=.12, color=COLORS.get(m))
    ax.set_xlabel('Communication round'); ax.set_ylabel('Model accuracy'); ax.set_title(title)
axes[0].legend(fontsize=8, loc='lower right')
fig.tight_layout(); fig.savefig(os.path.join(PAPER, 'sim_accuracy.png')); plt.close(fig)

# 2) DP study: gap to DP optimum + runtime scaling
s = pd.read_csv(os.path.join(ROOT, 'outputs/dp_scheduler_study.csv'))
grid = s[s.clients == 500]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
ax = axes[0]
sc = {(300.0, 40.0): ['300 (slack)', '40 (tight)'], 12.0: ['12 (binding)']}
for meth, name in [('greedy_vb', 'Greedy value/byte'), ('greedy_v', 'Greedy value'), ('r16', 'Fixed r=16')]:
    xs, ys, es = [], [], []
    for dl in (1.0, 0.5, 0.25):
        for b in (300.0, 40.0, 12.0):
            r_ = grid[(grid.deadline == dl) & (grid.budget == b)]
            xs.append(f'{dl}/{int(b)}'); ys.append(100 * r_[meth + '_gap'].mean())
            es.append(100 * r_[meth + '_gap'].std() / 5 ** .5)
    ax.bar([x + 0.25 * ['greedy_vb', 'greedy_v', 'r16'].index(meth) for x in range(len(xs))], ys,
           .25, label=name, color=COLORS[meth])
ax.set_xticks(range(9)); ax.set_xticklabels([f'{dl}/{int(b)}' for dl in (1.0, .5, .25) for b in (300, 40, 12)], fontsize=8)
ax.set_xlabel('Deadline / budget'); ax.set_ylabel('Optimality gap vs DP (%)')
ax.legend(fontsize=8); ax.set_title('(a) gap to the exact DP optimum')
sc = s[(s.deadline == 1.0) & (s.budget == 300.0)].sort_values('clients')
ax = axes[1]
ax.plot(sc.clients, sc.dp_ms, 'o-', color=COLORS['DP'])
ax.set_xlabel('Number of clients'); ax.set_ylabel('DP solver time (ms)')
ax.set_title('(b) scheduler runtime (K=25, B=300 MB)')
fig.tight_layout(); fig.savefig(os.path.join(PAPER, 'dp_study.png')); plt.close(fig)

# 3) copy trainer figures
for src, dst in [('cifar10_accuracy.png', 'trainer_accuracy.png'),
                 ('cifar10_dl0.25_accuracy.png', 'trainer_dl025.png')]:
    shutil.copy(os.path.join(ROOT, 'outputs/real_fedlora_v3/figures', src), os.path.join(PAPER, dst))
print('figures written to', PAPER)

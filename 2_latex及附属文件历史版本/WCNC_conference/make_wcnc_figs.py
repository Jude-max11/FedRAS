"""Regenerate the two evaluation figures for the WCNC conference version
from the FROZEN data snapshot in ./data_snapshot (single-column legible sizing).
Legends are placed OUTSIDE the axes so they never cover the curves.

ISOLATION: this script reads ONLY ./data_snapshot, never ../outputs.
If you re-run experiments for the journal version, the conference figures
are unaffected. To deliberately refresh the conference figures from new
results, copy the new CSVs into ./data_snapshot (see ISOLATION.md)."""
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data_snapshot')
TRAINER_CSV = os.path.join(DATA, 'real_fedlora_v3__all_metrics.csv')
SWEEP_CSV = os.path.join(DATA, 'paper_sweep__all_round_metrics.csv')
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .3,
                     'axes.linewidth': .8, 'figure.dpi': 200, 'savefig.bbox': 'tight'})

STYLE = {
    'FedAvg_full':    ('FedAvg (full)', '#c0392b', '-', 1.5),
    'FedAvg':         ('FedAvg (full)', '#c0392b', '-', 1.5),
    'FedLoRA_r4':     ('FedLoRA r=4',   '#7f8c8d', '-', 1.1),
    'FedLoRA_r16':    ('FedLoRA r=16',  '#2980b9', '-', 1.1),
    'Random_rank':    ('Random rank',   '#8e44ad', '-', 1.1),
    'Reliability_only': ('Reliability-only', '#f39c12', '-', 1.1),
    'SA_rank':        ('SA-annealing',  '#e67e22', '--', 1.2),
    'Adaptive_rank':  ('Adaptive (ours)', '#27ae60', '-', 1.6),
}

def legend_below(ax, ncol):
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.30), ncol=ncol,
              fontsize=6.5, frameon=False, columnspacing=0.9,
              handlelength=1.5, handletextpad=0.4)

# ---- Fig: real trainer under the 0.25 s deadline (CIFAR-10, five seeds) ----
d = pd.read_csv(TRAINER_CSV)
g = d[d.dataset == 'cifar10_dl0.25']
fig, ax = plt.subplots(figsize=(3.6, 1.4))
for m in ['FedAvg_full', 'FedLoRA_r4', 'FedLoRA_r16', 'Random_rank', 'Adaptive_rank']:
    label, color, ls, lw = STYLE[m]
    z = g[g['method'] == m].groupby('round')['accuracy'].agg(['mean', 'std'])
    ax.plot(z.index, z['mean'], label=label, color=color, ls=ls, lw=lw)
    ax.fill_between(z.index, z['mean'] - z['std'], z['mean'] + z['std'], alpha=.12, color=color)
ax.set_xlabel('Communication round')
ax.set_ylabel('Test accuracy')
ax.set_ylim(0.910, 0.917)
ax.set_xlim(0, 39)
legend_below(ax, 3)
fig.savefig(os.path.join(HERE, 'fig_trainer_stress.png'))
plt.close(fig)

# ---- Fig: 500-client scheduler simulation, 0.25 s deadline ----
d = pd.read_csv(SWEEP_CSV)
g = d[d.scenario == 'stress/deadline_025']
fig, ax = plt.subplots(figsize=(3.6, 1.4))
for m in ['FedAvg', 'FedLoRA_r4', 'FedLoRA_r16', 'Random_rank', 'Reliability_only', 'SA_rank', 'Adaptive_rank']:
    label, color, ls, lw = STYLE[m]
    z = g[g['method'] == m].groupby('round')['accuracy'].agg(['mean', 'std'])
    ax.plot(z.index, z['mean'], label=label, color=color, ls=ls, lw=lw)
    ax.fill_between(z.index, z['mean'] - z['std'], z['mean'] + z['std'], alpha=.10, color=color)
ax.set_xlabel('Communication round')
ax.set_ylabel('Scheduling accuracy')
ax.set_xlim(0, 149)
legend_below(ax, 3)
fig.savefig(os.path.join(HERE, 'fig_sim_stress.png'))
plt.close(fig)


# ---- Fig: delivered rank dynamics under three deadlines (+ SA schedule) ----
d = pd.read_csv(TRAINER_CSV)
fig, ax = plt.subplots(figsize=(3.6, 1.4))
for ds, dl, color in [('cifar10', 'nominal', '#27ae60'), ('cifar10_dl0.5', '0.5 s', '#2980b9'),
                      ('cifar10_dl0.25', '0.25 s', '#8e44ad')]:
    g = d[(d.dataset == ds) & (d['method'] == 'Adaptive_rank')]
    z = g.groupby('round')['mean_rank'].agg(['mean', 'std'])
    ax.plot(z.index, z['mean'], label='adaptive (' + dl + ')', color=color, lw=1.4)
    ax.fill_between(z.index, z['mean'] - z['std'], z['mean'] + z['std'], alpha=.12, color=color)
g = d[(d.dataset == 'cifar10_ext') & (d['method'] == 'SA_rank')]
z = g.groupby('round')['mean_rank'].mean()
ax.plot(z.index, z.values, '--', label='SA-annealing', color='#e67e22', lw=1.2)
ax.set_xlabel('Communication round')
ax.set_ylabel('Delivered mean rank')
ax.set_ylim(0, 18)
ax.set_xlim(0, 39)
legend_below(ax, 2)
fig.savefig(os.path.join(HERE, 'fig_rank_dynamics.png'))
plt.close(fig)

# ---- Fig: DP-joint vs fair cohort vs r16 in the real trainer (0.25 s) ----
d = pd.read_csv(os.path.join(HERE, 'data_snapshot', 'wcnc_dp__all_metrics.csv'))
dv3 = pd.read_csv(TRAINER_CSV)
g = d[(d.dataset == 'cifar10_dl0.25') & (d['method'] == 'DP_cohort')]
g3 = dv3[(dv3.dataset == 'cifar10_dl0.25')]
fig, ax = plt.subplots(figsize=(3.6, 1.4))
for src, m, label, color, ls, lw in [
        (g, 'DP_cohort', 'DP-joint (ours)', '#27ae60', '-', 1.6),
        (g3, 'Adaptive_rank', 'fair cohort', '#2980b9', '-', 1.2),
        (g3, 'FedLoRA_r16', 'fixed $r{=}16$', '#7f8c8d', '-', 1.2)]:
    z = src[src['method'] == m].groupby('round')['timely_ratio'].agg(['mean', 'std'])
    ax.plot(z.index, z['mean'] * 10, label=label, color=color, ls=ls, lw=lw)
    ax.fill_between(z.index, (z['mean'] - z['std']).clip(lower=0) * 10, (z['mean'] + z['std']).clip(upper=1) * 10, alpha=.12, color=color)
ax.set_xlabel('Communication round')
ax.set_ylabel('On-time updates')
ax.set_ylim(0, 4)
ax.set_xlim(0, 39)
legend_below(ax, 3)
fig.savefig(os.path.join(HERE, 'fig_dp_timely.png'))
plt.close(fig)
print('written: fig_rank_dynamics.png, fig_dp_timely.png')

print('written: fig_trainer_stress.png, fig_sim_stress.png, fig_rank_dynamics.png, fig_dp_timely.png')

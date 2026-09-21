"""Regenerate every journal figure from the v4 (real-byte) data.

Reads outputs/real_fedlora_v4/all_metrics.csv, the simulator round metrics,
and outputs/dp_scheduler_study_v4.csv; writes the paper figures into
'FedLoRA lunwen/CN_package_v4/'. Matplotlib only; deterministic order.
"""
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'FedLoRA lunwen', 'CN_package_v4')
os.makedirs(os.path.join(OUT, 'recolored_figures'), exist_ok=True)

COL = {'FedAvg_full': '#7f7f7f', 'FedLoRA_r4': '#2ca02c', 'FedLoRA_r16': '#1f77b4',
       'Random_rank': '#9467bd', 'Adaptive_rank': '#d62728', 'SA_rank': '#ff7f0e',
       'DP_cohort': '#8c564b', 'DP_same': '#e377c2', 'FedAvg': '#7f7f7f',
       'FedLoRA': '#2ca02c', 'FedRAS': '#d62728'}
LBL = {'FedAvg_full': 'FedAvg (full)', 'FedLoRA_r4': 'LoRA r=4', 'FedLoRA_r16': 'LoRA r=16',
       'Random_rank': 'Random rank', 'Adaptive_rank': 'FedRAS (ours)', 'SA_rank': 'SA-annealing',
       'DP_cohort': 'Full DP', 'DP_same': 'Same-cohort DP', 'FedAvg': 'FedAvg (full)'}


def seeds_mean(d, ds, col):
    """Per-round mean and std over seeds for one dataset tag and method."""
    g = d[(d.dataset == ds) & (d.method == col)]
    if g.empty:
        return None, None, None
    p = g.groupby('round').agg(m=(col if col in g else 'accuracy'), )
    return p


def traj(d, ds, method, value='accuracy'):
    g = d[(d.dataset == ds) & (d.method == method)]
    if g.empty:
        return None, None
    m = g.groupby('round')[value].mean()
    s = g.groupby('round')[value].std()
    return m, s


def wrank_traj(d, ds, method):
    g = d[(d.dataset == ds) & (d.method == method)].copy()
    g = g.dropna(subset=['mean_rank'])
    if g.empty:
        return None, None
    grp = g.groupby('round')
    m = grp.apply(lambda x: (x.mean_rank * x.aggregated).sum() / max(1, x.aggregated.sum()),
                  include_groups=False)
    return m, None


def band(ax, m, s, color, label):
    ax.plot(m.index, m.values, color=color, lw=1.6, label=label)
    if s is not None:
        ax.fill_between(m.index, m.values - s.values, m.values + s.values, color=color, alpha=.15)


def main():
    d = pd.read_csv(os.path.join(ROOT, 'outputs/real_fedlora_v4/all_metrics.csv'))
    d['mean_rank'] = pd.to_numeric(d.mean_rank, errors='coerce')
    M5 = ['FedAvg_full', 'FedLoRA_r4', 'FedLoRA_r16', 'Random_rank', 'Adaptive_rank']

    # 1) trainer accuracy trajectories: nominal and 0.1 s
    for ds, fname, title in [('cifar10', 'fig_trainer_nominal.png', 'nominal deadline 2.0 s'),
                             ('cifar10_dl0.1', 'fig_trainer_stress.png', 'stress deadline 0.1 s')]:
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        for m in M5:
            mm, ss = traj(d, ds, m)
            if mm is not None:
                band(ax, mm, ss, COL[m], LBL[m])
        ax.set_xlabel('federated round'); ax.set_ylabel('test accuracy')
        ax.set_title(title, fontsize=9); ax.legend(fontsize=6.5, loc='lower right')
        ax.set_ylim(.90, .92); fig.tight_layout()
        fig.savefig(os.path.join(OUT, fname), dpi=200); plt.close(fig)

    # 2) rank dynamics under three deadlines + SA fixed path
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for ds, color, lb in [('cifar10', '#1f77b4', '2.0 s'), ('cifar10_dl0.1', '#ff7f0e', '0.1 s'),
                          ('cifar10_dl0.05', '#d62728', '0.05 s')]:
        mm, _ = wrank_traj(d, ds, 'Adaptive_rank')
        if mm is not None:
            ax.plot(mm.index, mm.values, color=color, lw=1.6, label=f'FedRAS {lb}')
    mm, _ = wrank_traj(d, 'cifar10_dl0.1', 'SA_rank')
    if mm is not None:
        ax.plot(mm.index, mm.values, color='#7f7f7f', lw=1.2, ls='--', label='SA-annealing 0.1 s')
    ax.set_xlabel('federated round'); ax.set_ylabel('delivered mean rank')
    ax.legend(fontsize=6.5); fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'recolored_figures', 'fig_rank_dynamics.png'), dpi=200)
    plt.close(fig)

    # 3) cross-dataset final accuracy bars
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    datasets = [('cifar10', 'CIFAR-10'), ('cifar100', 'CIFAR-100'), ('mnist', 'MNIST')]
    xs = np.arange(len(datasets)); w = .16
    for j, m in enumerate(M5):
        vals = []
        for ds, _ in datasets:
            g = d[d.dataset == ds]
            acc = g[g.method == m].sort_values('round').groupby('run').tail(1).accuracy
            vals.append(acc.mean() if len(acc) else np.nan)
        ax.bar(xs + (j - 2) * w, vals, w, color=COL[m], label=LBL[m])
    ax.set_xticks(xs); ax.set_xticklabels([n for _, n in datasets])
    ax.set_ylabel('final accuracy'); ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'recolored_figures', 'fig_cross_dataset.png'), dpi=200)
    plt.close(fig)

    # 4) simulator trajectories nominal / 0.1 s
    sim_frames = {}
    for tag in ['nominal', 'dl0.1', 'dl0.05', 'budget1.2', 'mob0.7']:
        fs = sorted(glob.glob(os.path.join(ROOT, f'outputs/fedlora_sim_v4/{tag}/round_metrics_seed*.csv')))
        if fs:
            x = pd.concat([pd.read_csv(f) for f in fs])
            x['tag'] = tag
            sim_frames[tag] = x
    SIMM = ['FedAvg', 'FedLoRA_r4', 'FedLoRA_r16', 'Random_rank', 'SA_rank', 'Adaptive_rank']
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    for ax, tag, title in [(axes[0], 'nominal', 'simulator, nominal 2.0 s'),
                           (axes[1], 'dl0.1', 'simulator, stress 0.1 s')]:
        dd = sim_frames[tag]
        for m in SIMM:
            g = dd[dd.method == m]
            mm = g.groupby('round').accuracy.mean()
            band(ax, mm, g.groupby('round').accuracy.std(), COL.get(m, '#333'), LBL.get(m, m))
        ax.set_title(title, fontsize=9); ax.set_xlabel('round'); ax.set_ylabel('utility state')
        ax.legend(fontsize=6, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'recolored_figures', 'fig_sim_accuracy.png'), dpi=200)
    plt.close(fig)

    # 5) simulator regime summary: timely ratio and bytes per method per regime
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    tags = ['nominal', 'dl0.1', 'dl0.05', 'budget1.2', 'mob0.7']
    tnames = ['2.0s', '0.1s', '0.05s', '1.2MB', 'mob.7']
    xs = np.arange(len(tags)); w = .15
    for j, m in enumerate(SIMM):
        t_vals, b_vals = [], []
        for tag in tags:
            dd = sim_frames[tag]; last = dd[dd['round'] == dd['round'].max()]
            row = last[last.method == m]
            t_vals.append(row.timely_ratio.mean()); b_vals.append(row.bytes_mb.mean())
        axes[0].bar(xs + (j - 2) * w, t_vals, w, color=COL.get(m, '#333'), label=LBL.get(m, m))
        axes[1].bar(xs + (j - 2) * w, b_vals, w, color=COL.get(m, '#333'))
    for ax, yl in [(axes[0], 'timely ratio'), (axes[1], 'MB per round')]:
        ax.set_xticks(xs); ax.set_xticklabels(tnames, fontsize=8); ax.set_ylabel(yl)
    axes[0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'recolored_figures', 'fig_sim_regimes.png'), dpi=200)
    plt.close(fig)

    # 6) solver study: gaps and runtime
    sv = pd.read_csv(os.path.join(ROOT, 'outputs/dp_scheduler_study_v4.csv'))
    grid = sv[(sv.clients == 500) & (sv.DP.notna())]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    labels = [f"{int(100*r)}s/{b:g}MB" for r, b in zip(grid.deadline, grid.budget)]
    xs = np.arange(len(grid))
    for j, (m, c) in enumerate([('greedy_vb_gap', '#d62728'), ('greedy_v_gap', '#1f77b4'),
                                ('myopic_gap', '#ff7f0e'), ('r16_gap', '#7f7f7f')]):
        axes[0].bar(xs + (j - 1.5) * .2, 100 * grid[m], .2, color=c,
                    label={'greedy_vb_gap': 'value/byte greedy', 'greedy_v_gap': 'value greedy',
                           'myopic_gap': 'cohort myopic', 'r16_gap': 'fixed r16'}[m])
    axes[0].set_xticks(xs); axes[0].set_xticklabels(labels, rotation=45, fontsize=6.5)
    axes[0].set_ylabel('objective loss vs DP (%)'); axes[0].legend(fontsize=6)
    sc = sv[sv.DP.isna()].dropna(subset=['dp_ms']).sort_values('clients')
    axes[1].plot(sc.clients, sc.dp_ms, 'o-', color='#1f77b4')
    axes[1].set_xlabel('clients'); axes[1].set_ylabel('DP runtime (ms)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'recolored_figures', 'fig_dp_study.png'), dpi=200)
    plt.close(fig)

    # 7) sensitivity: rmax, cohort, alpha
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.7))
    panels = [('rmax', [('cifar10_rmax8', '8'), ('cifar10', '16'), ('cifar10_rmax32', '32')]),
              ('cohort', [('cifar10_pr5', '5'), ('cifar10', '10'), ('cifar10_pr20', '20')]),
              ('alpha', [('cifar10_alpha0.1', '0.1'), ('cifar10', '0.5'), ('cifar10_alpha1.0', '1.0')])]
    for ax, (ttl, cells) in zip(axes, panels):
        accs, byts = [], []
        names = []
        for ds, nm in cells:
            g = d[d.dataset == ds]
            acc = g[(g.method == 'Adaptive_rank')].sort_values('round').groupby('run').tail(1).accuracy
            accs.append(acc.mean() if len(acc) else np.nan)
            gg = g[(g.method == 'Adaptive_rank') & (g['round'] >= 30)]
            byts.append(gg.bytes_mb.mean() if len(gg) else np.nan)
            names.append(nm)
        ax2 = ax.twinx()
        ax.bar(range(len(cells)), accs, .5, color='#1f77b4')
        ax2.plot(range(len(cells)), byts, 'o--', color='#d62728', lw=1.2)
        ax.set_xticks(range(len(cells))); ax.set_xticklabels(names)
        ax.set_title(f'{ttl}', fontsize=9); ax.set_ylabel('accuracy', color='#1f77b4')
        ax2.set_ylabel('MB/round', color='#d62728'); ax2.tick_params(axis='y', colors='#d62728')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'recolored_figures', 'fig_sensitivity.png'), dpi=200)
    plt.close(fig)

    print('figures written to', OUT)


if __name__ == '__main__':
    main()

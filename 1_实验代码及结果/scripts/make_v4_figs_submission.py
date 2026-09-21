"""Submission-grade figures for the FedRAS manuscript (v4, real-byte era).

Backend: Python (matplotlib), per the nature-figure backend gate (user-selected).
Export contract (Computer Networks / Elsevier): 90 mm single-column and
190 mm double-column widths, Arial/Helvetica fallback chain, 7 pt base type,
5 pt glyph floor, vector PDF primary plus 600 dpi PNG raster.

Every multi-panel figure runs the render-time panel-alignment gate
(require_matplotlib_panel_alignment, 1.5 pt tolerance, strict) before export.
Run the skill QA tools on the exported PDFs afterwards:
  validate_figure.py <script> ; audit_pdf_text.py <pdf> ;
  audit_figure_collisions.py <pdf> --json-out <pdf>.json
"""
import glob
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
SINGLE, DOUBLE = 90 * MM, 190 * MM

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'pdf.fonttype': 42,
    'svg.fonttype': 'none',
    'font.size': 7,
    'axes.titlesize': 7.5,
    'axes.labelsize': 7,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.linewidth': 0.8,
    'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'legend.frameon': False,
    'legend.fontsize': 6.2,
    'lines.linewidth': 1.4,
})

# Method families, unified across every panel (signal red reserved for FedRAS).
COL = {'FedAvg_full': '#7f7f7f', 'FedAvg': '#7f7f7f',
       'FedLoRA_r4': '#2ca02c', 'FedLoRA': '#2ca02c',
       'FedLoRA_r16': '#1f77b4', 'FedLoRA': '#1f77b4',
       'Random_rank': '#9467bd', 'Random': '#9467bd',
       'Reliability_only': '#8c6d31', 'Reliability-only': '#8c6d31',
       'SA_rank': '#ff7f0e', 'SA-annealing': '#ff7f0e',
       'Adaptive_rank': '#d62728', 'FedRAS': '#d62728',
       'DP_cohort': '#8c564b', 'DP_same': '#e377c2'}
LBL = {'FedAvg_full': 'FedAvg (full model)', 'FedAvg': 'FedAvg (full model)',
       'FedLoRA_r4': 'LoRA $r{=}4$', 'FedLoRA_r16': 'LoRA $r{=}16$',
       'Random_rank': 'Random rank', 'Reliability_only': 'Reliability-only',
       'Reliability-only': 'Reliability-only', 'SA_rank': 'SA-annealing',
       'Adaptive_rank': 'FedRAS (ours)', 'DP_cohort': 'Full DP',
       'DP_same': 'Same-cohort DP'}
M5 = ['FedAvg_full', 'FedLoRA_r4', 'FedLoRA_r16', 'Random_rank', 'Adaptive_rank']
SIMM = ['FedAvg', 'FedLoRA_r4', 'FedLoRA_r16', 'Random_rank', 'SA_rank', 'Adaptive_rank']

PANEL_KW = dict(fontsize=8, fontweight='bold', ha='left', va='bottom')


def save_pub(fig, name, panels=1):
    """Alignment gate (multi-panel) then PDF + PNG export at final size."""
    if panels > 1:
        require_matplotlib_panel_alignment(
            fig,
            json_out=os.path.join(OUT, f'{name}.alignment.json'),
            overlay_svg=os.path.join(OUT, f'{name}.alignment.svg'),
            tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
        )
    fig.savefig(os.path.join(OUT, name + '.pdf'))
    fig.savefig(os.path.join(OUT, name + '.png'), dpi=600)
    plt.close(fig)
    print('saved', name)


def panel_label(fig, letter, x=0.005, y=0.98):
    fig.text(x, y, f'({letter})', **PANEL_KW)


def load():
    d = pd.read_csv(os.path.join(ROOT, 'outputs/real_fedlora_v4/all_metrics.csv'))
    d['mean_rank'] = pd.to_numeric(d.mean_rank, errors='coerce')
    return d


def traj(d, ds, method, value='accuracy'):
    g = d[(d.dataset == ds) & (d.method == method)]
    if g.empty:
        return None, None
    return (g.groupby('round')[value].mean(), g.groupby('round')[value].std())


def wrank_traj(d, ds, method):
    g = d[(d.dataset == ds) & (d.method == method)].dropna(subset=['mean_rank'])
    if g.empty:
        return None
    return g.groupby('round').apply(
        lambda x: (x.mean_rank * x.aggregated).sum() / max(1, x.aggregated.sum()),
        include_groups=False)


def band(ax, m, s, color, label):
    ax.plot(m.index, m.values, color=color, lw=1.2, label=label)
    if s is not None:
        ax.fill_between(m.index, m.values - s.values, m.values + s.values,
                        color=color, alpha=.15, lw=0)


def load_sim():
    frames = {}
    for tag in ['nominal', 'dl0.1', 'dl0.05', 'budget1.2', 'mob0.7']:
        fs = sorted(glob.glob(os.path.join(
            ROOT, f'outputs/fedlora_sim_v4/{tag}/round_metrics_seed*.csv')))
        if fs:
            x = pd.concat([pd.read_csv(f) for f in fs])
            x['tag'] = tag
            frames[tag] = x
    return frames


def main():
    d = load()
    sim = load_sim()

    # -- 1. trainer trajectories, nominal ------------------------------------
    fig, ax = plt.subplots(figsize=(SINGLE, 2.35))
    for m in M5:
        mm, ss = traj(d, 'cifar10', m)
        if mm is not None:
            band(ax, mm, ss, COL[m], LBL[m])
    ax.set_xlabel('Federated round'); ax.set_ylabel('Test accuracy')
    ax.set_ylim(.903, .9185); ax.legend(loc='lower right', handlelength=1.4)
    fig.tight_layout(pad=.4)
    save_pub(fig, 'fig_trainer_nominal', panels=1)

    # -- 2. trainer trajectories, 0.1 s --------------------------------------
    fig, ax = plt.subplots(figsize=(SINGLE, 2.35))
    for m in M5:
        mm, ss = traj(d, 'cifar10_dl0.1', m)
        if mm is not None:
            band(ax, mm, ss, COL[m], LBL[m])
    ax.set_xlabel('Federated round'); ax.set_ylabel('Test accuracy')
    ax.set_ylim(.903, .9185); ax.legend(loc='lower right', handlelength=1.4)
    fig.tight_layout(pad=.4)
    save_pub(fig, 'fig_trainer_stress', panels=1)

    # -- 3. delivered rank dynamics ------------------------------------------
    fig, ax = plt.subplots(figsize=(SINGLE, 2.35))
    for ds, color, lb in [('cifar10', '#1f77b4', '2.0 s'),
                          ('cifar10_dl0.1', '#ff7f0e', '0.1 s'),
                          ('cifar10_dl0.05', '#d62728', '0.05 s')]:
        mm = wrank_traj(d, ds, 'Adaptive_rank')
        if mm is not None:
            ax.plot(mm.index, mm.values, color=color, lw=1.3, label=f'FedRAS, {lb}')
    mm = wrank_traj(d, 'cifar10_dl0.1', 'SA_rank')
    if mm is not None:
        ax.plot(mm.index, mm.values, color='#7f7f7f', lw=1.0, ls='--',
                label='SA-annealing, 0.1 s')
    ax.set_xlabel('Federated round'); ax.set_ylabel('Delivered mean rank')
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.02), ncol=2,
              handlelength=1.6, columnspacing=1.0)
    fig.tight_layout(pad=.4)
    save_pub(fig, 'fig_rank_dynamics', panels=1)

    # -- 4. cross-dataset final accuracy -------------------------------------
    datasets = [('cifar10', 'CIFAR-10'), ('cifar100', 'CIFAR-100'), ('mnist', 'MNIST')]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.2))
    xs = np.arange(len(datasets)); w = .15
    for j, m in enumerate(M5):
        vals = []
        for ds, _ in datasets:
            acc = d[d.dataset == ds].query('method == @m') \
                .sort_values('round').groupby('run').tail(1).accuracy
            vals.append(acc.mean() if len(acc) else np.nan)
        ax.bar(xs + (j - 2) * w, vals, w, color=COL[m], label=LBL[m])
    ax.set_xticks(xs); ax.set_xticklabels([n for _, n in datasets])
    ax.tick_params(axis='x', which='major', pad=3)
    ax.set_ylabel('Final accuracy'); ax.set_ylim(.60, 1.06)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.17), ncol=3,
              handlelength=1.2, columnspacing=.9)
    fig.tight_layout(pad=.4)
    save_pub(fig, 'fig_cross_dataset', panels=1)

    # -- 5. simulator trajectories, nominal | 0.1 s --------------------------
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.5))
    for ax, tag, title in [(axes[0], 'nominal', '(a) Nominal deadline (2.0 s)'),
                           (axes[1], 'dl0.1', '(b) Stress deadline (0.1 s)')]:
        dd = sim[tag]
        for m in SIMM:
            g = dd[dd.method == m]
            band(ax, g.groupby('round').accuracy.mean(),
                 g.groupby('round').accuracy.std(), COL.get(m, '#333'), LBL.get(m, m))
        ax.set_title(title, loc='left', fontsize=7); ax.set_xlabel('Round')
        ax.set_ylim(.0, 1.0)
    axes[0].set_ylabel('Utility state')
    axes[0].legend(loc='upper left', handlelength=1.3, ncol=1, fontsize=5.8)
    fig.tight_layout(pad=.4)
    save_pub(fig, 'fig_sim_accuracy', panels=2)

    # -- 6. simulator regime summary ------------------------------------------
    tags = ['nominal', 'dl0.1', 'dl0.05', 'budget1.2', 'mob0.7']
    tnames = ['2.0 s', '0.1 s', '0.05 s', '1.2 MB', 'mob 0.7']
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.5))
    xs = np.arange(len(tags)); w = .13
    for j, m in enumerate(SIMM):
        t_vals, b_vals = [], []
        for tag in tags:
            dd = sim[tag]; last = dd[dd['round'] == dd['round'].max()]
            row = last[last.method == m]
            t_vals.append(row.timely_ratio.mean()); b_vals.append(row.bytes_mb.mean())
        axes[0].bar(xs + (j - 2.5) * w, t_vals, w, color=COL.get(m, '#333'),
                    label=LBL.get(m, m))
        axes[1].bar(xs + (j - 2.5) * w, b_vals, w, color=COL.get(m, '#333'))
    for ax, yl in [(axes[0], 'Timely delivery ratio'), (axes[1], 'MB per round')]:
        ax.set_xticks(xs); ax.set_xticklabels(tnames); ax.set_ylabel(yl)
    axes[0].set_title('(a)', loc='left', fontsize=8, fontweight='bold')
    axes[1].set_title('(b)', loc='left', fontsize=8, fontweight='bold')
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc='upper center', ncol=6, fontsize=5.8,
               handlelength=1.1, columnspacing=1.0, bbox_to_anchor=(.5, 1.0))
    fig.tight_layout(pad=.4, rect=(0, 0, 1, .88))
    save_pub(fig, 'fig_sim_regimes', panels=2)

    # -- 7. solver study: gaps | runtime ---------------------------------------
    sv = pd.read_csv(os.path.join(ROOT, 'outputs/dp_scheduler_study_v4.csv'))
    grid = sv[(sv.clients == 500) & (sv.DP.notna())]
    labels = [f'{r:g}s\n{b:g}MB' for r, b in zip(grid.deadline, grid.budget)]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.5))
    xs = np.arange(len(grid))
    heur = [('greedy_vb_gap', 'Value/byte greedy', '#d62728'),
            ('greedy_v_gap', 'Value greedy', '#1f77b4'),
            ('myopic_gap', 'Cohort myopic', '#ff7f0e'),
            ('r16_gap', 'Fixed $r{=}16$', '#7f7f7f')]
    for j, (m, lb, c) in enumerate(heur):
        axes[0].bar(xs + (j - 1.5) * .2, 100 * grid[m], .2, color=c, label=lb)
    axes[0].set_xticks(xs); axes[0].set_xticklabels(labels, fontsize=5.8)
    axes[0].set_ylabel('Objective loss vs DP (%)')
    sc = sv[sv.DP.isna()].dropna(subset=['dp_ms']).sort_values('clients')
    axes[1].plot(sc.clients, sc.dp_ms, 'o-', color='#1f77b4', ms=3.5)
    axes[1].set_xlabel('Clients'); axes[1].set_ylabel('DP runtime (ms)')
    axes[0].set_title('(a)', loc='left', fontsize=8, fontweight='bold')
    axes[1].set_title('(b)', loc='left', fontsize=8, fontweight='bold')
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc='upper center', ncol=4, fontsize=5.8,
               handlelength=1.2, columnspacing=1.0, bbox_to_anchor=(.5, 1.02))
    fig.tight_layout(pad=.4, rect=(0, 0, 1, .90))
    save_pub(fig, 'fig_dp_study', panels=2)

    # -- 8. sensitivity: rmax | cohort | alpha ---------------------------------
    panels = [('Provisioned width $r_{max}$',
               [('cifar10_rmax8', '8'), ('cifar10', '16'), ('cifar10_rmax32', '32')]),
              ('Cohort size $K$',
               [('cifar10_pr5', '5'), ('cifar10', '10'), ('cifar10_pr20', '20')]),
              ('Dirichlet $\\alpha$',
               [('cifar10_alpha0.1', '0.1'), ('cifar10', '0.5'), ('cifar10_alpha1.0', '1.0')])]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.2))
    for k, (ttl, cells) in enumerate(panels):
        ax = axes[k]; ax2 = ax.twinx()
        ax2.spines['right'].set_visible(True)
        accs, byts, names = [], [], []
        for ds, nm in cells:
            g = d[d.dataset == ds]
            acc = g.query('method == "Adaptive_rank"') \
                .sort_values('round').groupby('run').tail(1).accuracy
            accs.append(acc.mean() if len(acc) else np.nan)
            gg = g.query('method == "Adaptive_rank" and 30 <= `round` <= 39')
            byts.append(gg.bytes_mb.mean() if len(gg) else np.nan)
            names.append(nm)
        ax.bar(range(len(cells)), accs, .55, color='#1f77b4')
        ax2.plot(range(len(cells)), byts, 'o--', color='#d62728', lw=1.1, ms=3)
        ax.set_xticks(range(len(cells))); ax.set_xticklabels(names)
        ax.set_title(f'({"abc"[k]}) {ttl}', loc='left', fontsize=7)
        ax.set_ylim(.90, .92)
        if k == 0:
            ax.set_ylabel('Accuracy', color='#1f77b4')
            ax.tick_params(axis='y', colors='#1f77b4')
        else:
            ax.set_yticklabels([])
        if k == 2:
            ax2.set_ylabel('MB per round', color='#d62728')
            ax2.tick_params(axis='y', colors='#d62728')
        else:
            ax2.set_yticklabels([])
    fig.tight_layout(pad=.4)
    save_pub(fig, 'fig_sensitivity', panels=3)

    # -- 9. calibration: reliability | objective cost --------------------------
    curves_f = os.path.join(ROOT, 'outputs/journal_stats/calibration_curves.csv')
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.5))
    if os.path.exists(curves_f):
        cv = pd.read_csv(curves_f)
        cell = cv[(cv.deadline == 0.1) & (cv.budget == 30.0)]
        ccol = {'oracle': '#27ae60', 'gamma0.5': '#2980b9', 'gamma2': '#8e44ad',
                'noise0.2': '#e67e22'}
        clab = {'oracle': 'Oracle', 'gamma0.5': r'$\gamma$=0.5',
                'gamma2': r'$\gamma$=2', 'noise0.2': r'noise $\sigma$=0.2'}
        for name, c in ccol.items():
            g = cell[cell.variant == name].sort_values('pred_mean')
            if g.empty:
                continue
            axes[0].plot(g.pred_mean, g.obs_mean, 'o-', color=c, ms=2.5,
                         lw=1.0, label=clab[name])
        axes[0].plot([0, 1], [0, 1], 'k--', lw=.7, label='Perfect')
        axes[0].set_xlabel('Predicted $\\hat{P}_{on}$')
        axes[0].set_ylabel('Observed delivery frequency')
        axes[0].legend(loc='upper left', fontsize=5.8, handlelength=1.4)
    cal = pd.read_csv(os.path.join(ROOT, 'outputs/journal_stats/calibration.csv'))
    cells = [(2.0, 30.0), (0.1, 30.0), (0.1, 1.2)]
    xlab = ['2 s/30 MB', '0.1 s/30 MB', '0.1 s/1.2 MB']
    xs = np.arange(len(cells)); w = .2
    for j, (name, c) in enumerate([('gamma0.5', '#2980b9'), ('gamma2', '#8e44ad'),
                                   ('noise0.2', '#e67e22')]):
        vals = [100 * cal[(cal.variant == name) & (cal.deadline == dl) &
                          (cal.budget == b)].obj_loss_mean.iloc[0]
                for dl, b in cells]
        axes[1].bar(xs + (j - 1) * w, vals, w, color=c,
                    label={'gamma0.5': r'$\gamma$=0.5', 'gamma2': r'$\gamma$=2',
                           'noise0.2': r'noise $\sigma$=0.2'}[name])
    axes[1].set_xticks(xs); axes[1].set_xticklabels(xlab)
    axes[1].set_ylabel('Assignment-objective loss (%)')
    axes[1].legend(loc='upper right', fontsize=5.8, handlelength=1.3)
    axes[0].set_title('(a)', loc='left', fontsize=8, fontweight='bold')
    axes[1].set_title('(b)', loc='left', fontsize=8, fontweight='bold')
    fig.tight_layout(pad=.4)
    save_pub(fig, 'fig_calibration', panels=2)

    print('all submission figures written to', OUT)


if __name__ == '__main__':
    main()

"""fig_exchange: the capacity--delivery exchange in one figure.

(a) supply: measured delivered-update norm vs rank (the sqrt law anchor);
(b) price: payload-conditioned delivery probability vs rank per SAGIN link
    class at the 0.1 s stress deadline;
(c) law: the product sqrt(r/rmax) * P_on(r) peaking at r* = R*tau/(16c),
    with the trainer's measured per-class delivered ranks overlaid;
(d) conversion: accuracy gain of the preserved capacity vs backbone quality
    (FedRAS over the link-oblivious SA schedule; fixed r16 over fixed r4),
    stress deadline; fills in as the bb2/bb3/bb10 sweep lands.

Style contract matches make_v4_figs_submission.py: 190 mm double-column,
Arial/Helvetica chain, 7 pt base, fonttype 42, signal red reserved for FedRAS.
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
else:
    def require_matplotlib_panel_alignment(*a, **k):
        pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, 'outputs', 'real_fedlora_v4')
STATS = os.path.join(ROOT, 'outputs', 'journal_stats')
OUT = os.path.join(os.path.dirname(ROOT), '3_ComputerNetworks期刊latex版本',
                   'CN_package_v4', 'recolored_figures')
os.makedirs(OUT, exist_ok=True)

MM = 1 / 25.4
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'pdf.fonttype': 42, 'svg.fonttype': 'none', 'font.size': 7,
    'axes.titlesize': 7.5, 'axes.labelsize': 7,
    'axes.spines.right': False, 'axes.spines.top': False, 'axes.linewidth': 0.8,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'legend.frameon': False, 'legend.fontsize': 6.2, 'lines.linewidth': 1.4,
})
PANEL_KW = dict(fontsize=8, fontweight='bold', ha='left', va='bottom')
CLASS_COL = {'ground': '#08519c', 'aerial': '#6baed6', 'satellite': '#969696'}
CLASS_LBL = {'ground': 'ground', 'aerial': 'aerial', 'satellite': 'satellite'}
# SAGIN laws from the trainer (draw_links): (lograte mu, sigma) and Beta(a, b)
CLASS_LAW = {'ground': (2.8, 0.5, 2, 8), 'aerial': (2.3, 0.7, 2, 5), 'satellite': (1.6, 0.5, 2, 3)}
TAU, C_PER_RANK, RMAX = 0.1, 5632 / 1e6, 16
RANKS = [2, 4, 8, 16]
COL = {'FedLoRA_r4': '#2ca02c', 'FedLoRA_r16': '#1f77b4',
       'SA_rank': '#ff7f0e', 'Adaptive_rank': '#d62728'}
LBL = {'FedLoRA_r4': 'fixed $r{=}4$', 'FedLoRA_r16': 'fixed $r{=}16$',
       'SA_rank': 'SA-anneal', 'Adaptive_rank': 'FedRAS'}


def panel_supply(ax):
    """Measured delivered-update norm vs rank: nrm_r2/nrm_r8 + cifar10 runs."""
    d = pd.read_csv(os.path.join(STATS, 'delivered_capacity.csv'))
    src = [('nrm_r2', 'FedLoRA_r2'), ('cifar10', 'FedLoRA_r4'),
           ('nrm_r8', 'FedLoRA_r8'), ('cifar10', 'FedLoRA_r16')]
    xs, ys, es = [], [], []
    for fam, meth in src:
        h = d[(d.family == fam) & (d.method == meth) & d.update_norm.notna()]
        per_run = h.groupby('run')['update_norm'].mean()
        xs.append(int(meth.split('_r')[1]))
        ys.append(per_run.mean())
        es.append(per_run.std())
        ax.errorbar(int(meth.split('_r')[1]), per_run.mean(), yerr=per_run.std(),
                    fmt='o', ms=3, mfc='k', mec='k', capsize=2, lw=0.9, zorder=3)
    rr = np.linspace(1, 17, 100)
    k = np.polyfit(np.sqrt(RANKS), ys, 1)[0]
    ax.plot(rr, k * np.sqrt(rr), ls='--', c='#bbbbbb', lw=1.0, zorder=2)
    ax.set_xlabel('rank $r$')
    ax.set_ylabel('delivered-update norm')
    ax.set_xticks(RANKS)
    ax.text(0.05, 0.92, '$\\propto\\sqrt{r}$', transform=ax.transAxes, fontsize=6.5)


def hazard(r, rate, mob):
    return np.exp(-mob) * np.exp(-8.0 * C_PER_RANK * r / (rate * TAU))


def panel_price(ax):
    rr = np.linspace(1, 16, 200)
    for g, (mu, sg, a, b) in CLASS_LAW.items():
        rate, mob = np.exp(mu), a / (a + b)
        ax.plot(rr, hazard(rr, rate, mob), c=CLASS_COL[g], label=CLASS_LBL[g])
    ax.set_xlabel('rank $r$')
    ax.set_ylabel('delivery prob. $\\hat P_{\\rm on}(r)$')
    ax.legend(loc='lower left', handlelength=1.4)
    ax.set_ylim(bottom=0)
    ax.text(0.05, 0.92, '$\\tau=0.1$ s', transform=ax.transAxes, fontsize=6.5)


def panel_law(ax):
    rr = np.linspace(1, 16, 300)
    peaks = {}
    for g, (mu, sg, a, b) in CLASS_LAW.items():
        rate, mob = np.exp(mu), a / (a + b)
        u = np.sqrt(rr / RMAX) * hazard(rr, rate, mob)
        ax.plot(rr, u, c=CLASS_COL[g])
        cand = np.array([np.sqrt(r / RMAX) * hazard(r, rate, mob) for r in RANKS])
        pk = RANKS[int(np.argmax(cand))]
        peaks[g] = pk
        ax.plot(pk, np.sqrt(pk / RMAX) * hazard(pk, rate, mob), marker='v', ms=4,
                mfc=CLASS_COL[g], mec='k', mew=0.5, zorder=4)
    # measured per-class delivered ranks, grouped-profile stress runs
    got = {g: [] for g in CLASS_LAW}
    for f in sorted(glob.glob(f'{RUNS}/cifar10_sagin_dl0.1_seed*')) + \
             sorted(glob.glob(f'{RUNS}/cifar10_sagin_dl0.1_seed[0-9]')):
        p = os.path.join(f, 'link_outcomes.csv')
        if not os.path.exists(p):
            continue
        lo = pd.read_csv(p)
        a = lo[(lo.method == 'Adaptive_rank') & (lo.delivered == 1)]
        for g, v in a.groupby('group')['rank'].mean().items():
            if g in got:
                got[g].append(v)
    for g, vs in got.items():
        if vs:
            ax.plot(np.mean(vs), 0.02, marker='D', ms=3.5, mfc='w', mec=CLASS_COL[g],
                    mew=1.2, clip_on=False, zorder=5)
    ax.set_xlabel('rank $r$')
    ax.set_ylabel('$\\sqrt{r/r_{\\max}}\\,\\hat P_{\\rm on}(r)$')
    ax.set_xticks(RANKS)
    ax.annotate('$r^{\\star}=\\bar R\\tau/16c$', xy=(0.42, 0.86), xycoords='axes fraction',
                fontsize=6.5)
    ax.annotate('', xy=(0.97, 0.02), xytext=(0.62, 0.02),
                xycoords='axes fraction', arrowprops=dict(arrowstyle='-', lw=0))
    ax.text(0.30, 0.04, 'measured $\\triangleleft$', transform=ax.transAxes, fontsize=6)


def panel_conversion(ax):
    d = pd.read_csv(os.path.join(STATS, 'delivered_capacity.csv'))
    d = d[(d['round'] >= 30) & (d['round'] <= 39)]
    done = d.groupby(['family', 'run'])['round'].max().reset_index()
    done = done[done['round'] == 39]['run']
    d = d[d['run'].isin(set(done))]
    fams = {'weak': 1, 'bb2': 2, 'bb3': 3, 'weak5': 5, 'bb10': 10, 'cifar10': 30}
    pairs = [('Adaptive_rank', 'SA_rank', 'FedRAS $-$ SA-anneal'),
             ('FedLoRA_r16', 'FedLoRA_r4', 'fixed $r{=}16-r{=}4$')]
    plotted = False
    alias = {'cifar10_dl0.1': ['cifar10_dl0.1', 'sa_strong_dl0.1']}  # SA runs live in sa_strong*
    for ma, mb, lbl in pairs:
        xs, ys, es = [], [], []
        for fam in fams:
            h = d[d.family.isin(alias.get(f'{fam}_dl0.1', [f'{fam}_dl0.1']))]
            if not {ma, mb} <= set(h.method):
                continue
            ps = (h.groupby(['method', 'seed'])
                    .agg(acc=('accuracy', 'mean'), bb=('backbone_acc', 'mean')).reset_index())
            wa = ps[ps.method == ma].set_index('seed')
            wb = ps[ps.method == mb].set_index('seed')
            seeds = sorted(set(wa.index) & set(wb.index))
            if len(seeds) < 5:
                continue
            diffs = np.array([(wa.loc[s, 'acc'] - wb.loc[s, 'acc']) * 100 for s in seeds])
            xs.append(wa['bb'].mean())
            ys.append(diffs.mean())
            es.append(diffs.std(ddof=1) / np.sqrt(len(diffs)))
        if xs:
            ax.errorbar(xs, ys, yerr=es, fmt='o-', ms=3, lw=1.2, capsize=2,
                        c=COL[ma], label=lbl)
            plotted = True
    ax.axhline(0, c='#cccccc', lw=0.7, zorder=1)
    ax.set_xlabel('backbone accuracy')
    ax.set_ylabel('accuracy gain (pp)')
    if plotted:
        ax.legend(loc='upper right', handlelength=1.4)
    ax.text(0.05, 0.06, 'stress $\\tau=0.1$ s', transform=ax.transAxes, fontsize=6.5)


def main():
    fig, axes = plt.subplots(1, 4, figsize=(190 * MM, 42 * MM))
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.21, top=0.86, wspace=0.46)
    panel_supply(axes[0])
    panel_price(axes[1])
    panel_law(axes[2])
    panel_conversion(axes[3])
    for ax, tag in zip(axes, 'abcd'):
        ax.text(-0.30, 1.10, f'({tag})', transform=ax.transAxes, **PANEL_KW)
    require_matplotlib_panel_alignment(fig)
    fig.savefig(os.path.join(OUT, 'fig_exchange.pdf'))
    fig.savefig(os.path.join(OUT, 'fig_exchange.png'), dpi=600)
    print('wrote', os.path.join(OUT, 'fig_exchange.pdf'))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Figure 4 (sensitivity): accuracy change vs the reference setting."""
import glob, os, sys
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
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.2, 'ytick.major.size': 2.2,
    'xtick.labelsize': 6.2, 'ytick.labelsize': 6.2,
    'axes.labelsize': 6.6, 'pdf.fonttype': 42,
})
REF = 'p1_tab3_cifar10_dl2.0'
M = 'Adaptive_rank'


def per_seed(tag):
    out = {}
    for d in sorted(glob.glob(os.path.join(ROOT, tag + '_seed*'))):
        s = stats(d, M)
        if s:
            seed = int(d.rsplit('seed', 1)[1])
            out[seed] = s
    return out


ref = per_seed(REF)
PANELS = [
    ('(a)', 'provisioned width', [('p3_fig4_rmax8', '8'), (REF, '16'), ('p3_fig4_rmax32', '32')]),
    ('(b)', 'cohort size', [('p3_fig4_k5', '5'), (REF, '10'), ('p3_fig4_k20', '20')]),
    ('(c)', r'Dirichlet $\alpha$', [('p2_tab5d_cifar10_alpha0.1', '0.1'), (REF, '0.5'),
                                    ('p2_tab5d_cifar10_alpha1.0', '1.0')]),
]
for letter, xlab, spec in PANELS:
    fig, ax = plt.subplots(figsize=(1.75, 1.60))
    for i, (tag, lab) in enumerate(spec):
        cur = per_seed(tag)
        seeds = sorted(set(cur) & set(ref))
        if not seeds:
            continue
        d = np.array([(cur[s]['acc'] - ref[s]['acc']) * 100 for s in seeds])
        ax.plot(i + np.linspace(-.16, .16, len(d)), d, 'o', ms=1.6, color='#c0392b',
                alpha=.55, mew=0)
        ax.plot([i - .22, i + .22], [d.mean(), d.mean()], '-', color='#c0392b', lw=1.1)
        b = np.mean([cur[s]['mb'] for s in seeds])
        y = ax.get_ylim()[0]
        ax.text(i, d.max() + .07, f'{b:.2f} MB', ha='center', va='bottom', fontsize=4.6)
    ax.axhline(0, color='k', lw=.35)
    for y in (-.5, .5):
        ax.axhline(y, color='k', lw=.30, ls=(0, (3, 3)), alpha=.6)
    ax.set_xticks(range(len(spec)))
    ax.set_xticklabels([l for _, l in spec])
    ax.set_xlabel(xlab)
    ax.set_ylabel('accuracy change (pp)')
    ax.set_ylim(-1.0, 1.0)
    ax.tick_params(pad=1.5)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.text(.04, .94, letter, transform=ax.transAxes, fontsize=6.2, va='top')
    fig.tight_layout(pad=.22)
    name = {'(a)': 'fig_s6_sens_a.pdf', '(b)': 'fig_s6_sens_b.pdf', '(c)': 'fig_s6_sens_c.pdf'}[letter]
    fig.savefig(os.path.join(OUT, name), bbox_inches='tight', pad_inches=.02)
    plt.close(fig)
print('fig 4 written')

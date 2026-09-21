"""Concept figure: why the LoRA rank is a scheduling variable.

Purely ANALYTIC curves from the same model constants the paper documents for
the solver study (Section Optimality): hazard estimator
P(r) = exp(-mu) * exp(-8*B(r)/(rate*tau)), payload B(r) = 0.02r + 0.5 MB,
gain G(r) = sqrt(min(r, 16)/16).  No experimental data is read, so this
script is independent of data_snapshot and of the journal version.
Output: fig_tradeoff.png (single-column, legend below the axes).
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({'font.size': 8, 'axes.grid': True, 'grid.alpha': .3,
                     'axes.linewidth': .8, 'figure.dpi': 200, 'savefig.bbox': 'tight'})

TAU = 0.25                     # binding deadline used throughout the paper
RANKS = np.array([2, 4, 8, 16])
LINKS = [('stable link (16 Mbps, $\\mu{=}0.1$)', 16.0, 0.1, '#2980b9', 'o'),
         ('degraded link (6 Mbps, $\\mu{=}0.4$)', 6.0, 0.4, '#c0392b', 's')]

r = np.linspace(2, 16, 200)
B = 0.02 * r + 0.5             # MB, simulator payload model

def P_on(rate, mu):
    return np.exp(-mu - 8.0 * B / (rate * TAU))

def G(r_):
    return np.sqrt(np.minimum(r_, 16) / 16)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.6, 1.50))
for label, rate, mu, color, mk in LINKS:
    p = P_on(rate, mu)
    u = G(r) * p
    ax1.plot(r, p, color=color, lw=1.4, label=label)
    ax1.plot(RANKS, np.exp(-mu - 8.0 * (0.02 * RANKS + 0.5) / (rate * TAU)),
             mk, color=color, ms=3, mfc='none')
    ax2.plot(r, u / u.max(), color=color, lw=1.4)
    ax2.plot(RANKS, G(RANKS) * np.exp(-mu - 8.0 * (0.02 * RANKS + 0.5) / (rate * TAU))
             / (u.max()), mk, color=color, ms=3, mfc='none')
    u_disc = G(RANKS) * np.exp(-mu - 8.0 * (0.02 * RANKS + 0.5) / (rate * TAU))
    r_best = int(RANKS[int(np.argmax(u_disc))])   # scheduler picks from discrete R
    ax2.annotate('$r^{\\star}{=}%d$' % r_best,
                 xy=(r_best, 1.0), ha='center',
                 xytext=(r_best - 3.0 if r_best > 8 else r_best + 0.5, 1.07),
                 fontsize=7, color=color)

ax1.set_xlabel('rank $r$')
ax1.set_ylabel('on-time\nprobability')
ax1.set_ylim(0, 0.35)
ax1.set_xlim(1.4, 16.6)
ax1.set_xticks(RANKS)
ax2.set_xlabel('rank $r$')
ax2.set_ylabel('relative useful\nvalue')
ax2.set_ylim(0.45, 1.16)
ax2.set_yticks([0.5, 0.75, 1.0])
ax2.set_xlim(1.4, 16.6)
ax2.set_xticks(RANKS)

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.06),
           ncol=2, fontsize=7, frameon=False, columnspacing=0.9,
           handlelength=1.5, handletextpad=0.4)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(os.path.join(HERE, 'fig_tradeoff.png'))
print('written: fig_tradeoff.png')

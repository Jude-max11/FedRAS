#!/usr/bin/env python3
"""Figure 5(b): delivery probability and its product with the capacity term.

Computed from the paper's hazard model (eq:hazard) with the link parameters
stated in Sec. 6.1: rate ~ LogNormal(2.5, 0.7) Mbps clipped to [1, 40],
mobility ~ Beta(2, 5), payload size(r) = 5632*r bytes, deadline 0.1 s.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else 'figs_v6'
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['TeX Gyre Termes', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.2, 'ytick.major.size': 2.2,
    'xtick.labelsize': 6.4, 'ytick.labelsize': 6.4,
    'axes.labelsize': 6.8, 'pdf.fonttype': 42,
})

rng = np.random.default_rng(0)
N = 40000
tau = 0.1
rate = np.clip(rng.lognormal(2.5, 0.7, N), 1, 40)   # Mbps
mob = rng.beta(2, 5, N)
BYTES_PER_RANK = 5632
RMAX = 16

r = np.arange(1, 17)
size = r * BYTES_PER_RANK / 1e6                          # MB
pon = np.exp(-mob[:, None]) * np.exp(-8.0 * size[None, :] / (rate[:, None] * tau))
mean_pon = pon.mean(axis=0)
cap = np.sqrt(np.minimum(r, RMAX) / RMAX)
prod = mean_pon * cap

fig, ax = plt.subplots(figsize=(1.95, 1.72))
ax.plot(r, mean_pon, 'o--', color='#333333', ms=2.6, lw=.8, mfc='w', mew=.7,
        label=r'$\hat p_{\mathrm{on}}$')
ax.plot(r, prod, 'o-', color='#c0392b', ms=2.6, lw=1.0, label=r'$\hat p_{\mathrm{on}}\sqrt{r/r_{\max}}$')
ax.set_xlabel('rank $r$')
ax.set_ylabel(r'$\hat p_{\mathrm{on}}$ and $\hat p_{\mathrm{on}}\sqrt{r/r_{\max}}$')
ax.set_xlim(1, 16)
ax.set_ylim(0, 0.75)
ax.legend(frameon=False, loc='center right', handlelength=1.8, borderpad=.2)
ax.tick_params(pad=1.5)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=.22)
fig.savefig(os.path.join(OUT, 'fig_s6_mech_b.pdf'), bbox_inches='tight', pad_inches=.02)
print('mean pon at r=2,4,8,16:', [round(float(mean_pon[i-1]), 3) for i in (2, 4, 8, 16)])
print('product at r=2,4,8,16:', [round(float(prod[i-1]), 3) for i in (2, 4, 8, 16)])
print('wrote', os.path.join(OUT, 'fig_s6_mech_b.pdf'))

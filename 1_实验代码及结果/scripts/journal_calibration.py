"""P_on calibration study for the journal revision (reviewer point 2).

For the paper's link model (LogNormal(2.5,0.7) clipped [1,40] Mbps rates,
Beta(2,5) mobility, hazard-form surrogate) this measures, per (deadline,
budget) cell and predictor variant:
  - the reliability curve of the deployed predictor: predicted P_on versus
    observed deadline-success frequency (oracle link state -> calibrated by
    construction; reported as the ideal reference);
  - deliberately mis-calibrated predictors P~ = P^gamma (gamma 0.5 / 2) and
    multiplicative lognormal state noise (sigma 0.2, as in the robustness
    study): empirical ECE and the scheduling cost of optimizing against the
    distorted predictor (relative loss of the exact-DP assignment objective
    versus the oracle-predictor assignment, evaluated under true P_on).

Writes outputs/journal_stats/calibration.csv and
'FedLoRA lunwen/fig_calibration.png' (journal folder only).
"""
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'outputs', 'journal_stats')
PAPER = os.path.join(os.path.dirname(ROOT), '3_ComputerNetworks期刊latex版本', 'CN_package_v4')
os.makedirs(OUT, exist_ok=True)

RANKS = [2, 4, 8, 16]
RMAX = 16
CLIENTS, CAP = 500, 25
ROUNDS, SEEDS = 50, 3
STEP = 0.01  # MB budget grid: the same conservative ceil grid as the trainer
CELLS = [(2.0, 30.0), (0.1, 30.0), (0.1, 1.2)]  # (deadline s, budget real MB)
VARIANTS = [('oracle', {}), ('gamma0.5', {'gamma': 0.5}), ('gamma2', {'gamma': 2.0}),
            ('noise0.2', {'sigma': 0.2})]


def size(r):
    # Real serialized B-only upload (FP32), as in the v4 trainer/simulator.
    return r * 5632 / 1e6


def p_true(mob, rate, r, dl):
    return np.exp(-mob) * np.exp(-8.0 * size(r) / (rate * dl))


def dp_assign(vals, costs, cap, Bu):
    """Exact multiple-choice knapsack with admission cap; vals[i][r], costs in grid units."""
    NEG = -1e9
    V = np.full((cap + 1, Bu + 1), NEG)
    V[0, 0] = 0.0
    back = []
    for i in range(len(vals)):
        best = V.copy()
        act = np.full((cap + 1, Bu + 1), -1, dtype=np.int64)
        for ri, r in enumerate(RANKS):
            c = costs[i][ri]
            if c > Bu:
                continue
            shifted = np.full_like(V, NEG)
            shifted[1:, c:] = V[:-1, :Bu + 1 - c] + vals[i][ri]
            upd = shifted > best
            best = np.where(upd, shifted, best)
            act[upd] = r
        back.append(act)
        V = best
    k, b = np.unravel_index(np.argmax(V), V.shape)
    assign = np.zeros(len(vals), dtype=int)
    for i in reversed(range(len(vals))):
        r = int(back[i][k, b])
        if r > 0:
            assign[i] = r
            b -= costs[i][RANKS.index(r)]
            k -= 1
    return assign


def ece(pred, obs, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(pred, edges) - 1, 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum():
            tot += m.mean() * abs(pred[m].mean() - obs[m].mean())
    return tot


def main():
    rng = np.random.default_rng(0)
    rows = []
    curves = {}  # (variant, dl) -> (mean pred, freq) for the figure
    for dl, budget in CELLS:
        Bu = max(1, int(math.ceil(budget / STEP - 1e-9)))
        costs = [[max(1, int(math.ceil(size(r) / STEP - 1e-9))) for r in RANKS] for _ in range(CLIENTS)]
        capg = np.array([math.sqrt(min(r, RMAX) / RMAX) for r in RANKS])
        acc = {v[0]: {'ece': [], 'loss': [], 'pred': [], 'obs': []} for v in VARIANTS}
        for seed in range(SEEDS):
            for rnd in range(ROUNDS):
                g = np.random.default_rng(100000 * seed + rnd)
                rate = np.clip(g.lognormal(2.5, .7, CLIENTS), 1, 40)
                mob = g.beta(2, 5, CLIENTS)
                pt16 = p_true(mob, rate, 16, dl)                       # true success prob (rank 16)
                pt = np.stack([p_true(mob, rate, r, dl) for r in RANKS], axis=1)
                outcomes = (g.random(CLIENTS) < pt16).astype(float)    # realized rank-16 outcomes
                # oracle assignment objective (evaluated under true P_on)
                assign_o = dp_assign(pt * capg, costs, CAP, Bu)
                obj_o = sum(pt[i, RANKS.index(assign_o[i])] * capg[RANKS.index(assign_o[i])]
                            for i in range(CLIENTS) if assign_o[i] > 0)
                for name, kw in VARIANTS:
                    if 'gamma' in kw:
                        pred16 = np.power(pt16, kw['gamma'])
                        pred = np.power(pt, kw['gamma'])
                    elif 'sigma' in kw:
                        xi = np.exp(rng.normal(0, kw['sigma'], CLIENTS))
                        pred16 = np.minimum(1.0, pt16 * xi)
                        pred = np.minimum(pt, np.exp(rng.normal(0, kw['sigma'], (CLIENTS, len(RANKS)))))
                    else:
                        pred16, pred = pt16, pt
                    a = dp_assign(pred * capg, costs, CAP, Bu)
                    obj = sum(pt[i, RANKS.index(a[i])] * capg[RANKS.index(a[i])]
                              for i in range(CLIENTS) if a[i] > 0)
                    acc[name]['ece'].append(pred16)      # pooled ECE computed after the loop
                    acc[name]['obs'].append(outcomes)
                    acc[name]['loss'].append(1.0 - obj / max(obj_o, 1e-12))
                    acc[name]['pred'].append(pred16)
        for name, _ in VARIANTS:
            a = acc[name]
            pred = np.concatenate(a['pred']); obs = np.concatenate(a['obs'])
            pooled_ece = ece(pred, obs)   # ECE on the pooled sample (per-round ECE is noise-dominated)
            edges = np.linspace(0, 1, 11)
            idx = np.clip(np.digitize(pred, edges) - 1, 0, 9)
            bp, bo, bw = [], [], []
            for b in range(10):
                m = idx == b
                if m.sum() > 30:
                    bp.append(pred[m].mean()); bo.append(obs[m].mean()); bw.append(m.mean())
            curves[(name, dl, budget)] = (np.array(bp), np.array(bo), np.array(bw))
            rows.append(dict(deadline=dl, budget=budget, variant=name,
                             ece_mean=pooled_ece, ece_max=pooled_ece,
                             obj_loss_mean=np.mean(a['loss']), obj_loss_max=np.max(a['loss'])))
            print(f'dl={dl} B={budget} {name:9s} ECE={pooled_ece:.4f}  obj_loss={100 * np.mean(a["loss"]):.2f}%')
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'calibration.csv'), index=False)
    curve_rows = []
    for (name, dl, budget), (bp, bo, bw) in curves.items():
        for x, y, w in zip(bp, bo, bw):
            curve_rows.append(dict(variant=name, deadline=dl, budget=budget,
                                   pred_mean=x, obs_mean=y, weight=w))
    pd.DataFrame(curve_rows).to_csv(os.path.join(OUT, 'calibration_curves.csv'), index=False)
    print('written:', os.path.join(OUT, 'calibration.csv'))

    # figure: reliability curves (slack stress cell) + objective loss bars
    STRESS_DL, STRESS_BUDGET = 0.1, 30.0
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', lw=.8, label='perfect calibration')
    for name, color in [('oracle', '#27ae60'), ('gamma0.5', '#2980b9'), ('gamma2', '#c0392b'), ('noise0.2', '#f39c12')]:
        bp, bo, bw = curves[(name, STRESS_DL, STRESS_BUDGET)]
        ece_val = df[(df.variant == name) & (df.deadline == STRESS_DL) & (df.budget == STRESS_BUDGET)].ece_mean.iloc[0]
        ax.plot(bp, bo, 'o-', color=color, ms=4, lw=1.2,
                label=f'{name} (ECE {ece_val:.3f})')
    ax.set_xlabel('predicted $\\hat P_{on}$'); ax.set_ylabel('observed success frequency')
    ax.set_title('(a) reliability, 0.1 s deadline'); ax.legend(fontsize=7)
    ax = axes[1]
    w = 0.2
    for j, (name, color) in enumerate([('gamma0.5', '#2980b9'), ('gamma2', '#c0392b'), ('noise0.2', '#f39c12')]):
        xs = np.arange(len(CELLS))
        vals = [100 * df[(df.variant == name) & (df.deadline == dl) & (df.budget == b)].obj_loss_mean.iloc[0]
                for dl, b in CELLS]
        ax.bar(xs + (j - 1) * w, vals, w, label=name, color=color)
    ax.set_xticks(range(len(CELLS)))
    ax.set_xticklabels([f'{dl} s / {int(b)} MB' for dl, b in CELLS])
    ax.set_ylabel('assignment-objective loss (%)')
    ax.set_title('(b) cost of mis-calibrated $\\hat P_{on}$'); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PAPER, 'fig_calibration.png'), dpi=200)
    plt.close(fig)
    print('written:', os.path.join(PAPER, 'fig_calibration.png'))


if __name__ == '__main__':
    main()

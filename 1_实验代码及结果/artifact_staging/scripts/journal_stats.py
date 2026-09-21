"""Paired statistics for the journal revision (Computer Networks review point 8).

Combines the five-seed main run (seeds 0-4, outputs/real_fedlora_v3) with the
five-seed replication (seeds 5-9, outputs/wcnc_extra/real_v3_ext) into ten
common-random-number seeds, and reports paired Wilcoxon signed-rank tests with
rank-biserial effect sizes and bootstrap 95% CIs for:
  Adaptive_rank vs FedLoRA_r16 and Adaptive_rank vs FedAvg_full
on CIFAR-10 (nominal deadline) and CIFAR-10 under the 0.25 s stress deadline.

Per-seed metrics: final-round accuracy; mean timely ratio and bytes over rounds
30-39; delivered mean rank over rounds 30-39.

Writes nothing outside outputs/journal_stats/.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'outputs', 'journal_stats')
os.makedirs(OUT, exist_ok=True)


def load():
    a = pd.read_csv(os.path.join(ROOT, 'outputs/real_fedlora_v3/all_metrics.csv'))
    b = pd.read_csv(os.path.join(ROOT, 'outputs/wcnc_extra/real_v3_ext/all_metrics.csv'))
    d = pd.concat([a, b], ignore_index=True)
    return d[d.seed < 10]


def per_seed(d, ds):
    g = d[d.dataset == ds]
    acc = g.sort_values('round').groupby(['seed', 'method']).tail(1)[['seed', 'method', 'accuracy']]
    net = g[g['round'] >= 30].groupby(['seed', 'method'], as_index=False)[
        ['timely_ratio', 'bytes_norm', 'mean_rank']].mean()
    return acc.merge(net, on=['seed', 'method'])


def rank_biserial(x, y):
    """Matched-pairs rank-biserial correlation (positive = x > y)."""
    diff = x - y
    diff = diff[diff != 0]
    if len(diff) == 0:
        return 0.0
    ranks = pd.Series(np.abs(diff)).rank()
    return float((ranks[diff > 0].sum() - ranks[diff < 0].sum()) / ranks.sum())


def boot_ci(x, y, n=100000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n, len(x)))
    vals = (x[idx] - y[idx]) / y[idx]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    d = load()
    rows = []
    for ds in ('cifar10', 'cifar10_dl0.25'):
        ps = per_seed(d, ds)
        for n_seeds in (5, 10):
            sub = ps[ps.seed < n_seeds]
            for base in ('FedLoRA_r16', 'FedAvg_full'):
                for col in ('accuracy', 'timely_ratio', 'bytes_norm'):
                    x = sub[sub.method == 'Adaptive_rank'].sort_values('seed')[col].to_numpy()
                    y = sub[sub.method == base].sort_values('seed')[col].to_numpy()
                    stat, pv = wilcoxon(x - y)
                    rel = (x - y) / y
                    lo, hi = boot_ci(x, y)
                    rows.append(dict(dataset=ds, seeds=n_seeds, pair=f'Adaptive vs {base}',
                                     metric=col, mean_x=x.mean(), mean_y=y.mean(),
                                     rel_diff_pct=100 * rel.mean(), ci_lo_pct=100 * lo, ci_hi_pct=100 * hi,
                                     p=pv, r_rb=rank_biserial(x, y)))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, 'paired_stats.csv'), index=False)
    pd.set_option('display.width', 250)
    print(out.to_string(index=False, float_format=lambda v: f'{v:.4g}'))
    print('\nwritten:', os.path.join(OUT, 'paired_stats.csv'))


if __name__ == '__main__':
    main()

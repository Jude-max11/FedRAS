"""Paired statistics for the v4 (real-byte) journal revision.

Combines the five-seed main run (outputs/real_fedlora_v4) with the five-seed
replication (--ext, default outputs/wcnc_extra/real_v4_ext) into ten
common-random-number seeds, and reports paired Wilcoxon signed-rank tests
with rank-biserial effect sizes, bootstrap 95% CIs, and - for accuracy parity -
a TOST equivalence test against a pre-registered margin, for:
  Adaptive_rank vs FedLoRA_r16 and Adaptive_rank vs FedAvg_full
on CIFAR-10 (nominal deadline) and CIFAR-10 under the stress deadline.

Statistical fixes vs v3:
- bootstrap CIs now average WITHIN each resample (distribution of the mean),
  for both the mean of per-seed relative differences and the ratio of means;
- accuracy parity is claimed only if TOST rejects the pre-registered +/-0.5
  percentage-point margin (non-significance alone is never evidence of parity);
- mean rank is pooled delivered-weighted (zero-delivery rounds carry no rank).

Writes outputs/journal_stats/paired_stats_v4.csv. Writes nothing else.
"""
import argparse, os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ttest_1samp
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TOST_MARGIN = 0.005  # +/- 0.5 percentage points on accuracy


def load(main_csv, ext_csv):
    frames = [pd.read_csv(main_csv)]
    if os.path.abspath(ext_csv) != os.path.abspath(main_csv):
        frames.append(pd.read_csv(ext_csv))
    d = pd.concat(frames, ignore_index=True)
    return d[d.seed < 10]


def pooled_rank(g):
    """Delivered-weighted mean rank: zero-delivery rounds carry no rank."""
    x = g.dropna(subset=['mean_rank'])
    if x.empty:
        return np.nan
    return float((x.mean_rank * x.aggregated).sum() / x.aggregated.sum())


def per_seed(d, ds):
    g = d[d.dataset == ds]
    acc = g.sort_values('round').groupby(['seed', 'method']).tail(1)[['seed', 'method', 'accuracy']]
    net = g[g['round'] >= 30].groupby(['seed', 'method']).apply(
        lambda x: pd.Series({
            'timely_ratio': x.timely_ratio.mean(),
            'bytes_mb': x.bytes_mb.mean(),
            'mean_rank': pooled_rank(x),
        }), include_groups=False).reset_index()
    return acc.merge(net, on=['seed', 'method'])


def rank_biserial(x, y):
    """Matched-pairs rank-biserial correlation (positive = x > y)."""
    diff = x - y
    diff = diff[diff != 0]
    if len(diff) == 0:
        return 0.0
    ranks = pd.Series(np.abs(diff)).rank()
    return float((ranks[diff > 0].sum() - ranks[diff < 0].sum()) / ranks.sum())


def boot_mean_of_ratios(x, y, n=100000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n, len(x)))
    vals = ((x[idx] - y[idx]) / y[idx]).mean(axis=1)  # mean WITHIN each resample
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def boot_ratio_of_means(x, y, n=100000, seed=1):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n, len(x)))
    vals = (x[idx].mean(axis=1) - y[idx].mean(axis=1)) / y[idx].mean(axis=1)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def tost_paired(x, y, margin=TOST_MARGIN):
    """Two one-sided paired t-tests against +/- margin. Equivalence holds when
    the 90% CI of the mean difference lies inside (-margin, +margin)."""
    diff = x - y
    n = len(diff); sd = diff.std(ddof=1); mean = diff.mean()
    se = sd / np.sqrt(n)
    if se == 0:
        return (0.0 if abs(mean) < margin else 1.0), (mean - margin, mean + margin)
    p_lower = stats.t.sf((mean + margin) / se, df=n - 1)  # H0: mean <= -margin
    p_upper = stats.t.cdf((mean - margin) / se, df=n - 1)  # H0: mean >= +margin
    ci90 = (mean - stats.t.ppf(.95, n - 1) * se, mean + stats.t.ppf(.95, n - 1) * se)
    return float(max(p_lower, p_upper)), ci90


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--main', default=os.path.join(ROOT, 'outputs/real_fedlora_v4/all_metrics.csv'))
    p.add_argument('--ext', default=os.path.join(ROOT, 'outputs/wcnc_extra/real_v4_ext/all_metrics.csv'))
    p.add_argument('--stress', default='cifar10_dl0.1')
    p.add_argument('--out', default=os.path.join(ROOT, 'outputs/journal_stats/paired_stats_v4.csv'))
    a = p.parse_args()
    d = load(a.main, a.ext)
    rows = []
    for ds in ('cifar10', a.stress):
        ps = per_seed(d, ds)
        for n_seeds in (5, 10):
            sub = ps[ps.seed < n_seeds]
            for base in ('FedLoRA_r16', 'FedAvg_full'):
                for col in ('accuracy', 'timely_ratio', 'bytes_mb'):
                    x = sub[sub.method == 'Adaptive_rank'].sort_values('seed')[col].to_numpy()
                    y = sub[sub.method == base].sort_values('seed')[col].to_numpy()
                    stat, pv = wilcoxon(x - y)
                    lo_m, hi_m = boot_mean_of_ratios(x, y)
                    lo_r, hi_r = boot_ratio_of_means(x, y)
                    row = dict(dataset=ds, seeds=n_seeds, pair=f'Adaptive vs {base}',
                               metric=col, mean_x=x.mean(), mean_y=y.mean(),
                               rel_diff_mean_of_ratios_pct=100 * ((x - y) / y).mean(),
                               rel_diff_ratio_of_means_pct=100 * (x.mean() - y.mean()) / y.mean(),
                               ci_mean_of_ratios_lo_pct=100 * lo_m, ci_mean_of_ratios_hi_pct=100 * hi_m,
                               ci_ratio_of_means_lo_pct=100 * lo_r, ci_ratio_of_means_hi_pct=100 * hi_r,
                               p=pv, r_rb=rank_biserial(x, y))
                    if col == 'accuracy':
                        p_tost, ci90 = tost_paired(x, y)
                        row |= dict(tost_margin_pct=100 * TOST_MARGIN, tost_p=p_tost,
                                    acc_diff_ci90_lo_pct=100 * ci90[0], acc_diff_ci90_hi_pct=100 * ci90[1])
                    rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    pd.set_option('display.width', 250)
    print(out.to_string(index=False, float_format=lambda v: f'{v:.4g}'))
    print('\nwritten:', a.out)


if __name__ == '__main__':
    main()

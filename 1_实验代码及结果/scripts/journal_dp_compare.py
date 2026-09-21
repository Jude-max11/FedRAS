"""DP_cohort vs fair-cohort comparison in the real trainer (review point 5).

Loads the newly run full-scheduling arm (outputs/journal_dp/dp_seed*/metrics.csv,
exact per-round client-rank DP over all 20 clients under cap K=10 and byte
budget 8.2 MB) and the existing fair-cohort runs (Adaptive_rank, FedLoRA_r16,
FedAvg_full from outputs/real_fedlora_v3 seeds 0-4 + outputs/wcnc_extra/
real_v3_ext seeds 5-9), combines them into ten common-random-number seeds,
and reports per-seed metrics with paired statistics.

Writes outputs/journal_stats/dp_vs_faircohort.csv. Writes nothing else.
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'outputs', 'journal_stats')


def load_dp():
    frames = []
    for s in range(10):
        f = os.path.join(ROOT, f'outputs/journal_dp/dp_seed{s}/metrics.csv')
        d = pd.read_csv(f)
        d['dataset'] = 'cifar10'  # raw trainer CSV has no dataset column
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def main():
    dp = load_dp()
    a = pd.read_csv(os.path.join(ROOT, 'outputs/real_fedlora_v3/all_metrics.csv'))
    b = pd.read_csv(os.path.join(ROOT, 'outputs/wcnc_extra/real_v3_ext/all_metrics.csv'))
    fc = pd.concat([a, b], ignore_index=True)
    fc = fc[fc.seed < 10]
    d = pd.concat([fc, dp], ignore_index=True)
    d = d[d.dataset == 'cifar10']

    def per_seed(g, method):
        h = g[g.method == method]
        acc = h.sort_values('round').groupby('seed').tail(1).set_index('seed')['accuracy']
        net = h[h['round'] >= 30].groupby('seed')[['timely_ratio', 'bytes_norm', 'mean_rank', 'update_norm']].mean()
        return acc.to_frame('accuracy').join(net)

    ps = {m: per_seed(d, m) for m in ('DP_cohort', 'Adaptive_rank', 'FedLoRA_r16', 'FedAvg_full')}
    methods = [m for m in ('DP_cohort', 'Adaptive_rank', 'FedLoRA_r16', 'FedAvg_full') if m in ps and len(ps[m])]
    rows = []
    for m in methods:
        t = ps[m]
        rows.append(dict(method=m, n=len(t), acc_mean=t.accuracy.mean(), acc_std=t.accuracy.std(),
                         timely=t.timely_ratio.mean(), bytes=t.bytes_norm.mean(),
                         rank=t.mean_rank.mean(), upd_norm=t.update_norm.mean()))
    tab = pd.DataFrame(rows)
    print('== CIFAR-10 nominal, ten seeds (rounds 30-39 for network metrics) ==')
    print(tab.round(4).to_string(index=False))

    print('\n== paired tests (n=10) ==')
    tests = []
    for m in ('Adaptive_rank', 'FedLoRA_r16', 'FedAvg_full'):
        for col in ('accuracy', 'timely_ratio', 'bytes_norm'):
            x = ps['DP_cohort'][col].sort_index().to_numpy()
            y = ps[m][col].sort_index().to_numpy()
            if abs(ps['DP_cohort'][col].mean() - y.mean()) < 1e-12:
                continue
            stat, pv = wilcoxon(x - y)
            tests.append(dict(pair=f'DP vs {m}', metric=col,
                              mean_diff=x.mean() - y.mean(), p=pv))
    t = pd.DataFrame(tests)
    print(t.to_string(index=False))

    tab.to_csv(os.path.join(OUT, 'dp_vs_faircohort.csv'), index=False)
    t.to_csv(os.path.join(OUT, 'dp_vs_faircohort_tests.csv'), index=False)
    print('\nwritten:', os.path.join(OUT, 'dp_vs_faircohort.csv'))


if __name__ == '__main__':
    main()

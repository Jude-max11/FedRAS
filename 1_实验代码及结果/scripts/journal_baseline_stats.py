"""Ten-seed replication statistics for the external baselines (review point 8).

Combines the five-seed baseline runs (seeds 0-4, outputs/real_fedlora_v3,
datasets cifar10_ext / cifar10_ext_dl0.25) with the newly run seeds 5-9
(outputs/journal_ext/, raw trainer CSVs mapped to the same dataset names) and
reports:
  - five-seed means for cross-checking against Table tab:ext;
  - ten-seed per-method means and paired Wilcoxon tests of each baseline
    against Adaptive_rank (accuracy / timely ratio / bytes).

Writes outputs/journal_stats/baseline_10seed.csv. Writes nothing else.
"""
import os

import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'outputs', 'journal_stats')
NEW = {'cifar10_ext': 'cifar10', 'cifar10_ext_dl0.25': 'cifar10_dl0.25'}


def load():
    old = pd.read_csv(os.path.join(ROOT, 'outputs/real_fedlora_v3/all_metrics.csv'))
    rep = pd.read_csv(os.path.join(ROOT, 'outputs/wcnc_extra/real_v3_ext/all_metrics.csv'))
    keep = list(NEW) + ['cifar10', 'cifar10_dl0.25']  # mains hold the reference methods
    frames = [old[old.dataset.isin(keep)], rep[rep.dataset.isin(['cifar10', 'cifar10_dl0.25'])]]
    for ds_raw, ds_new in NEW.items():
        dl = '_dl0.25' if 'dl0.25' in ds_new else ''
        for s in range(5, 10):
            d = pd.read_csv(os.path.join(ROOT, f'outputs/journal_ext/{ds_new}_seed{s}/metrics.csv'))
            d['dataset'] = ds_raw
            frames.append(d)
    return pd.concat(frames, ignore_index=True)


def per_seed(d, ds, method):
    g = d[(d.dataset == ds) & (d.method == method)]
    acc = g.sort_values('round').groupby('seed').tail(1).set_index('seed')['accuracy']
    net = g[g['round'] >= 30].groupby('seed')[['timely_ratio', 'bytes_norm', 'mean_rank']].mean()
    return acc.to_frame('accuracy').join(net)


def main():
    d = load()
    for ds, dl in [('cifar10_ext', 'nominal'), ('cifar10_ext_dl0.25', '0.25 s')]:
        ds_ref = ds.replace('_ext', '')  # Adaptive_rank / FedLoRA_r16 live in the main datasets
        print(f'\n== {ds} ({dl}) ==')
        rows = []
        for m in ('SA_rank', 'Oort_utility', 'JointAB', 'Adaptive_rank', 'FedLoRA_r16'):
            src = ds if m in ('SA_rank', 'Oort_utility', 'JointAB') else ds_ref
            p5 = per_seed(d[d.seed < 5], src, m)
            p10 = per_seed(d, src, m)
            rows.append(dict(method=m, n5_acc=p5.accuracy.mean(), n5_timely=p5.timely_ratio.mean(),
                             n5_bytes=p5.bytes_norm.mean(), n5_rank=p5.mean_rank.mean(),
                             n10_acc=p10.accuracy.mean(), n10_timely=p10.timely_ratio.mean(),
                             n10_bytes=p10.bytes_norm.mean(), n10_rank=p10.mean_rank.mean()))
        t = pd.DataFrame(rows)
        print(t.round(4).to_string(index=False))
        base = per_seed(d, ds_ref, 'Adaptive_rank')
        print(f'\n  paired vs Adaptive_rank (10 seeds):')
        for m in ('SA_rank', 'Oort_utility', 'JointAB'):
            pm = per_seed(d, ds, m)
            for col in ('accuracy', 'timely_ratio', 'bytes_norm'):
                x = pm[col].sort_index().to_numpy()
                y = base[col].sort_index().to_numpy()
                _, pv = wilcoxon(x - y)
                print(f'    {m:13s} {col:12s} diff={x.mean() - y.mean():+.4f}  p={pv:.4f}')
    d[d.seed >= 5].to_csv(os.path.join(OUT, 'baseline_ext_seeds59.csv'), index=False)
    print('\nwritten:', os.path.join(OUT, 'baseline_ext_seeds59.csv'))


if __name__ == '__main__':
    main()

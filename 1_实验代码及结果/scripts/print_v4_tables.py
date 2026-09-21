"""Generate every journal-paper number from the v4 (real-byte) data.

Tables produced (all means over seeds; network metrics averaged over rounds
30-39; mean rank is delivered-weighted; acc measured at the final round):
  tab_trainer  : nominal 2.0 s, five seeds, five methods
  tab_stress   : deadlines 0.1 s and 0.05 s
  tab_extended : loose 10 s, mobility 0.7, alpha 0.1, cross-dataset
  tab_ext      : external baselines (SA/Oort/JointAB) nominal + stress
  tab_dpvfair  : DP_cohort vs DP_same vs fair-cohort vs r16 (10 seeds)
  tab_sens     : rmax 8/32, cohort 5/20, alpha 1.0 (nominal)
Writes outputs/journal_stats/v4_tables.txt (paper-ready layout) and also
prints to stdout. Read-only w.r.t. the paper directories.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'outputs', 'journal_stats')

METHODS_MAIN = ['FedAvg_full', 'FedLoRA_r4', 'FedLoRA_r16', 'Random_rank', 'Adaptive_rank']
LABEL = {'FedAvg_full': 'FedAvg (full model)', 'FedLoRA_r4': 'FedLoRA r=4',
         'FedLoRA_r16': 'FedLoRA r=16', 'Random_rank': 'Random rank',
         'Adaptive_rank': 'FedRAS (ours)', 'SA_rank': 'SA-FedLoRA-style annealing',
         'Oort_utility': 'Oort-style utility, fixed rank', 'JointAB': 'Joint A/B, plain averaging',
         'DP_cohort': 'Full DP (grid assignment, 1.0 MB)', 'DP_same': 'Same-cohort DP (1.0 MB)',
         'FedAvg_offline': 'FedAvg offline control'}


def load():
    d = pd.read_csv(os.path.join(ROOT, 'outputs/real_fedlora_v4/all_metrics.csv'))
    d['mean_rank'] = pd.to_numeric(d.mean_rank, errors='coerce')
    return d


def net_stats(g):
    gm = g[g['round'] >= 30]
    wr = (gm.mean_rank * gm.aggregated).sum() / max(1, gm.aggregated.sum())
    acc = g.sort_values('round').groupby(['run', 'seed']).tail(1).accuracy
    return dict(acc=acc.mean(), acc_std=acc.std(ddof=1) if len(acc) > 1 else 0.0,
                timely=gm.timely_ratio.mean(), bytes=gm.bytes_mb.mean(), rank=wr,
                deliv=gm.aggregated.mean())


def cell(d, ds, methods, seeds=None):
    g = d[d.dataset == ds]
    if seeds is not None:
        g = g[g.seed.isin(seeds)]
    rows = {}
    for m in methods:
        gm = g[g.method == m]
        if gm.empty:
            continue
        s = net_stats(gm)
        rows[m] = f'{s["acc"]:.4f}±{s["acc_std"]:.4f}  {s["timely"]:.2f}  {s["bytes"]:.2f}  {s["rank"]:.1f}'
    return rows


def main():
    d = load()
    out = []

    out.append('== tab_trainer: nominal 2.0 s (5 seeds) ==')
    out.append('method                        acc±std        timely bytes  rank')
    for m, s in cell(d, 'cifar10', METHODS_MAIN).items():
        out.append(f'{LABEL[m]:28s} {s}')

    out.append('\n== tab_stress: deadline 0.1 s | 0.05 s (5 seeds) ==')
    for m in METHODS_MAIN:
        a = cell(d, 'cifar10_dl0.1', [m]).get(m, '-')
        b = cell(d, 'cifar10_dl0.05', [m]).get(m, '-')
        out.append(f'{LABEL[m]:28s} {a}   |   {b}')

    out.append('\n== tab_extended: acc±std only (5 seeds) ==')
    for ds in ['cifar10_dl10.0', 'cifar10_mob0.7', 'cifar10_alpha0.1', 'cifar10_alpha1.0',
               'cifar100', 'cifar100_dl0.1', 'mnist', 'mnist_dl0.1']:
        c = cell(d, ds, METHODS_MAIN)
        if not c:
            out.append(f'{ds}: (missing)')
            continue
        parts = [f'{LABEL[m].split()[0]}={v.split("±")[0]}' for m, v in c.items()]
        out.append(f'{ds:20s} ' + '  '.join(parts))

    out.append('\n== tab_ext: external baselines, nominal | 0.1 s (5 seeds) ==')
    ext = ['SA_rank', 'Oort_utility', 'JointAB', 'FedLoRA_r16', 'Adaptive_rank']
    for m in ext:
        a = cell(d, 'cifar10', [m]).get(m, '-')
        b = cell(d, 'cifar10_dl0.1', [m]).get(m, '-')
        out.append(f'{LABEL[m]:28s} {a}   |   {b}')
    for m in ['SA_rank', 'Oort_utility', 'JointAB']:
        a = cell(d, 'cifar10_ext', [m]).get(m, '-')
        b = cell(d, 'cifar10_ext_dl0.1', [m]).get(m, '-')
        out.append(f'{LABEL[m]:28s} {a}   |   {b}')

    out.append('\n== tab_dpvfair: 10 seeds, nominal ==')
    for m, ds_tag in [('DP_cohort', 'dp'), ('DP_same', 'dpsame'),
                      ('Adaptive_rank', 'cifar10'), ('FedLoRA_r16', 'cifar10')]:
        s = cell(d, ds_tag, [m], seeds=range(10)).get(m, '-')
        out.append(f'{LABEL[m]:32s} {s}')
    out.append('\n== dp arms under stress 0.1 s (5 seeds) ==')
    for m, ds_tag in [('DP_cohort', 'dp_dl0.1'), ('DP_same', 'dpsame_dl0.1'),
                      ('Adaptive_rank', 'cifar10_dl0.1')]:
        s = cell(d, ds_tag, [m]).get(m, '-')
        out.append(f'{LABEL[m]:32s} {s}')

    out.append('\n== tab_sens: nominal, acc / timely / bytes / rank ==')
    for ds in ['cifar10_rmax8', 'cifar10_rmax32', 'cifar10_pr5', 'cifar10_pr20', 'cifar10_sagin']:
        for m, s in cell(d, ds, ['Adaptive_rank']).items():
            out.append(f'{ds:18s} {s}')

    # sagin per-group delivery (from link_outcomes.csv of the sagin runs)
    out.append('\n== sagin per-group delivered ratio ==')
    import glob, csv
    frames = []
    for f in glob.glob(os.path.join(ROOT, 'outputs/real_fedlora_v4/cifar10_sagin_seed*/link_outcomes.csv')):
        frames.append(pd.read_csv(f))
    if frames:
        o = pd.concat(frames)
        grp = o.groupby('group').agg(delivered=('delivered', 'mean'), n=('delivered', 'size'))
        for gname, r in grp.iterrows():
            out.append(f'  {gname:10s} delivered={r.delivered:.3f} (n={int(r.n)})')

    text = '\n'.join(out)
    with open(os.path.join(OUT, 'v4_tables.txt'), 'w') as f:
        f.write(text + '\n')
    print(text)
    print('\nwritten:', os.path.join(OUT, 'v4_tables.txt'))


if __name__ == '__main__':
    main()

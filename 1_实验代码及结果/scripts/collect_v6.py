#!/usr/bin/env python3
"""Aggregate outputs/real_v6 into the paper's tables.

Conventions (Sec. 6.1): accuracy = mean over the final 10 rounds; timely ratio,
attempted MB and delivered rank = mean over rounds 30-39, rank weighted by the
number of aggregated updates; kappa = timely * sqrt(rank / r_max).

Usage: python scripts/collect_v6.py [--root outputs/real_v6] [--latex]
"""
import argparse, csv, glob, math, os, statistics as st

LAB = {'FedLoRA_r4': 'FedLoRA-4', 'FedLoRA_r16': 'FedLoRA-16', 'Random_rank': 'Random',
       'SA_mono': 'SA-FedLoRA', 'Oort_utility': 'Oort', 'ROBA_style': 'EL',
       'Adaptive_rank': 'FedRAS', 'FedAvg_full': 'FedAvg', 'DP_cohort': 'Full DP',
       'DP_same': 'Same-cohort DP', 'Greedy_value': 'Value greedy',
       'Greedy_vpb': 'Value/byte greedy', 'JointAB': 'Joint A/B'}


def num(x):
    try:
        return float(x)
    except Exception:
        return None


def stats(d, method, window=30, rmax=16.0):
    f = os.path.join(d, 'metrics.csv')
    if not os.path.exists(f):
        return None
    rows = [r for r in csv.DictReader(open(f)) if r['method'] == method]
    rows = [r for r in rows if num(r['accuracy']) is not None]
    if len(rows) < 20:
        return None
    rounds = sorted(int(r['round']) for r in rows)
    last10 = set(rounds[-10:])
    acc = st.mean([num(r['accuracy']) for r in rows if int(r['round']) in last10])
    net = [r for r in rows if int(r['round']) >= window]
    if not net:
        return None
    tim = st.mean([num(r['timely_ratio']) for r in net])
    mb = st.mean([num(r['bytes_mb']) for r in net])
    rk = [(num(r['mean_rank']), num(r['aggregated'])) for r in net
          if num(r['mean_rank']) is not None and num(r['aggregated']) is not None]
    w = sum(a for _, a in rk)
    rank = sum(r * a for r, a in rk) / w if w > 0 else float('nan')
    return dict(acc=acc, timely=tim, mb=mb, rank=rank,
                kappa=tim * math.sqrt(rank / rmax), n=len(rows))


def table(root, pattern, methods, title):
    dirs = sorted(glob.glob(os.path.join(root, pattern)))
    cells = {}
    for d in dirs:
        tag = os.path.basename(d)
        for m in methods:
            s = stats(d, m)
            if s:
                cells.setdefault(m, []).append(s)
    if not cells:
        print(f"\n## {title}: (no runs yet)")
        return
    print(f"\n## {title}  ({len(dirs)} runs)")
    print(f"{'method':16s} {'n':>3s} {'acc':>7s} {'sd':>7s} {'timely':>7s} {'MB':>6s} {'rank':>6s} {'kappa':>6s}")
    for m in methods:
        if m not in cells:
            continue
        v = cells[m]
        a = [x['acc'] for x in v]
        sd = st.stdev(a) if len(a) > 1 else 0.0
        print(f"{LAB.get(m,m):16s} {len(v):3d} {st.mean(a):7.4f} {sd:7.4f} "
              f"{st.mean([x['timely'] for x in v]):7.3f} {st.mean([x['mb'] for x in v]):6.2f} "
              f"{st.mean([x['rank'] for x in v]):6.2f} {st.mean([x['kappa'] for x in v]):6.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='outputs/real_v6')
    a = ap.parse_args()
    R = a.root
    M7 = ['FedLoRA_r4', 'FedLoRA_r16', 'Random_rank', 'SA_mono', 'Oort_utility', 'ROBA_style', 'Adaptive_rank']
    M4 = ['FedLoRA_r4', 'FedLoRA_r16', 'Random_rank', 'Adaptive_rank']
    for ds in ('cifar10', 'cifar100', 'mnist'):
        for dl in ('2.0', '0.1', '0.05'):
            table(R, f'p1_tab3_{ds}_dl{dl}_seed*', M7 + ['FedAvg_full'], f'Table 3  {ds} @ {dl}s')
    for tag in ('weakdisj_2.0', 'weakdisj_0.1', 'vit_0.5'):
        table(R, f'p1_tab4_{tag}_seed*', M7, f'Table 4  {tag}')
    for dl in ('2.0', '0.1'):
        table(R, f'p2_tab5c_sagin_dl{dl}_seed*', M7, f'Table 5c SAGIN @ {dl}s')
    for ds in ('cifar10', 'cifar100', 'mnist'):
        for sc in ('dl10.0', 'mob0.7', 'alpha0.1', 'alpha1.0'):
            table(R, f'p2_tab5d_{ds}_{sc}_seed*', M4, f'Table 5d {ds} {sc}')
    for tag in ('nocap', 'nomom', 'lr02_frozen', 'lr02_joint', 'jointab_ab', 'rrnomask_dl0.1'):
        m = 'JointAB' if 'joint' in tag else ('Random_rank' if 'rrnomask' in tag else 'Adaptive_rank')
        table(R, f'p3_tab6a_{tag}_seed*', [m], f'Table 6a {tag}')
    for tag, m in (('dp_cohort_1.0', 'DP_cohort'), ('dp_same_1.0', 'DP_same'),
                   ('perclient_1.0', 'Adaptive_rank'), ('dp_cohort_0.1', 'DP_cohort'),
                   ('dp_same_0.1', 'DP_same'), ('perclient_0.1', 'Adaptive_rank'),
                   ('dp_0.4', 'DP_cohort'), ('greedy_v_0.4', 'Greedy_value'),
                   ('greedy_vb_0.4', 'Greedy_vpb'), ('perclient_0.4', 'Adaptive_rank'),
                   ('fedlora16_0.4', 'FedLoRA_r16')):
        table(R, f'p3_tab6b_{tag}_seed*', [m], f'Table 6 {tag}')
        table(R, f'p3_tab6c_{tag}_seed*', [m], f'Table 6c {tag}')
    for tag in ('rmax8', 'rmax32', 'k5', 'k20'):
        table(R, f'p3_fig4_{tag}_seed*', ['Adaptive_rank'], f'Fig.4 {tag}')
    for tag in ('weak1', 'weak5'):
        table(R, f'p3_fig5_{tag}_seed*', ['Adaptive_rank'], f'Fig.5 {tag}')


if __name__ == '__main__':
    main()

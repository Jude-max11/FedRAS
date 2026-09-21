"""Print LaTeX-ready stress-table values from the v3 summary (5 seeds each)."""
import pandas as pd

d = pd.read_csv('outputs/real_fedlora_v3/all_metrics.csv')
d = d[d['round'] >= 30]  # average network metrics over the last 10 rounds
last = d.sort_values('round').groupby(['dataset', 'seed', 'method']).tail(1)
net = d.groupby(['dataset', 'seed', 'method'])[['timely_ratio', 'bytes_norm', 'mean_rank', 'aggregated']].mean().reset_index()
METH = ['FedAvg_full', 'FedLoRA_r4', 'FedLoRA_r16', 'Random_rank', 'Adaptive_rank']
NAME = {'FedAvg_full': 'FedAvg (full model)', 'FedLoRA_r4': 'FedLoRA $r{=}4$',
        'FedLoRA_r16': 'FedLoRA $r{=}16$', 'Random_rank': 'Random rank',
        'Adaptive_rank': 'Adaptive-rank (ours)'}

for ds in ['cifar10_dl0.5', 'cifar10_dl0.25', 'cifar10_dl2.0', 'cifar10_mob0.7',
           'cifar10_alpha0.1', 'cifar100_dl0.25', 'mnist_dl0.25']:
    g = last[last.dataset == ds]
    if g.empty:
        print(f'== {ds}: no data yet'); continue
    n = g.seed.nunique()
    print(f'== {ds} ({n} seeds)')
    for m in METH:
        h = g[g.method == m]
        n = net[(net.dataset == ds) & (net.method == m)]
        if h.empty or n.empty: continue
        acc, accs = h.accuracy.mean(), h.accuracy.std()
        print(f"  {NAME[m]:<22} {acc:.4f}+-{accs:.4f}  timely {n.timely_ratio.mean():.2f}  bytes {n.bytes_norm.mean():.2f}  rank {n['mean_rank'].mean():.1f}  agg {n.aggregated.mean():.1f}")

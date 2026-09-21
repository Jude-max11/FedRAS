"""Paired statistical tests for the paper: Adaptive_rank vs fixed r16 and vs
FedAvg_full on CIFAR-10 (nominal + dl0.25), five seeds. Wilcoxon signed-rank
on per-seed final-round accuracy; network metrics are last-10-round means.
Also prints the scheduler-decision overhead recorded in run configs.
"""
import glob, json
import pandas as pd
from scipy.stats import wilcoxon

d = pd.read_csv('outputs/real_fedlora_v3/all_metrics.csv')
d = d[d['round'] >= 30]
acc = d.sort_values('round').groupby(['dataset', 'seed', 'method']).tail(1)
net = d.groupby(['dataset', 'seed', 'method'])[['timely_ratio', 'bytes_norm']].mean().reset_index()

def paired(ds, a, b, col, src):
    x = src[(src.dataset == ds) & (src.method == a)].sort_values('seed')[col].values
    y = src[(src.dataset == ds) & (src.method == b)].sort_values('seed')[col].values
    n = min(len(x), len(y))
    if n < 3:
        return None
    stat, pv = wilcoxon(x[:n] - y[:n])
    return pv, float((x[:n] - y[:n]).mean()), n

for ds in ('cifar10', 'cifar10_dl0.25'):
    print(f'== {ds} ==')
    for b in ('FedLoRA_r16', 'FedAvg_full'):
        for col in ('accuracy', 'timely_ratio', 'bytes_norm'):
            src = acc if col == 'accuracy' else net
            r = paired(ds, 'Adaptive_rank', b, col, src)
            if r:
                pv, diff, n = r
                print(f'  Adaptive vs {b:<12} {col:<12} diff {diff:+.4f}  Wilcoxon p={pv:.3f} (n={n})')

times = []
for f in glob.glob('outputs/real_fedlora_v3/*/config.json'):
    c = json.load(open(f))
    if 'sched_ms_total' in c and c.get('dataset') == 'cifar10' and 'seed' in c and c.get('rounds') == 40:
        per_round = c.get('per_round', 10)
        times.append(c['sched_ms_total'] / (40 * per_round) * 1000)  # us per decision
if times:
    print(f'\nscheduler decision overhead: mean {sum(times)/len(times):.1f} us per client-rank decision '
          f'({len(times)} runs), vs DP full-scheduling 33-836 ms per round')

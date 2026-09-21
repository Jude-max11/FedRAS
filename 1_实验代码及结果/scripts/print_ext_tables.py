"""LaTeX-ready values for the extended-evaluation tables (external baselines,
component ablations, sensitivity) from the v3 summary."""
import pandas as pd

d = pd.read_csv('outputs/real_fedlora_v3/all_metrics.csv')
d['is_stress'] = d.dataset.str.contains('_')
acc = d.sort_values('round').groupby(['dataset', 'seed', 'method']).tail(1)
net = d[d['round'] >= 30].groupby(['dataset', 'seed', 'method'])[['timely_ratio', 'bytes_norm', 'mean_rank', 'aggregated']].mean().reset_index()
accm = acc.groupby(['dataset', 'method'])['accuracy'].agg(['mean', 'std'])
netm = net.groupby(['dataset', 'method'])[['timely_ratio', 'bytes_norm', 'mean_rank', 'aggregated']].mean()

def row(ds, m):
    a = accm.loc[(ds, m)] if (ds, m) in accm.index else None
    n = netm.loc[(ds, m)] if (ds, m) in netm.index else None
    if a is None or n is None:
        return None
    return (f"{a['mean']:.4f}+-{a['std']:.4f}", f"{n.timely_ratio:.2f}", f"{n.bytes_norm:.2f}", f"{n['mean_rank']:.1f}", f"{n.aggregated:.1f}")

print('== Table 6: external baselines (nominal cifar10) ==')
for m in ['SA_rank', 'Oort_utility', 'JointAB']:
    print(f"  {m:<14}", row('cifar10_ext', m))
for m in ['Adaptive_rank', 'FedLoRA_r16', 'FedAvg_full']:
    print(f"  {m:<14}", row('cifar10', m))
print('== Table 6: external baselines (dl0.25) ==')
for m in ['SA_rank', 'Oort_utility', 'JointAB']:
    print(f"  {m:<14}", row('cifar10_ext_dl0.25', m))
for m in ['Adaptive_rank', 'FedLoRA_r16', 'FedAvg_full']:
    print(f"  {m:<14}", row('cifar10_dl0.25', m))
print('== Table 7: ablations (nominal cifar10) ==')
for ds, name in [('cifar10', 'ours (full)'), ('cifar10_nomask', 'w/o rank mask'),
                 ('cifar10_nocap', 'w/o capacity term'), ('cifar10_nomom', 'w/o server momentum'),
                 ('cifar10_ext', 'w/o frozen A (JointAB)')]:
    key = 'JointAB' if name.startswith('w/o frozen') else 'Adaptive_rank'
    print(f"  {name:<24}", row(ds, key))
print('== Sensitivity: rmax (Adaptive_rank) ==')
for ds in ['cifar10_rmax8', 'cifar10', 'cifar10_rmax32']:
    print(f"  {ds:<16}", row(ds, 'Adaptive_rank'))
print('== Sensitivity: alpha (Adaptive_rank) ==')
for ds in ['cifar10_alpha0.1', 'cifar10', 'cifar10_alpha1.0']:
    print(f"  {ds:<20}", row(ds, 'Adaptive_rank'))
print('== Sensitivity: cohort K (Adaptive_rank) ==')
for ds in ['cifar10_pr5', 'cifar10', 'cifar10_pr20']:
    print(f"  {ds:<16}", row(ds, 'Adaptive_rank'))

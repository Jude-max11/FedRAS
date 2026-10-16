"""Analyze the four review-fix batches once runs complete."""
import glob, os
import pandas as pd
from scipy.stats import wilcoxon

V4 = 'outputs/real_fedlora_v4'
def load(tag):
    fs = sorted(glob.glob(f'{V4}/{tag}/metrics.csv'))
    d = pd.concat([pd.read_csv(f) for f in fs])
    d = d[d['round'] >= 30]
    g = d.groupby(['method', 'seed'])[['accuracy', 'timely_ratio', 'bytes_mb', 'mean_rank']].mean()
    return g, len(fs)

print("=== A. de-circularized mismatch arm (0.1 s) ===")
g, n = load('mm_dl0.1_seed?')
for m in ['FedAvg_full', 'FedLoRA_r4', 'FedLoRA_r16', 'Random_rank', 'Adaptive_rank']:
    try:
        s = g.xs(m)
        print(f"  {m:14s} acc {s.accuracy.mean():.4f}+-{s.accuracy.std():.4f}  timely {s.timely_ratio.mean():.3f}+-{s.timely_ratio.std():.3f}  MB {s.bytes_mb.mean():.3f}  rank {s['mean_rank'].mean():.1f}")
    except KeyError: pass
try:
    fr, r16 = g.xs('Adaptive_rank'), g.xs('FedLoRA_r16')
    d1 = fr.timely_ratio - r16.timely_ratio
    print(f"  paired vs r16: timely +{(fr.timely_ratio/r16.timely_ratio).sub(1).mean()*100:.0f}% ({(d1>0).sum()}/{n}); bytes {1-(fr.bytes_mb/r16.bytes_mb).mean():.0%} saved")
except KeyError: pass

print("\n=== B. weak backbone, 10 seeds ===")
for tag, lab in [('weak_seed?', 'nominal'), ('weak_dl0.1_seed?', 'stress')]:
    g, n = load(tag)
    print(f"-- {lab} ({n} seeds)")
    for m in ['FedLoRA_r4', 'FedLoRA_r16', 'Adaptive_rank', 'SA_rank', 'Oort_utility', 'JointAB']:
        try:
            s = g.xs(m)
            print(f"  {m:14s} acc {s.accuracy.mean():.4f}+-{s.accuracy.std():.4f}  timely {s.timely_ratio.mean():.2f}  MB {s.bytes_mb.mean():.2f}")
        except KeyError: pass
    try:
        fr, sa, r16, r4 = g.xs('Adaptive_rank'), g.xs('SA_rank'), g.xs('FedLoRA_r16'), g.xs('FedLoRA_r4')
        for nm, o in [('SA', sa), ('r4', r4), ('r16', r16)]:
            d = fr.accuracy - o.accuracy
            print(f"  FedRAS-{nm}: {d.mean()*100:+.2f}pp ({(d>0).sum()}/{n})")
    except KeyError: pass

print("\n=== C. SAGIN at 0.1 s ===")
for f in sorted(glob.glob(f'{V4}/cifar10_sagin_dl0.1_seed?/link_outcomes.csv')):
    d = pd.read_csv(f)
    fin = d[d['round'] >= 30]
    print(' ', os.path.basename(os.path.dirname(f)), fin.groupby('group')['delivered'].mean().round(3).to_dict())

print("\n=== D. unmasked normalizer at weak stress ===")
gm, _ = load('weak_dl0.1_seed?')
gu, _ = load('weak_dl0.1_nomask_seed?')
for m in ['FedLoRA_r4', 'FedLoRA_r16', 'Adaptive_rank']:
    try:
        a, b = gm.xs(m), gu.xs(m)
        print(f"  {m:14s} masked {a.accuracy.mean():.4f}  unmasked {b.accuracy.mean():.4f}  diff {(a.accuracy-b.accuracy).mean()*100:+.2f}pp")
    except KeyError: pass

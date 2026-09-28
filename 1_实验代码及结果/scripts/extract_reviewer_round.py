"""Extract reviewer-round experiment results into paper-ready numbers."""
import csv, glob
import numpy as np

V4 = 'outputs/real_fedlora_v4'

def runs(pat):
    return sorted(glob.glob(f'{V4}/{pat}/metrics.csv'))

def finals(fs, method, rounds=range(30, 40)):
    """Per-seed (final_acc, timely, bytes, mean_rank) averaged over `rounds`."""
    out = []
    for f in fs:
        rows = [r for r in csv.DictReader(open(f)) if r['method'] == method and int(r['round']) in rounds]
        if not rows:
            continue
        last = [r for r in csv.DictReader(open(f)) if r['method'] == method and int(r['round']) == 39]
        out.append(dict(
            acc=float(last[0]['accuracy']),
            acc_all=[float(r['accuracy']) for r in csv.DictReader(open(f)) if r['method'] == method],
            timely=float(np.mean([float(r['timely_ratio']) for r in rows])),
            bytes=float(np.mean([float(r['bytes_mb']) for r in rows])),
            rank=float(np.mean([float(r['mean_rank']) for r in rows if r['mean_rank'] != ''])),
        ))
    return out

def show(label, fs, method):
    rs = finals(fs, method)
    if not rs:
        print(f'{label} {method}: NO DATA'); return
    acc = np.array([r['acc'] for r in rs]); t = np.array([r['timely'] for r in rs])
    b = np.array([r['bytes'] for r in rs]); rk = np.array([r['rank'] for r in rs])
    print(f'{label:24s} {method:16s} n={len(rs)} acc={acc.mean():.4f}+-{acc.std(ddof=1):.4f} '
          f'timely={t.mean():.2f} bytes={b.mean():.3f} rank={rk.mean():.1f}')
    return rs

for tag, pat in (('dl0.1', 'fp16_dl0.1_seed*'), ('nominal', 'fp16_seed*')):
    fs = runs(pat)
    show(f'FP16 {tag}', fs, 'FedLoRA_r16')
for tag, pat in (('dl0.1', 'fedras_fp16_dl0.1_seed*'), ('nominal', 'fedras_fp16_seed*')):
    fs = runs(pat)
    show(f'FedRAS-FP16 {tag}', fs, 'Adaptive_rank')
# weak backbone: capacity payoff
for tag, pat in (('weak nominal', 'weak_seed*'), ('weak dl0.1', 'weak_dl0.1_seed*')):
    fs = runs(pat)
    for m in ('FedLoRA_r4', 'FedLoRA_r16', 'Adaptive_rank'):
        show(f'{tag}', fs, m)
# FedAvg lr probe
fs = runs('fedavg_lr10_seed*')
show('lr1.0 nominal', fs, 'FedAvg_full')
# trajectory shape for the lr probe: min acc over rounds (stability)
for f in fs:
    accs = [float(r['accuracy']) for r in csv.DictReader(open(f)) if r['method'] == 'FedAvg_full']
    if accs:
        print(f'  {f.split("/")[-2]}: min_acc={min(accs):.4f} final={accs[-1]:.4f} drop={accs[0]-min(accs):+.4f}')

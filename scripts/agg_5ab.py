#!/usr/bin/env python3
"""Aggregate the 500-client simulator and the solver study into Table 5(a)/(b) rows."""
import csv, glob, os, statistics as st, sys

R = '/root/autodl-tmp/FedRAS/outputs/real_v6'
LABELS = [('FedAvg', 'FedAvg'), ('FedLoRA_r4', r'\mbox{FedLoRA-4}'),
          ('FedLoRA_r16', r'\mbox{FedLoRA-16}'), ('Random_rank', 'Random'),
          ('Reliability_only', '仅送达'), ('SA_mono', r'\mbox{SA-FedLoRA}'),
          ('ROBA_style', 'EL'), ('Adaptive_rank', 'FedRAS（本文）')]


def sim_rows(tag, lo=None):
    files = glob.glob(os.path.join(R, 'sim', tag, 'round_metrics_seed*.csv'))
    if not files:
        return None
    per = {}
    for f in files:
        rows = list(csv.DictReader(open(f)))
        if not rows:
            continue
        nmax = max(int(r['round']) for r in rows)
        cut = (nmax - 29) if lo is None else lo
        for r in rows:
            if int(r['round']) < cut:
                continue
            m = r['method']
            per.setdefault(m, {'acc': [], 'tim': [], 'mb': []})
            per[m]['acc'].append(float(r['accuracy']) * 100)
            per[m]['tim'].append(float(r['timely_ratio']))
            per[m]['mb'].append(float(r['bytes_mb']))
    out = {}
    for m, v in per.items():
        out[m] = (st.mean(v['acc']) / 100, st.mean(v['tim']), st.mean(v['mb']))
    return out


TAGS = [('nominal', '名义 2.0 s'), ('dl0.1', '截止 0.1 s'), ('budget1.2', '预算 1.2 MB')]
print('=== Table 5(a): sim (last 30 rounds) ===')
res = {t: sim_rows(t) for t, _ in TAGS}
for key, lab in LABELS:
    cells = []
    for t, _ in TAGS:
        v = (res[t] or {}).get(key)
        cells.append(f'{v[0]:.3f} & {v[1]:.2f} & {v[2]:.1f}' if v else '-- & & ')
    print(f'{lab} & ' + ' & '.join(cells) + r' \\')

print()
print('=== solver_study.csv ===')
p = os.path.join(R, 'solver_study.csv')
if not os.path.exists(p):
    print('(not yet)')
    sys.exit(0)
rows = list(csv.DictReader(open(p)))
print('columns:', sorted(rows[0].keys()) if rows else 'empty')
keys = sorted({(r.get('deadline'), r.get('budget')) for r in rows}, key=lambda x: (float(x[0]), float(x[1])))
methods = sorted({r.get('method') for r in rows})
print('methods:', methods)
for dl, bud in keys:
    sub = [r for r in rows if r.get('deadline') == dl and r.get('budget') == bud]
    line = f'dl={dl} budget={bud}  n={len(sub)}'
    print(line)
    for m in methods:
        mm = [r for r in sub if r.get('method') == m]
        if not mm:
            continue
        obj = st.mean([float(r['objective']) for r in mm]) if 'objective' in mm[0] else None
        gap = st.mean([float(r['gap_pct']) for r in mm]) if 'gap_pct' in mm[0] else None
        ms = st.mean([float(r['ms']) for r in mm]) if 'ms' in mm[0] else None
        print(f'   {m:22s} obj={obj} gap={gap} ms={ms}')

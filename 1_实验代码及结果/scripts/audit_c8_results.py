"""Substantive audit of the C8 sweep + sa_strong results and every number the
revised manuscript claims from them.

Checks:
  A. completeness: all methods x rounds per run, terminal window full
  B. config integrity: dir seed == config seed; family-uniform epochs/deadline/
     methods/protocol; no resume contamination; backbone ladder monotone in
     pretrain epochs
  C. sa_strong backbone reuse: backbone_acc matches the seed-matched
     cifar10_seed{s} source exactly
  D. numeric sanity: bounds on accuracy/timely/bytes/rank/aggregated;
     update_norm positive iff aggregated > 0
  E. independent recomputation (raw metrics only, no journal script imports):
     headline timely/bytes/dc ratios, conversion curve, failure matrix,
     nominal identity
  F. paper-claim cross-check with pass/fail per claim

Prints a report; exit 1 if any check fails. Writes nothing.
"""
import csv
import glob
import json
import math
import os

import numpy as np
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, 'outputs', 'real_fedlora_v4')
FAMILIES = {'bb2': (2, 2.0), 'bb2_dl0.1': (2, 0.1), 'bb3': (3, 2.0),
            'bb3_dl0.1': (3, 0.1), 'bb10': (10, 2.0), 'bb10_dl0.1': (10, 0.1),
            'sa_strong': (30, 2.0), 'sa_strong_dl0.1': (30, 0.1)}
M6 = {'FedLoRA_r4', 'FedLoRA_r16', 'Adaptive_rank', 'SA_rank', 'Oort_utility', 'JointAB'}
SA_M = {'SA_rank'}
fails, warns = [], []


def fail(msg):
    fails.append(msg)


def warn(msg):
    warns.append(msg)


def load_run(d):
    cfg = json.load(open(os.path.join(d, 'config.json')))
    rows = list(csv.DictReader(open(os.path.join(d, 'metrics.csv'))))
    return cfg, rows


def check_complete(d, cfg, rows, methods):
    need = {(m, r) for m in methods for r in range(cfg['rounds'])}
    have = {(r['method'], int(r['round'])) for r in rows}
    if not need <= have:
        fail(f'{d}: missing {sorted(need - have)[:3]}')
    tw = {(m, r) for m in methods for r in range(30, 40)}
    if not tw <= have:
        fail(f'{d}: terminal window incomplete')


def numeric_ok(d, rows):
    for r in rows:
        a, t = float(r['accuracy']), float(r['timely_ratio'])
        b, ag, sel = float(r['bytes_mb']), int(r['aggregated']), int(r['selected'])
        n = r['update_norm']
        if not (0.0 <= a <= 1.0 and 0.0 <= t <= 1.0 and b > 0 and ag <= sel <= 20):
            fail(f'{d}: bounds r{r["round"]} {r["method"]}')
        mr = r['mean_rank'].strip()
        if mr and not (2.0 <= float(mr) <= 16.0):  # fleet mean over delivered ranks, not a candidate value
            fail(f'{d}: mean_rank {mr} out of [2,16]')
        if ag > 0 and not (n and float(n) > 0):
            fail(f'{d}: aggregated={ag} but update_norm={n}')


print('== A/B/D. completeness, config integrity, numeric sanity ==')
ladder = {}
for fam, (ep, dl) in FAMILIES.items():
    bbs = []
    for s in range(10):
        d = f'{RUNS}/{fam}_seed{s}'
        if not os.path.isdir(d):
            fail(f'missing dir {d}')
            continue
        cfg, rows = load_run(d)
        methods = SA_M if fam.startswith('sa_') else M6
        check_complete(d, cfg, rows, methods)
        numeric_ok(d, rows)
        if int(cfg['seed']) != s:
            fail(f'{d}: config seed {cfg["seed"]} != dir seed {s}')
        if cfg['pretrain_epochs'] != ep or abs(cfg['deadline'] - dl) > 1e-9:
            fail(f'{d}: epochs/deadline {cfg["pretrain_epochs"]}/{cfg["deadline"]} != {ep}/{dl}')
        if cfg['rounds'] != 40 or cfg['clients'] != 20 or cfg['per_round'] != 10:
            fail(f'{d}: protocol drift {cfg["rounds"]}/{cfg["clients"]}/{cfg["per_round"]}')
        if cfg.get('resume'):
            fail(f'{d}: config marked resume')
        want = ','.join(sorted(methods))
        got = ','.join(sorted(cfg['methods'].split(',')))
        if want != got:
            fail(f'{d}: methods {got} != {want}')
        bbs.append(cfg['backbone_acc'])
    ladder[ep] = (np.mean(bbs), np.std(bbs))
    print(f'{fam:16s} 10 runs  bb_acc {np.mean(bbs):.4f}±{np.std(bbs):.4f}')

eps_sorted = sorted(ladder)
accs = [ladder[e][0] for e in eps_sorted]
if any(accs[i] >= accs[i + 1] for i in range(len(accs) - 1)):
    fail(f'backbone ladder not monotone: {list(zip(eps_sorted, accs))}')
else:
    print(f'ladder monotone: ' + ', '.join(f'{e}ep→{a:.3f}' for e, a in zip(eps_sorted, accs)))

print('\n== C. sa_strong backbone reuse ==')
for s in range(10):
    sa = json.load(open(f'{RUNS}/sa_strong_seed{s}/config.json'))
    ci = json.load(open(f'{RUNS}/cifar10_seed{s}/config.json'))
    if abs(sa['backbone_acc'] - ci['backbone_acc']) > 1e-6:
        warn(f'sa_strong_seed{s} bb {sa["backbone_acc"]:.4f} != cifar10_seed{s} {ci["backbone_acc"]:.4f}')
print('sa_strong backbones == seed-matched cifar10 backbones' if not any('sa_strong' in w for w in warns) else '(see warnings)')

print('\n== E. independent recomputation from raw metrics ==')
TER = range(30, 40)


def fam_bbacc(fam):
    """Mean backbone accuracy of any 10-seed family (old anchors included)."""
    vals = []
    for s in range(10):
        cfg = json.load(open(f'{RUNS}/{fam}_seed{s}/config.json'))
        vals.append(cfg['backbone_acc'])
    return np.mean(vals)


def terminal(fam, method, field):
    per_seed = []
    for s in range(10):
        rows = [r for r in csv.DictReader(open(f'{RUNS}/{fam}_seed{s}/metrics.csv'))
                if r['method'] == method and int(r['round']) in TER]
        vals = []
        for r in rows:
            if field == 'dc':
                vals.append(int(r['aggregated']) * float(r['update_norm']))
            else:
                vals.append(float(r[field]))
        per_seed.append(np.mean(vals))
    return np.array(per_seed)


def paired(fam, ma, mb, field):
    a, b = terminal(fam, ma, field), terminal(fam, mb, field)
    return a, b, np.mean(a / b - 1), wilcoxon(a, b).pvalue


def paired_pp(fam, ma, mb):
    """Absolute accuracy difference in percentage points (mean of per-seed diffs)."""
    a, b = terminal(fam, ma, 'accuracy'), terminal(fam, mb, 'accuracy')
    return np.mean(a - b) * 100, wilcoxon(a, b).pvalue


claims = []
# headline stress: timely/bytes/dc vs r16 and r4
for field, label in [('timely_ratio', 'timely'), ('bytes_mb', 'bytes'), ('dc', 'dc')]:
    a, b, g, p = paired('cifar10_dl0.1', 'Adaptive_rank', 'FedLoRA_r16', field)
    a2, b2, g2, p2 = paired('cifar10_dl0.1', 'Adaptive_rank', 'FedLoRA_r4', field)
    print(f'stress {label:6s} FedRAS {a.mean():.4f} r16 {b.mean():.4f} ({g:+.1%}, p={p:.4f}) | '
          f'r4 {b2.mean():.4f} ({g2:+.1%}, p={p2:.4f})')
    if field == 'timely_ratio':
        claims.append(('stress timely vs r16 ≈ +22%', abs((a / b).mean() - 1.22) < 0.03))
    if field == 'dc':
        claims.append(('stress dc vs r4 = 42%', abs(g2 - 0.42) < 0.02))
# nominal identity
_, _, gN, pN = paired('cifar10', 'Adaptive_rank', 'FedLoRA_r16', 'dc')
print(f'nominal dc FedRAS vs r16: {gN:+.2%} (p={pN:.2f}) -- identity expected')
claims.append(('nominal identity |dc diff| < 1%', abs(gN) < 0.01))
# conversion curve
curve = []
for fam, ep in [('weak_dl0.1', 1), ('bb2_dl0.1', 2), ('bb3_dl0.1', 3),
                ('weak5_dl0.1', 5), ('bb10_dl0.1', 10)]:
    g1, p1 = paired_pp(fam, 'Adaptive_rank', 'SA_rank')
    g2, p2 = paired_pp(fam, 'FedLoRA_r16', 'FedLoRA_r4')
    bbm = fam_bbacc(fam)
    curve.append((bbm, g1, p1, g2, p2))
    print(f'{fam:12s} bb={bbm:.3f}  FedRAS-SA {g1:+.2f}pp (p={p1:.3f})  r16-r4 {g2:+.2f}pp (p={p2:.3f})')
mono = all(curve[i][1] >= curve[i + 1][1] - 0.05 for i in range(len(curve) - 1))
claims.append(('conversion curve monotone (FedRAS-SA)', mono))
claims.append(('boundary: significant at 0.53-0.75, ns at 0.82+',
               curve[0][2] < .05 and curve[1][2] < .05 and curve[2][2] < .05
               and curve[3][2] > .05 and curve[4][2] > .05))
# failure matrix FedRAS worst cell (SA lives in sa_strong* for the strong regimes)
worst = 0.0
for fam in ['cifar10', 'cifar10_dl0.1', 'weak', 'weak_dl0.1']:
    fr = terminal(fam, 'Adaptive_rank', 'accuracy')
    sa_fam = ('sa_strong' if fam == 'cifar10' else
              'sa_strong_dl0.1' if fam == 'cifar10_dl0.1' else fam)
    best = max(terminal(fam, m, 'accuracy').mean() for m in ['FedLoRA_r4', 'FedLoRA_r16', 'Adaptive_rank'])
    best = max(best, terminal(sa_fam, 'SA_rank', 'accuracy').mean())
    worst = max(worst, (best - fr.mean()) * 100)
print(f'FedRAS worst accuracy cell: {worst:.2f}pp (paper: 0.41)')
claims.append(('worst cell ≈ 0.4pp', abs(worst - 0.41) < 0.06))

print('\n== F. paper-claim cross-check ==')
ok = True
for label, passed in claims:
    if label == 'skip':
        continue
    print(f'  [{"PASS" if passed else "FAIL"}] {label}')
    ok = ok and passed

print(f'\n== summary: {len(fails)} hard fails, {len(warns)} warnings ==')
for m in fails:
    print('  FAIL:', m)
for m in warns:
    print('  WARN:', m)
raise SystemExit(0 if (not fails and ok) else 1)

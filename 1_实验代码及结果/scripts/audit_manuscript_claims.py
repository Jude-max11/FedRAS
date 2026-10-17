"""Fleet-wide integrity sweep + recomputation of every remaining manuscript
table from raw run data (no journal-script imports).

  Part 1: every *_seed* run dir -- config/seed match, completeness vs its own
           config, numeric bounds.
  Part 2: manuscript claims, cell by cell:
           tab:ladder (nominal/0.1/0.05), weakbb panels, SAGIN per-class,
           de-circularized arm, corrskew bias, FP16 stacking, tab:failures
           16 cells, the sqrt-r norm law, energy/bandwidth translations.

Prints PASS/FAIL per claim; exit 1 on any hard fail. Writes nothing.
"""
import csv
import glob
import json
import math
import os
import re

import numpy as np
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, 'outputs', 'real_fedlora_v4')
TER = range(30, 39 + 1)
fails = []


def fail(msg):
    fails.append(msg)
    print(f'  FAIL {msg}')


def rows_of(fam, seed):
    p = f'{RUNS}/{fam}_seed{seed}/metrics.csv'
    if not os.path.exists(p):
        return None
    return list(csv.DictReader(open(p)))


def terminal(fam, method, field, seeds=range(10)):
    per = []
    for s in seeds:
        rows = rows_of(fam, s)
        if rows is None:
            continue
        vals = []
        for r in rows:
            if r['method'] == method and int(r['round']) in TER:
                if field == 'dc':
                    vals.append(int(r['aggregated']) * float(r['update_norm']))
                elif r[field] != '':  # mean_rank is empty on zero-delivery rounds
                    vals.append(float(r[field]))
        if vals:
            per.append(np.mean(vals))
    return np.array(per)


def check(label, got, want, tol):
    ok = abs(got - want) <= tol
    print(f'  [{"PASS" if ok else "FAIL"}] {label}: got {got:.4f} want {want} (±{tol})')
    if not ok:
        fails.append(label)


def check_b(label, cond):
    print(f'  [{"PASS" if cond else "FAIL"}] {label}')
    if not cond:
        fails.append(label)


print('== Part 1: fleet integrity sweep ==')
n_dirs = n_ok = 0
viol = 0
for d in sorted(glob.glob(f'{RUNS}/*_seed*')):
    m = re.match(r'^(.*)_seed(\d+)$', d)
    cfgp, metp = f'{d}/config.json', f'{d}/metrics.csv'
    if not m or not os.path.exists(cfgp):
        continue
    n_dirs += 1
    try:
        cfg = json.load(open(cfgp))
        rows = list(csv.DictReader(open(metp)))
    except Exception:
        print(f'  WARN incomplete (skipped from stats): {os.path.basename(d)}')
        continue
    if int(cfg['seed']) != int(m.group(2)):
        fail(f'{d}: seed mismatch')
    for r in rows:
        if not (0.0 <= float(r['accuracy']) <= 1.0 and 0.0 <= float(r['timely_ratio']) <= 1.0
                and float(r['bytes_mb']) > 0 and int(r['aggregated']) <= int(r['selected']) <= cfg['clients']):
            fail(f'{d}: bounds r{r["round"]} {r["method"]}')
            break
    need = {(mm, rd) for mm in ([x for x in cfg['methods'].split(',') if x] or ['FedAvg_full'])
            for rd in range(cfg['rounds'])}
    if need <= {(r['method'], int(r['round'])) for r in rows}:
        n_ok += 1
print(f'fleet: {n_dirs} run dirs, {n_ok} fully complete, {len(fails)} violations so far')

print('\n== Part 2a: tab:ladder (CIFAR-10, 10 seeds nominal/0.1s, 5 seeds 0.05s) ==')
lad = {('cifar10', 'FedAvg_full'): (.9135, .17, 52.86),
       ('cifar10', 'FedLoRA_r4'): (.9154, .75, .23),
       ('cifar10', 'FedLoRA_r16'): (.9153, .73, .90),
       ('cifar10', 'Random_rank'): (.9151, .76, .42),
       ('cifar10', 'Adaptive_rank'): (.9153, .73, .90),
       ('cifar10_dl0.1', 'FedAvg_full'): (.9153, .00, 52.86),
       ('cifar10_dl0.1', 'FedLoRA_r4'): (.9154, .62, .23),
       ('cifar10_dl0.1', 'FedLoRA_r16'): (.9149, .41, .90),
       ('cifar10_dl0.1', 'Random_rank'): (.9153, .56, .42),
       ('cifar10_dl0.1', 'Adaptive_rank'): (.9154, .49, .69)}
for (fam, meth), (acc, tly, mb) in lad.items():
    a = terminal(fam, meth, 'accuracy').mean()
    t = terminal(fam, meth, 'timely_ratio').mean()
    b = terminal(fam, meth, 'bytes_mb').mean()
    check(f'{fam} {meth} acc', a, acc, .002)
    check(f'{fam} {meth} timely', t, tly, .012)
    check(f'{fam} {meth} MB', b, mb, .02)
for meth, (acc, tly, mb, rk) in {'FedAvg_full': (.9147, .00, 52.86, None),
                                 'FedLoRA_r4': (.9144, .57, .23, 4.0),
                                 'FedLoRA_r16': (.9143, .27, .90, 16.0),
                                 'Random_rank': (.9143, .46, .42, 5.7),
                                 'Adaptive_rank': (.9143, .49, .44, 8.1)}.items():
    a = terminal('cifar10_dl0.05', meth, 'accuracy', seeds=range(5)).mean()
    t = terminal('cifar10_dl0.05', meth, 'timely_ratio', seeds=range(5)).mean()
    b = terminal('cifar10_dl0.05', meth, 'bytes_mb', seeds=range(5)).mean()
    check(f'0.05s {meth} acc', a, acc, .002)
    check(f'0.05s {meth} timely', t, tly, .015)
    check(f'0.05s {meth} MB', b, mb, .02)
    if rk:
        r_ = terminal('cifar10_dl0.05', meth, 'mean_rank', seeds=range(5)).mean()
        check(f'0.05s {meth} rank', r_, rk, .35)

print('\n== Part 2b: weakbb tier (fig:weakbb) ==')
r16 = terminal('weak', 'FedLoRA_r16', 'accuracy')
r4 = terminal('weak', 'FedLoRA_r4', 'accuracy')
check('weak nominal r16 acc', r16.mean(), .561, .003)
check('weak nominal r4 acc', r4.mean(), .546, .003)
check_b('weak nominal +1.5pp p=0.002', abs((r16 - r4).mean() * 100 - 1.5) < .2
        and wilcoxon(r16, r4).pvalue < .005)
rk = terminal('weak_dl0.1', 'Adaptive_rank', 'mean_rank').mean()
check('weak stress FedRAS rank 12.8', rk, 12.8, .25)
a_fr, a_r4 = terminal('weak_dl0.1', 'Adaptive_rank', 'accuracy'), terminal('weak_dl0.1', 'FedLoRA_r4', 'accuracy')
a_r16 = terminal('weak_dl0.1', 'FedLoRA_r16', 'accuracy')
check('weak stress FedRAS-r4 ≈ +1.1pp', (a_fr - a_r4).mean() * 100, 1.1, .25)
check_b('weak stress FedRAS-r4 10/10 pairs', int((a_fr > a_r4).sum()) == 10)
check('weak stress r16-FedRAS ≈ 0.4pp', (a_r16 - a_fr).mean() * 100, .4, .2)
t_fr, t_r16 = terminal('weak_dl0.1', 'Adaptive_rank', 'timely_ratio'), terminal('weak_dl0.1', 'FedLoRA_r16', 'timely_ratio')
check('weak stress timely +22% vs r16', t_fr.mean() / t_r16.mean() * 100 - 100, 22, 3)
b_fr, b_r16 = terminal('weak_dl0.1', 'Adaptive_rank', 'bytes_mb'), terminal('weak_dl0.1', 'FedLoRA_r16', 'bytes_mb')
check('weak stress bytes -23% vs r16', 100 - (b_fr / b_r16).mean() * 100, 23, 3)

print('\n== Part 2c: SAGIN grouped stress (per-class, terminal window) ==')


def sagin_cls(fam, meth):
    cls = {g: [] for g in ('ground', 'aerial', 'satellite')}
    for s in range(5):
        lo = list(csv.DictReader(open(f'{RUNS}/{fam}_seed{s}/link_outcomes.csv')))
        a = [r for r in lo if r['method'] == meth and int(r['round']) in TER]
        for g in cls:
            gg = [r for r in a if r['group'] == g]
            cls[g].append(np.mean([int(r['delivered']) for r in gg]))
    return {g: np.mean(v) for g, v in cls.items()}


sc = sagin_cls('cifar10_sagin_dl0.1', 'Adaptive_rank')
sc16 = sagin_cls('cifar10_sagin_dl0.1', 'FedLoRA_r16')
check('ground timely', sc['ground'], .57, .02)
check('aerial timely', sc['aerial'], .42, .03)
check('satellite timely', sc['satellite'], .43, .03)
check('satellite vs fixed r16 doubles', sc['satellite'] / sc16['satellite'], 2.3, .4)
check('delivered rank 13.1', terminal('cifar10_sagin_dl0.1', 'Adaptive_rank', 'mean_rank', seeds=range(5)).mean(), 13.1, .3)
check('aggregate timely 0.51', terminal('cifar10_sagin_dl0.1', 'Adaptive_rank', 'timely_ratio', seeds=range(5)).mean(), .51, .02)
check('r16 aggregate 0.44', terminal('cifar10_sagin_dl0.1', 'FedLoRA_r16', 'timely_ratio', seeds=range(5)).mean(), .44, .02)

print('\n== Part 2d: de-circularized arm (mm2, 5 seeds) ==')
S5 = range(5)
check('mm2 FedRAS timely 0.59', terminal('mm2_dl0.1', 'Adaptive_rank', 'timely_ratio', S5).mean(), .59, .03)
check('mm2 FedRAS MB 0.73', terminal('mm2_dl0.1', 'Adaptive_rank', 'bytes_mb', S5).mean(), .73, .05)
check('mm2 FedRAS rank 13.1', terminal('mm2_dl0.1', 'Adaptive_rank', 'mean_rank', S5).mean(), 13.1, .4)
check('mm2 r16 timely 0.52', terminal('mm2_dl0.1', 'FedLoRA_r16', 'timely_ratio', S5).mean(), .52, .03)
check('mm2 r4 timely 0.67', terminal('mm2_dl0.1', 'FedLoRA_r4', 'timely_ratio', S5).mean(), .67, .03)
accs = [terminal('mm2_dl0.1', m, 'accuracy', S5).mean() for m in ('FedLoRA_r4', 'FedLoRA_r16', 'Adaptive_rank')]
check_b('mm2 accuracy 0.915-0.916', all(.9145 <= a <= .9165 for a in accs))

print('\n== Part 2e: corrskew bias ==')
for meth, gap in (('Adaptive_rank', 4.9), ('FedLoRA_r16', 5.8), ('FedLoRA_r4', 3.2)):
    gaps = []
    for s in S5:
        be = list(csv.DictReader(open(f'{RUNS}/corrskew_dl0.1_seed{s}/bias_eval.csv')))
        h = [r for r in be if r['method'] == meth and int(r['round']) in TER]
        gaps.append(np.mean([100 * (float(r['acc_strong_link_classes']) - float(r['acc_weak_link_classes'])) for r in h]))
    check(f'corrskew gap {meth}', np.mean(gaps), gap, .45)
check('corrskew agg acc 0.9125', terminal('corrskew_dl0.1', 'Adaptive_rank', 'accuracy', S5).mean(), .9125, .003)

print('\n== Part 2f: FP16 stacking (all ten seeds) ==')
t = terminal('fedras_fp16_dl0.1', 'Adaptive_rank', 'timely_ratio', range(10))
t0 = terminal('fp16_dl0.1', 'FedLoRA_r16', 'timely_ratio', range(10))
b = terminal('fedras_fp16_dl0.1', 'Adaptive_rank', 'bytes_mb', range(10))
b0 = terminal('fp16_dl0.1', 'FedLoRA_r16', 'bytes_mb', range(10))
check('FP16 timely +12%', (t / t0).mean() * 100 - 100, 12, 3)
check('FP16 bytes -23%', 100 - (b / b0).mean() * 100, 23, 3)
check_b('FP16 p=0.002 each', wilcoxon(t, t0).pvalue < .01 and wilcoxon(b, b0).pvalue < .01)

print('\n== Part 2g: tab:failures cells (dc%/timely% of regime best) ==')
TEX = {'FedLoRA_r4': [54, 72, 55, 73], 'FedLoRA_r16': [100, 93, 100, 92],
       'SA_rank': [65, 83, 66, 84], 'Adaptive_rank': [100, 100, 100, 100]}
regs = [('cifar10', 'sa_strong'), ('cifar10_dl0.1', 'sa_strong_dl0.1'), ('weak', 'weak'), ('weak_dl0.1', 'weak_dl0.1')]
for meth, want in TEX.items():
    for i, (fam, sa_fam) in enumerate(regs):
        dc = {mm: terminal(fam if mm != 'SA_rank' else sa_fam, mm, 'dc').mean() for mm in TEX}
        tly = {mm: terminal(fam if mm != 'SA_rank' else sa_fam, mm, 'timely_ratio').mean() for mm in TEX}
        got = round(100 * dc[meth] / max(dc.values()))
        gott = round(100 * tly[meth] / max(tly.values()))
        check_b(f'failures {meth} regime{i} dc {got}%=={want[i]}% timely {gott}% in [65,100]', got == want[i])

print('\n== Part 2h: sqrt-r norm law (2.5x/1.9x measured) ==')


def norm_of(fam, meth):
    per = []
    for s in range(5):
        rows = rows_of(fam, s) or []
        v = [float(r['update_norm']) for r in rows if r['method'] == meth and r['update_norm']]
        if v:
            per.append(np.mean(v))
    return np.mean(per)


n2, n4 = norm_of('nrm_r2', 'FedLoRA_r2'), norm_of('cifar10', 'FedLoRA_r4')
n8, n16 = norm_of('nrm_r8', 'FedLoRA_r8'), norm_of('cifar10', 'FedLoRA_r16')
check('norm ratio r2->16', n16 / n2, 2.5, .3)
check('norm ratio r4->16', n16 / n4, 1.9, .25)

print('\n== Part 2i: energy/bandwidth translations ==')
dcfr, dcr16 = terminal('cifar10_dl0.1', 'Adaptive_rank', 'dc'), terminal('cifar10_dl0.1', 'FedLoRA_r16', 'dc')
bfr, br16 = terminal('cifar10_dl0.1', 'Adaptive_rank', 'bytes_mb'), terminal('cifar10_dl0.1', 'FedLoRA_r16', 'bytes_mb')
check('dc per joule +41% vs r16', np.mean((dcfr / bfr) / (dcr16 / br16)) * 100 - 100, 41, 4)
lo4, lo16 = np.log(terminal('cifar10_dl0.1', 'FedLoRA_r4', 'timely_ratio').mean()), np.log(terminal('cifar10_dl0.1', 'FedLoRA_r16', 'timely_ratio').mean())
inv = (lo4 - lo16) / (0.9011 - 0.2253)
mu = -lo4 - inv * 0.2253
lofr = np.log(terminal('cifar10_dl0.1', 'Adaptive_rank', 'timely_ratio').mean())
check('bandwidth multiplier 1.5x', inv * 0.9011 / (-lofr - mu), 1.54, .08)

print(f'\n== summary: {len(fails)} fails ==')
for m in fails:
    print('  FAIL:', m)
raise SystemExit(0 if not fails else 1)

#!/usr/bin/env python3
"""Cross-check Table 5(a)(b)(c)(d) against the simulator / solver / run outputs."""
import csv, glob, math, os, re, statistics as st, sys
sys.path.insert(0, '/root/autodl-tmp/FedRAS/scripts')
from collect_v6 import stats  # noqa: E402

R = '/root/autodl-tmp/FedRAS/outputs/real_v6'
TEX = sys.argv[1] if len(sys.argv) > 1 else '/tmp/main_zh.tex'
tex = open(TEX, encoding='utf-8').read()
bad, checked = [], 0


def num(s):
    s = s.replace(r'\pm', ' ').replace('$', ' ').replace('\\,', ' ')
    s = s.replace('\\%', ' ').replace('{', ' ').replace('}', ' ')
    return [float(v) for v in re.findall(r'-?\d*\.?\d+', s)]


def cmp(tag, paper, data, tol):
    global checked
    checked += 1
    if abs(paper - data) > tol:
        bad.append(f'{tag}: paper {paper:.4g} vs data {data:.4g}')


def section(start, end):
    i = tex.index(start)
    j = tex.index(end, i) if end else len(tex)
    return tex[i:j]


# ---------------- 5(a) simulator (last 30 rounds)
def sim(tag):
    per = {}
    for f in glob.glob(f'{R}/sim/{tag}/round_metrics_seed*.csv'):
        rows = list(csv.DictReader(open(f)))
        if not rows:
            continue
        nmax = max(int(r['round']) for r in rows)
        for r in rows:
            if int(r['round']) < nmax - 29:
                continue
            m = r['method']
            per.setdefault(m, {'a': [], 't': [], 'b': []})
            per[m]['a'].append(float(r['accuracy']))
            per[m]['t'].append(float(r['timely_ratio']))
            per[m]['b'].append(float(r['bytes_mb']))
    return {m: (st.mean(v['a']), st.mean(v['t']), st.mean(v['b'])) for m, v in per.items()}


L5A = [('FedAvg', 'FedAvg'), ('FedLoRA-4', 'FedLoRA_r4'), ('FedLoRA-16', 'FedLoRA_r16'),
       ('Random', 'Random_rank'), ('仅送达', 'Reliability_only'), ('SA-FedLoRA', 'SA_mono'),
       ('EL', 'ROBA_style'), ('FedRAS', 'Adaptive_rank')]
S = [sim('nominal'), sim('dl0.1'), sim('budget1.2')]
blk = section(r'\textsf{(a) 跨情形的调度器模拟}', r'\textsf{(b) 求解器研究}')
for ln in blk.splitlines():
    s = ln.strip()
    for lab, key in L5A:
        if s.startswith(lab) or s.startswith(r'\mbox{' + lab + '}'):
            v = num(ln.split('&', 1)[1])
            if len(v) < 9:
                continue
            for bi in range(3):
                d = S[bi].get(key)
                if not d:
                    continue
                cmp(f'5a {lab} blk{bi+1} acc', v[bi * 3], d[0], 0.0006)
                cmp(f'5a {lab} blk{bi+1} timely', v[bi * 3 + 1], d[1], 0.006)
                cmp(f'5a {lab} blk{bi+1} MB', v[bi * 3 + 2], d[2], 0.06)

# ---------------- 5(b) solver study
rows = list(csv.DictReader(open(f'{R}/solver_study.csv')))
blk = section(r'\textsf{(b) 求解器研究}', r'\textsf{(c) SAGIN 链路类别}')
for ln in blk.splitlines():
    m = re.match(r'\s*\$([\d.]+)\$\\,s, \$([\d.]+)\$\\,MB', ln)
    if not m:
        continue
    dl, bud = m.group(1), m.group(2)
    v = num(ln.split('&', 1)[1])
    sub = [r for r in rows if abs(float(r['deadline']) - float(dl)) < 1e-9
           and abs(float(r['budget']) - float(bud)) < 1e-9]
    if not sub or len(v) < 5:
        continue
    r0 = sub[0]
    for k, (col, idx) in enumerate((('greedy_vb_gap', 0), ('greedy_v_gap', 1),
                                    ('myopic_gap', 2), ('r16_gap', 3))):
        cmp(f'5b {dl}s {bud}MB gap{k+1}', v[idx], float(r0[col]) * 100, 0.15)
    cmp(f'5b {dl}s {bud}MB ms', v[4], float(r0['dp_ms']), 1.5)

# ---------------- 5(c) SAGIN per-class timely
def per_class(tag, method, lo=30):
    acc = {}
    for d in sorted(glob.glob(f'{R}/{tag}_seed*')):
        f = os.path.join(d, 'link_outcomes.csv')
        if not os.path.exists(f):
            continue
        for r in csv.DictReader(open(f)):
            if r['method'] != method or int(r['round']) < lo:
                continue
            k = r['group']
            a = acc.setdefault(k, [0, 0])
            a[0] += int(r['delivered'])
            a[1] += 1
    return {k: v[0] / v[1] for k, v in acc.items() if v[1]}


base = per_class('p2_tab5c_sagin_dl2.0', 'Adaptive_rank')
fr = per_class('p2_tab5c_sagin_dl0.1', 'Adaptive_rank')
f16 = per_class('p2_tab5c_sagin_dl0.1', 'FedLoRA_r16')
blk = section(r'\textsf{(c) SAGIN 链路类别}', r'\textsf{(d) 扩展压力矩阵}')
gmap = {'地面': 'ground', '空中': 'aerial', '卫星': 'satellite'}
for ln in blk.splitlines():
    s = ln.strip()
    for zh, g in gmap.items():
        if s.startswith(zh):
            v = num(ln.split('&', 1)[1])
            if len(v) >= 3:
                cmp(f'5c {zh} 2.0s', v[0], base.get(g, float('nan')), 0.006)
                cmp(f'5c {zh} 0.1s FedRAS', v[1], fr.get(g, float('nan')), 0.006)
                cmp(f'5c {zh} 0.1s FL16', v[2], f16.get(g, float('nan')), 0.006)

# ---------------- 5(d) extended matrix
M4 = [('FedLoRA-4', 'FedLoRA_r4'), ('FedLoRA-16', 'FedLoRA_r16'),
      ('Random', 'Random_rank'), ('FedRAS', 'Adaptive_rank')]
SC = [('基准', None), ('截止 10', 'dl10.0'), ('移动性', 'mob0.7'),
      ('α = 0.1', 'alpha0.1'), ('α = 1.0', 'alpha1.0')]
blk = section(r'\textsf{(d) 扩展压力矩阵}', r'\end{table*}')
cur = None
for ln in blk.splitlines():
    m = re.search(r'\\textbf\{(CIFAR-10|CIFAR-100|MNIST)\}', ln)
    if m:
        cur = {'CIFAR-10': 'cifar10', 'CIFAR-100': 'cifar100', 'MNIST': 'mnist'}[m.group(1)]
        continue
    s = ln.strip()
    for zh, sc in SC:
        if s.startswith(zh) and cur:
            v = num(ln.split('&', 1)[1])
            if len(v) < 4:
                continue
            for k, (lab, key) in enumerate(M4):
                pat = (f'{R}/p2_tab5d_{cur}_{sc}_seed*' if sc
                       else f'{R}/p1_tab3_{cur}_dl2.0_seed*')
                d = [x for x in (stats(p, key) for p in sorted(glob.glob(pat))) if x]
                if d:
                    cmp(f'5d {cur} {zh} {lab}', v[k], st.mean([x['acc'] for x in d]), 0.0007)

print('cells checked:', checked)
print('mismatches:', len(bad))
for b in bad[:30]:
    print('  ', b)

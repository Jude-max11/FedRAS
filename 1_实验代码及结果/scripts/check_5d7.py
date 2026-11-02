#!/usr/bin/env python3
"""Check Table 5(d) after extending it to seven methods (105 cells)."""
import glob, re, statistics as st, sys
sys.path.insert(0, '/root/autodl-tmp/FedRAS/scripts')
from collect_v6 import stats  # noqa: E402

R = '/root/autodl-tmp/FedRAS/outputs/real_v6'
tex = open('/tmp/main_zh.tex', encoding='utf-8').read()
M = [('FedLoRA_r4', 'p2_tab5d'), ('FedLoRA_r16', 'p2_tab5d'), ('Random_rank', 'p2_tab5d'),
     ('Adaptive_rank', 'p2_tab5d'), ('SA_mono', 'p2_tab5df'), ('Oort_utility', 'p2_tab5df'),
     ('ROBA_style', 'p2_tab5df')]
SC = [('基准', None), ('截止 10', 'dl10.0'), ('移动性', 'mob0.7'), ('α = 0.1', 'alpha0.1'),
      ('α = 1.0', 'alpha1.0')]
i = tex.index(r'\textsf{(d) 扩展压力矩阵}')
j = tex.index(r'\end{table*}', i)
blk = tex[i:j]
bad, n = [], 0
cur = None
for ln in blk.splitlines():
    m = re.search(r'\\textbf\{(CIFAR-10|CIFAR-100|MNIST)\}', ln)
    if m:
        cur = {'CIFAR-10': 'cifar10', 'CIFAR-100': 'cifar100', 'MNIST': 'mnist'}[m.group(1)]
        continue
    s = ln.strip()
    for zh, sc in SC:
        if s.startswith(zh) and cur:
            vals = [float(x) for x in re.findall(r'0\.\d{4}', ln.split('&', 1)[1])]
            if len(vals) < 7:
                continue
            for k, (key, pref) in enumerate(M):
                pat = (f'{R}/{pref}_{cur}_{sc}_seed*' if sc
                       else (f'{R}/p1_tab3_{cur}_dl2.0_seed*' if k < 4
                             else f'{R}/p1_tab3_{cur}_dl2.0_seed*'))
                v = [x['acc'] for x in (stats(d, key) for d in sorted(glob.glob(pat))) if x]
                if not v:
                    bad.append(f'{cur} {zh} {key}: no data')
                    continue
                n += 1
                if abs(st.mean(v) - vals[k]) > 0.0006:
                    bad.append(f'{cur} {zh} {key}: paper {vals[k]:.4f} vs data {st.mean(v):.4f}')
print('cells checked:', n, 'mismatches:', len(bad))
for b in bad[:20]:
    print('  ', b)

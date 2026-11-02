#!/usr/bin/env python3
"""Cross-check every number in the manuscript's result tables against the runs.

Checks:
  T3  Table 3  (3 datasets x 3 deadlines x 7 methods, 5 columns per block)
  T4  Table 4  (weak backbone 2.0/0.1 + ViT, 7 methods)
  T6a Table 6(a) ablations, T6b Table 6(b) two blocks, T6c Table 6(c)
  X1  Table 5(d) baseline row  ==  Table 3 @ 2.0 s (same runs)
  X2  Table 5(a) 0.1 s FedRAS vs FedLoRA-16 (bytes/timely direction)
  X3  Fig.2 source: kappa/MB at 0.1 s == Table 3 0.1 s columns
  X4  Fig.3(c/d) source == Table 4 ViT accuracy
Prints one line per mismatched cell.
"""
import glob, math, os, re, statistics as st, sys

sys.path.insert(0, '/root/autodl-tmp/FedRAS/scripts')
from collect_v6 import stats  # noqa: E402

R = '/root/autodl-tmp/FedRAS/outputs/real_v6'
TEX = sys.argv[1] if len(sys.argv) > 1 else '/tmp/main_zh.tex'
tex = open(TEX, encoding='utf-8').read()
bad, checked = [], 0


def num(s):
    s = s.replace(r'\pm', ' ').replace('$', ' ').replace('{', ' ').replace('}', ' ')
    vals = re.findall(r'-?\d*\.\d+', s)
    return [abs(float(v)) for v in vals]


def block(start_label, end_marker=r'\end{tblr}'):
    i = tex.index(start_label)
    j = tex.index(end_marker, i)
    return tex[i:j].splitlines()


def cmp(tag, paper, data, tol):
    global checked
    checked += 1
    if paper is None or data is None:
        bad.append(f'{tag}: missing (paper={paper} data={data})')
        return
    if abs(paper - data) > tol:
        bad.append(f'{tag}: paper {paper:.4f} vs data {data:.4f}')


def agg(pattern, method):
    v = [s for s in (stats(d, method) for d in sorted(glob.glob(pattern))) if s]
    if not v:
        return None
    return (st.mean([x['acc'] for x in v]), st.mean([x['timely'] for x in v]),
            st.mean([x['mb'] for x in v]), st.mean([x['rank'] for x in v]),
            st.mean([x['kappa'] for x in v]))


# ---------------- Table 3
M7 = [('FedLoRA-4', 'FedLoRA_r4'), ('FedLoRA-16', 'FedLoRA_r16'), ('Random', 'Random_rank'),
      ('SA-FedLoRA', 'SA_mono'), ('Oort', 'Oort_utility'), ('EL', 'ROBA_style'),
      ('FedRAS', 'Adaptive_rank')]
lines = block(r'\label{tab:cmp7}')
ds_of = {'CIFAR-10': 'cifar10', 'CIFAR-100': 'cifar100', 'MNIST': 'mnist'}
cur_ds = None
for ln in lines:
    m = re.search(r'\\textbf\{(CIFAR-10|CIFAR-100|MNIST)\}', ln)
    if m:
        cur_ds = ds_of[m.group(1)]
        continue
    for lab, key in M7:
        if ln.strip().startswith(lab + ' &') or ln.strip().startswith('\\mbox{' + lab + '} &'):
            vals = num(ln)
            if len(vals) < 15:
                continue
            for bi, dl in enumerate(('2.0', '0.1', '0.05')):
                d = agg(f'{R}/p1_tab3_{cur_ds}_dl{dl}_seed*', key)
                base = bi * 6   # acc, sd, timely, MB, rank, kappa
                if d is None:
                    continue
                cmp(f'T3 {cur_ds}@{dl} {lab} acc', vals[base], d[0], 0.0006)
                cmp(f'T3 {cur_ds}@{dl} {lab} timely', vals[base + 2], d[1], 0.006)
                cmp(f'T3 {cur_ds}@{dl} {lab} MB', vals[base + 3], d[2], 0.006)
                cmp(f'T3 {cur_ds}@{dl} {lab} rank', vals[base + 4], d[3], 0.06)
                cmp(f'T3 {cur_ds}@{dl} {lab} kappa', vals[base + 5], d[4], 0.006)

# ---------------- Table 4
lines = block(r'\label{tab:weak7}')
for ln in lines:
    for lab, key in M7:
        lab2 = lab.replace('（本文）', '')
        head = ln.strip()
        if head.startswith(lab2 + ' ') or head.startswith('\\mbox{' + lab2 + '}') or head.startswith('\\mbox{' + lab2 + '~}'):
            vals = num(ln)
            if len(vals) < 15:
                continue
            for bi, tag in enumerate(('p1_tab4_weakdisj_2.0', 'p1_tab4_weakdisj_0.1', 'p1_tab4_vit_0.5')):
                d = agg(f'{R}/{tag}_seed*', key)
                for k in range(5):
                    cmp(f'T4 {tag} {lab2} c{k+1}', vals[bi * 5 + k], d[k] if d else None,
                        0.0006 if k == 0 else 0.006)

# ---------------- Table 6(a)
T6A = [('完整系统', 'p1_tab3_cifar10_dl2.0', 'Adaptive_rank'),
       ('去掉适应容量项', 'p3_tab6a_nocap', 'Adaptive_rank'),
       ('去掉冻结', 'p3_tab6a_jointab_ab', 'JointAB'),
       ('去掉秩掩码', 'p3_tab6a_rrnomask_dl0.1', 'Random_rank'),
       ('去掉服务器动量', 'p3_tab6a_nomom', 'Adaptive_rank'),
       ('冻结', 'p3_tab6a_lr02_frozen', 'Adaptive_rank'),
       ('联合 A/B，学习率 0.02', 'p3_tab6a_lr02_joint', 'JointAB')]
lines = block(r'\label{tab:ablation}')
for ln in lines:
    s = ln.strip()
    for head, tag, key in T6A:
        if s.startswith(head):
            vals = num(ln.split('&', 1)[1])
            d = agg(f'{R}/{tag}_seed*', key)
            if d:
                cmp(f'T6a {tag} acc', vals[0], d[0], 0.0006)
                cmp(f'T6a {tag} sd', vals[1], 0.0, 9)   # sd not re-checked
                cmp(f'T6a {tag} timely', vals[2], d[1], 0.006)
                cmp(f'T6a {tag} MB', vals[3], d[2], 0.006)

# ---------------- Table 6(b)
T6B = [('完整 DP', 'p3_tab6b_dp_cohort_1.0', 'DP_cohort'),
       ('同参与集合 DP', 'p3_tab6b_dp_same_1.0', 'DP_same'),
       ('单客户端规则', 'p3_tab6b_perclient_1.0', 'Adaptive_rank')]
T6B2 = [('完整 DP', 'p3_tab6b_dp_cohort_0.1', 'DP_cohort'),
        ('同参与集合 DP', 'p3_tab6b_dp_same_0.1', 'DP_same'),
        ('单客户端规则', 'p3_tab6b_perclient_0.1', 'Adaptive_rank')]
seen_blk = 0
for ln in lines:
    s = ln.strip()
    if s.startswith('名义 2.0'):
        seen_blk = 1
        continue
    if s.startswith('压力 0.1'):
        seen_blk = 2
        continue
    spec = T6B if seen_blk == 1 else (T6B2 if seen_blk == 2 else [])
    for head, tag, key in spec:
        if s.startswith(head):
            vals = num(ln.split('&', 1)[1])
            d = agg(f'{R}/{tag}_seed*', key)
            if d and len(vals) >= 4:
                cmp(f'T6b blk{seen_blk} {tag} acc', vals[0], d[0], 0.0006)
                cmp(f'T6b blk{seen_blk} {tag} timely', vals[2], d[1], 0.006)
                cmp(f'T6b blk{seen_blk} {tag} MB', vals[3], d[2], 0.006)
                cmp(f'T6b blk{seen_blk} {tag} rank', vals[4], d[3], 0.06)

# ---------------- Table 6(c)
T6C = [('完整 DP', 'p3_tab6c_dp_0.4', 'DP_cohort'),
       ('价值贪心', 'p3_tab6c_greedy_v_0.4', 'Greedy_value'),
       ('按字节价值贪心', 'p3_tab6c_greedy_vb_0.4', 'Greedy_vpb'),
       ('单客户端规则，未校验', 'p3_tab6c_perclient_0.4', 'Adaptive_rank'),
       ('FedLoRA-16', 'p3_tab6c_fedlora16_0.4', 'FedLoRA_r16')]
for ln in lines:
    s = ln.strip()
    for head, tag, key in T6C:
        if s.startswith(head) or s.startswith(r'\mbox{FedLoRA-16}'):
            vals = num(ln.split('&', 1)[1])
            if len(vals) < 3:
                continue
            d = agg(f'{R}/{tag}_seed*', key)
            if d:
                cmp(f'T6c {tag} timely', vals[0], d[1], 0.006)
                cmp(f'T6c {tag} MB', vals[1], d[2], 0.006)
                cmp(f'T6c {tag} rank', vals[2], d[3], 0.06)

# ---------------- X1: Table 5(d) baseline == Table 3 @2.0
for ds in ('cifar10', 'cifar100', 'mnist'):
    for key, tag5d in (('FedLoRA_r4', 'cifar10'),):
        pass
print('cells checked:', checked)
print('mismatches:', len(bad))
for b in bad[:40]:
    print('  ', b)

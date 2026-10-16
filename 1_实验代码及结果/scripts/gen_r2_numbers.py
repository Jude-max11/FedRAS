#!/usr/bin/env python3
"""Unified R2 number generator: emits LaTeX table rows (last-10-round window,
matching the paper protocol) and paired statistics for every family touched
by the revision. Writes a digest to stdout for backfilling main.tex."""
import csv
import glob
import json
import math
import os
from collections import defaultdict

OUT = 'outputs/real_fedlora_v4'


def per_seed(pat, methods=None):
    """{method: [(acc, timely, mb, rank) per seed]} over rounds 30-39."""
    by = defaultdict(list)
    for d in sorted(glob.glob(f'{OUT}/{pat}')):
        f = os.path.join(d, 'metrics.csv')
        if not os.path.exists(f):
            continue
        rows = defaultdict(list)
        for r in csv.DictReader(open(f)):
            if 30 <= int(r['round']) <= 39:
                rows[r['method']].append(r)
        for m, rs in rows.items():
            if methods and m not in methods:
                continue
            mu = lambda k: sum(float(r[k]) for r in rs) / len(rs)
            rk = [float(r['mean_rank']) for r in rs if r['mean_rank']]
            by[m].append((mu('accuracy'), mu('timely_ratio'), mu('bytes_mb'),
                          sum(rk) / len(rk) if rk else float('nan')))
    return by


def row(by, m, prec=4):
    xs = by[m]
    def ms(i):
        v = [x[i] for x in xs]
        mu = sum(v) / len(v)
        sd = math.sqrt(sum((y - mu) ** 2 for y in v) / max(1, len(v) - 1)) if len(v) > 1 else 0
        return mu, sd
    a, asd = ms(0)
    t, tsd = ms(1)
    b, bsd = ms(2)
    rk, _ = ms(3)
    return (f'$.{str(round(a,4))[2:4]}{str(round(a,4))[4:5]}\\pm.{str(round(asd,4))[2:4]}{str(round(asd,4))[4:5]}$ '
            f'& {t:.2f} & {b:.2f} & {rk:.1f}', len(xs))


def paired(byA, byB, key=1):
    """paired per-seed values (same seed index) for timely(1)/bytes(2)/acc(0)."""
    n = min(len(byA), len(byB))
    return [byA[i][key] for i in range(n)], [byB[i][key] for i in range(n)]


def wilcoxon_p(a, b):
    try:
        from scipy.stats import wilcoxon
        return wilcoxon(a, b).pvalue
    except Exception:
        diffs = [x - y for x, y in zip(a, b)]
        w = sum(1 for d in diffs if d > 0)
        from math import comb
        n = len(diffs)
        return min(1.0, 2 * sum(comb(n, k) for k in range(w + 1)) / 2 ** n) if w < n / 2 else \
            min(1.0, 2 * sum(comb(n, k) for k in range(w, n + 1)) / 2 ** n)


def merge_fams(*pats, methods=None):
    """union per_seed dicts (seed-aligned across families of same length)."""
    by = defaultdict(list)
    for p in pats:
        for m, xs in per_seed(p, methods).items():
            by[m].extend(xs)
    return by


if __name__ == '__main__':
    print('%% ---- tab:ladder (last-10-round window) ----')
    nom = per_seed('cifar10_seed*')
    s01 = per_seed('cifar10_dl0.1_seed*')
    s005 = per_seed('cifar10_dl0.05_seed*')
    for m in ('FedAvg_full', 'FedLoRA_r4', 'FedLoRA_r16', 'Random_rank', 'Adaptive_rank'):
        r1, n1 = row(nom, m); r2, n2 = row(s01, m); r3, n3 = row(s005, m)
        print(f'{m} & {r1} & {r2} & {r3} \\\\  % seeds {n1}/{n2}/{n3}')

    print('\n%% ---- tab:extbaselines ----')
    ext = merge_fams('cifar10_ext_seed*', 'cifar10_ext_dl0.1_seed*')
    for m in ('SA_rank', 'Oort_utility', 'JointAB'):
        r1, _ = row(ext_n := per_seed('cifar10_ext_seed*'), m); r2, _ = row(per_seed('cifar10_ext_dl0.1_seed*'), m)
        print(f'{m} & {r1} & {r2} \\\\')
    for tag, m in (('fp16_r16', 'FedLoRA_r16'), ('fp16_fedras', 'Adaptive_rank')):
        pass
    a1 = per_seed('fp16_seed*'); a2 = per_seed('fp16_dl0.1_seed*')
    b1 = per_seed('fedras_fp16_seed*'); b2 = per_seed('fedras_fp16_dl0.1_seed*')
    r, _ = row(a1, 'FedLoRA_r16'); r2, _ = row(a2, 'FedLoRA_r16')
    print(f'FP16 r16 & {r} & {r2} \\\\')
    r, _ = row(b1, 'Adaptive_rank'); r2, _ = row(b2, 'Adaptive_rank')
    print(f'FP16 FedRAS & {r} & {r2} \\\\')
    ta, tb = paired(b2['Adaptive_rank'], a2['FedLoRA_r16'], 1)
    ba, bb = paired(b2['Adaptive_rank'], a2['FedLoRA_r16'], 2)
    print(f'% FP16 paired timely: {sum(1 for x,y in zip(ta,tb) if x>y)}/{len(ta)} p={wilcoxon_p(ta,tb):.4f}; '
          f'bytes mean saving {(1-sum(ba)/sum(bb))*100:.0f}% p={wilcoxon_p(ba,bb):.4f}')

    print('\n%% ---- tab:arms (1.0MB) ----')
    dp_n, dps_n = per_seed('dp_seed*'), per_seed('dpsame_seed*')
    dp_s, dps_s = per_seed('dp_dl0.1_seed*'), per_seed('dpsame_dl0.1_seed*')
    full = {'DP_cohort': dp_n, 'DP_same': dps_n, 'Adaptive_rank': nom, 'FedLoRA_r16': nom}
    for m, src in (('Full DP', dp_n), ('Same-cohort DP', dps_n), ('Fair cohort', nom)):
        r1, _ = row(src, list(src)[0] if m.startswith(('Full', 'Same')) else 'Adaptive_rank')
        src2 = dp_s if m == 'Full DP' else (dps_s if m == 'Same-cohort DP' else s01)
        m2 = ('DP_cohort' if m == 'Full DP' else 'DP_same' if m == 'Same-cohort DP' else 'Adaptive_rank')
        r2, _ = row(src2, m2)
        print(f'{m} & {r1} & {r2} \\\\')

    print('\n%% ---- bind04 (0.4MB, dl0.1) ----')
    bd = per_seed('bind04_dl0.1_seed*')
    for m in ('DP_cohort', 'DP_same', 'Greedy_value', 'Greedy_vpb', 'Adaptive_rank', 'FedLoRA_r16'):
        r, n = row(bd, m)
        print(f'{m} & {r} \\\\  % n={n}')

    print('\n%% ---- newbase / litbase / jointab ----')
    for pat, ms in (('newbase_dl0.1_seed*', ('SA_mono', 'Rate_thresh')),
                    ('newbase_seed*', ('SA_mono', 'Rate_thresh')),
                    ('litbase_dl0.1_seed*', ('ROBA_style', 'FedARA_svd')),
                    ('litbase_seed*', ('ROBA_style', 'FedARA_svd')),
                    ('jointab_ab_dl0.1_seed*', ('JointAB',)), ('jointab_ab_seed*', ('JointAB',))):
        d = per_seed(pat)
        for m in ms:
            r, n = row(d, m)
            print(f'{pat.split("_seed")[0]} {m} & {r} % n={n}')

    print('\n%% ---- adaptive tiers (zero-comm from configs) ----')
    for pat in ('vitshift0.5_seed*', 'vitshift2.0_seed*', 'vitclean2.0_seed*', 'vitclean0.5_seed*',
                'weakdisj_dl0.1_seed*', 'weakdisj_seed*', 'shiftc_dl0.1_seed*', 'shiftc_seed*',
                'weak_dl0.1_seed*', 'weak_seed*'):
        d = per_seed(pat)
        bbs = [json.load(open(f'{dd}/config.json')).get('backbone_acc', float('nan'))
               for dd in sorted(glob.glob(f'{OUT}/{pat}'))]
        print(f'{pat.split("_seed")[0]}: zero-comm={sum(bbs)/len(bbs):.4f}')
        for m in sorted(d):
            r, n = row(d, m)
            print(f'   {m} & {r} % n={n}')

    print('\n%% ---- paired stats for claims ----')
    # vitshift FedRAS vs r16 (0.5s): timely, bytes
    v = per_seed('vitshift0.5_seed*')
    ta, tb = paired(v['Adaptive_rank'], v['FedLoRA_r16'], 1)
    ba, bb = paired(v['Adaptive_rank'], v['FedLoRA_r16'], 2)
    print(f'vitshift0.5 FedRAS vs r16: timely {sum(1 for x,y in zip(ta,tb) if x>y)}/{len(ta)} p={wilcoxon_p(ta,tb):.4f}; '
          f'rel gain {(sum(ta)/sum(tb)-1)*100:.0f}%; bytes {(1-sum(ba)/sum(bb))*100:.0f}% saved')
    # weakdisj FedRAS vs zero-comm / vs SA
    wd = per_seed('weakdisj_dl0.1_seed*')
    accs = [x[0] for x in wd['Adaptive_rank']]
    print(f'weakdisj FedRAS acc over zero-comm: mean +{(sum(accs)/len(accs)-0.3652)*100:.1f}pp')
    # weak backbone vs ROBA
    wl = per_seed('weaklit_dl0.1_seed*')
    print(f'weaklit ROBA acc {sum(x[0] for x in wl["ROBA_style"])/len(wl["ROBA_style"]):.4f} '
          f'timely {sum(x[1] for x in wl["ROBA_style"])/len(wl["ROBA_style"]):.3f} '
          f'MB {sum(x[2] for x in wl["ROBA_style"])/len(wl["ROBA_style"]):.2f}')
    # bind04 DP vs greedy paired
    ta, tb = paired(bd['DP_cohort'], bd['Greedy_value'], 1)
    print(f'bind04 DP vs value-greedy timely: {sum(1 for x,y in zip(ta,tb) if x>y)}/{len(ta)} p={wilcoxon_p(ta,tb):.4f}')
    # W_j hold vs zero-fill (weak dl0.1)
    wh = per_seed('weakhold_dl0.1_seed*'); wz = per_seed('weak_dl0.1_seed*')
    for m in ('Adaptive_rank', 'FedLoRA_r16'):
        ah = [x[0] for x in wh[m]][:5]; az = [x[0] for x in wz[m]][:5]
        print(f'W_j hold vs zero {m}: mean {sum(ah)/5:.4f} vs {sum(az)/5:.4f} ({(sum(ah)-sum(az))/5*100:+.1f}pp)')

    print('\n%% ---- measured train time ----')
    tts = [json.load(open(f'{dd}/config.json')).get('train_ms_total', 0) for dd in sorted(glob.glob(f'{OUT}/cifar10_seed*'))]
    eps = [json.load(open(f'{dd}/config.json')).get('local_epochs_total', 1) for dd in sorted(glob.glob(f'{OUT}/cifar10_seed*'))]
    if tts and eps:
        print(f'train_ms_total mean {sum(tts)/len(tts):.0f} over {sum(eps)/len(eps):.0f} client-epochs '
              f'-> {sum(tts)/len(tts)/(sum(eps)/len(eps)):.0f} ms/client-epoch')

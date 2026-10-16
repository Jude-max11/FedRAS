#!/usr/bin/env python3
"""Round-2 result collector: aggregates every new run family into paper-ready rows.
Usage: python scripts/collect_r2.py [family ...]   (default: all found families)"""
import csv, glob, json, math, os, sys
from collections import defaultdict

OUT = 'outputs/real_fedlora_v4'
TAIL = (30, 39)  # pre-fixed terminal window, matches the paper protocol


def run_stats(d, tail=TAIL):
    """per-method dict: acc (last-10 mean), timely, mb, rank (rounds tail), n_rounds."""
    f = os.path.join(d, 'metrics.csv')
    if not os.path.exists(f):
        return None, 0
    rows = list(csv.DictReader(open(f)))
    by = defaultdict(list)
    for r in rows:
        by[r['method']].append(r)
    out = {}
    for m, rs in by.items():
        rs = sorted(rs, key=lambda r: int(r['round']))
        n = len(rs)
        w = [r for r in rs if tail[0] <= int(r['round']) <= tail[1]]
        if not w:
            w = rs[-10:]
        accs = [float(r['accuracy']) for r in w]
        tim = [float(r['timely_ratio']) for r in w]
        mb = [float(r['bytes_mb']) for r in w]
        ranks = [float(r['mean_rank']) for r in w if r['mean_rank']]
        mean = lambda xs: sum(xs) / len(xs) if xs else float('nan')
        out[m] = dict(acc=mean(accs), acc_sd=math.sqrt(sum((x - mean(accs)) ** 2 for x in accs) / max(1, len(accs) - 1)) if len(accs) > 1 else 0.0,
                      timely=mean(tim), mb=mean(mb), rank=mean(ranks) if ranks else float('nan'))
    return out, n


def agg(pattern, methods=None):
    """aggregate seed dirs matching pattern: seed-level means then cross-seed mean/sd."""
    dirs = sorted(glob.glob(os.path.join(OUT, pattern)))
    per_seed = []
    for d in dirs:
        st, n = run_stats(d)
        if st and n >= 40:
            per_seed.append(st)
    if not per_seed:
        return None, 0
    ms = methods or sorted({m for st in per_seed for m in st})
    res = {}
    for m in ms:
        rows = [st[m] for st in per_seed if m in st]
        if not rows:
            continue
        def ms_(k):
            xs = [r[k] for r in rows]
            mu = sum(xs) / len(xs)
            sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / max(1, len(xs) - 1)) if len(xs) > 1 else 0.0
            return mu, sd
        res[m] = dict(n=len(rows), **{f'{k}_{p}': v for k in ('acc', 'timely', 'mb', 'rank') for p, v in zip(('mu', 'sd'), ms_(k))})
    return res, len(per_seed)


def paired(pattern, a, b, key='timely'):
    """per-seed paired values of metric `key` for methods a vs b (same run dir)."""
    dirs = sorted(glob.glob(os.path.join(OUT, pattern)))
    pa = []
    for d in dirs:
        st, n = run_stats(d)
        if st and n >= 40 and a in st and b in st:
            pa.append((st[a][key], st[b][key]))
    return pa


def show(title, res, n):
    print(f'\n== {title} (seeds={n}) ==')
    if not res:
        print('  (no completed runs yet)')
        return
    for m, v in res.items():
        print(f'  {m:16s} acc {v["acc_mu"]:.4f}±{v["acc_sd"]:.4f}  timely {v["timely_mu"]:.3f}±{v["timely_sd"]:.3f}'
              f'  MB {v["mb_mu"]:.2f}  rank {v["rank_mu"]:.1f}')


def main():
    fams = sys.argv[1:] or ['mm2', 'fp16', 'weak5', 'bind04', 'newbase', 'weakhold', 'corrskew', 'mob21', 'jointab', 'shift', 'vit']
    if 'mm2' in fams:
        res, n = agg('mm2_dl0.1_seed*')
        show('mismatch arm rerun (mean-stationary AR1), dl=0.1', res, n)
        pa = paired('mm2_dl0.1_seed*', 'Adaptive_rank', 'FedLoRA_r4')
        if pa:
            wins = sum(1 for x, y in pa if x > y)
            print(f'  FedRAS vs r4 timely: {wins}/{len(pa)} seeds')
    if 'fp16' in fams:
        for pat, t in (('fp16_dl0.1_seed*', 'r16 FP16 dl0.1'), ('fedras_fp16_dl0.1_seed*', 'FedRAS FP16 dl0.1'),
                       ('fp16_seed*', 'r16 FP16 nominal'), ('fedras_fp16_seed*', 'FedRAS FP16 nominal')):
            res, n = agg(pat); show(t, res, n)
        pa = paired('fedras_fp16_dl0.1_seed*', 'Adaptive_rank', None, ) if False else None
    if 'weak5' in fams:
        for pat, t in (('weak5_seed*', 'weak backbone ep5 nominal'), ('weak5_dl0.1_seed*', 'weak backbone ep5 dl0.1')):
            res, n = agg(pat); show(t, res, n)
    if 'bind04' in fams:
        res, n = agg('bind04_dl0.1_seed*')
        show('binding budget 0.4 MB arms, dl=0.1', res, n)
    if 'newbase' in fams:
        for pat, t in (('newbase_dl0.1_seed*', 'SA-mono + rate-thresh dl0.1'), ('newbase_seed*', 'SA-mono + rate-thresh nominal')):
            res, n = agg(pat); show(t, res, n)
    if 'weakhold' in fams:
        res, n = agg('weakhold_dl0.1_seed*'); show('W_j hold-state ablation (weak dl0.1)', res, n)
        res2, n2 = agg('weak_dl0.1_seed*'); show('  reference: zero-fill default (weak dl0.1)', res2, n2)
    if 'corrskew' in fams:
        for pat, t in (('corrskew_dl0.1_seed*', 'correlated skew dl0.1'), ('corrskew_seed*', 'correlated skew nominal')):
            res, n = agg(pat); show(t, res, n)
        # per-class-half accuracy from bias_eval.csv (final-10 mean)
        for pat in ('corrskew_dl0.1_seed*',):
            rows = defaultdict(list)
            for f in glob.glob(os.path.join(OUT, pat, 'bias_eval.csv')):
                for r in csv.DictReader(open(f)):
                    if 30 <= int(r['round']) <= 39:
                        rows[r['method']].append((float(r['acc_weak_link_classes']), float(r['acc_strong_link_classes'])))
            if rows:
                print(f'  bias_eval ({pat}):')
                for m, xs in rows.items():
                    w = sum(x[0] for x in xs) / len(xs); s = sum(x[1] for x in xs) / len(xs)
                    print(f'    {m:14s} weak-link classes {w:.4f}  strong-link classes {s:.4f}  gap {s-w:+.4f}')
    if 'mob21' in fams:
        res, n = agg('mob21_seed*'); show('mobility Beta(2,1) nominal', res, n)
    if 'jointab' in fams:
        for pat, t in (('jointab_ab_dl0.1_seed*', 'JointAB at true A+B payload dl0.1'), ('jointab_ab_seed*', 'JointAB A+B nominal')):
            res, n = agg(pat); show(t, res, n)
    if 'shift' in fams:
        for pat in sorted(set(glob.glob(os.path.join(OUT, 'shift*_seed*')) + glob.glob(os.path.join(OUT, 'shift*_dl0.1_seed*')))):
            name = os.path.basename(pat)
            res, n = agg(name); show(f'shift task {name}', res, n)
            bb = os.path.join(pat, 'config.json')
            if os.path.exists(bb) and n:
                print(f'    backbone(zero-comm) acc on shifted test: {json.load(open(bb)).get("backbone_acc")}')
    if 'vit' in fams:
        for pat in sorted(glob.glob(os.path.join(OUT, 'vit_*_seed*'))):
            name = os.path.basename(pat)
            res, n = agg(name); show(f'ViT {name}', res, n)
            cfg = os.path.join(pat, 'config.json')
            if os.path.exists(cfg) and n:
                c = json.load(open(cfg))
                print(f'    backbone acc {c.get("backbone_acc")}  bytes/rank {c.get("bytes_per_rank_b_only")}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Emit the fresh-run plan for the R2 pipeline: ViT cache warm-up first, then
every incomplete or missing run in priority order. Incomplete runs are
DELETED (not resumed) -- cross-round state (mismatch-arm AR(1), Oort utility,
FedARA rank) is not checkpointed, so a resume would contaminate the terminal
window; correctness requires a fresh run."""
import glob
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
from audit_r2 import FAMILIES, complete, live_outs, PY, BASE  # noqa: E402



OUT = 'outputs/real_fedlora_v4'
# priority order: cheap batch-1 leftovers first, then 2a, 2b, 2c tiers
ORDER = ['mm2', 'weak5', 'bind04', 'newbase', 'corrskew', 'mob21', 'jointab_ab',
         'weakdisj', 'vitclean', 'shiftc', 'vitshift', 'litbase', 'weaklit',
         'bb2', 'bb3', 'bb10']


def seeds_of(pat):
    if '5-9' in pat:
        return range(5, 10)
    if '0-4' in pat:
        return range(5)
    return range(10)


def main():
    cmds = []
    # 0. ViT backbone cache warm-up (cache hits are fast no-ops)
    for frac in ('1.0', '0.5'):
        for s in range(10):
            cmds.append(f'{PY} scripts/train_torch_fedlora.py --data data --dataset cifar10 '
                        f'--arch vit --pretrain-epochs 80 --pretrain-lr 0.02 --pretrain-frac {frac} '
                        f'--seed {s} --pretrain-only --backbone-cache-dir {OUT}/vit_bcache '
                        f'--out {OUT}/vit_bcache/warm_{frac}_s{s}')
    fams = {f[0].split('_')[0].split('2.')[0].split('0.')[0]: None for f in FAMILIES}
    ordered = sorted(FAMILIES, key=lambda f: next((i for i, k in enumerate(ORDER)
                                                   if f[0].startswith(k)), 99))
    dropped = 0
    live = live_outs()
    for pat, extra, methods in ordered:
        base = pat.split('_seed')[0]
        for s in seeds_of(pat):
            d = f'{OUT}/{base}_seed{s}'
            if d in live:
                continue  # an in-flight run owns this dir; verify next pass
            ms = [m for m in methods.split(',') if m] if methods else None
            if os.path.isdir(d):
                if complete(d, want_bias=('corrskew' in pat), expected=ms):
                    continue
                shutil.rmtree(d)
                dropped += 1
            nb = '' if (methods and 'FedAvg_full' in methods) else ' --no-full-baseline'
            cmds.append(f'{PY} scripts/train_torch_fedlora.py {BASE} {extra} --seed {s} --out {d}'
                        + (f' --methods {methods}{nb}' if methods else ''))
    with open('logs/r2_plan.txt', 'w') as f:
        f.write('\n'.join(cmds) + '\n')
    print(f'plan: {len(cmds)} runs to execute ({dropped} partial runs dropped for fresh restart)')


if __name__ == '__main__':
    main()

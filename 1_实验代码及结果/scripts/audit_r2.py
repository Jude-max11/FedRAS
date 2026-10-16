#!/usr/bin/env python3
"""R2 pipeline auditor: enumerate every expected run of the review batches,
verify completeness (all 40 rounds x all configured methods), and emit repair
commands -- rebuilt from each run's own config.json with --resume, or from the
family template when the run never started."""
import csv
import glob
import json
import os
import shutil
import subprocess
import sys

OUT = 'outputs/real_fedlora_v4'
PY = '/home/jovyan/FedLoRA_remote_package/.venv/bin/python'
BASE = '--data data --dataset cifar10 --rounds 40 --clients 20 --per-round 10'

# family template: (glob of expected dirs, extra args, methods or '' for default)
FAMILIES = [
    ('mm2_dl0.1_seed[0-4]', '--pretrain-epochs 30 --deadline 0.1 --delivery-mismatch', ''),
    ('fp16_dl0.1_seed[5-9]', '--pretrain-epochs 30 --deadline 0.1 --fp16-upload', 'FedLoRA_r16'),
    ('fp16_seed[5-9]', '--pretrain-epochs 30 --deadline 2.0 --fp16-upload', 'FedLoRA_r16'),
    ('fedras_fp16_dl0.1_seed[5-9]', '--pretrain-epochs 30 --deadline 0.1 --fp16-upload', 'Adaptive_rank'),
    ('fedras_fp16_seed[5-9]', '--pretrain-epochs 30 --deadline 2.0 --fp16-upload', 'Adaptive_rank'),
    ('weak5_seed[0-9]', '--pretrain-epochs 5 --deadline 2.0', 'FedLoRA_r4,FedLoRA_r16,Adaptive_rank,SA_rank,Oort_utility,JointAB'),
    ('weak5_dl0.1_seed[0-9]', '--pretrain-epochs 5 --deadline 0.1', 'FedLoRA_r4,FedLoRA_r16,Adaptive_rank,SA_rank,Oort_utility,JointAB'),
    ('bind04_dl0.1_seed[0-9]', '--pretrain-epochs 30 --deadline 0.1 --dp-budget 0.4', 'DP_cohort,DP_same,Greedy_value,Greedy_vpb,Adaptive_rank,FedLoRA_r16'),
    ('newbase_dl0.1_seed[0-9]', '--pretrain-epochs 30 --deadline 0.1', 'SA_mono,Rate_thresh'),
    ('newbase_seed[0-9]', '--pretrain-epochs 30 --deadline 2.0', 'SA_mono,Rate_thresh'),
    ('weakhold_dl0.1_seed[0-4]', '--pretrain-epochs 1 --deadline 0.1 --empty-cols hold', 'FedLoRA_r4,FedLoRA_r16,Adaptive_rank,SA_rank,Oort_utility,JointAB'),
    ('corrskew_dl0.1_seed[0-4]', '--pretrain-epochs 30 --deadline 0.1 --link-profile sagin --corr-skew', 'FedLoRA_r4,FedLoRA_r16,Adaptive_rank'),
    ('corrskew_seed[0-4]', '--pretrain-epochs 30 --deadline 2.0 --link-profile sagin --corr-skew', 'FedLoRA_r4,FedLoRA_r16,Adaptive_rank'),
    ('mob21_seed[0-4]', '--pretrain-epochs 30 --deadline 2.0 --mobility-shape 2,1', 'FedLoRA_r4,FedLoRA_r16,Adaptive_rank'),
    ('jointab_ab_dl0.1_seed[0-9]', '--pretrain-epochs 30 --deadline 0.1 --jointab-ab', 'JointAB'),
    ('jointab_ab_seed[0-9]', '--pretrain-epochs 30 --deadline 2.0 --jointab-ab', 'JointAB'),
    ('weakdisj_seed[0-9]', '--pretrain-epochs 1 --pretrain-frac 0.5 --deadline 2.0', 'FedLoRA_r4,FedLoRA_r16,Adaptive_rank,SA_rank,Oort_utility,JointAB'),
    ('weakdisj_dl0.1_seed[0-9]', '--pretrain-epochs 1 --pretrain-frac 0.5 --deadline 0.1', 'FedLoRA_r4,FedLoRA_r16,Adaptive_rank,SA_rank,Oort_utility,JointAB'),
    ('vitclean2.0_seed[0-9]', '--pretrain-epochs 80 --pretrain-lr 0.02 --arch vit --backbone-cache-dir outputs/real_fedlora_v4/vit_bcache --deadline 2.0', ''),
    ('vitclean0.5_seed[0-9]', '--pretrain-epochs 80 --pretrain-lr 0.02 --arch vit --backbone-cache-dir outputs/real_fedlora_v4/vit_bcache --deadline 0.5', ''),
    ('vitclean0.25_seed[0-4]', '--pretrain-epochs 80 --pretrain-lr 0.02 --arch vit --backbone-cache-dir outputs/real_fedlora_v4/vit_bcache --deadline 0.25', ''),
    ('shiftc_dl0.1_seed[0-9]', '--pretrain-epochs 30 --shift-severity 0.1 --pretrain-frac 0.5 --deadline 0.1', 'FedLoRA_r4,FedLoRA_r16,Random_rank,Adaptive_rank,SA_rank,SA_mono,Rate_thresh,Oort_utility,JointAB'),
    ('shiftc_seed[0-9]', '--pretrain-epochs 30 --shift-severity 0.1 --pretrain-frac 0.5 --deadline 2.0', 'FedLoRA_r4,FedLoRA_r16,Random_rank,Adaptive_rank,SA_rank,SA_mono,Rate_thresh,Oort_utility,JointAB'),
    ('shiftcf_dl0.1_seed[0-9]', '--pretrain-epochs 30 --shift-severity 0.1 --pretrain-frac 0.5 --deadline 0.1', 'FedAvg_full'),
    ('shiftcf_seed[0-9]', '--pretrain-epochs 30 --shift-severity 0.1 --pretrain-frac 0.5 --deadline 2.0', 'FedAvg_full'),
    ('vitshift0.5_seed[0-9]', '--pretrain-epochs 80 --pretrain-lr 0.02 --arch vit --pretrain-frac 0.5 --shift-severity 0.1 --backbone-cache-dir outputs/real_fedlora_v4/vit_bcache --deadline 0.5', ''),
    ('vitshift2.0_seed[0-9]', '--pretrain-epochs 80 --pretrain-lr 0.02 --arch vit --pretrain-frac 0.5 --shift-severity 0.1 --backbone-cache-dir outputs/real_fedlora_v4/vit_bcache --deadline 2.0', ''),
    ('litbase_dl0.1_seed[0-9]', '--pretrain-epochs 30 --deadline 0.1', 'ROBA_style,FedARA_svd'),
    ('litbase_seed[0-9]', '--pretrain-epochs 30 --deadline 2.0', 'ROBA_style,FedARA_svd'),
    ('weaklit_dl0.1_seed[0-9]', '--pretrain-epochs 1 --deadline 0.1', 'ROBA_style,FedARA_svd'),
    # C8 backbone-quality sweep: fill 2/3/10-epoch tiers between the 1/5/30 anchors
    ('bb2_seed[0-9]', '--pretrain-epochs 2 --deadline 2.0', 'FedLoRA_r4,FedLoRA_r16,Adaptive_rank,SA_rank,Oort_utility,JointAB'),
    ('bb2_dl0.1_seed[0-9]', '--pretrain-epochs 2 --deadline 0.1', 'FedLoRA_r4,FedLoRA_r16,Adaptive_rank,SA_rank,Oort_utility,JointAB'),
    ('bb3_seed[0-9]', '--pretrain-epochs 3 --deadline 2.0', 'FedLoRA_r4,FedLoRA_r16,Adaptive_rank,SA_rank,Oort_utility,JointAB'),
    ('bb3_dl0.1_seed[0-9]', '--pretrain-epochs 3 --deadline 0.1', 'FedLoRA_r4,FedLoRA_r16,Adaptive_rank,SA_rank,Oort_utility,JointAB'),
    ('bb10_seed[0-9]', '--pretrain-epochs 10 --deadline 2.0', 'FedLoRA_r4,FedLoRA_r16,Adaptive_rank,SA_rank,Oort_utility,JointAB'),
    ('bb10_dl0.1_seed[0-9]', '--pretrain-epochs 10 --deadline 0.1', 'FedLoRA_r4,FedLoRA_r16,Adaptive_rank,SA_rank,Oort_utility,JointAB'),
]


def tmpl(extra, methods, seed, out):
    """Fresh-run command from a family template (mirrors plan_r2.py emission)."""
    nb = '' if (methods and 'FedAvg_full' in methods) else ' --no-full-baseline'
    return (f'{PY} scripts/train_torch_fedlora.py {BASE} {extra} --seed {seed} --out {out}'
            + (f' --methods {methods}{nb}' if methods else ''))



def live_outs():
    """--out targets of currently running trainers (never touch these)."""
    try:
        out = subprocess.check_output(['ps', '-eo', 'args'], text=True)
    except Exception:
        return set()
    return {l.split('--out ')[1].split()[0] for l in out.splitlines()
            if 'train_torch_fedlora.py' in l and '--out ' in l}

BOOL_FLAGS = [('delivery_mismatch', '--delivery-mismatch'), ('fp16_upload', '--fp16-upload'),
              ('corr_skew', '--corr-skew'), ('jointab_ab', '--jointab-ab'), ('resume', '--resume')]


def complete(d, want_bias=False, expected=None):
    cfg = os.path.join(d, 'config.json')
    met = os.path.join(d, 'metrics.csv')
    if not os.path.exists(cfg) or not os.path.exists(met):
        return False
    try:
        c = json.load(open(cfg))
        rows = list(csv.DictReader(open(met)))
        rounds_by = {(r['method'], int(r['round'])) for r in rows}
        if expected is not None:
            ms = expected
        else:
            ms = [m for m in c['methods'].split(',') if m] if c.get('methods') else \
                ['FedLoRA_r4', 'FedLoRA_r16', 'Random_rank', 'Adaptive_rank', 'FedAvg_full']
        need = {(m, rd) for m in ms for rd in range(c['rounds'])}
        if not need <= rounds_by:
            return False
        if want_bias and not os.path.exists(os.path.join(d, 'bias_eval.csv')):
            return False
        return True
    except Exception:
        return False


def cmd_from_cfg(c):
    a = [PY, 'scripts/train_torch_fedlora.py', '--data', c['data'], '--dataset', c['dataset'],
         '--out', c['out'], '--rounds', str(c['rounds']), '--clients', str(c['clients']),
         '--per-round', str(c['per_round']), '--local-epochs', str(c['local_epochs']),
         '--pretrain-epochs', str(c['pretrain_epochs']), '--pretrain-lr', str(c['pretrain_lr']),
         '--batch-size', str(c['batch_size']), '--lr', str(c['lr']), '--seed', str(c['seed']),
         '--alpha', str(c['alpha']), '--deadline', str(c['deadline']),
         '--mobility-shape', str(c['mobility_shape']), '--link-profile', c['link_profile'],
         '--arch', c.get('arch', 'cnn'), '--dp-budget', str(c.get('dp_budget', 1.0)), '--resume']
    if c.get('methods'):
        a += ['--methods', c['methods']]
    for k, f in BOOL_FLAGS:
        if c.get(k) and k != 'resume':
            a.append(f)
    if c.get('shift_severity', 0) > 0:
        a += ['--shift-severity', str(c['shift_severity'])]
    if c.get('pretrain_frac', 1.0) < 1.0:
        a += ['--pretrain-frac', str(c['pretrain_frac'])]
    if c.get('empty_cols', 'zero') != 'zero':
        a += ['--empty-cols', c['empty_cols']]
    if c.get('arch') == 'vit' or c.get('backbone_cache_dir'):
        a += ['--backbone-cache-dir', c.get('backbone_cache_dir') or 'outputs/real_fedlora_v4/vit_bcache']
    return ' '.join(a)


def main():
    repairs = []
    live = live_outs()
    for pat, extra, methods in FAMILIES:
        ms = [m for m in methods.split(',') if m] if methods else None
        for d in sorted(glob.glob(f'{OUT}/{pat}')):
            if complete(d, want_bias=('corrskew' in pat), expected=ms):
                continue
            if d in live:
                continue
            shutil.rmtree(d, ignore_errors=True)
            seed = d.rsplit('seed', 1)[1]
            repairs.append(tmpl(extra, methods, seed, d))
    for pat, extra, methods in FAMILIES:
        base = pat.split('_seed')[0]
        seeds = range(5, 10) if '5-9' in pat else (range(5) if '0-4' in pat else range(10))
        for s in seeds:
            cand = f'{OUT}/{base}_seed{s}'
            if not os.path.isdir(cand) and not glob.glob(f'{OUT}/{base}_seed{s}'):
                repairs.append(tmpl(extra, methods, str(s), cand))
    open('logs/r2_repairs.txt', 'w').write(chr(10).join(sorted(set(repairs))) + chr(10))
    print(f'audit: {len(set(repairs))} repairs needed')
    return 0 if not repairs else 1

if __name__ == '__main__':
    sys.exit(main())

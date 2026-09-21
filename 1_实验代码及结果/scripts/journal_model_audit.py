"""Model / payload audit for the journal revision (reviewer point 6).

Reports, for the exact CNN used in the paper (6 adapted 1x1-conv LoRA blocks +
fc1/fc2 LoRALinear, r_max provisioned, frozen backbone):
  - backbone (frozen) parameter count and FP32/FP16 update size;
  - per adapted layer the LoRA factor shapes and the per-rank factor cost
    (d+k parameters per rank column);
  - actual serialized update size at rank r (FP32 and FP16);
  - the stylized payload model s(r)=0.02r+0.5 normalized MB used throughout
    the paper, and how the two relate;
  - the full-model update size (FedAvg_full trains all backbone + head params).

Writes only outputs/journal_stats/. Read-only w.r.t. the paper directories.
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
os.chdir(ROOT)
from train_torch_fedlora import Net  # noqa: E402

OUT = os.path.join(ROOT, 'outputs', 'journal_stats')
os.makedirs(OUT, exist_ok=True)


def main():
    m = Net(c=10, in_ch=3, width=256, rmax=16, base_ch=64)
    lora = {k: v.shape for k, v in m.state_dict().items() if k.endswith(('.A', '.B'))}
    base = {k: p.numel() for k, p in m.named_parameters() if not (k.endswith('.A') or k.endswith('.B'))}
    n_base = sum(base.values())

    # Per-rank factor cost: sum over adapted layers of (d_l + k_l). The rank
    # axis is dim 0 of A and dim 1 of B (both layouts), so A contributes
    # shape[1:] (in channels x spatial) and B contributes d[0] * prod(d[2:])
    # (out channels x spatial). For 1x1 convs and linears the spatial product
    # is 1: per-rank cost = in + out per layer.
    per_col = 0     # A + B parameters per rank column
    per_col_b = 0   # B only (the actual frozen-A upload per rank column)
    layers = []
    for k, sh in sorted(lora.items()):
        if k.endswith('.A'):
            d = tuple(lora[k.replace('.A', '.B')])
            a_col = int(torch.tensor(sh[1:]).prod())
            b_col = int(d[0]) * int(torch.tensor(d[2:]).prod())
            per_col += a_col + b_col
            per_col_b += b_col
            layers.append((k[:-2], tuple(sh), d, a_col + b_col))
    for name, a_sh, b_sh, col in layers:
        print(f'{name:12s} A={a_sh} B={b_sh}  per-rank params={col}')
    print(f'\nadapted layers: {len(layers)};  A+B params per rank column: {per_col}')
    print(f'{" ":18}B-only params per rank column (frozen-A upload): {per_col_b}')
    print(f'frozen backbone + classifier params: {n_base:,}  '
          f'(FP32 {4 * n_base / 1e6:.2f} MB, FP16 {2 * n_base / 1e6:.2f} MB)')

    print('\n rank | A+B FP32 MB | B-only FP32 MB | B-only FP16 MB | stylized s(r) MB')
    rows = []
    for r in (2, 4, 8, 16, 32, 64):
        fp32_ab = 4 * r * per_col / 1e6
        fp32_b = 4 * r * per_col_b / 1e6
        fp16_b = 2 * r * per_col_b / 1e6
        s = 0.02 * r + 0.5
        print(f'{r:5d} | {fp32_ab:11.4f} | {fp32_b:14.4f} | {fp16_b:14.4f} | {s:16.2f}')
        rows.append((r, fp32_ab, fp32_b, fp16_b, s))
    n_adapt = 16 * per_col
    r16_b_mb = 4 * 16 * per_col_b / 1e6
    full_fp32 = 4 * (n_base + n_adapt) / 1e6  # FedAvg_full transmits base + both factors
    print(f'\nFedAvg_full actual transmitted-update size (FP32): {full_fp32:.3f} MB '
          f'(base {4 * n_base / 1e6:.3f} + adapters {4 * n_adapt / 1e6:.3f}); '
          f'ratio to B-only rank-16 upload {r16_b_mb:.4f} MB = {full_fp32 / r16_b_mb:.1f}x')

    with open(os.path.join(OUT, 'model_audit.csv'), 'w') as f:
        f.write('rank,ab_fp32_mb,bonly_fp32_mb,bonly_fp16_mb,stylized_mb\n')
        for r, ab, b32, b16, s in rows:
            f.write(f'{r},{ab:.4f},{b32:.4f},{b16:.4f},{s:.2f}\n')
        f.write(f'backbone_params,{n_base},,,\n')
        f.write(f'params_per_rank_ab,{per_col},,,\n')
        f.write(f'params_per_rank_b_only,{per_col_b},,,\n')
        f.write(f'fedavg_full_mb,{full_fp32:.4f},,,\n')
    print('written:', os.path.join(OUT, 'model_audit.csv'))


if __name__ == '__main__':
    main()

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
    base = {k: v.numel() for k, v in m.state_dict().items() if not (k.endswith('.A') or k.endswith('.B'))}
    n_base = sum(base.values())

    # per-rank factor cost: sum over adapted layers of (d_l + k_l); conv
    # factors are 1x1 (spatial dims 1), so column cost = in + out channels.
    per_col = 0
    layers = []
    for k, sh in sorted(lora.items()):
        if k.endswith('.A'):
            d = tuple(lora[k.replace('.A', '.B')])
            col = int(torch.tensor(sh[1:]).prod()) + int(torch.tensor(d[1:]).prod())
            per_col += col
            layers.append((k[:-2], tuple(sh), d, col))
    for name, a_sh, b_sh, col in layers:
        print(f'{name:12s} A={a_sh} B={b_sh}  per-rank params={col}')
    print(f'\nadapted layers: {len(layers)};  factor params per rank column: {per_col}')
    print(f'frozen backbone + classifier params: {n_base:,}  '
          f'(FP32 {4 * n_base / 1e6:.2f} MB, FP16 {2 * n_base / 1e6:.2f} MB)')

    print('\n rank | actual FP32 MB | actual FP16 MB | stylized s(r) MB')
    rows = []
    for r in (2, 4, 8, 16, 32, 64):
        fp32 = 4 * r * per_col / 1e6
        fp16 = 2 * r * per_col / 1e6
        s = 0.02 * r + 0.5
        print(f'{r:5d} | {fp32:14.3f} | {fp16:14.3f} | {s:16.2f}')
        rows.append((r, fp32, fp16, s))
    full_fp32 = (n_base + 16 * per_col) * 4 / 1e6  # FedAvg_full trains backbone + head adapters
    print(f'\nFedAvg_full actual trainable-update size (FP32): {full_fp32:.2f} MB '
          f'vs stylized rank-64 payload 1.78 MB')

    with open(os.path.join(OUT, 'model_audit.csv'), 'w') as f:
        f.write('rank,actual_fp32_mb,actual_fp16_mb,stylized_mb\n')
        for r, a32, a16, s in rows:
            f.write(f'{r},{a32:.4f},{a16:.4f},{s:.2f}\n')
        f.write(f'backbone_params,{n_base},,\n')
        f.write(f'params_per_rank,{per_col},,\n')
    print('written:', os.path.join(OUT, 'model_audit.csv'))


if __name__ == '__main__':
    main()

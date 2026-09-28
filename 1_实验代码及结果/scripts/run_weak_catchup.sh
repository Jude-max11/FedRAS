#!/usr/bin/env bash
set -u
PY=/opt/conda/bin/python
OUT=outputs/real_fedlora_v4
cd "$(dirname "$0")/.."
for spec in "weak_seed2:2.0" "weak_seed3:2.0" "weak_dl0.1_seed1:0.1" "weak_dl0.1_seed2:0.1" "weak_dl0.1_seed3:0.1"; do
  out="${spec%%:*}"; dl="${spec##*:}"
  $PY scripts/train_torch_fedlora.py --data data --dataset cifar10 --out "$OUT/$out" \
    --rounds 40 --clients 20 --pretrain-epochs 1 --deadline "$dl" \
    --methods "FedLoRA_r4,FedLoRA_r16,Adaptive_rank" > "logs/rr_${out}.log" 2>&1 &
done
wait
echo "WEAK CATCHUP DONE"

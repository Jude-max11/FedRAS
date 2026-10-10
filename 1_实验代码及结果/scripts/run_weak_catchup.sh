#!/usr/bin/env bash
set -u
PY=${PY:-python3}
OUT=outputs/real_fedlora_v4
cd "$(dirname "$0")/.."
for s in 0 1 2 3 4; do
  for spec in "weak_seed${s}:2.0" "weak_dl0.1_seed${s}:0.1"; do
    out="${spec%%:*}"; dl="${spec##*:}"
    $PY scripts/train_torch_fedlora.py --data data --dataset cifar10 --out "$OUT/$out" \
      --rounds 40 --clients 20 --pretrain-epochs 1 --deadline "$dl" --seed "$s" \
      --methods "FedLoRA_r4,FedLoRA_r16,Adaptive_rank,SA_rank,Oort_utility,JointAB" > "logs/rr_${out}.log" 2>&1 &
  done
done
wait
echo "WEAK RUNS DONE"

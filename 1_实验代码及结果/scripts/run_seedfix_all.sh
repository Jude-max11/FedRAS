#!/usr/bin/env bash
set -u
PY=/home/jovyan/FedLoRA_remote_package/.venv/bin/python
OUT=outputs/real_fedlora_v4
cd "$(dirname "$0")/.."

# ---- batch 1: weak backbone, 6 arms, seeds 0-4, both deadlines ----
for s in 0 1 2 3 4; do
  for spec in "weak_seed${s}:2.0" "weak_dl0.1_seed${s}:0.1"; do
    out="${spec%%:*}"; dl="${spec##*:}"
    $PY scripts/train_torch_fedlora.py --data data --dataset cifar10 --out "$OUT/$out" \
      --rounds 40 --clients 20 --pretrain-epochs 1 --deadline "$dl" --seed "$s" \
      --methods "FedLoRA_r4,FedLoRA_r16,Adaptive_rank,SA_rank,Oort_utility,JointAB" > "logs/rr_${out}.log" 2>&1 &
  done
done
wait
echo "BATCH1 WEAK DONE"

# ---- batch 2: fp16 arms, seeds 1-4 ----
for s in 1 2 3 4; do
  for spec in "fp16_seed${s}:2.0:FedLoRA_r16" "fp16_dl0.1_seed${s}:0.1:FedLoRA_r16" "fedras_fp16_seed${s}:2.0:Adaptive_rank" "fedras_fp16_dl0.1_seed${s}:0.1:Adaptive_rank"; do
    IFS=: read -r out dl meth <<< "$spec"
    mkdir -p "$OUT/$out"
    cp "$OUT/cifar10_seed${s}/backbone_pretrained.pt" "$OUT/cifar10_seed${s}/backbone_meta.json" "$OUT/$out/" 2>/dev/null || true
    $PY scripts/train_torch_fedlora.py --data data --dataset cifar10 --out "$OUT/$out" \
      --rounds 40 --clients 20 --pretrain-epochs 30 --deadline "$dl" --seed "$s" \
      --methods "$meth" --fp16-upload > "logs/rr_${out}.log" 2>&1 &
  done
done
wait
echo "BATCH2 FP16 DONE"

# ---- batch 3: fedavg lr 1.0 check, seeds 1-2 ----
for s in 1 2; do
  out="fedavg_lr10_seed${s}"
  mkdir -p "$OUT/$out"
  cp "$OUT/cifar10_seed${s}/backbone_pretrained.pt" "$OUT/cifar10_seed${s}/backbone_meta.json" "$OUT/$out/" 2>/dev/null || true
  $PY scripts/train_torch_fedlora.py --data data --dataset cifar10 --out "$OUT/$out" \
    --rounds 40 --clients 20 --pretrain-epochs 30 --deadline 2.0 --seed "$s" \
    --methods FedAvg_full --full-lr-scale 1.0 > "logs/rr_${out}.log" 2>&1 &
done
wait
echo "BATCH3 LR10 DONE"
echo "SEEDFIX ALL DONE"

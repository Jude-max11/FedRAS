#!/usr/bin/env bash
set -u
PY=/home/jovyan/FedLoRA_remote_package/.venv/bin/python
OUT=outputs/real_fedlora_v4
cd "$(dirname "$0")/.."

# ---- batch A: de-circularized mismatch arm, 0.1 s, seeds 0-4 ----
for s in 0 1 2 3 4; do
  $PY scripts/train_torch_fedlora.py --data data --dataset cifar10 --out "$OUT/mm_dl0.1_seed${s}" \
    --rounds 40 --clients 20 --pretrain-epochs 30 --deadline 0.1 --seed "$s" \
    --delivery-mismatch > "logs/rr_mm_dl0.1_seed${s}.log" 2>&1 &
done
wait
echo "BATCH A MISMATCH DONE"

# ---- batch B: weak backbone seeds 5-9, both deadlines, six arms ----
for s in 5 6 7 8 9; do
  for spec in "weak_seed${s}:2.0" "weak_dl0.1_seed${s}:0.1"; do
    out="${spec%%:*}"; dl="${spec##*:}"
    $PY scripts/train_torch_fedlora.py --data data --dataset cifar10 --out "$OUT/$out" \
      --rounds 40 --clients 20 --pretrain-epochs 1 --deadline "$dl" --seed "$s" \
      --methods "FedLoRA_r4,FedLoRA_r16,Adaptive_rank,SA_rank,Oort_utility,JointAB" > "logs/rr_${out}.log" 2>&1 &
  done
done
wait
echo "BATCH B WEAK DONE"

# ---- batch C: SAGIN profile at 0.1 s, seeds 0-4 ----
for s in 0 1 2 3 4; do
  $PY scripts/train_torch_fedlora.py --data data --dataset cifar10 --out "$OUT/cifar10_sagin_dl0.1_seed${s}" \
    --rounds 40 --clients 20 --pretrain-epochs 30 --deadline 0.1 --seed "$s" \
    --link-profile sagin > "logs/rr_cifar10_sagin_dl0.1_seed${s}.log" 2>&1 &
done
wait
echo "BATCH C SAGIN DONE"

# ---- batch D: unmasked (unified-denominator) aggregation, weak stress ----
for s in 0 1 2 3 4; do
  $PY scripts/train_torch_fedlora.py --data data --dataset cifar10 --out "$OUT/weak_dl0.1_nomask_seed${s}" \
    --rounds 40 --clients 20 --pretrain-epochs 1 --deadline 0.1 --seed "$s" \
    --methods "FedLoRA_r4,FedLoRA_r16,Adaptive_rank" --no-rank-mask > "logs/rr_weak_dl0.1_nomask_seed${s}.log" 2>&1 &
done
wait
echo "BATCH D NOMASK DONE"
echo "REVIEWFIX ALL DONE"

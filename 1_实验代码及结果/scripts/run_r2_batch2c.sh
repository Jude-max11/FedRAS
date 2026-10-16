#!/usr/bin/env bash
# Batch 2c: literature-mechanism baselines (after ROBA/TSFA and after FedARA),
# chained after batch 2b. Strong backbone both deadlines + weak-backbone tier.
set -u
PY=/home/jovyan/FedLoRA_remote_package/.venv/bin/python
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
OUT=outputs/real_fedlora_v4
LOG=logs
BASE="--data data --dataset cifar10 --rounds 40 --clients 20 --per-round 10 --pretrain-epochs 30"

while ! grep -q "R2 BATCH2B ALL DONE" "$LOG/r2_batch2b_main.log" 2>/dev/null; do sleep 120; done
echo "batch2b done; starting batch2c"

launch() {
  while [ "$(jobs -rp | wc -l)" -ge 5 ]; do sleep 20; done
  echo "launch $1"
  taskset -c 64-103 "$@" &
}

# ---- strong backbone, both deadlines ----
for s in 0 1 2 3 4 5 6 7 8 9; do
  for spec in "litbase_dl0.1_seed${s}:0.1" "litbase_seed${s}:2.0"; do
    IFS=: read -r out dl <<< "$spec"
    launch $PY scripts/train_torch_fedlora.py $BASE --deadline "$dl" --seed "$s" \
      --methods ROBA_style,FedARA_svd --no-full-baseline --out "$OUT/$out" > "$LOG/r2_${out}.log" 2>&1
  done
done
wait; echo "LITBASE STRONG DONE"

# ---- weak-backbone tier (paired with existing weak runs by seed) ----
for s in 0 1 2 3 4 5 6 7 8 9; do
  launch $PY scripts/train_torch_fedlora.py --data data --dataset cifar10 --rounds 40 --clients 20 \
    --per-round 10 --pretrain-epochs 1 --deadline 0.1 --seed "$s" \
    --methods ROBA_style,FedARA_svd --no-full-baseline \
    --out "$OUT/weaklit_dl0.1_seed${s}" > "$LOG/r2_weaklit_s${s}.log" 2>&1
done
wait; echo "LITBASE WEAK DONE"
echo "R2 BATCH2C ALL DONE"

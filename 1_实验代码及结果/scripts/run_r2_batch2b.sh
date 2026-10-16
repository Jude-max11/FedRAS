#!/usr/bin/env bash
# Batch 2b (shift tiers), chained after batch 2a.
# (1) CNN covariate-shift tier at the standard protocol (boundary framing),
# (2) ViT + disjoint pretrain + shift tier (transformer adaptive task).
set -u
PY=/home/jovyan/FedLoRA_remote_package/.venv/bin/python
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
OUT=outputs/real_fedlora_v4
LOG=logs
BASE="--data data --dataset cifar10 --rounds 40 --clients 20 --per-round 10 --pretrain-epochs 30"

while ! grep -q "R2 BATCH2A ALL DONE" "$LOG/r2_batch2a_main.log" 2>/dev/null; do sleep 120; done
echo "batch2a done; starting batch2b"

launch() {
  while [ "$(jobs -rp | wc -l)" -ge 5 ]; do sleep 20; done
  echo "launch $1"
  taskset -c 64-103 "$@" &
}

# ---- 1. CNN covariate shift (sev 0.1, disjoint pretrain, standard protocol) ----
for s in 0 1 2 3 4 5 6 7 8 9; do
  for spec in "shiftc_dl0.1_seed${s}:0.1" "shiftc_seed${s}:2.0"; do
    IFS=: read -r out dl <<< "$spec"
    launch $PY scripts/train_torch_fedlora.py $BASE --shift-severity 0.1 --pretrain-frac 0.5 \
      --deadline "$dl" --seed "$s" \
      --methods FedLoRA_r4,FedLoRA_r16,Random_rank,Adaptive_rank,SA_rank,SA_mono,Rate_thresh,Oort_utility,JointAB,FedAvg_full \
      --out "$OUT/$out" > "$LOG/r2_${out}.log" 2>&1
  done
done
wait; echo "CNNSHIFT DONE"

# ---- 2. ViT + disjoint pretrain + shift (sev 0.1) ----
for s in 0 1 2 3 4 5 6 7 8 9; do
  for spec in "vitshift0.5_seed${s}:0.5" "vitshift2.0_seed${s}:2.0"; do
    IFS=: read -r out dl <<< "$spec"
    launch $PY scripts/train_torch_fedlora.py $BASE --arch vit --pretrain-epochs 80 --pretrain-lr 0.02 \
      --pretrain-frac 0.5 --shift-severity 0.1 --backbone-cache-dir "$OUT/vit_bcache" \
      --deadline "$dl" --seed "$s" \
      --out "$OUT/$out" > "$LOG/r2_${out}.log" 2>&1
  done
done
wait; echo "VITSHIFT DONE"
echo "R2 BATCH2B ALL DONE"

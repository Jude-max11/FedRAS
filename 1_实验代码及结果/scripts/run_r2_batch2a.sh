#!/usr/bin/env bash
# Batch 2a (pilot-independent de-risked tiers), chained after batch 1.
# Tiers: (1) weak backbone + disjoint pretrain corpus (CNN), (2) ViT clean ladder.
set -u
PY=/home/jovyan/FedLoRA_remote_package/.venv/bin/python
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
OUT=outputs/real_fedlora_v4
LOG=logs
BASE="--data data --dataset cifar10 --rounds 40 --clients 20 --per-round 10"

# wait for batch 1 to finish (marker written by run_r2_review.sh)
while ! grep -q "R2 BATCH1 ALL DONE" "$LOG/r2_batch1_main.log" "$LOG/r2_v2_launcher.log" 2>/dev/null; do sleep 120; done
echo "batch1 done; starting batch2a"

launch() {
  while [ "$(jobs -rp | wc -l)" -ge 5 ]; do sleep 20; done
  echo "launch $1"
  taskset -c 64-103 "$@" &
}

# ---- 1. weak backbone (1-epoch pretrain) with DISJOINT pretrain corpus ----
for s in 0 1 2 3 4 5 6 7 8 9; do
  for spec in "weakdisj_seed${s}:2.0" "weakdisj_dl0.1_seed${s}:0.1"; do
    IFS=: read -r out dl <<< "$spec"
    launch $PY scripts/train_torch_fedlora.py $BASE --pretrain-epochs 1 --pretrain-frac 0.5 \
      --deadline "$dl" --seed "$s" \
      --methods FedLoRA_r4,FedLoRA_r16,Adaptive_rank,SA_rank,Oort_utility,JointAB \
      --no-full-baseline --out "$OUT/$out" > "$LOG/r2_${out}.log" 2>&1
  done
done
wait; echo "WEAKDISJ DONE"

# ---- 2. ViT clean ladder (backbones from the shared cache) ----
for s in 0 1 2 3 4 5 6 7 8 9; do
  for spec in "vitclean2.0_seed${s}:2.0" "vitclean0.5_seed${s}:0.5"; do
    IFS=: read -r out dl <<< "$spec"
    launch $PY scripts/train_torch_fedlora.py $BASE --arch vit --pretrain-epochs 80 --pretrain-lr 0.02 \
      --backbone-cache-dir "$OUT/vit_bcache" --deadline "$dl" --seed "$s" \
      --out "$OUT/$out" > "$LOG/r2_${out}.log" 2>&1
  done
done
for s in 0 1 2 3 4; do
  launch $PY scripts/train_torch_fedlora.py $BASE --arch vit --pretrain-epochs 80 --pretrain-lr 0.02 \
    --backbone-cache-dir "$OUT/vit_bcache" --deadline 0.25 --seed "$s" \
    --out "$OUT/vitclean0.25_seed${s}" > "$LOG/r2_vitclean0.25_s${s}.log" 2>&1
done
wait; echo "VITCLEAN DONE"
echo "R2 BATCH2A ALL DONE"

#!/usr/bin/env bash
# Reviewer-round-2 batch 1: B-class reruns and new baselines (CNN trainer).
# Groups run with bounded concurrency; each group waits before the next.
set -u
PY=/home/jovyan/FedLoRA_remote_package/.venv/bin/python
cd "$(dirname "$0")/.."
OUT=outputs/real_fedlora_v4
LOG=logs
BASE="--data data --dataset cifar10 --rounds 40 --clients 20 --per-round 10 --pretrain-epochs 30"

launch() {  # limit-checked background launch
  while [ "$(jobs -rp | wc -l)" -ge 5 ]; do sleep 20; done
  echo "launch $1"
  "$@" &
}

# ---- A. mismatch arm rerun with mean-stationary AR(1) (0.1 s, seeds 0-4) ----
for s in 0 1 2 3 4; do
  launch $PY scripts/train_torch_fedlora.py $BASE --deadline 0.1 --seed "$s" \
    --delivery-mismatch --out "$OUT/mm2_dl0.1_seed${s}" > "$LOG/r2_mm2_s${s}.log" 2>&1
done
wait; echo "A MISMATCH2 DONE"

# ---- B. FP16 seeds 5-9 (both formats, both deadlines) ----
for s in 5 6 7 8 9; do
  for spec in "fp16_dl0.1_seed${s}:0.1:FedLoRA_r16" "fp16_seed${s}:2.0:FedLoRA_r16" \
              "fedras_fp16_dl0.1_seed${s}:0.1:Adaptive_rank" "fedras_fp16_seed${s}:2.0:Adaptive_rank"; do
    IFS=: read -r out dl m <<< "$spec"
    launch $PY scripts/train_torch_fedlora.py $BASE --deadline "$dl" --seed "$s" \
      --fp16-upload --methods "$m" --out "$OUT/$out" > "$LOG/r2_${out}.log" 2>&1
  done
done
wait; echo "B FP16 DONE"

# ---- C. weak backbone, second strength (5-epoch pretrain), seeds 0-9 ----
for s in 0 1 2 3 4 5 6 7 8 9; do
  for spec in "weak5_seed${s}:2.0" "weak5_dl0.1_seed${s}:0.1"; do
    IFS=: read -r out dl <<< "$spec"
    launch $PY scripts/train_torch_fedlora.py --data data --dataset cifar10 --rounds 40 --clients 20 \
      --per-round 10 --pretrain-epochs 5 --deadline "$dl" --seed "$s" \
      --methods FedLoRA_r4,FedLoRA_r16,Adaptive_rank,SA_rank,Oort_utility,JointAB \
      --no-full-baseline --out "$OUT/$out" > "$LOG/r2_${out}.log" 2>&1
  done
done
wait; echo "C WEAK5 DONE"

# ---- D. binding budget 0.4 MB arms (0.1 s, seeds 0-9) ----
for s in 0 1 2 3 4 5 6 7 8 9; do
  launch $PY scripts/train_torch_fedlora.py $BASE --deadline 0.1 --seed "$s" --dp-budget 0.4 \
    --methods DP_cohort,DP_same,Greedy_value,Greedy_vpb,Adaptive_rank,FedLoRA_r16 \
    --no-full-baseline --out "$OUT/bind04_dl0.1_seed${s}" > "$LOG/r2_bind04_s${s}.log" 2>&1
done
wait; echo "D BIND04 DONE"

# ---- E. new baselines SA_mono + Rate_thresh (both deadlines, seeds 0-9) ----
for s in 0 1 2 3 4 5 6 7 8 9; do
  for spec in "newbase_dl0.1_seed${s}:0.1" "newbase_seed${s}:2.0"; do
    IFS=: read -r out dl <<< "$spec"
    launch $PY scripts/train_torch_fedlora.py $BASE --deadline "$dl" --seed "$s" \
      --methods SA_mono,Rate_thresh --no-full-baseline --out "$OUT/$out" > "$LOG/r2_${out}.log" 2>&1
  done
done
wait; echo "E NEWBASE DONE"

# ---- F. W_j=0 hold-state ablation on the weak backbone (0.1 s, seeds 0-4) ----
for s in 0 1 2 3 4; do
  launch $PY scripts/train_torch_fedlora.py --data data --dataset cifar10 --rounds 40 --clients 20 \
    --per-round 10 --pretrain-epochs 1 --deadline 0.1 --seed "$s" --empty-cols hold \
    --methods FedLoRA_r4,FedLoRA_r16,Adaptive_rank,SA_rank,Oort_utility,JointAB \
    --no-full-baseline --out "$OUT/weakhold_dl0.1_seed${s}" > "$LOG/r2_weakhold_s${s}.log" 2>&1
done
wait; echo "F HOLD DONE"

# ---- G. link-class correlated label skew (both deadlines, seeds 0-4) ----
for s in 0 1 2 3 4; do
  for spec in "corrskew_dl0.1_seed${s}:0.1" "corrskew_seed${s}:2.0"; do
    IFS=: read -r out dl <<< "$spec"
    launch $PY scripts/train_torch_fedlora.py $BASE --deadline "$dl" --seed "$s" \
      --link-profile sagin --corr-skew \
      --methods FedLoRA_r4,FedLoRA_r16,Adaptive_rank --no-full-baseline \
      --out "$OUT/$out" > "$LOG/r2_${out}.log" 2>&1
  done
done
wait; echo "G CORRSKEW DONE"

# ---- H. genuinely higher-mean mobility Beta(2,1) at the nominal deadline ----
for s in 0 1 2 3 4; do
  launch $PY scripts/train_torch_fedlora.py $BASE --deadline 2.0 --seed "$s" \
    --mobility-shape "2,1" --methods FedLoRA_r4,FedLoRA_r16,Adaptive_rank \
    --no-full-baseline --out "$OUT/mob21_seed${s}" > "$LOG/r2_mob21_s${s}.log" 2>&1
done
wait; echo "H MOB21 DONE"

# ---- I. JointAB billed at its true A+B payload (both deadlines, seeds 0-9) ----
for s in 0 1 2 3 4 5 6 7 8 9; do
  for spec in "jointab_ab_dl0.1_seed${s}:0.1" "jointab_ab_seed${s}:2.0"; do
    IFS=: read -r out dl <<< "$spec"
    launch $PY scripts/train_torch_fedlora.py $BASE --deadline "$dl" --seed "$s" \
      --jointab-ab --methods JointAB --no-full-baseline --out "$OUT/$out" > "$LOG/r2_${out}.log" 2>&1
  done
done
wait; echo "I JOINTAB DONE"
echo "R2 BATCH1 ALL DONE"

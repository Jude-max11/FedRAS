#!/usr/bin/env bash
# Journal revision runs (Computer Networks review): (1) full-DP scheduler arm
# DP_cohort on CIFAR-10 nominal, seeds 0-9, vs the existing fair-cohort runs;
# (2) external baselines (SA_rank, Oort_utility, JointAB) extended to seeds 5-9
# on cifar10 nominal + 0.25 s stress, completing ten-seed coverage of the
# baseline table.
# ISOLATION: writes ONLY to outputs/journal_dp and outputs/journal_ext.
# Per-seed backbone caches are reused (copied) from the existing run dirs,
# mirroring scripts/wcnc_run_seeds59.sh; the shared trainer is used read-only.
set -u
PY=/home/jovyan/FedLoRA_remote_package/.venv/bin/python
ROOT=/home/jovyan/FedLoRA_remote_package
DP="$ROOT/outputs/journal_dp"
EXT="$ROOT/outputs/journal_ext"
V3="$ROOT/outputs/real_fedlora_v3"
EXTO="$ROOT/outputs/wcnc_extra/real_v3_ext"
cd "$ROOT"
mkdir -p "$DP" "$EXT" logs

done_stamp () { [ -f "$1/metrics.csv" ] && [ "$(tail -1 "$1/metrics.csv" | cut -d, -f2)" = "39" ]; }

# ---- set 1: DP_cohort, seeds 0-9, cifar10 nominal ----
CONC=4
running=0
for seed in 0 1 2 3 4 5 6 7 8 9; do
  out="$DP/dp_seed${seed}"
  mkdir -p "$out"
  cp "$V3/cifar10_seed${seed}/backbone_pretrained.pt" "$V3/cifar10_seed${seed}/backbone_meta.json" "$out/" 2>/dev/null || true
  done_stamp "$out" && { echo "skip finished $out"; continue; }
  echo "RUN dp_cohort seed$seed"
  $PY scripts/train_torch_fedlora.py --data data --dataset cifar10 --out "$out" \
    --rounds 40 --clients 20 --pretrain-epochs 30 --seed "$seed" \
    --methods DP_cohort --no-full-baseline \
    > "logs/journal_dp_seed${seed}.log" 2>&1 &
  running=$((running+1))
  if [ $((running % CONC)) -eq 0 ]; then wait; fi
done
wait

# ---- set 2: baselines SA_rank/Oort_utility/JointAB, seeds 5-9 ----
for seed in 5 6 7 8 9; do
  out="$EXT/cifar10_seed${seed}"
  mkdir -p "$out"
  cp "$EXTO/cifar10_seed${seed}/backbone_pretrained.pt" "$EXTO/cifar10_seed${seed}/backbone_meta.json" "$out/" 2>/dev/null || true
  if done_stamp "$out"; then echo "skip finished $out"; continue; fi
  echo "RUN baselines cifar10 seed$seed"
  $PY scripts/train_torch_fedlora.py --data data --dataset cifar10 --out "$out" \
    --rounds 40 --clients 20 --pretrain-epochs 30 --seed "$seed" \
    --methods SA_rank,Oort_utility,JointAB --no-full-baseline \
    > "logs/journal_ext_cifar10_seed${seed}.log" 2>&1 &
  running=$((running+1))
  if [ $((running % CONC)) -eq 0 ]; then wait; fi
done
wait

# 0.25 s stress with per-seed backbone reuse (protocol of real_fedlora_v3)
for seed in 5 6 7 8 9; do
  out="$EXT/cifar10_dl0.25_seed${seed}"
  mkdir -p "$out"
  cp "$EXTO/cifar10_seed${seed}/backbone_pretrained.pt" "$EXTO/cifar10_seed${seed}/backbone_meta.json" "$out/" 2>/dev/null || true
  if done_stamp "$out"; then echo "skip finished $out"; continue; fi
  echo "RUN baselines cifar10_dl0.25 seed$seed"
  $PY scripts/train_torch_fedlora.py --data data --dataset cifar10 --out "$out" \
    --rounds 40 --clients 20 --pretrain-epochs 30 --deadline 0.25 --seed "$seed" \
    --methods SA_rank,Oort_utility,JointAB --no-full-baseline \
    > "logs/journal_ext_cifar10_dl0.25_seed${seed}.log" 2>&1 &
  running=$((running+1))
  if [ $((running % CONC)) -eq 0 ]; then wait; fi
done
wait

echo ALL_JOURNAL_RUNS_DONE

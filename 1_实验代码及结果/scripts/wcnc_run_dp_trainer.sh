#!/usr/bin/env bash
# WCNC full-DP-in-trainer experiment: run the existing DP_cohort arm
# (exact joint client-rank DP over all 20 clients, byte budget 8.2 MB = the
# fixed-r16 per-round payload, cap 10) on cifar10 nominal + 0.25 s stress,
# seeds 0-4, paired with the v3 fair-cohort runs by seed and trace.
# ISOLATION: writes ONLY to outputs/wcnc_extra/real_v3_dp; the shared trainer
# is used read-only (DP_cohort was already implemented there); journal data
# directories are never touched. Per-seed backbones are reused from
# outputs/real_fedlora_v3 (same protocol as the v3 stress runs).
set -u
PY=/home/jovyan/FedLoRA_remote_package/.venv/bin/python
ROOT=/home/jovyan/FedLoRA_remote_package
OUT="$ROOT/outputs/wcnc_extra/real_v3_dp"
SRC="$ROOT/outputs/real_fedlora_v3"
cd "$ROOT"
mkdir -p "$OUT" logs

run_one () {  # seed [extra...]
  local seed=$1; shift
  local out="$OUT/cifar10_seed${seed}"
  if [ -f "$out/metrics.csv" ] && [ "$(tail -1 "$out/metrics.csv" | cut -d, -f2)" = "39" ]; then
    echo "skip finished $out"; return
  fi
  mkdir -p "$out"
  cp "$SRC/cifar10_seed${seed}/backbone_pretrained.pt" \
     "$SRC/cifar10_seed${seed}/backbone_meta.json" "$out/" 2>/dev/null || true
  echo "RUN cifar10 DP_cohort seed$seed"
  $PY scripts/train_torch_fedlora.py --data data --dataset cifar10 --out "$out" \
    --rounds 40 --clients 20 --pretrain-epochs 30 --seed "$seed" \
    --methods DP_cohort "$@" \
    > "logs/wcnc_dp_cifar10_seed${seed}.log" 2>&1
}

CONC=3
running=0
for seed in 0 1 2 3 4; do
  run_one "$seed" &
  running=$((running+1))
  if [ $((running % CONC)) -eq 0 ]; then wait; fi
done
wait

for seed in 0 1 2 3 4; do
  out="$OUT/cifar10_dl0.25_seed${seed}"
  if [ -f "$out/metrics.csv" ] && [ "$(tail -1 "$out/metrics.csv" | cut -d, -f2)" = "39" ]; then
    echo "skip finished $out"; continue
  fi
  mkdir -p "$out"
  cp "$SRC/cifar10_seed${seed}/backbone_pretrained.pt" \
     "$SRC/cifar10_seed${seed}/backbone_meta.json" "$out/" 2>/dev/null || true
  echo "RUN cifar10_dl0.25 DP_cohort seed$seed"
  $PY scripts/train_torch_fedlora.py --data data --dataset cifar10 --out "$out" \
    --rounds 40 --clients 20 --pretrain-epochs 30 --deadline 0.25 --seed "$seed" \
    --methods DP_cohort \
    > "logs/wcnc_dp_cifar10_dl0.25_seed${seed}.log" 2>&1 &
  running=$((running+1))
  if [ $((running % CONC)) -eq 0 ]; then wait; fi
done
wait

echo ALL_RUNS_DONE
$PY scripts/summarize_real_fedlora.py --root "$OUT"

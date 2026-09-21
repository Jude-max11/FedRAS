#!/usr/bin/env bash
# WCNC seed extension: cifar10 nominal + 0.25 s deadline stress, seeds 5-9
# (extends the 5-seed v3 study to 10 seeds for the statistical claims).
# Writes only to outputs/wcnc_extra. Protocol mirrors scripts/run_real_v3.sh
# (same flags; per-seed backbone reuse for the stress runs).
set -u
PY="${PY:-python3}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/outputs/wcnc_extra/real_v3_ext"
cd "$ROOT"
mkdir -p "$OUT" logs

run_one () {  # seed [extra...]
  local seed=$1; shift
  local out="$OUT/cifar10_seed${seed}"
  if [ -f "$out/metrics.csv" ] && [ "$(tail -1 "$out/metrics.csv" | cut -d, -f2)" = "39" ]; then
    echo "skip finished $out"; return
  fi
  echo "RUN cifar10 seed$seed"
  $PY scripts/train_torch_fedlora.py --data data --dataset cifar10 --out "$out" \
    --rounds 40 --clients 20 --pretrain-epochs 30 --seed "$seed" "$@" \
    > "logs/wcnc_ext_cifar10_seed${seed}.log" 2>&1
}

CONC=3
running=0
for seed in 5 6 7 8 9; do
  run_one "$seed" &
  running=$((running+1))
  if [ $((running % CONC)) -eq 0 ]; then wait; fi
done
wait

# 0.25 s deadline stress with per-seed backbone reuse (protocol of real_fedlora_v3)
for seed in 5 6 7 8 9; do
  out="$OUT/cifar10_dl0.25_seed${seed}"
  mkdir -p "$out"
  cp "$OUT/cifar10_seed${seed}/backbone_pretrained.pt" \
     "$OUT/cifar10_seed${seed}/backbone_meta.json" "$out/" 2>/dev/null || true
  if [ -f "$out/metrics.csv" ] && [ "$(tail -1 "$out/metrics.csv" | cut -d, -f2)" = "39" ]; then
    echo "skip finished $out"; continue
  fi
  echo "RUN cifar10_dl0.25 seed$seed"
  $PY scripts/train_torch_fedlora.py --data data --dataset cifar10 --out "$out" \
    --rounds 40 --clients 20 --pretrain-epochs 30 --deadline 0.25 --seed "$seed" \
    > "logs/wcnc_ext_cifar10_dl0.25_seed${seed}.log" 2>&1 &
  running=$((running+1))
  if [ $((running % CONC)) -eq 0 ]; then wait; fi
done
wait

echo ALL_RUNS_DONE
$PY scripts/summarize_real_fedlora.py --root "$OUT"

#!/usr/bin/env bash
# Estimation-error arm on the SAME 6-epoch backbone as the main runs.
# (The first attempt passed --pretrain-epochs 30, so its accuracies were not
#  comparable; the decision-side quantities were also mixed with a backbone the
#  paper no longer reports.)
set -u
ROOT=/root/autodl-tmp/FedRAS
PY="${PY:-/root/miniconda3/bin/python}"
OUT="$ROOT/outputs/real_v6"; L="$ROOT/logs"
mkdir -p "$L" "$OUT"
CONC="${CONC:-8}"; running=0

while ! grep -q "\[fedavg6\] done" "$L/fedavg6.log" 2>/dev/null; do sleep 60; done
echo "[bias6] start $(date -u +%H:%M)" >> "$L/bias6.log"

launch () {
  local gpu=$1 tag=$2; shift 2
  local d="$OUT/$tag"
  if [ -f "$d/metrics.csv" ] && [ "$(tail -1 "$d/metrics.csv" | cut -d, -f2)" = "39" ]; then return; fi
  mkdir -p "$d"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    $PY scripts/train_torch_fedlora.py --data data --dataset cifar10 --out "$d" \
      --backbone-cache-dir "$OUT/bb_cache" --rounds 40 --clients 20 --per-round 10 \
      --methods Adaptive_rank "$@" \
      > "$L/v6bias6_$(echo "$tag" | tr '/' '_').log" 2>&1
}
tick () { running=$((running+1)); [ $((running % CONC)) -eq 0 ] && wait; }

for sc in 0.5 2.0; do
  for dl in 2.0 0.1; do
    for s in 0 1 2 3 4 5 6 7 8 9; do
      launch $(( s % 2 )) "p6b_mu${sc}_dl${dl}_seed${s}" --seed "$s" --deadline "$dl" --mu-est-scale "$sc" & tick
    done
  done
done
wait; echo "[bias6] mu done $(date -u +%H:%M)" >> "$L/bias6.log"

for eps in -0.3 -0.15 0.15 0.3; do
  for dl in 2.0 0.1; do
    for s in 0 1 2 3 4 5 6 7 8 9; do
      launch $(( s % 2 )) "p6b_rate${eps}_dl${dl}_seed${s}" --seed "$s" --deadline "$dl" --rate-est-bias "$eps" & tick
    done
  done
done
wait
echo "[bias6] done $(date -u +%H:%M)" >> "$L/bias6.log"

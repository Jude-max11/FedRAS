#!/usr/bin/env bash
# Reference point re-run on the SAME 6-epoch backbone as every other v6 run.
# (The original run_v6_fedavg.sh passed --pretrain-epochs 30, which made the
#  reference point incomparable with the LoRA runs.)
set -u
ROOT=/root/autodl-tmp/FedRAS
PY="${PY:-/root/miniconda3/bin/python}"
OUT="$ROOT/outputs/real_v6"; L="$ROOT/logs"
mkdir -p "$L" "$OUT"
echo "[fedavg6] start $(date -u +%H:%M)" >> "$L/fedavg6.log"
CONC="${CONC:-6}"; running=0

launch () {
  local gpu=$1 dl=$2 s=$3
  local d="$OUT/p5_fedavg6_dl${dl}_seed${s}"
  if [ -f "$d/metrics.csv" ] && [ "$(tail -1 "$d/metrics.csv" | cut -d, -f2)" = "39" ]; then return; fi
  mkdir -p "$d"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    $PY scripts/train_torch_fedlora.py --data data --dataset cifar10 --out "$d" \
      --backbone-cache-dir "$OUT/bb_cache" --rounds 40 --clients 20 --per-round 10 \
      --seed "$s" --deadline "$dl" --methods FedAvg_full \
      > "$L/v6fedavg6_dl${dl}_seed${s}.log" 2>&1
}
tick () { running=$((running+1)); [ $((running % CONC)) -eq 0 ] && wait; }
for dl in 2.0 0.1 0.05; do
  for s in 0 1 2 3 4 5 6 7 8 9; do
    launch $(( s % 2 )) "$dl" "$s" & tick
  done
done
wait
echo "[fedavg6] done $(date -u +%H:%M)" >> "$L/fedavg6.log"

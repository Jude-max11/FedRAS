#!/usr/bin/env bash
# Targeted ablations: (a) frozen-A vs joint A/B stability at elevated lr 0.02;
# (b) rank-mask on/off under the 0.25 s deadline where ranks are heterogeneous.
set -u
PY="${1:-/opt/conda/bin/python}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
OUT=outputs/real_fedlora_v3
mkdir -p "$OUT" logs

run_one () {  # name seed extra...
  local name=$1 seed=$2; shift 2
  local out="$OUT/${name}_seed${seed}"
  if [ -f "$out/metrics.csv" ] && [ "$(tail -1 "$out/metrics.csv" | cut -d, -f2)" = "39" ]; then
    echo "skip finished $out"; return
  fi
  mkdir -p "$out"
  cp "$OUT/cifar10_seed${seed}/backbone_pretrained.pt" "$OUT/cifar10_seed${seed}/backbone_meta.json" "$out/" 2>/dev/null || true
  echo "RUN $out"
  $PY scripts/train_torch_fedlora.py --data data --dataset cifar10 --out "$out" \
    --rounds 40 --clients 20 --pretrain-epochs 30 --seed "$seed" "$@" \
    > "logs/v3y_$(basename $out).log" 2>&1
}

CONC=8; running=0
for seed in 0 1 2 3 4; do
  run_one cifar10_frA_lr02 "$seed" --methods Adaptive_rank --no-full-baseline --lr 0.02 &
  running=$((running+1)); [ $((running % CONC)) -eq 0 ] && wait
  run_one cifar10_joint_lr02 "$seed" --methods JointAB --no-full-baseline --lr 0.02 &
  running=$((running+1)); [ $((running % CONC)) -eq 0 ] && wait
  run_one cifar10_nomask_dl0.25 "$seed" --methods Adaptive_rank --no-full-baseline --no-rank-mask --deadline 0.25 &
  running=$((running+1)); [ $((running % CONC)) -eq 0 ] && wait
done
wait
echo ABL2_DONE

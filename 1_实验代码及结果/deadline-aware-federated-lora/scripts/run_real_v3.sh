#!/usr/bin/env bash
# v3 real-training matrix: widened backbone (64/128/256), long pretraining
# (30 ep CIFAR-10 / 45 ep CIFAR-100 / 3 ep MNIST), bf16 AMP, RandomErasing.
# Usage: bash scripts/run_real_v3.sh [PY] [OUT_ROOT]
set -u
PY="${1:-python3}"
OUT="${2:-outputs/real_fedlora_v3}"
cd "$(dirname "$0")/.."
mkdir -p "$OUT" logs

run_one () {  # dataset seed [extra...]
  local ds=$1 seed=$2; shift 2
  local out="$OUT/${ds}_seed${seed}"
  if [ -f "$out/metrics.csv" ] && [ "$(tail -1 "$out/metrics.csv" | cut -d, -f2)" = "39" ]; then
    echo "skip finished $out"; return
  fi
  local data=data pre=30
  [ "$ds" = cifar100 ] && pre=45
  [ "$ds" = mnist ] && data=data/mnist && pre=3
  echo "RUN $ds seed$seed"
  $PY scripts/train_torch_fedlora.py --data "$data" --dataset "$ds" --out "$out" \
    --rounds 40 --clients 20 --pretrain-epochs "$pre" --seed "$seed" "$@" \
    > "logs/v3_${ds}_seed${seed}.log" 2>&1
}

CONC=5
running=0
for ds in cifar10 cifar100 mnist; do
  for seed in 0 1 2 3 4; do
    run_one "$ds" "$seed" &
    running=$((running+1))
    if [ $((running % CONC)) -eq 0 ]; then wait; fi
  done
done
wait

# deadline-stress runs on the validated seed-0 backbone
for dl in 0.5 0.25; do
  out="$OUT/cifar10_dl${dl}_seed0"
  mkdir -p "$out"
  cp "$OUT/cifar10_seed0/backbone_pretrained.pt" "$OUT/cifar10_seed0/backbone_meta.json" "$out/" 2>/dev/null || true
  echo "RUN stress dl=$dl"
  $PY scripts/train_torch_fedlora.py --data data --dataset cifar10 --out "$out" \
    --rounds 40 --clients 20 --pretrain-epochs 30 --deadline "$dl" --seed 0 \
    > "logs/v3_cifar10_dl${dl}_seed0.log" 2>&1
done

echo ALL_RUNS_DONE
$PY scripts/summarize_real_fedlora.py --root "$OUT"

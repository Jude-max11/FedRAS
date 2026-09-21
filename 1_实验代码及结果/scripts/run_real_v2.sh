#!/usr/bin/env bash
# Final v2 real-training matrix: 3 datasets x 5 seeds, 40 federated rounds.
# Recipe: frozen-A LoRA (FFA-LoRA style), lr 0.005, 10 clients/round,
# server momentum 0.3, centralized backbone pretraining (6 ep CIFAR / 3 ep MNIST).
# Usage: bash scripts/run_real_v2.sh [PY] [OUT_ROOT]
set -u
PY="${1:-/opt/conda/bin/python}"
OUT="${2:-outputs/real_fedlora_v2}"
cd "$(dirname "$0")/.."
mkdir -p "$OUT" logs

run_one () {  # dataset seed [extra args...]
  local ds=$1 seed=$2; shift 2
  local out="$OUT/${ds}_seed${seed}"
  if [ -f "$out/metrics.csv" ] && [ "$(tail -1 "$out/metrics.csv" | cut -d, -f2)" = "39" ]; then
    echo "skip finished $out"; return
  fi
  local data=data pre=6
  [ "$ds" = mnist ] && data=data/mnist && pre=3
  echo "RUN $ds seed$seed"
  $PY scripts/train_torch_fedlora.py --data "$data" --dataset "$ds" --out "$out" \
    --rounds 40 --clients 20 --pretrain-epochs "$pre" --seed "$seed" "$@" \
    > "logs/${ds}_seed${seed}.log" 2>&1
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
echo ALL_RUNS_DONE
$PY scripts/summarize_real_fedlora.py --root "$OUT"

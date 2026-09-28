#!/usr/bin/env bash
# Reviewer-response batch: FP16 compression baseline, weak-backbone capacity
# payoff, FedAvg full-lr stability probe.
# - Standard 30-epoch backbone reused per seed where the comparison requires
#   it (the trainer's fingerprint check enforces config agreement).
# - Partial runs resume from checkpoint.pt; finished runs (round 39) are skipped.
set -u
PY=/opt/conda/bin/python
OUT=outputs/real_fedlora_v4
cd "$(dirname "$0")/.."
mkdir -p logs
CONC="${CONC:-10}"; running=0
tick () { running=$((running + 1)); [ $((running % CONC)) -eq 0 ] && wait; }

run_one () {  # out deadline pre_epochs methods extra...
  local out=$1 dl=$2 pe=$3 methods=$4; shift 4
  local last=""; local RES=""
  if [ -f "$OUT/$out/metrics.csv" ]; then
    last=$(tail -1 "$OUT/$out/metrics.csv" | cut -d, -f2)
  fi
  if [ "$last" = "39" ]; then echo "skip finished $out"; return; fi
  if [ -n "$last" ] && [ -f "$OUT/$out/checkpoint.pt" ]; then RES="--resume"; fi
  mkdir -p "$OUT/$out"
  echo "RUN $out (last=$last res=$RES)"
  $PY scripts/train_torch_fedlora.py --data data --dataset cifar10 --out "$OUT/$out" \
    --rounds 40 --clients 20 --pretrain-epochs "$pe" --deadline "$dl" \
    --methods "$methods" $RES "$@" > "logs/rr_${out}.log" 2>&1
  tick
}

copy_backbone () {  # dst_dir seed
  cp "$OUT/cifar10_seed$2/backbone_pretrained.pt" "$OUT/cifar10_seed$2/backbone_meta.json" "$1/" 2>/dev/null || true
}

# A) FP16 compression baseline: r16 at half payload with real fp16 round-trip.
for s in 0 1 2 3 4; do
  for cfg in "fp16_dl0.1_seed${s}:0.1" "fp16_seed${s}:2.0"; do
    out="${cfg%%:*}"; dl="${cfg##*:}"
    mkdir -p "$OUT/$out"; copy_backbone "$OUT/$out" "$s"
    run_one "$out" "$dl" 30 "FedLoRA_r16" --fp16-upload &
  done
done
wait

# B) Weak-backbone capacity payoff: 1-epoch pretraining, fresh per run.
for s in 0 1 2 3 4; do
  run_one "weak_seed${s}" 2.0 1 "FedLoRA_r4,FedLoRA_r16,Adaptive_rank" &
  run_one "weak_dl0.1_seed${s}" 0.1 1 "FedLoRA_r4,FedLoRA_r16,Adaptive_rank" &
done
wait

# C) FedAvg full-lr stability probe (nominal, 3 seeds, standard backbone).
for s in 0 1 2; do
  mkdir -p "$OUT/fedavg_lr10_seed${s}"; copy_backbone "$OUT/fedavg_lr10_seed${s}" "$s"
  run_one "fedavg_lr10_seed${s}" 2.0 30 "FedAvg_full" --full-lr-scale 1.0 &
done
wait
echo "BATCH DONE"

#!/usr/bin/env bash
# Staged reviewer-round runner: priority order, bounded concurrency per stage.
set -u
PY=/opt/conda/bin/python
OUT=outputs/real_fedlora_v4
cd "$(dirname "$0")/.."
mkdir -p logs

run_one () {  # out deadline pe methods extra...
  local out=$1 dl=$2 pe=$3 methods=$4; shift 4
  local last=""; local RES=""
  if [ -f "$OUT/$out/metrics.csv" ]; then last=$(tail -1 "$OUT/$out/metrics.csv" | cut -d, -f2); fi
  if [ "$last" = "39" ]; then echo "skip finished $out"; return; fi
  if [ -n "$last" ] && [ -f "$OUT/$out/checkpoint.pt" ]; then RES="--resume"; fi
  mkdir -p "$OUT/$out"
  echo "RUN $out (last=$last)"
  $PY scripts/train_torch_fedlora.py --data data --dataset cifar10 --out "$OUT/$out" \
    --rounds 40 --clients 20 --pretrain-epochs "$pe" --deadline "$dl" \
    --methods "$methods" $RES "$@" > "logs/rr_${out}.log" 2>&1
}

copy_backbone () { cp "$OUT/cifar10_seed$2/backbone_pretrained.pt" "$OUT/cifar10_seed$2/backbone_meta.json" "$1/" 2>/dev/null || true; }

stage () {  # conc, then specs "out:dl:pe:methods:extra"
  local conc=$1; shift
  local running=0
  local spec out dl pe methods extra
  for spec in "$@"; do
    IFS=':' read -r out dl pe methods extra <<< "$spec"
    run_one "$out" "$dl" "$pe" "$methods" $extra &
    running=$((running + 1))
    if [ $((running % conc)) -eq 0 ]; then wait; fi
  done
  wait
}

# Stage 1: complete the stress FP16 cells (composition claim + comparison row).
stage 7 \
  "fedras_fp16_dl0.1_seed2:0.1:30:Adaptive_rank:--fp16-upload" \
  "fedras_fp16_dl0.1_seed3:0.1:30:Adaptive_rank:--fp16-upload" \
  "fedras_fp16_dl0.1_seed4:0.1:30:Adaptive_rank:--fp16-upload" \
  "fp16_dl0.1_seed1:0.1:30:FedLoRA_r16:--fp16-upload" \
  "fp16_dl0.1_seed2:0.1:30:FedLoRA_r16:--fp16-upload" \
  "fp16_dl0.1_seed3:0.1:30:FedLoRA_r16:--fp16-upload" \
  "fp16_dl0.1_seed4:0.1:30:FedLoRA_r16:--fp16-upload"
echo "STAGE1 DONE"

# Stage 2: weak-backbone capacity payoff.
stage 10 \
  "weak_seed0:2.0:1:FedLoRA_r4,FedLoRA_r16,Adaptive_rank:" \
  "weak_seed1:2.0:1:FedLoRA_r4,FedLoRA_r16,Adaptive_rank:" \
  "weak_seed2:2.0:1:FedLoRA_r4,FedLoRA_r16,Adaptive_rank:" \
  "weak_seed3:2.0:1:FedLoRA_r4,FedLoRA_r16,Adaptive_rank:" \
  "weak_seed4:2.0:1:FedLoRA_r4,FedLoRA_r16,Adaptive_rank:" \
  "weak_dl0.1_seed0:0.1:1:FedLoRA_r4,FedLoRA_r16,Adaptive_rank:" \
  "weak_dl0.1_seed1:0.1:1:FedLoRA_r4,FedLoRA_r16,Adaptive_rank:" \
  "weak_dl0.1_seed2:0.1:1:FedLoRA_r4,FedLoRA_r16,Adaptive_rank:" \
  "weak_dl0.1_seed3:0.1:1:FedLoRA_r4,FedLoRA_r16,Adaptive_rank:" \
  "weak_dl0.1_seed4:0.1:1:FedLoRA_r4,FedLoRA_r16,Adaptive_rank:"
echo "STAGE2 DONE"

# Stage 3: nominal FP16 cells + FedAvg lr probe.
for s in 0 1 2 3 4; do
  mkdir -p "$OUT/fp16_seed${s}"; copy_backbone "$OUT/fp16_seed${s}" "$s"
  mkdir -p "$OUT/fedavg_lr10_seed${s}"; copy_backbone "$OUT/fedavg_lr10_seed${s}" "$s"
done
stage 8 \
  "fp16_seed0:2.0:30:FedLoRA_r16:--fp16-upload" \
  "fp16_seed1:2.0:30:FedLoRA_r16:--fp16-upload" \
  "fp16_seed2:2.0:30:FedLoRA_r16:--fp16-upload" \
  "fp16_seed3:2.0:30:FedLoRA_r16:--fp16-upload" \
  "fp16_seed4:2.0:30:FedLoRA_r16:--fp16-upload" \
  "fedavg_lr10_seed0:2.0:30:FedAvg_full:--full-lr-scale 1.0" \
  "fedavg_lr10_seed1:2.0:30:FedAvg_full:--full-lr-scale 1.0" \
  "fedavg_lr10_seed2:2.0:30:FedAvg_full:--full-lr-scale 1.0"
echo "STAGE3 DONE"
echo "BATCH DONE"

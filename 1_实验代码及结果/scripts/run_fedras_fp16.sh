#!/usr/bin/env bash
set -u
PY=${PY:-/home/jovyan/FedLoRA_remote_package/.venv/bin/python}
OUT=outputs/real_fedlora_v4
cd "$(dirname "$0")/.."
for s in 0 1 2 3 4; do
  for spec in "fedras_fp16_seed${s}:2.0:Adaptive_rank" "fedras_fp16_dl0.1_seed${s}:0.1:Adaptive_rank" "fp16_seed${s}:2.0:FedLoRA_r16" "fp16_dl0.1_seed${s}:0.1:FedLoRA_r16"; do
    IFS=: read -r out dl meth <<< "$spec"
    if [ -f "$OUT/$out/metrics.csv" ] && [ "$(tail -1 "$OUT/$out/metrics.csv" | cut -d, -f2)" = "39" ]; then echo "skip $out"; continue; fi
    mkdir -p "$OUT/$out"
    cp "$OUT/cifar10_seed${s}/backbone_pretrained.pt" "$OUT/cifar10_seed${s}/backbone_meta.json" "$OUT/$out/" 2>/dev/null || true
    $PY scripts/train_torch_fedlora.py --data data --dataset cifar10 --out "$OUT/$out" \
      --rounds 40 --clients 20 --pretrain-epochs 30 --deadline "$dl" --seed "$s" \
      --methods "$meth" --fp16-upload > "logs/rr_${out}.log" 2>&1 &
  done
done
wait
echo "FP16 ARMS DONE"

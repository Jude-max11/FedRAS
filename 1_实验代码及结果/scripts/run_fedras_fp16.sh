#!/usr/bin/env bash
set -u
PY=/opt/conda/bin/python
OUT=outputs/real_fedlora_v4
cd "$(dirname "$0")/.."
for s in 0 1 2 3 4; do
  for cfg in "fedras_fp16_dl0.1_seed${s}:0.1" "fedras_fp16_seed${s}:2.0"; do
    out="${cfg%%:*}"; dl="${cfg##*:}"
    if [ -f "$OUT/$out/metrics.csv" ] && [ "$(tail -1 "$OUT/$out/metrics.csv" | cut -d, -f2)" = "39" ]; then echo "skip $out"; continue; fi
    mkdir -p "$OUT/$out"
    cp "$OUT/cifar10_seed${s}/backbone_pretrained.pt" "$OUT/cifar10_seed${s}/backbone_meta.json" "$OUT/$out/" 2>/dev/null || true
    echo "RUN $out"
    $PY scripts/train_torch_fedlora.py --data data --dataset cifar10 --out "$OUT/$out" \
      --rounds 40 --clients 20 --pretrain-epochs 30 --deadline "$dl" \
      --methods Adaptive_rank --fp16-upload > "logs/rr_${out}.log" 2>&1 &
  done
done
wait
echo "FEDRAS_FP16 DONE"

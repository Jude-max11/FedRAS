#!/usr/bin/env bash
# Norm-vs-rank curve: fixed rank 8 and rank 2 arms, nominal + 0.1 s, 5 seeds.
set -u
PY=/opt/conda/bin/python
OUT=outputs/real_fedlora_v4
cd "$(dirname "$0")/.."
mkdir -p logs
CONC="${CONC:-10}"; running=0
for s in 0 1 2 3 4; do
  for cfg in "nrm_r8_seed${s}:2.0:FedLoRA_r8" "nrm_r2_seed${s}:2.0:FedLoRA_r2" \
             "nrm_r8_dl0.1_seed${s}:0.1:FedLoRA_r8" "nrm_r2_dl0.1_seed${s}:0.1:FedLoRA_r2"; do
    out=$(echo "$cfg" | cut -d: -f1); dl=$(echo "$cfg" | cut -d: -f2); me=$(echo "$cfg" | cut -d: -f3)
    if [ -f "$OUT/$out/metrics.csv" ] && [ "$(tail -1 "$OUT/$out/metrics.csv" | cut -d, -f2)" = "39" ]; then echo "skip $out"; continue; fi
    mkdir -p "$OUT/$out"
    cp "$OUT/cifar10_seed${s}/backbone_pretrained.pt" "$OUT/cifar10_seed${s}/backbone_meta.json" "$OUT/$out/" 2>/dev/null || true
    echo "RUN $out"
    $PY scripts/train_torch_fedlora.py --data data --dataset cifar10 --out "$OUT/$out" \
      --rounds 40 --clients 20 --pretrain-epochs 30 --seed "$s" --deadline "$dl" \
      --methods "$me" > "logs/rr_${out}.log" 2>&1 &
    running=$((running + 1)); [ $((running % CONC)) -eq 0 ] && wait
  done
done
wait
echo "NORM CURVE RUNS DONE"

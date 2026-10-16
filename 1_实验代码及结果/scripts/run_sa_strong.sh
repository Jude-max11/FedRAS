#!/usr/bin/env bash
# SA_rank arms on the strong backbone (nominal + 0.1 s, 10 seeds): completes the
# static-policy failure table. Single-method, no full baseline; seed-matched
# backbone copied from the cifar10 family (fingerprint includes the seed).
set -u
PY=/home/jovyan/FedLoRA_remote_package/.venv/bin/python
OUT=outputs/real_fedlora_v4
cd "$(dirname "$0")/.."
mkdir -p logs
CONC="${CONC:-4}"; running=0
for s in $(seq 0 9); do
  for cfg in "sa_strong_seed${s}:2.0" "sa_strong_dl0.1_seed${s}:0.1"; do
    out=$(echo "$cfg" | cut -d: -f1); dl=$(echo "$cfg" | cut -d: -f2)
    if [ -f "$OUT/$out/metrics.csv" ] && [ "$(tail -1 "$OUT/$out/metrics.csv" | cut -d, -f2)" = "39" ]; then echo "skip $out"; continue; fi
    rm -rf "$OUT/$out"; mkdir -p "$OUT/$out"
    cp "$OUT/cifar10_seed${s}/backbone_pretrained.pt" "$OUT/$out/" 2>/dev/null || true
    # cifar10_seed* metas predate the vit/pretrain_frac fingerprint fields; patch
    # the copy so the trainer's staleness check hits and loads the copied weights.
    $PY -c "
import json, sys
src = '$OUT/cifar10_seed${s}/backbone_meta.json'
dst = '$OUT/$out/backbone_meta.json'
try:
    fp = json.load(open(src))
except FileNotFoundError:
    sys.exit(0)
fp.setdefault('vit', None); fp.setdefault('pretrain_frac', 1.0)
fp['arch'] = fp['arch'].replace('cnn_lora_v3', 'cnn_lora')  # old fingerprint scheme
json.dump(fp, open(dst, 'w'))"
    echo "RUN $out"
    $PY scripts/train_torch_fedlora.py --data data --dataset cifar10 --out "$OUT/$out" \
      --rounds 40 --clients 20 --per-round 10 --pretrain-epochs 30 --seed "$s" --deadline "$dl" \
      --methods SA_rank --no-full-baseline > "logs/rr_${out}.log" 2>&1 &
    running=$((running + 1)); [ $((running % CONC)) -eq 0 ] && wait
  done
done
wait
echo "SA STRONG RUNS DONE"

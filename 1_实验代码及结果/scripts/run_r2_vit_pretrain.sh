#!/usr/bin/env bash
# Warm the ViT backbone cache: 10 seeds clean (pretrain-frac 1.0) then 10 seeds
# shift-corpus (frac 0.5). Concurrency 4. Runs after batch1 or alongside it.
set -u
PY=/home/jovyan/FedLoRA_remote_package/.venv/bin/python
cd "$(dirname "$0")/.."
OUT=outputs/real_fedlora_v4/vit_bcache
LOG=logs
mkdir -p "$OUT"

launch() {
  while [ "$(jobs -rp | wc -l)" -ge 4 ]; do sleep 30; done
  echo "launch $*"
  "$@" &
}

for s in 0 1 2 3 4 5 6 7 8 9; do
  launch $PY scripts/train_torch_fedlora.py --data data --dataset cifar10 --arch vit \
    --pretrain-epochs 80 --pretrain-lr 0.02 --pretrain-frac 1.0 --seed "$s" \
    --pretrain-only --backbone-cache-dir "$OUT" \
    --out "$OUT/warm_seed${s}" > "$LOG/r2_vitpre_c${s}.log" 2>&1
done
wait
echo "VIT CLEAN PRETRAIN DONE"
for s in 0 1 2 3 4 5 6 7 8 9; do
  launch $PY scripts/train_torch_fedlora.py --data data --dataset cifar10 --arch vit \
    --pretrain-epochs 80 --pretrain-lr 0.02 --pretrain-frac 0.5 --seed "$s" \
    --pretrain-only --backbone-cache-dir "$OUT" \
    --out "$OUT/warmf_seed${s}" > "$LOG/r2_vitpre_f${s}.log" 2>&1
done
wait
echo "VIT FRAC PRETRAIN DONE"

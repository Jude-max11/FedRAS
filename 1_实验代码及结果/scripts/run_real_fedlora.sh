#!/usr/bin/env bash
set -euo pipefail
DATA=${1:?dataset root required}; OUT=${2:-outputs/real_fedlora}; DATASET=${3:-cifar10}
python scripts/train_torch_fedlora.py --data "$DATA" --dataset "$DATASET" --out "$OUT/$DATASET" --rounds 30 --clients 20 --per-round 5 --local-epochs 1 --device auto

#!/usr/bin/env bash
# Extended evaluation matrix (P0+P1): external baselines, component ablations,
# and sensitivity sweeps on CIFAR-10, five seeds each, backbones reused from
# the main matrix (fingerprints match: rmax/cohort/alpha/flags are not part of
# the backbone fingerprint).
#   P0 ext:        SA_rank + Oort_utility + JointAB, nominal + dl 0.25   (10 runs)
#   P1 ablation:   no-rank-mask / no-capacity-term / no-server-momentum  (15 runs)
#   P1 sensitivity: rmax 8/32, per-round 5/20, alpha 1.0                  (25 runs)
# Usage: bash scripts/run_ext_v3.sh
set -u
PY="${1:-/opt/conda/bin/python}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
OUT=outputs/real_fedlora_v3
mkdir -p "$OUT" logs

run_one () {  # name seed extra...
  local name=$1 seed=$2; shift 2
  local out="$OUT/${name}_seed${seed}"
  if [ -f "$out/metrics.csv" ] && [ "$(tail -1 "$out/metrics.csv" | cut -d, -f2)" = "39" ]; then
    echo "skip finished $out"; return
  fi
  mkdir -p "$out"
  local src="$OUT/cifar10_seed${seed}"
  cp "$src/backbone_pretrained.pt" "$src/backbone_meta.json" "$out/" 2>/dev/null || true
  echo "RUN $out"
  $PY scripts/train_torch_fedlora.py --data data --dataset cifar10 --out "$out" \
    --rounds 40 --clients 20 --pretrain-epochs 30 --seed "$seed" "$@" \
    > "logs/v3x_$(basename $out).log" 2>&1
}

CONC=8; running=0
for seed in 0 1 2 3 4; do
  run_one cifar10_ext "$seed" --methods SA_rank,Oort_utility,JointAB --no-full-baseline &
  running=$((running+1)); [ $((running % CONC)) -eq 0 ] && wait
  run_one cifar10_ext_dl0.25 "$seed" --methods SA_rank,Oort_utility,JointAB --no-full-baseline --deadline 0.25 &
  running=$((running+1)); [ $((running % CONC)) -eq 0 ] && wait
  run_one cifar10_nomask "$seed" --no-rank-mask --no-full-baseline &
  running=$((running+1)); [ $((running % CONC)) -eq 0 ] && wait
  run_one cifar10_nocap "$seed" --no-capacity-term --no-full-baseline &
  running=$((running+1)); [ $((running % CONC)) -eq 0 ] && wait
  run_one cifar10_nomom "$seed" --server-momentum 0 --no-full-baseline &
  running=$((running+1)); [ $((running % CONC)) -eq 0 ] && wait
  run_one cifar10_rmax8 "$seed" --rmax 8 --no-full-baseline &
  running=$((running+1)); [ $((running % CONC)) -eq 0 ] && wait
  run_one cifar10_rmax32 "$seed" --rmax 32 --no-full-baseline &
  running=$((running+1)); [ $((running % CONC)) -eq 0 ] && wait
  run_one cifar10_pr5 "$seed" --per-round 5 --no-full-baseline &
  running=$((running+1)); [ $((running % CONC)) -eq 0 ] && wait
  run_one cifar10_pr20 "$seed" --per-round 20 --no-full-baseline &
  running=$((running+1)); [ $((running % CONC)) -eq 0 ] && wait
  run_one cifar10_alpha1.0 "$seed" --alpha 1.0 --no-full-baseline &
  running=$((running+1)); [ $((running % CONC)) -eq 0 ] && wait
done
wait
echo ALL_EXT_DONE
$PY scripts/summarize_real_fedlora.py --root "$OUT"

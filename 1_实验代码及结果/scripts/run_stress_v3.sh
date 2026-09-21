#!/usr/bin/env bash
# v3 stress matrix: five seeds per stress scenario, reusing the per-seed
# backbones pretrained in the main matrix (fingerprints match: deadline,
# mobility shape and Dirichlet alpha are not part of the backbone fingerprint).
#   cifar10: dl 0.5 / 0.25 / 2.0 (loose control) x5, mobility 0.7 x5, alpha 0.1 x5
#   cifar100: dl 0.25 x5        mnist: dl 0.25 x5
# Usage: bash scripts/run_stress_v3.sh
set -u
PY="${1:-/opt/conda/bin/python}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
OUT=outputs/real_fedlora_v3
mkdir -p "$OUT" logs

pre_for () {  # dataset -> pretrain epochs used by the main matrix
  case "$1" in
    cifar100) echo 45;;
    mnist) echo 3;;
    *) echo 30;;
  esac
}

run_one () {  # dataset variant seed  (variant: "dl 0.5" | "mob 0.7" | "alpha 0.1")
  local ds=$1 vkind=$2 vval=$3 seed=$4
  local tag extra
  case "$vkind" in
    dl)    tag="dl${vval}";  extra="--deadline ${vval}";;
    mob)   tag="mob${vval}"; extra="--mobility-shape ${vval}";;
    alpha) tag="alpha${vval}"; extra="--alpha ${vval}";;
  esac
  local out="$OUT/${ds}_${tag}_seed${seed}"
  if [ -f "$out/metrics.csv" ] && [ "$(tail -1 "$out/metrics.csv" | cut -d, -f2)" = "39" ]; then
    echo "skip finished $out"; return
  fi
  mkdir -p "$out"
  local src="$OUT/${ds}_seed${seed}"
  if [ -f "$src/backbone_pretrained.pt" ]; then
    cp "$src/backbone_pretrained.pt" "$src/backbone_meta.json" "$out/"
  fi
  local pre; pre=$(pre_for "$ds")
  local data=data
  [ "$ds" = mnist ] && data=data/mnist
  echo "RUN $out"
  $PY scripts/train_torch_fedlora.py --data "$data" --dataset "$ds" --out "$out" \
    --rounds 40 --clients 20 --pretrain-epochs "$pre" --seed "$seed" $extra \
    > "logs/v3s_$(basename $out).log" 2>&1
}

CONC=5; running=0
for seed in 0 1 2 3 4; do
  for v in "dl 0.5" "dl 0.25" "dl 2.0" "mob 0.7" "alpha 0.1"; do
    set -- $v; run_one cifar10 "$1" "$2" "$seed" &
    running=$((running+1))
    if [ $((running % CONC)) -eq 0 ]; then wait; fi
  done
  run_one cifar100 dl 0.25 "$seed" &
  running=$((running+1))
  if [ $((running % CONC)) -eq 0 ]; then wait; fi
  run_one mnist dl 0.25 "$seed" &
  running=$((running+1))
  if [ $((running % CONC)) -eq 0 ]; then wait; fi
done
wait
echo ALL_STRESS_DONE
$PY scripts/summarize_real_fedlora.py --root "$OUT"

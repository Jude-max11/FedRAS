#!/usr/bin/env bash
# v4 real-byte era matrix (teacher review response, Phase 0 decisions):
#   - REAL serialized payloads (B-only, 5,632 B/rank; FedAvg_full 5.286 MB)
#   - recalibrated deadline axis: nominal 2.0 s, stress 0.1 s, severe 0.05 s,
#     loose control 10.0 s
#   - uniform CONSERVATIVE ceil budget grid (0.01 MB) everywhere
#   - rank-r local training, truly frozen A, per-method evaluation
# Phases: A core CIFAR-10 deadlines | B baselines/DP arms/ablation controls
#         C full matrix (10-seed ext, cross-dataset, stress, sensitivity, sagin)
#         D CPU: solver study + simulator regimes + calibration
# Usage: bash scripts/run_v4_matrix.sh A [PYTHON]
set -u
PY="${2:-/opt/conda/bin/python}"; PHASE="${1:-A}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
OUT=outputs/real_fedlora_v4
V3=outputs/real_fedlora_v3; V3EXT=outputs/wcnc_extra/real_v3_ext
mkdir -p "$OUT" logs
CONC="${CONC:-8}"; running=0

pre_epochs () { case "$1" in cifar100) echo 45;; mnist) echo 3;; *) echo 30;; esac; }

run_one () {  # name seed dataset pre_from deadline extra...
  local name=$1 seed=$2 ds=$3 pre_from=$4 dl=$5; shift 5
  local out="$OUT/${name}_seed${seed}"
  if [ -f "$out/metrics.csv" ] && [ "$(tail -1 "$out/metrics.csv" | cut -d, -f2)" = "39" ]; then
    echo "skip finished $out"; return
  fi
  mkdir -p "$out"
  # Drop a possibly truncated trailing line before resuming (checkpoint is
  # saved every round, so at most one row is lost).
  if [ -f "$out/metrics.csv" ]; then
    awk -F, 'NR==1 || NF==10' "$out/metrics.csv" > "$out/metrics.csv.tmp" && mv "$out/metrics.csv.tmp" "$out/metrics.csv"
  fi
  cp "$pre_from/backbone_pretrained.pt" "$pre_from/backbone_meta.json" "$out/" 2>/dev/null || true
  if [ "$ds" = mnist ]; then DPATH=data/mnist; else DPATH=data; fi
  echo "RESUME $out"
  # flock guards against a second writer on the same run directory; setsid
  # detaches the trainer from this script's process tree so a harness-side
  # cleanup of the watchdog task cannot reap running experiments.
  (
    flock -n 9 || { echo "busy $out"; exit 0; }
    setsid $PY scripts/train_torch_fedlora.py --data "$DPATH" --dataset "$ds" --out "$out" \
      --rounds 40 --clients 20 --pretrain-epochs "$(pre_epochs "$ds")" --seed "$seed" \
      --deadline "$dl" --resume "$@" \
      > "logs/v4_$(basename $out).log" 2>&1
  ) 9>"$out/.lock"
}
tick () { running=$((running+1)); [ $((running % CONC)) -eq 0 ] && wait; }

case "$PHASE" in
A)
  for seed in 0 1 2 3 4; do
    run_one cifar10 "$seed" cifar10 "$V3/cifar10_seed$seed" 2.0 & tick
    run_one cifar10_dl0.1 "$seed" cifar10 "$V3/cifar10_seed$seed" 0.1 & tick
    run_one cifar10_dl0.05 "$seed" cifar10 "$V3/cifar10_seed$seed" 0.05 & tick
    run_one cifar10_dl10.0 "$seed" cifar10 "$V3/cifar10_seed$seed" 10.0 & tick
  done
  wait; echo PHASE_A_DONE ;;
B)
  for seed in 0 1 2 3 4; do
    run_one cifar10_ext "$seed" cifar10 "$V3/cifar10_seed$seed" 2.0 --methods SA_rank,Oort_utility,JointAB & tick
    run_one cifar10_ext_dl0.1 "$seed" cifar10 "$V3/cifar10_seed$seed" 0.1 --methods SA_rank,Oort_utility,JointAB & tick
    run_one dp "$seed" cifar10 "$V3/cifar10_seed$seed" 2.0 --methods DP_cohort --no-full-baseline & tick
    run_one dpsame "$seed" cifar10 "$V3/cifar10_seed$seed" 2.0 --methods DP_same --no-full-baseline & tick
    run_one cifar10_rrmask_dl0.1 "$seed" cifar10 "$V3/cifar10_seed$seed" 0.1 --methods Random_rank --no-full-baseline & tick
    run_one cifar10_rrnomask_dl0.1 "$seed" cifar10 "$V3/cifar10_seed$seed" 0.1 --methods Random_rank --no-full-baseline --no-rank-mask & tick
    run_one fedavg_offline "$seed" cifar10 "$V3/cifar10_seed$seed" 2.0 --methods FedAvg_offline --no-full-baseline & tick
  done
  wait; echo PHASE_B_DONE ;;
C)
  for seed in 5 6 7 8 9; do
    run_one cifar10 "$seed" cifar10 "$V3EXT/cifar10_seed$seed" 2.0 & tick
    run_one cifar10_dl0.1 "$seed" cifar10 "$V3EXT/cifar10_seed$seed" 0.1 & tick
    run_one cifar10_ext "$seed" cifar10 "$V3EXT/cifar10_seed$seed" 2.0 --methods SA_rank,Oort_utility,JointAB & tick
    run_one cifar10_ext_dl0.1 "$seed" cifar10 "$V3EXT/cifar10_seed$seed" 0.1 --methods SA_rank,Oort_utility,JointAB & tick
    run_one dp "$seed" cifar10 "$V3EXT/cifar10_seed$seed" 2.0 --methods DP_cohort --no-full-baseline & tick
    run_one dpsame "$seed" cifar10 "$V3EXT/cifar10_seed$seed" 2.0 --methods DP_same --no-full-baseline & tick
  done
  for seed in 0 1 2 3 4; do
    run_one cifar100 "$seed" cifar100 "$V3/cifar100_seed$seed" 2.0 & tick
    run_one cifar100_dl0.1 "$seed" cifar100 "$V3/cifar100_seed$seed" 0.1 & tick
    run_one mnist "$seed" mnist "$V3/mnist_seed$seed" 2.0 & tick
    run_one mnist_dl0.1 "$seed" mnist "$V3/mnist_seed$seed" 0.1 & tick
    run_one cifar10_mob0.7 "$seed" cifar10 "$V3/cifar10_seed$seed" 2.0 --mobility-shape 0.7 & tick
    run_one cifar10_alpha0.1 "$seed" cifar10 "$V3/cifar10_seed$seed" 2.0 --alpha 0.1 & tick
    run_one cifar10_alpha1.0 "$seed" cifar10 "$V3/cifar10_seed$seed" 2.0 --alpha 1.0 & tick
    run_one cifar10_rmax8 "$seed" cifar10 "$V3/cifar10_seed$seed" 2.0 --rmax 8 --no-full-baseline & tick
    run_one cifar10_rmax32 "$seed" cifar10 "$V3/cifar10_seed$seed" 2.0 --rmax 32 --no-full-baseline & tick
    run_one cifar10_pr5 "$seed" cifar10 "$V3/cifar10_seed$seed" 2.0 --per-round 5 --no-full-baseline & tick
    run_one cifar10_pr20 "$seed" cifar10 "$V3/cifar10_seed$seed" 2.0 --per-round 20 --no-full-baseline & tick
    run_one cifar10_sagin "$seed" cifar10 "$V3/cifar10_seed$seed" 2.0 --link-profile sagin & tick
  done
  wait; echo PHASE_C_DONE ;;
D)
  $PY scripts/dp_scheduler_study.py --grid-sweep > logs/v4_solver_study.log 2>&1
  for cfg in "nominal:2.0:30.0:-" "dl0.1:0.1:30.0:-" "dl0.05:0.05:30.0:-" "budget1.2:2.0:1.2:-" "mob0.7:2.0:30.0:0.7" "clients1000:2.0:30.0:-" "stylized_sens:0.25:12.0:-"; do
    IFS=: read -r tag dl bud mob <<< "$cfg"
    extra=""; [ "$mob" != "-" ] && extra="--mobility-shape $mob"
    pm=""; [ "$tag" = "stylized_sens" ] && pm="--payload-model stylized"
    cl=""; [ "$tag" = "clients1000" ] && cl="--clients 1000"
    echo "SIM $tag"
    $PY fedlora/run_experiment.py --out "outputs/fedlora_sim_v4/$tag" --deadline "$dl" --budget "$bud" $extra $pm $cl \
      > "logs/v4_sim_$tag.log" 2>&1
  done
  $PY scripts/journal_calibration.py > logs/v4_calibration.log 2>&1
  echo PHASE_D_DONE ;;
*) echo "unknown phase $PHASE"; exit 1 ;;
esac

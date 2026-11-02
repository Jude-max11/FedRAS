#!/usr/bin/env bash
# Table 5(d) top-up, fixed: (1) fresh output namespace so the skip check does not
# short-circuit, (2) no --pretrain-epochs override, so the backbone matches the
# main 6-epoch protocol, (3) waits for the bias arm to finish first.
set -u
ROOT=/root/autodl-tmp/FedRAS
PY="${PY:-/root/miniconda3/bin/python}"
OUT="$ROOT/outputs/real_v6"; L="$ROOT/logs"
mkdir -p "$L" "$OUT"
CONC="${CONC:-8}"; running=0
M3="SA_mono,Oort_utility,ROBA_style"

while ! grep -q "\[bias\] done" "$L/bias.log" 2>/dev/null; do sleep 60; done
echo "[5d-full2] start $(date -u +%H:%M)" >> "$L/tab5d_full2.log"

launch () {  # gpu tag extra...
  local gpu=$1 tag=$2; shift 2
  local d="$OUT/$tag"
  if [ -f "$d/metrics.csv" ] && [ "$(tail -1 "$d/metrics.csv" | cut -d, -f2)" = "39" ]; then return; fi
  mkdir -p "$d"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    $PY scripts/train_torch_fedlora.py --data data --out "$d" \
      --backbone-cache-dir "$OUT/bb_cache" --rounds 40 --clients 20 --per-round 10 \
      --methods "$M3" "$@" > "$L/v6d2_$(echo "$tag" | tr '/' '_').log" 2>&1
}
tick () { running=$((running+1)); [ $((running % CONC)) -eq 0 ] && wait; }

for ds in cifar10 cifar100 mnist; do
  for s in 0 1 2 3 4 5 6 7 8 9; do
    g=$(( s % 2 ))
    launch "$g" "p2_tab5df_${ds}_dl10.0_seed${s}"   --dataset "$ds" --seed "$s" --deadline 10.0 & tick
    launch "$g" "p2_tab5df_${ds}_mob0.7_seed${s}"   --dataset "$ds" --seed "$s" --deadline 2.0 --mobility-shape 0.7 & tick
    launch "$g" "p2_tab5df_${ds}_alpha0.1_seed${s}" --dataset "$ds" --seed "$s" --deadline 2.0 --alpha 0.1 & tick
    launch "$g" "p2_tab5df_${ds}_alpha1.0_seed${s}" --dataset "$ds" --seed "$s" --deadline 2.0 --alpha 1.0 & tick
  done
done
wait
echo "[5d-full2] done $(date -u +%H:%M)" >> "$L/tab5d_full2.log"

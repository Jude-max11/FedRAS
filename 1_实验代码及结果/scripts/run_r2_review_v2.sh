#!/usr/bin/env bash
# run_r2_review.sh 的接管版本（并发 5 -> 6）：
#  - 接管时 C 阶段已有 5 个任务在跑（weak5_seed0/1/2, weak5_dl0.1_seed0/1），跳过之
#  - 新增输出目录存在检查，任何已完成的任务不会重跑（原版没有，会重复跑）
#  - D-I 阶段命令与原版逐字一致
set -u
PY=/home/jovyan/FedLoRA_remote_package/.venv/bin/python
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=""   # 强制纯 CPU：防止每个任务偷偷抓 ~1GB 显存挤爆 LoRA 车道
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2   # 限线程：防止32进程×144线程的调度风暴
OUT=outputs/real_fedlora_v4
LOG=logs
BASE="--data data --dataset cifar10 --rounds 40 --clients 20 --per-round 10 --pretrain-epochs 30"
MAXJOBS=24
mkdir -p "$OUT" "$LOG"

launch() {  # 限并发 + 已完成或正在跑则跳过
  local outdir="$1"; shift
  if [ -e "$OUT/$outdir" ] || pgrep -f -- "--out $OUT/$outdir" >/dev/null 2>&1; then
    echo "skip $outdir (已有输出或正在跑)"; return 0
  fi
  while [ "$(jobs -rp | wc -l)" -ge $MAXJOBS ]; do sleep 20; done
  echo "[$(date '+%m-%d %H:%M')] launch $outdir"
  taskset -c 48-127 "$@" > "$LOG/r2_${outdir}.log" 2>&1 &
}

# ---- C 剩余：weak5（跳过在跑的 seed0/1/2 与 dl0.1_seed0/1）----
for s in 3 4 5 6 7 8 9; do
  launch "weak5_seed${s}" $PY scripts/train_torch_fedlora.py --data data --dataset cifar10 --rounds 40 --clients 20 \
    --per-round 10 --pretrain-epochs 5 --deadline 2.0 --seed "$s" \
    --methods FedLoRA_r4,FedLoRA_r16,Adaptive_rank,SA_rank,Oort_utility,JointAB \
    --no-full-baseline --out "$OUT/weak5_seed${s}"
done
for s in 2 3 4 5 6 7 8 9; do
  launch "weak5_dl0.1_seed${s}" $PY scripts/train_torch_fedlora.py --data data --dataset cifar10 --rounds 40 --clients 20 \
    --per-round 10 --pretrain-epochs 5 --deadline 0.1 --seed "$s" \
    --methods FedLoRA_r4,FedLoRA_r16,Adaptive_rank,SA_rank,Oort_utility,JointAB \
    --no-full-baseline --out "$OUT/weak5_dl0.1_seed${s}"
done
echo "C WEAK5 DONE"

# ---- D. binding budget 0.4 MB arms (0.1 s, seeds 0-9) ----
for s in 0 1 2 3 4 5 6 7 8 9; do
  launch "bind04_dl0.1_seed${s}" $PY scripts/train_torch_fedlora.py $BASE --deadline 0.1 --seed "$s" --dp-budget 0.4 \
    --methods DP_cohort,DP_same,Greedy_value,Greedy_vpb,Adaptive_rank,FedLoRA_r16 \
    --no-full-baseline --out "$OUT/bind04_dl0.1_seed${s}"
done
echo "D BIND04 DONE"

# ---- E. new baselines SA_mono + Rate_thresh (both deadlines, seeds 0-9) ----
for s in 0 1 2 3 4 5 6 7 8 9; do
  for spec in "newbase_dl0.1_seed${s}:0.1" "newbase_seed${s}:2.0"; do
    IFS=: read -r out dl <<< "$spec"
    launch "$out" $PY scripts/train_torch_fedlora.py $BASE --deadline "$dl" --seed "$s" \
      --methods SA_mono,Rate_thresh --no-full-baseline --out "$OUT/$out"
  done
done
echo "E NEWBASE DONE"

# ---- F. W_j=0 hold-state ablation (0.1 s, seeds 0-4) ----
for s in 0 1 2 3 4; do
  launch "weakhold_dl0.1_seed${s}" $PY scripts/train_torch_fedlora.py --data data --dataset cifar10 --rounds 40 --clients 20 \
    --per-round 10 --pretrain-epochs 1 --deadline 0.1 --seed "$s" --empty-cols hold \
    --methods FedLoRA_r4,FedLoRA_r16,Adaptive_rank,SA_rank,Oort_utility,JointAB \
    --no-full-baseline --out "$OUT/weakhold_dl0.1_seed${s}"
done
echo "F HOLD DONE"

# ---- G. link-class correlated label skew (both deadlines, seeds 0-4) ----
for s in 0 1 2 3 4; do
  for spec in "corrskew_dl0.1_seed${s}:0.1" "corrskew_seed${s}:2.0"; do
    IFS=: read -r out dl <<< "$spec"
    launch "$out" $PY scripts/train_torch_fedlora.py $BASE --deadline "$dl" --seed "$s" \
      --link-profile sagin --corr-skew \
      --methods FedLoRA_r4,FedLoRA_r16,Adaptive_rank --no-full-baseline --out "$OUT/$out"
  done
done
echo "G CORRSKEW DONE"

# ---- H. Beta(2,1) mobility at nominal deadline ----
for s in 0 1 2 3 4; do
  launch "mob21_seed${s}" $PY scripts/train_torch_fedlora.py $BASE --deadline 2.0 --seed "$s" \
    --mobility-shape "2,1" --methods FedLoRA_r4,FedLoRA_r16,Adaptive_rank \
    --no-full-baseline --out "$OUT/mob21_seed${s}"
done
echo "H MOB21 DONE"

# ---- I. JointAB true A+B payload (both deadlines, seeds 0-9) ----
for s in 0 1 2 3 4 5 6 7 8 9; do
  for spec in "jointab_ab_dl0.1_seed${s}:0.1" "jointab_ab_seed${s}:2.0"; do
    IFS=: read -r out dl <<< "$spec"
    launch "$out" $PY scripts/train_torch_fedlora.py $BASE --deadline "$dl" --seed "$s" \
      --jointab-ab --methods JointAB --no-full-baseline --out "$OUT/$out"
  done
done
echo "I JOINTAB DONE"
wait
echo "[$(date '+%m-%d %H:%M')] R2 BATCH1 ALL DONE (v2)"

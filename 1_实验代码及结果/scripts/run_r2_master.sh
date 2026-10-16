#!/usr/bin/env bash
# R2 master scheduler: robust concurrency gate (trainer-process count AND GPU
# free memory), plan-driven priority queue, three verify-repair passes, and a
# lightweight monitor. Replaces the failed chained batch scripts.
set -u
PY=/home/jovyan/FedLoRA_remote_package/.venv/bin/python
cd "$(dirname "$0")/.."
LOG=logs
MAXN=6          # concurrent trainers (A800 shared with another tenant)
MINFREE_MB=8000 # never launch below this much free GPU memory

gate() {
  while :; do
    n=$(pgrep -fc 'train_torch_fedlora.py' || true)
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    # adaptive cap: exclusive machine (>60GB free) -> 22; shared (40-60GB) -> 9; else 6
    cap=6; [ "${free:-0}" -gt 40000 ] && cap=9; [ "${free:-0}" -gt 60000 ] && cap=22
    if [ "${n:-0}" -lt "$cap" ] && [ "${free:-99999}" -gt "$MINFREE_MB" ]; then
      break
    fi
    sleep 10
  done
}

run_plan() {
  $PY scripts/plan_r2.py || return 1
  local i=0
  while IFS= read -r cmd; do
    [ -z "$cmd" ] && continue
    gate
    i=$((i+1))
    echo "[$(date +%H:%M:%S)] launch #$i: $cmd"
    nohup env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 bash -c "$cmd" > "$LOG/r2_master_${i}.log" 2>&1 &
  done < "$LOG/r2_plan.txt"
  wait
}

# monitor: proc count + gpu + completed-run count every 10 min
(
  while :; do
    n=$(pgrep -fc 'train_torch_fedlora.py' || true)
    done_n=$($PY - << 'EOF' 2>/dev/null
import csv, glob, json
n = 0
for d in glob.glob('outputs/real_fedlora_v4/*_seed*'):
    f = d + '/metrics.csv'
    try:
        cfg = json.load(open(d + '/config.json'))
        rows = list(csv.DictReader(open(f)))
        ms = [m for m in cfg['methods'].split(',') if m] if cfg.get('methods') else \
            ['FedLoRA_r4', 'FedLoRA_r16', 'Random_rank', 'Adaptive_rank', 'FedAvg_full']
        if {(r['method'], int(r['round'])) for r in rows} >= {(m, rd) for m in ms for rd in range(cfg['rounds'])}:
            n += 1
    except Exception:
        pass
print(n)
EOF
)
    echo "$(date '+%m-%d %H:%M') procs=${n:-0} gpu_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)MB complete=${done_n:-?}"
    sleep 600
  done
) >> "$LOG/r2_monitor.log" &

for pass in 1 2 3; do
  echo "=== MASTER PASS $pass ==="
  run_plan
  echo "=== PASS $pass DONE, verifying ==="
  if $PY scripts/audit_r2.py; then
    echo "R2 MASTER: ALL COMPLETE"
    break
  fi
done
$PY scripts/audit_r2.py && echo "R2 MASTER FINAL: ALL COMPLETE" || echo "R2 MASTER FINAL: gaps remain, see logs/r2_repairs.txt"
echo "vit_bcache: $(ls outputs/real_fedlora_v4/vit_bcache/*.pt 2>/dev/null | wc -l)/20"

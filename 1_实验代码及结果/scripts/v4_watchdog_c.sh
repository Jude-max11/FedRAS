#!/usr/bin/env bash
# Overnight watchdog v2: manifest-based completion check.
# Whenever no experiment process is alive, resume interrupted v4 matrix runs
# (checkpoint-based) until EVERY expected phase-A and phase-B run has reached
# round 39; then assemble all_metrics and guarantee calibration finished.
# Logs to logs/v4_watchdog.log. Idempotent: finished runs are skipped.
set -u
cd "$(dirname "$0")/.."
LOG=logs/v4_watchdog.log
PY=/opt/conda/bin/python
echo "[watchdog] start $(date)" >> "$LOG"

count_incomplete () {
  $PY - <<'EOF'
import os, csv
OUT = 'outputs/real_fedlora_v4'
phases = {
    'C': [(n, s) for n in ('cifar10', 'cifar10_dl0.1', 'cifar10_ext', 'cifar10_ext_dl0.1', 'dp', 'dpsame') for s in range(5, 10)]
         + [(n, s) for n in ('cifar100', 'cifar100_dl0.1', 'mnist', 'mnist_dl0.1') for s in range(5)]
         + [(n, s) for n in ('cifar10_mob0.7', 'cifar10_alpha0.1', 'cifar10_alpha1.0', 'cifar10_rmax8', 'cifar10_rmax32', 'cifar10_pr5', 'cifar10_pr20', 'cifar10_sagin') for s in range(5)],
    'A': [(n, s) for n in ('cifar10', 'cifar10_dl0.1', 'cifar10_dl0.05', 'cifar10_dl10.0') for s in range(5)],
    'B': [(n, s) for n in ('cifar10_ext', 'cifar10_ext_dl0.1', 'dp', 'dpsame',
                           'cifar10_rrmask_dl0.1', 'cifar10_rrnomask_dl0.1', 'fedavg_offline') for s in range(5)],
}
missing = []
for name, seed in phases['C']:
    f = os.path.join(OUT, f'{name}_seed{seed}', 'metrics.csv')
    try:
        rows = list(csv.reader(open(f)))
        if len(rows) > 1 and rows[-1] and rows[-1][1] == '39':
            continue
    except Exception:
        pass
    missing.append(f'{name}_seed{seed}')
print(len(missing))
if missing:
    print('MISSING: ' + ','.join(missing))
EOF
}

for i in $(seq 1 60); do
  # Heartbeat detection: any v4 metrics/checkpoint touched in the last
  # 10 minutes means experiments are running. (pgrep -f is unreliable here:
  # harness task wrappers embed the command text and match forever.)
  recent=$(find outputs/real_fedlora_v4 -name 'metrics.csv' -mmin -10 2>/dev/null | head -1)
  if [ -n "$recent" ]; then
    echo "[watchdog] cycle $i $(date +%H:%M): experiments running ($recent)" >> "$LOG"
  else
    read -r inc missing_list <<< "$(count_incomplete | tr '\n' ' ')"
    echo "[watchdog] cycle $i $(date +%H:%M): quiescent, $inc incomplete" >> "$LOG"
    if [ "$inc" -eq 0 ]; then
      $PY scripts/assemble_v4.py >> "$LOG" 2>&1
      if [ outputs/journal_stats/calibration.csv -ot scripts/journal_calibration.py ]; then
        echo "[watchdog] rerunning calibration" >> "$LOG"
        (
          flock -n 9 || { echo "[watchdog] calibration already running" >> "$LOG"; exit 0; }
          $PY scripts/journal_calibration.py > logs/v4_calibration.log 2>&1
        ) 9>outputs/journal_stats/.calib_lock
      fi
      echo "[watchdog] ALL PHASE-C RUNS COMPLETE $(date)" >> "$LOG"
      break
    fi
    echo "[watchdog] resuming phase C" >> "$LOG"
    bash scripts/run_v4_matrix_resume.sh C >> "$LOG" 2>&1
  fi
  sleep 900
done
echo "[watchdog] exit $(date)" >> "$LOG"

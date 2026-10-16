#!/usr/bin/env bash
# Final audit-and-repair: waits for the whole chained pipeline (batch1 ->
# 2a -> 2b -> 2c), then repairs any incomplete run at low concurrency (3),
# up to three passes. Also reports missing ViT backbone caches (those
# self-heal inline during repair runs).
set -u
cd "$(dirname "$0")/.."
LOG=logs

while ! grep -q "R2 BATCH2C ALL DONE" "$LOG/r2_batch2c_main.log" 2>/dev/null; do sleep 300; done
echo "pipeline drained; auditing"

for pass in 1 2 3; do
  if /home/jovyan/FedLoRA_remote_package/.venv/bin/python scripts/audit_r2.py; then
    echo "AUDIT PASS $pass: ALL COMPLETE"; break
  fi
  echo "AUDIT PASS $pass: repairing $(wc -l < $LOG/r2_repairs.txt) runs at concurrency 3"
  while IFS= read -r cmd; do
    [ -z "$cmd" ] && continue
    while [ "$(jobs -rp | wc -l)" -ge 3 ]; do sleep 30; done
    echo "repair: $cmd"
    bash -c "$cmd" > "$LOG/r2_repair_$(date +%s)_$$.log" 2>&1 &
  done < "$LOG/r2_repairs.txt"
  wait
done

/home/jovyan/FedLoRA_remote_package/.venv/bin/python scripts/audit_r2.py && echo "R2 AUDIT FINAL: ALL COMPLETE" || echo "R2 AUDIT FINAL: INCOMPLETE REMAINS - see logs/r2_repairs.txt"
echo "vit_bcache entries: $(ls outputs/real_fedlora_v4/vit_bcache/*.pt 2>/dev/null | wc -l) (expect 20; missing ones self-heal inline)"

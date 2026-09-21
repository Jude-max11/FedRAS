#!/usr/bin/env bash
# Build the reproducibility package for the journal submission (review point 9).
# Mirrors the repository layout (scripts/ + outputs/ + logs/) so that every
# command in README.md runs verbatim from the extracted package.
# ISOLATION: includes the journal experiment tree only; WCNC_conference and
# .pt checkpoints / datasets are excluded. Training data (CIFAR/MNIST) are
# downloaded by torchvision, not shipped.
set -u
ROOT=/home/jovyan/FedLoRA_remote_package
STAGE="$ROOT/artifact_staging"
PKG="$ROOT/fedras_code_package.zip"
cd "$ROOT"
rm -rf "$STAGE" "$PKG"
mkdir -p "$STAGE"/{scripts,outputs,logs}

# ---- code (repo layout: scripts/ + fedlora package) ----
cp scripts/train_torch_fedlora.py \
   scripts/run_real_v3.sh scripts/wcnc_run_seeds59.sh scripts/journal_dp_runs.sh \
   scripts/summarize_real_fedlora.py \
   scripts/run_sweep.py scripts/run_fedlora.py scripts/dp_scheduler_study.py \
   scripts/make_paper_figs.py scripts/make_more_figs.py scripts/make_sensitivity_fig.py \
   scripts/journal_stats.py scripts/journal_dp_compare.py scripts/journal_baseline_stats.py \
   scripts/journal_calibration.py scripts/journal_model_audit.py \
   scripts/wcnc_grid_sensitivity.py scripts/wcnc_scheduler_robustness.py \
   scripts/stat_tests.py "$STAGE/scripts/"
mkdir -p "$STAGE/scripts/fedlora"
cp fedlora/__init__.py fedlora/run_experiment.py "$STAGE/scripts/fedlora/"
cp requirements.txt "$STAGE/" 2>/dev/null || true

# ---- results (repo layout: outputs/...; metrics + configs only, no checkpoints) ----
O="$STAGE/outputs"
mkdir -p "$O"/{real_fedlora_v3,journal_dp,journal_ext,journal_stats,wcnc_extra,paper_sweep}
cp outputs/real_fedlora_v3/all_metrics.csv outputs/real_fedlora_v3/config.json "$O/real_fedlora_v3/" 2>/dev/null || true
for d in outputs/real_fedlora_v3/*/; do
  n=$(basename "$d"); mkdir -p "$O/real_fedlora_v3/$n"
  cp "$d"config.json "$O/real_fedlora_v3/$n/" 2>/dev/null || true
done
cp -r outputs/real_fedlora_v3/figures "$O/real_fedlora_v3/figures"
mkdir -p "$O/wcnc_extra/real_v3_ext"
cp outputs/wcnc_extra/real_v3_ext/all_metrics.csv "$O/wcnc_extra/real_v3_ext/"
cp outputs/wcnc_extra/grid_sensitivity.csv outputs/wcnc_extra/scheduler_robustness.csv "$O/wcnc_extra/"
for d in outputs/journal_dp/dp_seed*/; do
  n=$(basename "$d"); mkdir -p "$O/journal_dp/$n"
  cp "$d"metrics.csv "$d"config.json "$O/journal_dp/$n/" 2>/dev/null || true
done
for d in outputs/journal_ext/*/; do
  n=$(basename "$d"); mkdir -p "$O/journal_ext/$n"
  cp "$d"metrics.csv "$d"config.json "$O/journal_ext/$n/" 2>/dev/null || true
done
cp outputs/journal_stats/*.csv "$O/journal_stats/"
cp outputs/paper_sweep/all_round_metrics.csv "$O/paper_sweep/" 2>/dev/null || true
cp outputs/paper_sweep/sweep_summary.csv "$O/paper_sweep/" 2>/dev/null || true
cp outputs/dp_scheduler_study.csv "$O/" 2>/dev/null || true

# ---- raw logs ----
cp logs/v3_*.log logs/wcnc_ext_*.log logs/journal_*.log "$STAGE/logs/" 2>/dev/null || true

# ---- README ----
cp "$ROOT/FedLoRA lunwen/ARTIFACT_README.md" "$STAGE/README.md" || { echo "MISSING ARTIFACT_README.md"; exit 1; }

# ---- zip ----
( cd "$STAGE" && zip -qr "$PKG" . )
du -sh "$PKG"
echo "package: $PKG"

# Reproducibility Package — FedRAS: Deadline-Aware Rank Scheduling for Federated LoRA in Mobile Edge Networks

This package contains the code, per-run configurations, result tables, and raw
logs supporting the numerical claims of the manuscript. It covers the journal
(journal-version) experiments only; the IEEE WCNC conference version is
maintained and packaged separately.

> **命名说明**（2026-09-14）：论文中的机制名 FedRAS（federated rank-adaptive scheduling）
> 对应数据文件与方法键 `Adaptive_rank`（CSV 列名与脚本参数保持原键，未改动数据）。

## 1. Layout

```
scripts/    experiment, analysis, and figure scripts + fedlora simulator package
outputs/    experiment outputs: per-run metrics.csv + config.json, aggregated
            CSVs, and generated figures — no checkpoints, no training data
logs/       raw stdout logs of every training run
README.md   this file
requirements.txt  Python dependencies
```

## 2. Environment

- Python 3.12+, PyTorch 2.8.0+cu128, pandas, scipy, matplotlib, numpy
- One NVIDIA A800 80GB GPU (bf16 autocast); runs also work on CPU at much
  higher cost
- Training data: CIFAR-10 / CIFAR-100 / MNIST via `torchvision.datasets`
  (downloaded automatically into `--data data`; not shipped)
- Every run writes `config.json` (all hyperparameters) and `metrics.csv`
  (per-round: accuracy, timely_ratio, bytes_norm, mean_rank, selected,
  aggregated, update_norm)

## 3. What was run (experiment matrix)

| Run set | Script | Seeds | Output |
|---|---|---|---|
| Main trainer matrix (23 configurations × 5 methods) | `scripts/run_real_v3.sh` → `train_torch_fedlora.py` | 0–4 | `outputs/real_fedlora_v3/all_metrics.csv` |
| Ten-seed replication of the headline regimes (cifar10 nominal + 0.25 s) | `scripts/wcnc_run_seeds59.sh` | 5–9 | `outputs/wcnc_extra/real_v3_ext_all_metrics.csv` |
| Full-DP scheduler arm (DP_cohort, 8.2 MB budget) | `scripts/journal_dp_runs.sh` | 0–9 | `outputs/journal_dp/dp_seed*/metrics.csv` |
| External baselines to ten seeds (SA_rank, Oort_utility, JointAB) | `scripts/journal_dp_runs.sh` | 5–9 | `outputs/journal_ext/*/metrics.csv` |
| 500-client scheduling simulation, nine regimes | `scripts/run_sweep.py` → `fedlora/run_experiment.py` | 0–4 | `outputs/paper_sweep/all_round_metrics.csv` |
| Solver study (exact DP vs greedy, 30 instances/cell) | `scripts/dp_scheduler_study.py` | — | `outputs/dp_scheduler_study.csv` |
| Budget-grid sensitivity sweep | `scripts/wcnc_grid_sensitivity.py` | — | `outputs/wcnc_extra/grid_sensitivity.csv` |
| Predictor / utility-form robustness | `scripts/wcnc_scheduler_robustness.py` | — | `outputs/wcnc_extra/scheduler_robustness.csv` |

Common random numbers: link traces (rate ~ LogNormal(2.5, 0.7) clipped to
[1, 40] Mbps; mobility ~ Beta(2, 5)) are seeded per (seed, round) and shared
across methods; backbone weights are cached per seed and reused for the
deadline-stress and DP arms (copied `backbone_*` files, fingerprint-checked).

## 4. Reproducing each table / figure

| Paper element | Source data | Command |
|---|---|---|
| Trainer tables (accuracy parity, deadline stress, extended matrix, ablations) | `outputs/real_fedlora_v3/all_metrics.csv` (+ wcnc_extra replication) | `python scripts/summarize_real_fedlora.py --root outputs/real_fedlora_v3` |
| External-baseline table | `cifar10_ext`, `cifar10_ext_dl0.25` rows of `all_metrics.csv` (seeds 0–4) + `outputs/journal_ext/` (seeds 5–9) | `python scripts/journal_baseline_stats.py` |
| Full-DP vs fair-cohort table | `outputs/journal_dp/` + `all_metrics.csv` + `real_v3_ext` | `python scripts/journal_dp_compare.py` |
| Ten-seed paired statistics (Wilcoxon, bootstrap CIs, rank-biserial) | same as above | `python scripts/journal_stats.py` |
| Solver table (optimality gaps, runtime) | `outputs/dp_scheduler_study.csv` | `python scripts/dp_scheduler_study.py` (regenerates) |
| Simulator tables/figures | `outputs/paper_sweep/all_round_metrics.csv` | `python scripts/run_sweep.py` (regenerates) |
| Paper figures (trainer/simulator/DP/sensitivity) | result CSVs | `python scripts/make_paper_figs.py`, `make_more_figs.py`, `make_sensitivity_fig.py` (write PNGs next to the manuscript sources) |
| Budget-grid discretization error | `outputs/wcnc_extra/grid_sensitivity.csv` | `python scripts/wcnc_grid_sensitivity.py` |
| Robustness to prediction noise / utility form | `outputs/wcnc_extra/scheduler_robustness.csv` | `python scripts/wcnc_scheduler_robustness.py` |
| Calibration study (reliability curves, ECE, mis-calibration cost) | `outputs/journal_stats/calibration.csv` | `python scripts/journal_calibration.py` |
| Model / payload audit | architecture in `scripts/train_torch_fedlora.py` | `python scripts/journal_model_audit.py` |
| Five-seed paired tests (original table statistics) | `all_metrics.csv` | `python scripts/stat_tests.py` |

Analysis outputs land in `outputs/journal_stats/` (shipped) and are regenerated
by the commands above.

## 5. Rerunning training

Example — one seed of the full main configuration (nominal deadline):

```
python scripts/train_torch_fedlora.py --data data --dataset cifar10 \
  --out outputs/real_fedlora_v3/cifar10_seed0 \
  --rounds 40 --clients 20 --pretrain-epochs 30 --seed 0
```

Full-DP arm:

```
python scripts/train_torch_fedlora.py --data data --dataset cifar10 \
  --out outputs/journal_dp/dp_seed0 \
  --rounds 40 --clients 20 --pretrain-epochs 30 --seed 0 \
  --methods DP_cohort --no-full-baseline --dp-budget 8.2
```

Approximate cost per seed on one A800: 15–25 min for the five-method main set,
5–10 min for the DP arm, 20–30 min for the three-baseline set; central
backbone pretraining (30 epochs, ~10 min) is cached per seed and reused across
deadline settings. The scripts in §3 run the complete matrices with
checkpoint-aware skip logic, so re-invoking them only fills gaps.

## 6. Provenance notes

- All tables in the manuscript are transcriptions of the CSV outputs above;
  the five-seed table entries were cross-checked against the ten-seed merged
  data (see `outputs/journal_stats/baseline_10seed*`).
- `outputs/real_fedlora_v3/figures/` contains the per-configuration figures
  generated directly from the runs; manuscript figures are assembled from
  these and the sweep CSVs by the figure scripts.
- The package intentionally excludes model checkpoints (`*.pt`), datasets,
  and the conference-version tree; every figure can be regenerated from the
  shipped CSVs without GPU access.

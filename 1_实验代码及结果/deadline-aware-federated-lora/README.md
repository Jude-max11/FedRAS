# Deadline-Aware Adaptive-Rank Federated LoRA over Mobile Edge Networks

Code for the paper *"Deadline-Aware Adaptive-Rank Federated LoRA over Mobile
Edge Networks"* (IEEE WCNC 2026).

The paper treats the LoRA rank as a network-visible scheduling variable. Each
communication round is formulated as a joint client–rank assignment that
maximizes expected useful updates under a byte budget and a participation cap,
solved exactly on its budget grid by a pseudo-polynomial dynamic program.
Heterogeneous client ranks are aggregated by rank-masked averaging over a
frozen factor.

## Contents

| Path | Description |
|---|---|
| `fedlora/run_experiment.py` | 500-client trace-driven scheduler simulation (link traces, per-round client–rank assignment, budget admission, saturating accuracy proxy) |
| `scripts/train_torch_fedlora.py` | Real PyTorch federated LoRA trainer: frozen CNN backbone, rank-masked adapters, Dirichlet non-IID clients; arms = fixed rank (4/16), random rank, fair-cohort adaptive rule, DP-joint (exact joint client–rank DP), FedAvg full-update baseline |
| `scripts/dp_scheduler_study.py` | Solver study: exact DP vs greedy heuristics, optimality gaps and runtime over deadlines × budgets, instance scalability |
| `scripts/wcnc_scheduler_robustness.py` | Robustness of the schedule to noisy rate estimates and to the gain-shape choice |
| `scripts/wcnc_grid_sensitivity.py` | Sensitivity of DP objective/runtime to the budget-grid step |
| `scripts/run_real_v3.sh`, `run_seeds59.sh`, `run_dp_trainer.sh`, `run_sweep.py`, `collect_sweep.py` | Experiment runners (real-trainer matrix, 10-seed replication, DP-joint runs, 500-client sweep) |
| `scripts/make_wcnc_figs.py`, `make_mech_fig.py` | Figure generation (evaluation figures and the rank-trade-off concept figure) |
| `data/` | Aggregated per-round metrics from the paper's runs (see below) |
| `figures/` | Figures generated from `data/` by the figure scripts |

## Requirements

```
torch, torchvision, numpy, pandas, matplotlib
```
(CPU suffices for the simulator/solver study; the real trainer expects a CUDA GPU.)

## Data

`data/` contains the aggregated per-round metrics behind the paper's tables and
figures (no raw checkpoints):

- `real_fedlora_v3__all_metrics.csv` — real trainer, CIFAR-10/100 + MNIST,
  five seeds, plus deadline/mobility/heterogeneity stress variants
- `wcnc_seeds59__all_metrics.csv` — 10-seed replication (seeds 5–9)
- `wcnc_dp__all_metrics.csv` — DP-joint arm runs
- `paper_sweep__all_round_metrics.csv` — 500-client simulation, nine regimes
- `dp_scheduler_study.csv` — solver-study results (30 instances per cell)

## Reproducing

```bash
# solver study (CPU, ~minutes); writes outputs/dp_scheduler_study.csv
python scripts/dp_scheduler_study.py

# robustness + grid-step sensitivity
python scripts/wcnc_scheduler_robustness.py
python scripts/wcnc_grid_sensitivity.py

# 500-client simulation, one regime
python scripts/run_fedlora.py --out outputs/paper_sweep/main_cifar10 \
    --clients 500 --rounds 150 --budget 300 --cap 25 --deadline 1.0
# or the full nine-regime matrix, then aggregate
python scripts/run_sweep.py --root outputs/paper_sweep
python scripts/collect_sweep.py --root outputs/paper_sweep

# real trainer, single run (torchvision downloads the dataset into data/)
python scripts/train_torch_fedlora.py --data data --dataset cifar10 \
    --out outputs/real_fedlora_v3/cifar10_seed0 \
    --rounds 40 --clients 20 --pretrain-epochs 30 --seed 0

# full real-trainer matrix (three datasets x five seeds + stress runs, GPU)
bash scripts/run_real_v3.sh

# DP-joint arm and 10-seed replication (after the nominal runs)
bash scripts/run_dp_trainer.sh
bash scripts/run_seeds59.sh

# figures -> figures/
python scripts/make_wcnc_figs.py
python scripts/make_mech_fig.py
```

A real-trainer run trains a small CNN centrally (the "pretraining" phase) and
then runs 40 federated rounds; on a single A800 one run takes on the order of
tens of minutes depending on the dataset.

## Citation

A citation block will be added upon acceptance.

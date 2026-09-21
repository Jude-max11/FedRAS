# Deadline-aware Adaptive-rank FedLoRA remote package (v3)

This package is self-contained and CPU-runnable. It implements the experiment used to populate the manuscript's FedLoRA comparison table. No dataset download or GPU is required for the network simulator; the real trainer uses torchvision datasets already present under `data/` (CIFAR-10/CIFAR-100 at `data/`, MNIST at `data/mnist`).

## v3 changes (widened backbone, long pretraining)

- Backbone widened to 64/128/256-channel conv blocks (`--base-channels 64`), pretraining extended to 30 epochs (CIFAR-10) / 45 (CIFAR-100) / 3 (MNIST), RandomErasing added to the CIFAR augmentation stack, and bf16 autocast used for pretraining, local training and evaluation. Backbone test accuracy: **CIFAR-10 0.914, CIFAR-100 0.692**. Use `scripts/run_real_v3.sh` for the final matrix; v2 results (6-epoch pretraining, 32/64/128 backbone) remain under `outputs/real_fedlora_v2` for comparison.

## v2 changes (validated on an A800 GPU)

- **Real trainer (`scripts/train_torch_fedlora.py`)**: BatchNorm CNN backbone (augmented pretraining with cosine LR, Nesterov momentum, weight decay) replaces the raw-pixel MLP; LoRA adapters now sit on all six conv layers plus two wide head layers, so the rank r actually bounds update capacity. Following FFA-LoRA, the A factor is frozen at its shared random init and only B is trained — averaging B across clients then equals averaging the update, which removed the instability of jointly trained A/B factors. `Adaptive_rank` maximizes expected useful delivered capacity `sqrt(min(r,16)/16) * P_deliver(r)` with the deadline-aware `P_deliver(r) = exp(-mob) * exp(-8*size(r)/(rate*deadline))` (MB→Mbit factor fixed); the old score was monotone in r and always chose rank 2. Backbone caching is fingerprinted by config. New `update_norm` column approximates the update-exposure surface (gradient-inversion literature).
- **Simulator (`fedlora/run_experiment.py`)**: same capacity-aware objective; standard two-phase protocol (server samples the cohort uniformly, then admits under the byte budget by value-per-byte — the old top-utility selection hid all rank adaptivity); saturating bounded accuracy recursion; saturating capacity `sqrt(min(r,16)/16)` so FedAvg's rank-64 payload buys no extra capacity. `run_sweep.py` budgets are now binding (12/40/60 instead of 150/600/500).
- `scripts/run_real_v2.sh`/`run_real_v3.sh` run the final matrix (3 datasets x 5 seeds, 40 rounds + deadline-stress runs) and `scripts/summarize_real_fedlora.py` builds `summary.csv`, figures and the v1-vs-v2 comparison table.

## Run

```bash
python -m pip install -r requirements.txt
python scripts/run_fedlora.py --out outputs/fedlora --seeds 0,1,2,3,4 --rounds 150 --clients 500 --budget 300 --cap 25 --deadline 1.0
python scripts/summarize_fedlora.py --root outputs/fedlora
```

The primary output is `summary.csv`; per-seed traces are `round_metrics_seed*.csv`. Preserve the complete output directory when returning results.

## Paper-scale sweep (recommended)

```bash
python scripts/run_sweep.py --root outputs/paper_sweep --rounds 150 --seeds 0,1,2,3,4
python scripts/collect_sweep.py --root outputs/paper_sweep
python scripts/plot_sweep.py --root outputs/paper_sweep
```

This runs main CIFAR-10, a CIFAR-100 proxy configuration, a 1000-client scalability configuration, and deadline/budget stress scenarios. The simulator keeps the same deterministic client/link generation protocol across methods and seeds. The output contains `all_round_metrics.csv`, `sweep_summary.csv`, and a `figures/` directory. The CIFAR-100 and MNIST names denote experimental configurations in this lightweight package; for final manuscript claims they should be replaced or cross-validated with actual dataset loaders if real training is required.

Important validity note: this package is a controlled network-scheduler simulator, not a PyTorch LoRA trainer. Use it to validate the client--rank scheduling hypothesis and network metrics. Do not present the proxy accuracy numbers as results on CIFAR-10/CIFAR-100/MNIST until a real dataset loader and LoRA model have been run. All methods now share common random-number link traces within each seed; the mobility stress uses a separate mobility distribution.

## Real PyTorch FedLoRA training (v2)

The package includes `scripts/train_torch_fedlora.py`, a genuine federated LoRA trainer using torchvision datasets and weighted heterogeneous-rank aggregation. It requires a local dataset directory containing the already extracted torchvision files; downloads are disabled for reproducibility. Example:

```bash
python -m pip install torch torchvision pandas numpy matplotlib
# final matrix (3 datasets x 5 seeds; see scripts/run_real_v2.sh)
bash scripts/run_real_v2.sh
# single run
python scripts/train_torch_fedlora.py --data data --dataset cifar10 --out outputs/real_fedlora_v2/cifar10_seed0 --rounds 40 --clients 20 --seed 0
python scripts/train_torch_fedlora.py --data data/mnist --dataset mnist --out outputs/real_fedlora_v2/mnist_seed0 --rounds 40 --pretrain-epochs 3 --seed 0
# deadline stress (shows adaptive downshifting)
python scripts/train_torch_fedlora.py --data data --dataset cifar10 --out outputs/real_fedlora_v2/cifar10_dl05_seed0 --rounds 40 --deadline 0.5 --seed 0
python scripts/summarize_real_fedlora.py --root outputs/real_fedlora_v2
```

The trainer performs a deterministic centralized backbone pretraining stage (base weights only; adapters stay at zero delta), freezes the backbone, and then trains LoRA adapters federatively with frozen-A (FFA-LoRA style) local updates, masked rank-weighted aggregation in the shared rmax=16 coordinate system and server momentum. Client models always use `rmax=16`; lower ranks are masked before weighted aggregation. Use `--resume` to continue from `checkpoint.pt`. Methods: `FedLoRA_r4`, `FedLoRA_r16`, `Random_rank`, `Adaptive_rank` and the `FedAvg_full` whole-model baseline at rank-64 payload cost.

## Reproducibility

All random states are seeded. The methods use identical client states and random traces within each seed. `Adaptive_rank` chooses among ranks {2,4,8,16} using the expected timely useful-update objective. `FedAvg` is represented as a full-parameter 64-rank update for communication comparison. This is a scheduler experiment, not a claim of formal privacy.

## Stress sweeps

Repeat the command with `--deadline 0.5`, `--deadline 2.0`, different `--budget`, `--clients`, and `--cap`. Keep one output directory per setting.

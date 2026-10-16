# FedRAS: Network-Conditioned Rank Allocation for Federated LoRA

Code, experiment results, and LaTeX source for the paper
**"FedRAS: Network-Conditioned Rank Allocation for Federated LoRA"** (Computer Networks submission).

FedRAS formulates LoRA rank selection and client admission as a joint network
scheduling problem: it estimates a delivery probability and an adaptation
capacity for each client–rank pair, maximizes expected useful updates under
byte-budget and participation constraints, solves the assignment with a
dynamic program, and aggregates heterogeneous ranks with frozen-factor rank
masking. Under a 0.1 s stress deadline it delivers 22% more timely updates and
23% fewer bytes than fixed rank 16, while the dynamic program improves on
greedy scheduling by 28–55% when the budget binds.

## Repository layout

| Directory | Content |
|---|---|
| `1_实验代码及结果/` | Simulation and experiment code, configs, result CSVs/figures (see its own README) |
| `2_latex及附属文件历史版本/` | Historical versions of the manuscript and related files |
| `3_ComputerNetworks期刊latex版本/` | Current Computer Networks submission package (LaTeX + figures) |
| `release_staging/` | Anonymized release repos staged for review: `Deadline-Aware-FedLoRA` (IEEE WCNC 2026) and `FedRAS` (CN journal), each a standalone git repo with its own GitHub remote (moved here 2026-10-06) |

## Notes

- Datasets (MNIST / CIFAR-10 / CIFAR-100, ~750 MB) are **not** included; place
  them under `1_实验代码及结果/data/` as described in that directory's README.
- Model weight checkpoints (`*.pt` / `*.pth`) are excluded from version control
  by design; result files (JSON / CSV / PNG) are tracked.

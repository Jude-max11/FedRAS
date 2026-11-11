# Data description

`outputs/real_v6/` holds one directory per run, named `<family>_seed<k>`.

Each directory contains:

| file | content |
|---|---|
| `config.json` | the exact arguments of the run, plus the measured payload and timing constants |
| `metrics.csv` | per-round metrics: accuracy, timely-delivery ratio, bytes per round, delivered rank, delivered capacity, number of scheduled and aggregated clients |

Model checkpoints are not stored. Every table and figure in the paper is
regenerated from these two files.

## Run families

| family | runs | what it is | paper item |
|---|---|---|---|
| `p1_tab3_{cifar10,cifar100,mnist}_dl{2.0,0.1,0.05}` | 90 | main results, three datasets, three deadlines | Table 3 |
| `p1_tab4_vit_0.5` | 10 | shifted small ViT backbone | Table 4, Figure 3 |
| `p1_tab4_weakdisj_{0.1,2.0}` | 20 | disjoint weakly pretrained backbone | Table 4, Figure 3 |
| `p2_tab5c_sagin_dl{2.0,1.0,0.1,0.05}` | 40 | grouped ground, aerial and satellite links | Table 7 |
| `p2_tab5d_*` | 120 | stress matrix: dataset x {Dirichlet 0.1, Dirichlet 1.0, 10 s deadline, mobility 0.7} | Table 8 |
| `p2_tab5df_*` | 120 | same matrix with the frozen-`A` flag set | Table 8 |
| `p3_fig4_k{5,20}`, `p3_fig4_rmax{8,32}` | 40 | participation cap and largest-rank sweeps | Figure 4 |
| `p3_fig5_weak{1,5}` | 20 | weakly pretrained backbones | Figure 5 |
| `p3_tab6a_*` | 60 | component ablation: joint A/B, learning rate, no capacity term, no momentum, no rank mask | Table 9(a) |
| `p3_tab6b_*` | 60 | assignment rules: DP over cohort, DP over sampled clients, per-client rule | Table 9(b) |
| `p3_tab6c_*` | 50 | assignment rules at a 0.4 MB budget | Table 9(c) |
| `p5_fedavg_dl*`, `p5_fedavg6_dl*` | 60 | FedAvg baselines at three deadlines | Table 3 |
| `p6_*` | 120 | sensitivity sweep, strongly pretrained backbone (30 epochs, accuracy 0.92): rate bias +-0.15, +-0.3 and hazard scale 0.5, 2.0 at 2.0 s and 0.1 s | Figure 2 |
| `p6b_*` | 120 | the same sweep on the weakly pretrained backbone (6 epochs, accuracy 0.84) | Figure 2 |

Totals: 93 families, 930 runs, ten paired seeds per family.

Auxiliary entries in the same directory:

* `bb_cache/` — cached pretrained backbones (metadata only).
* `sim/` — scheduler-simulator output produced by `scripts/simulator.py`.
* `solver_study.csv` — output of `scripts/dp_scheduler_study.py`.

## Aggregated files

| file | content |
|---|---|
| `outputs/all_real_metrics.csv` | pooled per-round metrics across runs |
| `outputs/journal_stats/*.csv` | aggregated tables: calibration, delivered capacity, capacity exponent sweep, baseline statistics, solver study |

## Column meanings in `metrics.csv`

| column | meaning |
|---|---|
| `round` | training round |
| `accuracy` | test accuracy of the evaluated model at that round |
| `timely_ratio` | fraction of scheduled updates that arrived within the deadline |
| `bytes_mb` | bytes uploaded in the round, from the real serialized payload |
| `mean_rank` | size-weighted mean rank of the updates that actually arrived; empty when none arrived |
| `selected` | number of clients scheduled in the round |
| `aggregated` | number of updates that arrived in time |
| `delivered_capacity` | `aggregated` times the delivered rank term, in normalized units |

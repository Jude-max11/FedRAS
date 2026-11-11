# FedRAS: Delivery-Probability-Aware Rank-Adaptive Scheduling for Federated LoRA

Code and per-run results for the manuscript *"FedRAS: Delivery-Probability-Aware
Rank-Adaptive Scheduling for Federated LoRA in Mobile Edge Networks"* (Computer
Networks submission).

Every reported number comes from these scripts and the runs stored in
`outputs/real_v6/`. Model checkpoints are not included; every table and figure in
the paper is regenerated from the per-round metrics that are included.

## Repository layout

```
configs/                     run configurations (YAML)
scripts/                     experiment, analysis and figure code
  train_torch_fedlora.py     federated LoRA trainer (GPU, real serialized bytes)
  simulator.py               per-round scheduler simulator (NumPy only)
  collect_v6.py              aggregate outputs/real_v6 into the paper's tables
  emit_tab3_zh.py            Table 3 (main results, three datasets x three deadlines)
  dp_scheduler_study.py      exact DP vs greedy baselines (Table 6)
  journal_model_audit.py     payload accounting audit (1.64 MB -> 0.90 MB)
  make_v6_figs.py            Figures 2 and 3
  make_v6_fig4.py            Figure 4
  make_fig5a.py              Figure 5(a)
  make_fig5b.py              Figure 5(b)
  stat_tests.py              paired tests across seeds
  preflight.py               environment check
outputs/
  real_v6/<run>/config.json  exact configuration of the run
  real_v6/<run>/metrics.csv  per-round metrics of the run
  all_real_metrics.csv       pooled per-round metrics
  journal_stats/             aggregated tables produced by the scripts
data/
  fedras.db                  SQLite index: runs and results
  runs.csv                   the same index as a flat CSV
docs/
  DATA.md                    what every run family contains
  REPRODUCE.md               one command per table and figure
```

## Install

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`torch` and `torchvision` are needed only for retraining. The analysis,
simulator and figure scripts run on NumPy and Matplotlib alone.

## Quick start (no GPU, about one minute)

```bash
bash smoke_test.sh
```

This runs the simulator, rebuilds Table 3, and redraws Figures 2 to 5 from the
stored per-round metrics, writing everything to `smoke_out/`.

## Regenerating the paper's results

```bash
# Table 3 and the per-family summaries
python scripts/collect_v6.py --root outputs/real_v6
python scripts/emit_tab3_zh.py

# Figures 2 and 3, Figure 4, Figure 5
python scripts/make_v6_figs.py figs_v6
python scripts/make_v6_fig4.py figs_v6
python scripts/make_fig5a.py figs_v6
python scripts/make_fig5b.py figs_v6

# Table 6, exact DP against the greedy baselines
python scripts/dp_scheduler_study.py

# payload audit behind the 1.64 MB to 0.90 MB comparison
python scripts/journal_model_audit.py
```

## Querying the results

`data/fedras.db` (SQLite) and `data/runs.csv` index every run and every seed:

```
runs      one row per run, with its configuration
results   one row per run x method, with the terminal metrics used in the paper
```

The terminal metrics follow `scripts/collect_v6.py`: accuracy is the mean over the
last ten rounds, the timely-delivery ratio and bytes per round are means over
rounds 30 and later, and the delivered rank is weighted by the number of
aggregated updates.

Rebuild the index at any time:

```bash
python scripts/build_index.py        # writes data/runs.csv and data/fedras.db
```

Query it from the command line:

```bash
python scripts/query.py --list-datasets
python scripts/query.py --list-methods
python scripts/query.py --family p1_tab3_cifar10_dl0.1 --metric kappa
python scripts/query.py --dataset cifar10 --deadline 2.0 --metric timely
python scripts/query.py --sql "SELECT dataset, deadline, method, AVG(kappa) k
                               FROM results GROUP BY 1,2,3 ORDER BY k DESC LIMIT 10"
```

Method aliases are accepted, so `--method FedRAS` resolves to `Adaptive_rank` and
`--method FedLoRA-16` to `FedLoRA_r16`.

As a check, the query for `p1_tab3_cifar10_dl0.1` returns FedRAS
`0.8386 / 0.495 / 0.69 / 12.64 / 0.440` for accuracy, timely-delivery ratio, MB
per round, delivered rank and delivered capacity, the same row as Table 3.

## Retraining a run

Each run in `outputs/real_v6/` stores the `config.json` that produced it, so any
run can be repeated by passing the same arguments:

```bash
python scripts/train_torch_fedlora.py \
  --dataset cifar10 --rounds 40 --clients 20 --per-round 10 \
  --local-epochs 1 --batch-size 128 --lr 0.005 \
  --deadline 0.1 --mobility-shape 2.0 --arch cnn --seed 0 \
  --data ./data --out ./outputs/rerun_cifar10_dl0.1_seed0
```

`--link-profile sagin` switches to the ground, aerial and satellite link model
used in Table 7. `--arch vit` switches to the small vision transformer used in
Figure 3. Run `python scripts/train_torch_fedlora.py --help` for the full list.

A single CIFAR-10 run (40 rounds, 20 clients, 10 participants per round) takes
about 25 minutes on one NVIDIA A100; ten paired seeds take about four hours.

## Conventions that matter for reproducing the numbers

* **Real serialized bytes.** A rank-`r` upload carries only the trainable factor,
  5632 bytes per rank column (FP32), so rank 16 is 0.090 MB. The two-factor
  baseline transmits 1.64 MB and the frozen-`A` upload 0.90 MB per round.
* **Budget grid.** Byte budgets are discretized by `q = 0.01` MB: update sizes are
  rounded up and the budget is rounded down, so any plan feasible on the grid is
  feasible in real bytes.
* **Deadline.** The deadline starts when local training ends and covers only
  transmission, propagation and handover.
* **Zero-delivery rounds.** `mean_rank` is empty when no update arrived; `aggregated`
  is the per-round delivered count and is the correct weight when pooling.

## Data

`outputs/real_v6/` holds 930 runs (93 run families, 10 paired seeds each). See
[docs/DATA.md](docs/DATA.md) for what each family covers and
[docs/REPRODUCE.md](docs/REPRODUCE.md) for the mapping from scripts to paper
tables and figures.

## License

MIT. See [LICENSE](LICENSE).

# Mapping from scripts to paper items

Run every command from the repository root after `pip install -r requirements.txt`.

## Tables

| paper item | command | output |
|---|---|---|
| Table 3 (main results) | `python scripts/emit_tab3_zh.py` | LaTeX rows on stdout |
| Table 3, per-family summaries | `python scripts/collect_v6.py --root outputs/real_v6` | printed tables |
| Table 6 (solver study) | `python scripts/dp_scheduler_study.py` | `outputs/solver_study.csv` |
| payload audit (1.64 MB to 0.90 MB) | `python scripts/journal_model_audit.py` | printed audit |

## Figures

| paper item | command | output |
|---|---|---|
| Figures 2 and 3 | `python scripts/make_v6_figs.py figs_v6` | `fig_s6_main_3ds.pdf`, `fig_s6_weak_{a,b,c,d}.pdf`, `fig_s6_weak_legend.pdf` |
| Figure 4 | `python scripts/make_v6_fig4.py figs_v6` | `fig_s6_sens_{a,b,c}.pdf` |
| Figure 5(a) | `python scripts/make_fig5a.py figs_v6` | `fig_s6_mech_a.pdf` |
| Figure 5(b) | `python scripts/make_fig5b.py figs_v6` | `fig_s6_mech_b.pdf` |

Figure 1 is a drawing, not generated from data; the editable source is kept with
the manuscript.

## Simulator (Table 5, 500 clients)

```bash
python scripts/simulator.py --rounds 150 --clients 500 --seeds 0,1,2,3,4 \
    --budget 30 --cap 25 --deadline 2.0 --out outputs/sim_500
```

The output is `round_metrics_seed<k>.csv` per seed.

## Retraining

Every run stores the arguments in `config.json`. For example, the CIFAR-10 run at
a 0.1 s deadline:

```bash
python scripts/train_torch_fedlora.py --dataset cifar10 --rounds 40 --clients 20 \
    --per-round 10 --local-epochs 1 --batch-size 128 --lr 0.005 --deadline 0.1 \
    --mobility-shape 2.0 --arch cnn --link-profile uniform --seed 0 \
    --data ./data --out ./outputs/rerun_cifar10_dl0.1_seed0
```

The trainer downloads CIFAR-10 and CIFAR-100 through `torchvision` and pretrains
the backbone once per configuration; the pretrained backbone is cached in
`--backbone-cache-dir` and reused by later seeds.

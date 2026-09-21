"""Rerun only baselines affected by the group-size consistency correction."""
from mprbps.config import load_config
from mprbps.experiment import run_experiment

cfg = load_config("configs/full_cifar10.yaml")
cfg["experiment"]["methods"] = ["greedy_repair", "reactive"]
cfg["output_dir"] = "outputs/full_cifar10_corrections"
run_experiment(cfg)

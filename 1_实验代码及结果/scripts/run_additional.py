"""Run only experiments not present in the completed corrected archive."""
from copy import deepcopy
from pathlib import Path
from mprbps.config import load_config
from mprbps.experiment import run_experiment

base = load_config("configs/full_cifar10.yaml")
base["experiment"]["methods"] = ["fedavg"]
base["output_dir"] = "outputs/full_cifar10_fedavg"
run_experiment(base)

# Lightweight sensitivity sweep: one seed and 60 rounds, preserving all other settings.
for name, changes in {
    "budget_low": {"initial_budget": 8.0},
    "budget_high": {"initial_budget": 40.0},
}.items():
    cfg = deepcopy(base)
    cfg["experiment"]["methods"] = ["gcdp", "greedy_repair", "reactive"]
    cfg["experiment"]["seeds"] = [2026]
    cfg["experiment"]["rounds"] = 60
    cfg["utility"].update(changes)
    cfg["output_dir"] = f"outputs/sensitivity_{name}"
    run_experiment(cfg)

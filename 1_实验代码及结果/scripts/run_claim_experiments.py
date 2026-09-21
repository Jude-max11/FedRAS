"""Experiments targeted at the paper's actual claims, not accuracy cherry-picking.

Usage:
  python scripts/run_claim_experiments.py --phase screen
  python scripts/run_claim_experiments.py --phase cifar
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import pandas as pd

from mprbps.config import load_config
from mprbps.experiment import run_experiment

METHODS = ["gcdp", "greedy_repair", "two_stage", "reactive", "fixed_secagg"]


def multiply_mobility(cfg: dict, factor: float) -> None:
    for d in cfg["link"]["domains"].values():
        d["handover"] = min(0.85, d["handover"] * factor)
        d["outage"] = min(0.85, d["outage"] * factor)


def run_screen() -> None:
    base = load_config("configs/demo.yaml")
    base["experiment"].update(rounds=50, clients=100, admission_cap=15,
                              min_secagg=5, methods=METHODS,
                              seeds=[2026, 2027, 2028, 2029, 2030])
    base["link"]["monte_carlo_samples"] = 64
    suites = []
    for factor in [0.5, 1.0, 1.5, 2.0, 3.0]:
        cfg = deepcopy(base); multiply_mobility(cfg, factor)
        tag = f"mobility_{factor:g}"; cfg["output_dir"] = f"outputs/claims_screen/{tag}"
        run_experiment(cfg); suites.append(("mobility_factor", factor, tag))
    for deadline in [500, 650, 850, 1100, 1400]:
        cfg = deepcopy(base); cfg["link"]["deadline_ms"] = deadline
        tag = f"deadline_{deadline}"; cfg["output_dir"] = f"outputs/claims_screen/{tag}"
        run_experiment(cfg); suites.append(("deadline_ms", deadline, tag))
    for budget in [3, 5, 8, 12, 20, 40]:
        cfg = deepcopy(base); cfg["utility"]["initial_budget"] = budget
        tag = f"budget_{budget}"; cfg["output_dir"] = f"outputs/claims_screen/{tag}"
        run_experiment(cfg); suites.append(("initial_budget", budget, tag))
    collect(Path("outputs/claims_screen"), suites)


def run_cifar() -> None:
    base = load_config("configs/full_cifar10.yaml")
    base["experiment"].update(rounds=60, methods=METHODS, seeds=[2026, 2027, 2028])
    cases = []
    definitions = {
        "nominal": {},
        "high_mobility": {"mobility": 2.0},
        "tight_deadline": {"deadline": 1050},
        "low_budget": {"budget": 8.0},
    }
    for tag, change in definitions.items():
        cfg = deepcopy(base)
        if "mobility" in change: multiply_mobility(cfg, change["mobility"])
        if "deadline" in change: cfg["link"]["deadline_ms"] = change["deadline"]
        if "budget" in change: cfg["utility"]["initial_budget"] = change["budget"]
        cfg["output_dir"] = f"outputs/claims_cifar/{tag}"
        run_experiment(cfg); cases.append(("scenario", tag, tag))
    collect(Path("outputs/claims_cifar"), cases)


def collect(root: Path, definitions: list[tuple[str, object, str]]) -> None:
    rows = []
    for variable, value, tag in definitions:
        p = root / tag / "summary_by_seed.csv"
        x = pd.read_csv(p); x["variable"] = variable; x["value"] = value; x["scenario"] = tag
        rows.append(x)
    detail = pd.concat(rows, ignore_index=True)
    root.mkdir(parents=True, exist_ok=True); detail.to_csv(root / "all_results.csv", index=False)
    metrics = ["final_accuracy", "mean_objective", "success_ratio", "bytes_ratio",
               "energy", "mean_budget", "mean_solver_ms"]
    agg = detail.groupby(["variable", "value", "method"])[metrics].agg(["mean", "std"]).reset_index()
    agg.to_csv(root / "aggregate.csv", index=False)
    print(f"Claim-focused aggregate written to {root / 'aggregate.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", choices=["screen", "cifar"], required=True)
    args = ap.parse_args(); run_screen() if args.phase == "screen" else run_cifar()

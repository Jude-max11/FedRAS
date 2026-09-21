"""Run the multi-seed evidence matrix needed to support the paper's claims.

This script deliberately uses common seeds, data partitions, and link traces
within each scenario.  It does not turn the exposure score into a privacy
guarantee; a separate attack-calibration experiment is still required.

Examples
--------
python scripts/run_submission_evidence.py --phase screen
python scripts/run_submission_evidence.py --phase cifar --rounds 150
python scripts/report_submission_evidence.py --root outputs/submission_evidence
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import pandas as pd

from mprbps.config import load_config
from mprbps.experiment import run_experiment


METHODS = ["gcdp", "greedy_repair", "two_stage", "reactive", "fixed_secagg",
           "compression_only", "fixed_aead", "fedavg"]
SEEDS = [2026, 2027, 2028, 2029, 2030]


def scale_mobility(cfg: dict, multiplier: float) -> None:
    """Apply the same mobility stressor to each domain without changing data."""
    for domain in cfg["link"]["domains"].values():
        domain["handover"] = min(0.85, domain["handover"] * multiplier)
        domain["outage"] = min(0.85, domain["outage"] * multiplier)


def scenarios(phase: str) -> list[tuple[str, dict]]:
    # Each scenario changes exactly one scientific factor from the nominal case.
    if phase == "screen":
        return [
            ("nominal", {}),
            ("high_mobility", {"mobility": 2.0}),
            ("tight_deadline", {"deadline_ms": 650}),
            ("low_exposure_budget", {"initial_budget": 3.0}),
            ("high_exposure_budget", {"initial_budget": 20.0}),
            ("large_secagg_group", {"min_secagg": 5}),
            ("exposure_weight_high", {"delta_exposure": 0.20}),
        ]
    return [
        ("nominal", {}),
        ("high_mobility", {"mobility": 2.0}),
        ("tight_deadline", {"deadline_ms": 1050}),
        ("low_exposure_budget", {"initial_budget": 8.0}),
        ("large_secagg_group", {"min_secagg": 8}),
        ("exposure_weight_high", {"delta_exposure": 0.20}),
    ]


def configure(base: dict, phase: str, changes: dict, rounds: int | None) -> dict:
    cfg = deepcopy(base)
    cfg["experiment"]["methods"] = METHODS
    cfg["experiment"]["seeds"] = SEEDS
    if phase == "screen":
        cfg["experiment"].update(clients=100, admission_cap=15, min_secagg=5)
        cfg["experiment"]["rounds"] = 60 if rounds is None else rounds
        cfg["link"]["monte_carlo_samples"] = 128
    elif rounds is not None:
        cfg["experiment"]["rounds"] = rounds
    if "mobility" in changes:
        scale_mobility(cfg, float(changes["mobility"]))
    for key in ("deadline_ms",):
        if key in changes:
            cfg["link"][key] = changes[key]
    for key in ("initial_budget", "delta_exposure"):
        if key in changes:
            cfg["utility"][key] = changes[key]
    if "min_secagg" in changes:
        cfg["experiment"]["min_secagg"] = changes["min_secagg"]
    if cfg["experiment"]["min_secagg"] > cfg["experiment"]["admission_cap"]:
        raise ValueError("minimum SecAgg group cannot exceed the admission cap")
    return cfg


def collect(root: Path, completed: list[tuple[str, dict]]) -> None:
    rows = []
    for name, changes in completed:
        frame = pd.read_csv(root / name / "summary_by_seed.csv")
        frame.insert(0, "scenario", name)
        for key, value in changes.items():
            frame[key] = value
        rows.append(frame)
    all_results = pd.concat(rows, ignore_index=True)
    all_results.to_csv(root / "all_results.csv", index=False)
    print(f"Wrote {root / 'all_results.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["screen", "cifar"], required=True)
    parser.add_argument("--rounds", type=int, default=None,
                        help="Override rounds per scenario; use only for a pilot unless phase=cifar.")
    parser.add_argument("--root", default="outputs/submission_evidence")
    args = parser.parse_args()
    base = load_config("configs/demo.yaml" if args.phase == "screen" else "configs/full_cifar10.yaml")
    root = Path(args.root) / args.phase
    completed: list[tuple[str, dict]] = []
    for name, changes in scenarios(args.phase):
        cfg = configure(base, args.phase, changes, args.rounds)
        cfg["output_dir"] = str(root / name)
        print(f"\n=== {args.phase}: {name} ===", flush=True)
        run_experiment(cfg)
        completed.append((name, changes))
    collect(root, completed)


if __name__ == "__main__":
    main()

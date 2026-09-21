from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def reliability_scores(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty or "reliability" not in frame:
        return {"brier": float("nan"), "ece": float("nan")}
    p = frame["reliability"].to_numpy(float)
    y = frame["success"].to_numpy(float)
    brier = float(np.mean((p - y) ** 2))
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return {"brier": brier, "ece": float(ece)}


def summarize(rounds: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, seed), group in rounds.groupby(["method", "seed"]):
        action_group = actions[(actions.method == method) & (actions.seed == seed)]
        calibration = reliability_scores(action_group)
        rows.append({
            "method": method,
            "seed": seed,
            "final_accuracy": group.sort_values("round").accuracy.iloc[-1],
            "mean_accuracy": group.accuracy.mean(),
            "successful_updates": group.successful.sum(),
            "selected_updates": group.selected.sum(),
            "success_ratio": group.successful.sum() / max(1, group.selected.sum()),
            "mean_objective": group.objective.mean(),
            "mean_solver_ms": group.solver_ms.mean(),
            "bytes_ratio": action_group.bytes_ratio.sum(),
            "energy": action_group.energy.sum(),
            "mean_budget": group.mean_budget.iloc[-1],
            **calibration,
        })
    detail = pd.DataFrame(rows)
    numeric = [c for c in detail.columns if c not in {"method", "seed"}]
    aggregate = detail.groupby("method")[numeric].agg(["mean", "std"])
    aggregate.columns = [f"{a}_{b}" for a, b in aggregate.columns]
    aggregate = aggregate.reset_index()
    return detail, aggregate


def write_metrics(rounds: pd.DataFrame, actions: pd.DataFrame, outdir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    outdir.mkdir(parents=True, exist_ok=True)
    rounds.to_csv(outdir / "round_metrics.csv", index=False)
    actions.to_csv(outdir / "action_metrics.csv", index=False)
    detail, aggregate = summarize(rounds, actions)
    detail.to_csv(outdir / "summary_by_seed.csv", index=False)
    aggregate.to_csv(outdir / "summary.csv", index=False)
    return detail, aggregate


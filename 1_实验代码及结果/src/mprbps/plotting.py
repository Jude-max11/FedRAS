from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLORS = {
    "gcdp": "#0072B2", "greedy_repair": "#D55E00", "reactive": "#009E73",
    "two_stage": "#CC79A7", "fixed_secagg": "#E69F00", "compression_only": "#56B4E9",
    "fixed_aead": "#999999", "fedavg": "#000000",
}


def plot_results(rounds: pd.DataFrame, actions: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titlesize": 10, "axes.labelsize": 9,
                         "legend.frameon": False, "figure.dpi": 140})
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for method, group in rounds.groupby("method"):
        curve = group.groupby("round").accuracy.agg(["mean", "std"])
        x = curve.index.to_numpy(); y = curve["mean"].to_numpy(); s = curve["std"].fillna(0).to_numpy()
        ax.plot(x, y, label=method.replace("_", " "), color=COLORS.get(method), lw=1.8)
        ax.fill_between(x, y-s, y+s, color=COLORS.get(method), alpha=.10, linewidth=0)
    ax.set(xlabel="Round", ylabel="Test accuracy", title="Federated learning convergence")
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(outdir / "accuracy.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    summary = rounds.groupby("method")[["selected", "successful", "solver_ms"]].mean()
    ratio = (summary.successful / summary.selected.clip(lower=1)).sort_values()
    ratio.plot.bar(ax=axes[0], color=[COLORS.get(x, "#4C78A8") for x in ratio.index])
    runtime = summary.solver_ms.sort_values()
    runtime.plot.bar(ax=axes[1], color=[COLORS.get(x, "#F58518") for x in runtime.index])
    axes[0].set(ylabel="Success ratio", title="Timely update delivery", ylim=(0, 1))
    axes[1].set(ylabel="Milliseconds", title="Scheduler runtime")
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=35, labelsize=7); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(outdir / "delivery_runtime.png", dpi=220)
    plt.close(fig)

    if not actions.empty:
        policy = actions.groupby(["method", "policy"]).size().unstack(fill_value=0)
        policy = policy.div(policy.sum(axis=1), axis=0)
        ax = policy.plot.bar(stacked=True, figsize=(8, 4), color=[COLORS.get(x, "#777777") for x in policy.columns])
        ax.set(ylabel="Fraction", title="Selected policy distribution")
        ax.tick_params(axis="x", labelrotation=30)
        plt.tight_layout()
        plt.savefig(outdir / "policy_distribution.png", dpi=220)
        plt.close()

    # Additional publication-ready figures used by the manuscript.
    stats = rounds.groupby("method").agg(objective=("objective", "mean"),
        bytes=("selected", "size"), budget=("mean_budget", "last"), energy=("successful", "sum"))
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.0))
    for col, ylabel, title in [("objective", "Mean modeled utility", "Utility"),
                               ("budget", "Final mean budget", "Exposure budget"),
                               ("energy", "Successful updates (proxy)", "Learning throughput")]:
        s = stats[col].sort_values(); axes[list(["objective", "budget", "energy"]).index(col)].bar(
            [x.replace("_", " ") for x in s.index], s.values,
            color=[COLORS.get(x, "#777") for x in s.index]); i=list(["objective", "budget", "energy"]).index(col)
        axes[i].set_title(title); axes[i].set_ylabel(ylabel); axes[i].tick_params(axis="x", rotation=55, labelsize=6); axes[i].spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(outdir / "utility_budget_throughput.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    if not actions.empty and {"reliability", "success"}.issubset(actions.columns):
        fig, ax = plt.subplots(figsize=(5.5, 3.5)); bins = pd.cut(actions.reliability, bins=10, include_lowest=True)
        cal = actions.assign(bin=bins).groupby("bin", observed=True).agg(pred=("reliability", "mean"), obs=("success", "mean"))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration"); ax.plot(cal.pred, cal.obs, "o-", color=COLORS["gcdp"], label="Observed")
        ax.set(xlim=(0,1), ylim=(0,1), xlabel="Predicted timely-delivery probability", ylabel="Observed success rate", title="Reliability calibration"); ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(outdir / "reliability_calibration.png", dpi=300, bbox_inches="tight"); plt.close(fig)

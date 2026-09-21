"""Create confidence intervals and paired GCDP comparisons from evidence runs."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["final_accuracy", "mean_objective", "success_ratio", "bytes_ratio",
           "energy", "mean_budget", "mean_solver_ms", "brier", "ece"]


def ci(values: pd.Series) -> tuple[float, float, float]:
    values = values.dropna().to_numpy(float)
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    mean = float(values.mean())
    if len(values) == 1:
        return mean, mean, mean
    # t=2.776 is the two-sided 95% critical value for n=5 (df=4).
    half_width = 2.776 * float(values.std(ddof=1)) / np.sqrt(len(values))
    return mean, mean - half_width, mean + half_width


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True,
                        help="A phase directory containing all_results.csv.")
    args = parser.parse_args()
    root = Path(args.root)
    data = pd.read_csv(root / "all_results.csv")
    available = [metric for metric in METRICS if metric in data]
    rows = []
    for (scenario, method), group in data.groupby(["scenario", "method"]):
        for metric in available:
            mean, low, high = ci(group[metric])
            rows.append({"scenario": scenario, "method": method, "metric": metric,
                         "n_seeds": group.seed.nunique(), "mean": mean,
                         "ci95_low": low, "ci95_high": high})
    intervals = pd.DataFrame(rows)
    intervals.to_csv(root / "confidence_intervals.csv", index=False)

    paired = []
    gcdp = data[data.method == "gcdp"].set_index(["scenario", "seed"])
    for method in sorted(set(data.method) - {"gcdp"}):
        other = data[data.method == method].set_index(["scenario", "seed"])
        common = gcdp.index.intersection(other.index)
        for metric in available:
            delta = gcdp.loc[common, metric] - other.loc[common, metric]
            mean, low, high = ci(delta)
            paired.append({"baseline": method, "metric": metric, "n_pairs": len(delta),
                           "gcdp_minus_baseline": mean, "ci95_low": low, "ci95_high": high,
                           "all_scenarios": True})
        for scenario in sorted(data.scenario.unique()):
            index = common[common.get_level_values("scenario") == scenario]
            for metric in available:
                delta = gcdp.loc[index, metric] - other.loc[index, metric]
                mean, low, high = ci(delta)
                paired.append({"scenario": scenario, "baseline": method, "metric": metric,
                               "n_pairs": len(delta), "gcdp_minus_baseline": mean,
                               "ci95_low": low, "ci95_high": high, "all_scenarios": False})
    comparisons = pd.DataFrame(paired)
    comparisons.to_csv(root / "paired_gcdp_comparisons.csv", index=False)

    lines = ["# Submission-evidence report", "",
             "Positive paired differences favor GCDP. A 95% interval excluding zero is descriptive evidence under the five-seed design; it is not a substitute for an attack-based privacy evaluation.", ""]
    for scenario in sorted(data.scenario.unique()):
        lines.extend([f"## {scenario}", "", "| Baseline | Metric | GCDP - baseline | 95% CI |", "|---|---|---:|---|"])
        subset = comparisons[(comparisons.scenario == scenario) &
                             (comparisons.metric.isin(["mean_objective", "final_accuracy", "bytes_ratio", "success_ratio", "brier", "ece"]))]
        for _, row in subset.iterrows():
            lines.append(f"| {row.baseline} | {row.metric} | {row.gcdp_minus_baseline:.5g} | [{row.ci95_low:.5g}, {row.ci95_high:.5g}] |")
        lines.append("")
    (root / "submission_evidence_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {root / 'confidence_intervals.csv'}")
    print(f"Wrote {root / 'paired_gcdp_comparisons.csv'}")
    print(f"Wrote {root / 'submission_evidence_report.md'}")


if __name__ == "__main__":
    main()

"""Replace invalid baseline rows while preserving all unaffected full results."""
from pathlib import Path

import pandas as pd

from mprbps.metrics import write_metrics
from mprbps.plotting import plot_results

base = Path("outputs/full_cifar10")
fix = Path("outputs/full_cifar10_corrections")
out = Path("outputs/full_cifar10_corrected")
replace = {"greedy_repair", "reactive"}

rounds_old = pd.read_csv(base / "round_metrics.csv")
actions_old = pd.read_csv(base / "action_metrics.csv")
rounds_fix = pd.read_csv(fix / "round_metrics.csv")
actions_fix = pd.read_csv(fix / "action_metrics.csv")
assert set(rounds_fix.method.unique()) == replace
rounds = pd.concat([rounds_old[~rounds_old.method.isin(replace)], rounds_fix], ignore_index=True)
actions = pd.concat([actions_old[~actions_old.method.isin(replace)], actions_fix], ignore_index=True)
write_metrics(rounds, actions, out)
plot_results(rounds, actions, out)
print(f"Corrected results written to: {out.resolve()}")

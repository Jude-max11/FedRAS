from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .config import default_policies
from .fl_numpy import NumpyFederatedTask
from .link import CorrelatedLinkModel
from .metrics import write_metrics
from .plotting import plot_results
from .policies import ActionEvaluator, processing_ms, protect_update
from .solvers import solve_fixed, solve_gcdp, solve_greedy_repair, solve_reactive, solve_two_stage


def stable_seed(seed: int, *items: object) -> int:
    raw = "|".join(map(str, (seed,) + items)).encode()
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "little")


def _select(method, action_grid, states, cfg):
    k = cfg["experiment"]["admission_cap"]
    m = cfg["experiment"]["min_secagg"]
    if method == "gcdp":
        return solve_gcdp(action_grid, k, m)
    if method == "greedy_repair":
        return solve_greedy_repair(action_grid, k, m)
    if method == "reactive":
        return solve_reactive(action_grid, k)
    if method == "two_stage":
        return solve_two_stage(action_grid, states, k, m)
    if method == "fixed_aead":
        return solve_fixed(action_grid, k, "q1", m, method)
    if method == "fedavg":
        return solve_fixed(action_grid, k, "q0", m, method)
    if method == "fixed_secagg":
        return solve_fixed(action_grid, k, "q4", m, method)
    if method == "compression_only":
        return solve_fixed(action_grid, k, "q3", m, method)
    raise ValueError(f"Unknown method: {method}")


def _contributions(task, previous_norm, selected_domain, domains):
    norm = previous_norm / max(previous_norm.sum(), 1e-12)
    fairness = np.array([1.0 / (1.0 + selected_domain.get(d, 0)) for d in domains])
    fairness /= max(fairness.max(), 1e-12)
    return 0.30 * task.data_weights + 0.30 * norm + 0.20 * task.diversities + 0.20 * fairness


def run_numpy_method(cfg: dict, method: str, seed: int):
    n = int(cfg["experiment"]["clients"])
    task = NumpyFederatedTask(cfg["fl"], n, seed)
    link = CorrelatedLinkModel(cfg["link"], n, seed + 101)
    policies = default_policies(cfg)
    evaluator = ActionEvaluator(cfg, link, float(cfg["fl"]["model_megabits"]), seed + 303)
    budgets = np.full(n, float(cfg["utility"]["initial_budget"]))
    history = np.array([cfg["link"]["domains"][d]["loss"] for d in link.domains], dtype=float)
    previous_norm = np.full(n, 1.0 / n)
    selected_domain: dict[str, int] = {}
    round_rows, action_rows = [], []
    for round_idx in range(int(cfg["experiment"]["rounds"])):
        contributions = _contributions(task, previous_norm, selected_domain, link.domains)
        states = link.snapshot(budgets, history, contributions, task.data_weights)
        k = int(cfg["experiment"]["admission_cap"])
        m = int(cfg["experiment"]["min_secagg"])
        action_grid = {}
        for h in [0] + list(range(m, k + 1)):
            action_grid[h] = {s.client_id: [evaluator.evaluate(s, p, h, round_idx) for p in policies] for s in states}
        selection = _select(method, action_grid, states, cfg)
        successful = []
        for action in selection.actions:
            cid = action.client_id
            state = states[cid]
            update = task.local_update(cid, stable_seed(seed, round_idx, cid, "train"))
            proc = processing_ms(action.policy, selection.secagg_size, float(cfg["fl"]["model_megabits"]))
            rng = np.random.default_rng(stable_seed(seed, round_idx, cid, "link"))
            outcome = link.realize(
                state, float(cfg["fl"]["model_megabits"]) * action.policy.size_ratio, proc,
                float(cfg["link"]["deadline_ms"]), rng,
            )
            exposed = True if not action.policy.secagg else outcome.success
            if exposed:
                budgets[cid] = max(0.0, budgets[cid] - action.exposure_cost)
            history[cid] = 0.7 * history[cid] + 0.3 * (0.0 if outcome.success else 1.0)
            previous_norm[cid] = np.linalg.norm(update)
            selected_domain[state.domain] = selected_domain.get(state.domain, 0) + 1
            if outcome.success:
                protected = protect_update(update, action.policy, np.random.default_rng(stable_seed(seed, round_idx, cid, "protect")))
                successful.append((protected, state.data_weight))
            action_rows.append({
                "method": method, "seed": seed, "round": round_idx, "client": cid,
                "domain": state.domain, "policy": action.policy.name, "secagg_size": selection.secagg_size,
                "utility": action.utility, "reliability": action.reliability, "success": int(outcome.success),
                "delivery_ms": outcome.delivery_ms, "bytes_ratio": action.bytes_ratio,
                "energy": action.energy_cost, "distortion": action.distortion,
                "exposure_cost": action.exposure_cost, "budget_after": budgets[cid],
            })
        task.aggregate(successful)
        round_rows.append({
            "method": method, "seed": seed, "round": round_idx, "accuracy": task.accuracy(),
            "selected": len(selection.actions), "successful": len(successful), "objective": selection.objective,
            "secagg_size": selection.secagg_size, "solver_ms": selection.runtime_ms,
            "mean_budget": budgets.mean(),
        })
    return pd.DataFrame(round_rows), pd.DataFrame(action_rows)


def run_experiment(cfg: dict) -> Path:
    if cfg["fl"]["backend"] != "numpy":
        from .fl_torch import run_torch_experiment
        return run_torch_experiment(cfg)
    all_rounds, all_actions = [], []
    for seed in cfg["experiment"]["seeds"]:
        for method in cfg["experiment"]["methods"]:
            rounds, actions = run_numpy_method(copy.deepcopy(cfg), method, int(seed))
            all_rounds.append(rounds)
            all_actions.append(actions)
    rounds = pd.concat(all_rounds, ignore_index=True)
    actions = pd.concat(all_actions, ignore_index=True)
    outdir = Path(cfg["output_dir"])
    _, summary = write_metrics(rounds, actions, outdir)
    plot_results(rounds, actions, outdir)
    print(summary.to_string(index=False))
    return outdir

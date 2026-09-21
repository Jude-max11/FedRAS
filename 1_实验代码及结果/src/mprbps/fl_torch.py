from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def _imports():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Subset
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise RuntimeError("PyTorch backend requires: pip install -r requirements-full.txt") from exc
    return torch, nn, F, DataLoader, Subset, datasets, transforms


def _seed(seed: int, *items) -> int:
    raw = "|".join(map(str, (seed,) + items)).encode()
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "little") % (2**31 - 1)


def dirichlet_partition(labels: np.ndarray, n_clients: int, alpha: float, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    parts = [[] for _ in range(n_clients)]
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        proportions = rng.dirichlet(np.full(n_clients, alpha))
        cuts = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        for cid, chunk in enumerate(np.split(idx, cuts)):
            parts[cid].extend(chunk.tolist())
    # Guarantee nonempty clients by moving samples from the largest partitions.
    for cid, part in enumerate(parts):
        if not part:
            donor = max(range(n_clients), key=lambda j: len(parts[j]))
            part.append(parts[donor].pop())
    return parts


def run_torch_experiment(cfg: dict) -> Path:
    torch, nn, F, DataLoader, Subset, datasets, transforms = _imports()
    from .config import default_policies
    from .link import CorrelatedLinkModel
    from .metrics import write_metrics
    from .plotting import plot_results
    from .policies import ActionEvaluator, processing_ms, protect_update
    from .solvers import solve_fixed, solve_gcdp, solve_greedy_repair, solve_reactive, solve_two_stage

    class SmallCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.fc1 = nn.Linear(64 * 8 * 8, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = F.max_pool2d(F.relu(self.conv1(x)), 2)
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            return self.fc2(F.relu(self.fc1(torch.flatten(x, 1))))

    fl = cfg["fl"]
    device = fl.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    trainset = datasets.CIFAR10(fl["data_root"], train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(fl["data_root"], train=False, download=True, transform=transform)
    labels = np.asarray(trainset.targets)
    n = int(cfg["experiment"]["clients"])

    def flatten(model):
        return torch.cat([p.detach().reshape(-1).cpu() for p in model.parameters()]).numpy()

    def add_vector(model, vector):
        offset = 0
        with torch.no_grad():
            for p in model.parameters():
                count = p.numel()
                delta = torch.as_tensor(vector[offset:offset + count], dtype=p.dtype, device=p.device).reshape_as(p)
                p.add_(delta)
                offset += count

    def local_update(global_model, indices, seed):
        torch.manual_seed(seed)
        local = copy.deepcopy(global_model).to(device)
        before = flatten(global_model)
        loader = DataLoader(Subset(trainset, indices), batch_size=fl["batch_size"], shuffle=True, num_workers=fl["num_workers"])
        opt = torch.optim.SGD(local.parameters(), lr=fl["learning_rate"], momentum=fl.get("momentum", 0.0))
        local.train()
        for _ in range(fl["local_epochs"]):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(local(x), y)
                loss.backward()
                opt.step()
        return flatten(local) - before

    def accuracy(model):
        loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=fl["num_workers"])
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in loader:
                pred = model(x.to(device)).argmax(1).cpu()
                correct += int((pred == y).sum())
                total += y.numel()
        return correct / total

    all_rounds, all_actions = [], []
    for seed in cfg["experiment"]["seeds"]:
        partitions = dirichlet_partition(labels, n, fl["dirichlet_alpha"], int(seed))
        weights = np.array([len(p) for p in partitions], float)
        weights /= weights.sum()
        diversities = np.array([len(np.unique(labels[p])) / 10 for p in partitions])
        for method in cfg["experiment"]["methods"]:
            torch.manual_seed(int(seed))
            model = SmallCNN().to(device)
            link = CorrelatedLinkModel(cfg["link"], n, int(seed) + 101)
            evaluator = ActionEvaluator(cfg, link, float(fl["model_megabits"]), int(seed) + 303)
            policies = default_policies(cfg)
            budgets = np.full(n, cfg["utility"]["initial_budget"], float)
            history = np.array([cfg["link"]["domains"][d]["loss"] for d in link.domains], float)
            previous_norm = np.full(n, 1 / n)
            domain_counts = {}
            round_rows, action_rows = [], []
            for rnd in range(cfg["experiment"]["rounds"]):
                if rnd == 0 or (rnd + 1) % 10 == 0:
                    print(f"[progress] seed={seed} method={method} round={rnd + 1}/{cfg['experiment']['rounds']}", flush=True)
                norm = previous_norm / max(previous_norm.sum(), 1e-12)
                fairness = np.array([1 / (1 + domain_counts.get(d, 0)) for d in link.domains])
                fairness /= fairness.max()
                contribution = 0.3 * weights + 0.3 * norm + 0.2 * diversities + 0.2 * fairness
                states = link.snapshot(budgets, history, contribution, weights)
                k, m = cfg["experiment"]["admission_cap"], cfg["experiment"]["min_secagg"]
                grid = {h: {s.client_id: [evaluator.evaluate(s, p, h, rnd) for p in policies] for s in states}
                        for h in [0] + list(range(m, k + 1))}
                if method == "gcdp": selection = solve_gcdp(grid, k, m)
                elif method == "greedy_repair": selection = solve_greedy_repair(grid, k, m)
                elif method == "reactive": selection = solve_reactive(grid, k)
                elif method == "two_stage": selection = solve_two_stage(grid, states, k, m)
                elif method == "fixed_aead": selection = solve_fixed(grid, k, "q1", m, method)
                elif method == "fedavg": selection = solve_fixed(grid, k, "q0", m, method)
                elif method == "fixed_secagg": selection = solve_fixed(grid, k, "q4", m, method)
                elif method == "compression_only": selection = solve_fixed(grid, k, "q3", m, method)
                else: raise ValueError(method)
                updates = []
                for action in selection.actions:
                    cid, state = action.client_id, states[action.client_id]
                    delta = local_update(model, partitions[cid], _seed(int(seed), rnd, cid, "train"))
                    previous_norm[cid] = np.linalg.norm(delta)
                    outcome = link.realize(state, fl["model_megabits"] * action.policy.size_ratio,
                                           processing_ms(action.policy, selection.secagg_size, fl["model_megabits"]),
                                           cfg["link"]["deadline_ms"], np.random.default_rng(_seed(int(seed), rnd, cid, "link")))
                    exposed = not action.policy.secagg or outcome.success
                    if exposed: budgets[cid] = max(0, budgets[cid] - action.exposure_cost)
                    history[cid] = 0.7 * history[cid] + 0.3 * (not outcome.success)
                    domain_counts[state.domain] = domain_counts.get(state.domain, 0) + 1
                    if outcome.success:
                        delta = protect_update(delta, action.policy, np.random.default_rng(_seed(int(seed), rnd, cid, "protect")))
                        updates.append((delta, weights[cid]))
                    action_rows.append({"method": method, "seed": seed, "round": rnd, "client": cid,
                        "domain": state.domain, "policy": action.policy.name, "secagg_size": selection.secagg_size,
                        "utility": action.utility, "reliability": action.reliability, "success": int(outcome.success),
                        "delivery_ms": outcome.delivery_ms, "bytes_ratio": action.bytes_ratio, "energy": action.energy_cost,
                        "distortion": action.distortion, "exposure_cost": action.exposure_cost, "budget_after": budgets[cid]})
                if updates:
                    total = sum(w for _, w in updates)
                    mean = sum(u * w for u, w in updates) / total
                    add_vector(model, mean)
                round_rows.append({"method": method, "seed": seed, "round": rnd, "accuracy": accuracy(model),
                    "selected": len(selection.actions), "successful": len(updates), "objective": selection.objective,
                    "secagg_size": selection.secagg_size, "solver_ms": selection.runtime_ms, "mean_budget": budgets.mean()})
            all_rounds.append(pd.DataFrame(round_rows)); all_actions.append(pd.DataFrame(action_rows))
            checkpoint = Path(cfg["output_dir"]) / "checkpoints"
            checkpoint.mkdir(parents=True, exist_ok=True)
            pd.concat(all_rounds, ignore_index=True).to_csv(checkpoint / "round_metrics_partial.csv", index=False)
            pd.concat(all_actions, ignore_index=True).to_csv(checkpoint / "action_metrics_partial.csv", index=False)
            print(f"[checkpoint] completed seed={seed} method={method}", flush=True)
    rounds, actions = pd.concat(all_rounds, ignore_index=True), pd.concat(all_actions, ignore_index=True)
    outdir = Path(cfg["output_dir"])
    _, summary = write_metrics(rounds, actions, outdir)
    plot_results(rounds, actions, outdir)
    print(summary.to_string(index=False))
    return outdir

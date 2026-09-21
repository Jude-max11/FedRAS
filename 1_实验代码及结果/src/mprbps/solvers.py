from __future__ import annotations

import itertools
import time

from .model import ActionEstimate, Selection


NEG = float("-inf")


def _best(actions: list[ActionEstimate], secagg: bool) -> ActionEstimate | None:
    feasible = [a for a in actions if a.feasible and a.policy.secagg == secagg]
    return max(feasible, key=lambda a: a.utility, default=None)


def solve_gcdp(actions_by_h: dict[int, dict[int, list[ActionEstimate]]], k_cap: int,
               min_secagg: int, method: str = "gcdp") -> Selection:
    start = time.perf_counter()
    client_ids = sorted(next(iter(actions_by_h.values())))
    candidates = [0] + list(range(min_secagg, k_cap + 1))
    global_best = (NEG, [], 0)
    for h in candidates:
        source = actions_by_h[h]
        # State maps (total selected, SecAgg selected) to (value, chosen actions).
        dp: dict[tuple[int, int], tuple[float, list[ActionEstimate]]] = {(0, 0): (0.0, [])}
        for cid in client_ids:
            nonsec = _best(source[cid], False)
            sec = _best(source[cid], True) if h > 0 else None
            nxt = dict(dp)
            for (k, j), (value, chosen) in dp.items():
                if k >= k_cap:
                    continue
                for action, dj in ((nonsec, 0), (sec, 1)):
                    if action is None or j + dj > h:
                        continue
                    key = (k + 1, j + dj)
                    candidate = value + action.utility
                    if candidate > nxt.get(key, (NEG, []))[0]:
                        nxt[key] = (candidate, chosen + [action])
            dp = nxt
        options = [(value, chosen) for (k, j), (value, chosen) in dp.items() if j == h and k <= k_cap]
        if options:
            value, chosen = max(options, key=lambda x: x[0])
            if value > global_best[0]:
                global_best = (value, chosen, h)
    runtime = (time.perf_counter() - start) * 1000
    if global_best[0] == NEG:
        return Selection(method, [], 0.0, 0, runtime, {"status": "infeasible"})
    return Selection(method, global_best[1], global_best[0], global_best[2], runtime)


def solve_greedy_repair(actions_by_h: dict[int, dict[int, list[ActionEstimate]]], k_cap: int,
                        min_secagg: int) -> Selection:
    start = time.perf_counter()
    # Top-h SecAgg followed by greedy non-SecAgg fill.  Every action is taken
    # from the grid evaluated at the *actual* target h, so reported utility and
    # group-dependent overhead remain consistent.  Client collisions make this
    # a heuristic; GCDP is guaranteed to dominate it on the same grid.
    best_value, chosen, best_h = 0.0, [], 0
    for h in [0] + list(range(min_secagg, k_cap + 1)):
        source = actions_by_h[h]
        if h == 0:
            nonsec = [a for acts in source.values() if (a := _best(acts, False)) is not None and a.utility > 0]
            candidate = sorted(nonsec, key=lambda a: a.utility, reverse=True)[:k_cap]
        else:
            sec = [a for acts in source.values() if (a := _best(acts, True)) is not None]
            if len(sec) < h:
                continue
            sec_chosen = sorted(sec, key=lambda a: a.utility, reverse=True)[:h]
            used = {a.client_id for a in sec_chosen}
            nonsec = [a for cid, acts in source.items()
                      if cid not in used and (a := _best(acts, False)) is not None and a.utility > 0]
            candidate = sec_chosen + sorted(nonsec, key=lambda a: a.utility, reverse=True)[:k_cap - h]
        value = sum(a.utility for a in candidate)
        if value > best_value:
            best_value, chosen, best_h = value, candidate, h
    runtime = (time.perf_counter() - start) * 1000
    return Selection("greedy_repair", chosen, best_value, best_h, runtime)


def solve_reactive(actions_by_h: dict[int, dict[int, list[ActionEstimate]]], k_cap: int) -> Selection:
    """Current-snapshot adaptive baseline without historical/group coupling."""
    start = time.perf_counter()
    source = actions_by_h[0]
    candidates = [a for acts in source.values()
                  if (a := _best(acts, False)) is not None and a.utility > 0]
    chosen = sorted(candidates, key=lambda a: a.utility, reverse=True)[:k_cap]
    return Selection("reactive", chosen, sum(a.utility for a in chosen), 0,
                     (time.perf_counter() - start) * 1000)


def solve_two_stage(actions_by_h: dict[int, dict[int, list[ActionEstimate]]], states,
                    k_cap: int, min_secagg: int) -> Selection:
    start = time.perf_counter()
    admitted = {s.client_id for s in sorted(states, key=lambda s: s.contribution * (1 - s.history_risk), reverse=True)[:k_cap]}
    restricted = {h: {cid: acts for cid, acts in by_client.items() if cid in admitted}
                  for h, by_client in actions_by_h.items()}
    result = solve_gcdp(restricted, k_cap, min_secagg, method="two_stage")
    result.runtime_ms += (time.perf_counter() - start) * 1000
    return result


def solve_fixed(actions_by_h: dict[int, dict[int, list[ActionEstimate]]], k_cap: int,
                policy_name: str, min_secagg: int, method: str) -> Selection:
    start = time.perf_counter()
    is_secagg = policy_name in {"q4", "q5"}
    sizes = range(min_secagg, k_cap + 1) if is_secagg else [0]
    best_value, chosen, best_h = NEG, [], 0
    for h in sizes:
        source = actions_by_h[h]
        candidates = [a for actions in source.values() for a in actions
                      if a.policy.name == policy_name and a.feasible]
        count = h if is_secagg else min(k_cap, len(candidates))
        if is_secagg and len(candidates) < h:
            continue
        candidate = sorted(candidates, key=lambda a: a.utility, reverse=True)[:count]
        value = sum(a.utility for a in candidate)
        if value > best_value:
            best_value, chosen, best_h = value, candidate, h
    if best_value == NEG:
        best_value = 0.0
    return Selection(method, chosen, sum(a.utility for a in chosen), sum(a.policy.secagg for a in chosen),
                     (time.perf_counter() - start) * 1000)


def brute_force(actions_by_h: dict[int, dict[int, list[ActionEstimate]]], k_cap: int,
                min_secagg: int) -> Selection:
    """Reference solver for tiny instances; enumerates skip/non-SecAgg/SecAgg actions."""
    start = time.perf_counter()
    client_ids = sorted(next(iter(actions_by_h.values())))
    best_value, best_actions, best_h = 0.0, [], 0
    for h, source in actions_by_h.items():
        if h != 0 and h < min_secagg:
            continue
        choices = []
        for cid in client_ids:
            n, s = _best(source[cid], False), _best(source[cid], True)
            choices.append([None] + ([n] if n else []) + ([s] if s and h > 0 else []))
        for assignment in itertools.product(*choices):
            picked = [a for a in assignment if a is not None]
            if len(picked) > k_cap or sum(a.policy.secagg for a in picked) != h:
                continue
            value = sum(a.utility for a in picked)
            if value > best_value:
                best_value, best_actions, best_h = value, picked, h
    return Selection("brute_force", best_actions, best_value, best_h, (time.perf_counter() - start) * 1000)

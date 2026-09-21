from __future__ import annotations

import hashlib

import numpy as np

from .model import ActionEstimate, ClientState, Policy


def predicted_distortion(policy: Policy) -> float:
    quant = 0.0 if policy.bits >= 32 else (32.0 / policy.bits - 1.0) * 0.018
    return float(np.clip(0.55 * policy.pruning + quant + 0.7 * policy.noise, 0, 1))


def processing_ms(policy: Policy, group_size: int, model_megabits: float) -> float:
    compression = 5.0 + 16.0 * policy.pruning + (0 if policy.bits == 32 else 4.0)
    crypto = 4.0 if policy.aead else 0.0
    secagg = 0.0
    if policy.secagg:
        secagg = 10.0 + 1.8 * group_size + 0.04 * group_size * group_size
    noise = 6.0 if policy.noise > 0 else 0.0
    return (compression + crypto + secagg + noise) * max(0.25, model_megabits / 2.0)


def _stable_seed(seed: int, *items: object) -> int:
    raw = "|".join(map(str, (seed,) + items)).encode()
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "little")


class ActionEvaluator:
    def __init__(self, cfg: dict, link_model, model_megabits: float, seed: int):
        self.cfg = cfg
        self.link = link_model
        self.model_megabits = model_megabits
        self.seed = seed

    def evaluate(self, state: ClientState, policy: Policy, group_size: int, round_idx: int) -> ActionEstimate:
        ucfg = self.cfg["utility"]
        lcfg = self.cfg["link"]
        if policy.secagg and group_size < self.cfg["experiment"]["min_secagg"]:
            return self._infeasible(state, policy, group_size, "SecAgg group too small")
        ratio = policy.size_ratio
        proc = processing_ms(policy, group_size, self.model_megabits)
        rng = np.random.default_rng(_stable_seed(self.seed, round_idx, state.client_id, policy.name, group_size))
        future = self.link.draw_futures(state, int(lcfg["monte_carlo_samples"]), rng)
        delivery = proc + self.model_megabits * ratio / future["rate"] * 1000.0 + future["delay"]
        delivery += future["handover"] * lcfg["handover_penalty_ms"]
        success = (delivery <= lcfg["deadline_ms"]) & ~future["lost"] & ~future["outage"]
        reliability = float(np.mean(success))
        exposure_prob = reliability if policy.secagg else 1.0
        exposure_class = "noise" if policy.noise > 0 else ("secagg" if policy.secagg else "aead")
        exposure_cost = float(ucfg["exposure_cost"][exposure_class])
        distortion = predicted_distortion(policy)
        energy = float(np.clip(0.18 + 0.55 * ratio + proc / 800.0, 0, 2))
        cpu_need = 0.25 + 0.25 * policy.pruning + (0.18 if policy.secagg else 0) + (0.08 if policy.noise else 0)
        memory_need = 0.25 + 0.20 * ratio + (0.12 if policy.secagg else 0)
        reasons = []
        if reliability < ucfg["reliability_threshold"]:
            reasons.append("reliability")
        if exposure_cost > state.budget:
            reasons.append("budget")
        if cpu_need > state.cpu:
            reasons.append("cpu")
        if memory_need > state.memory:
            reasons.append("memory")
        if energy > state.energy:
            reasons.append("energy")
        utility = (
            state.contribution * reliability
            - ucfg["alpha_bytes"] * ratio
            - ucfg["beta_energy"] * energy
            - ucfg["gamma_distortion"] * distortion
            - ucfg["delta_exposure"] * exposure_prob * exposure_cost
        )
        return ActionEstimate(
            state.client_id, policy, group_size, float(utility), reliability, exposure_prob,
            exposure_cost, ratio, energy, distortion, float(np.mean(delivery)), not reasons, ",".join(reasons),
        )

    @staticmethod
    def _infeasible(state: ClientState, policy: Policy, group_size: int, reason: str) -> ActionEstimate:
        return ActionEstimate(state.client_id, policy, group_size, float("-inf"), 0, 0, 0, 0, 0, 0, float("inf"), False, reason)


def protect_update(update: np.ndarray, policy: Policy, rng: np.random.Generator) -> np.ndarray:
    flat = update.reshape(-1).copy()
    if policy.pruning > 0:
        keep = max(1, int(round((1.0 - policy.pruning) * flat.size)))
        threshold_idx = np.argpartition(np.abs(flat), -keep)[-keep:]
        mask = np.zeros(flat.size, dtype=bool)
        mask[threshold_idx] = True
        flat[~mask] = 0
    if policy.bits < 32:
        levels = 2 ** (policy.bits - 1) - 1
        scale = np.max(np.abs(flat)) / max(levels, 1)
        if scale > 0:
            flat = np.round(flat / scale) * scale
    if policy.noise > 0:
        norm = np.linalg.norm(flat)
        if norm > 1.0:
            flat /= norm
        flat += rng.normal(0, policy.noise / np.sqrt(max(1, flat.size)), flat.size)
    return flat.reshape(update.shape)


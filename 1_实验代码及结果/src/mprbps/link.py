from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import ClientState


@dataclass
class LinkOutcome:
    success: bool
    delivery_ms: float
    handover: bool
    lost: bool
    outage: bool


class CorrelatedLinkModel:
    """Time-correlated domain link model with common random-number support."""

    def __init__(self, cfg: dict, n_clients: int, seed: int):
        self.cfg = cfg
        self.n_clients = n_clients
        self.rng = np.random.default_rng(seed)
        self.domains = self._assign_domains(cfg["domains"], n_clients)
        self.quality = self.rng.normal(0.0, 0.35, size=n_clients)
        self.handover_prev = np.zeros(n_clients, dtype=bool)
        self.outage_prev = np.zeros(n_clients, dtype=bool)

    def _assign_domains(self, specs: dict, n_clients: int) -> list[str]:
        names = list(specs)
        counts = [int(round(specs[name]["share"] * n_clients)) for name in names]
        counts[-1] += n_clients - sum(counts)
        labels = [name for name, count in zip(names, counts) for _ in range(count)]
        self.rng.shuffle(labels)
        return labels

    def snapshot(self, budgets: np.ndarray, history: np.ndarray, contributions: np.ndarray,
                 data_weights: np.ndarray) -> list[ClientState]:
        rho = float(self.cfg["correlation"])
        self.quality = rho * self.quality + np.sqrt(1.0 - rho * rho) * self.rng.normal(0, 0.35, self.n_clients)
        states: list[ClientState] = []
        for i, domain in enumerate(self.domains):
            p = self.cfg["domains"][domain]
            q = float(self.quality[i])
            rate = max(0.25, p["rate_mbps"] * np.exp(0.25 * q))
            delay = max(1.0, p["delay_ms"] * np.exp(-0.12 * q))
            loss = float(np.clip(p["loss"] * np.exp(-0.20 * q), 0, 0.8))
            handover = float(np.clip(p["handover"] * (1.0 + 0.5 * self.handover_prev[i]), 0, 0.9))
            outage = float(np.clip(p["outage"] * (1.0 + 1.5 * self.outage_prev[i]), 0, 0.9))
            states.append(ClientState(
                client_id=i, domain=domain, cpu=p["cpu"], memory=p["memory"], energy=p["energy"],
                rate_mbps=rate, delay_ms=delay, loss_prob=loss, handover_prob=handover,
                outage_prob=outage, history_risk=float(history[i]), contribution=float(contributions[i]),
                budget=float(budgets[i]), data_weight=float(data_weights[i]), latent_quality=q,
            ))
        return states

    def draw_futures(self, state: ClientState, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        p = self.cfg["domains"][state.domain]
        rho = float(self.cfg["correlation"])
        innovation = rng.normal(0, 0.35, n)
        q = rho * state.latent_quality + np.sqrt(1 - rho * rho) * innovation
        rate = np.maximum(0.25, p["rate_mbps"] * np.exp(0.25 * q))
        delay = np.maximum(1.0, p["delay_ms"] * np.exp(-0.12 * q))
        handover = rng.random(n) < state.handover_prob
        outage = rng.random(n) < state.outage_prob
        loss_prob = np.clip(p["loss"] * np.exp(-0.20 * q) + handover * self.cfg["handover_loss_penalty"], 0, 0.95)
        lost = rng.random(n) < loss_prob
        return {"rate": rate, "delay": delay, "handover": handover, "outage": outage, "lost": lost}

    def realize(self, state: ClientState, update_megabits: float, processing_ms: float,
                deadline_ms: float, rng: np.random.Generator) -> LinkOutcome:
        future = self.draw_futures(state, 1, rng)
        handover = bool(future["handover"][0])
        delivery = processing_ms + update_megabits / future["rate"][0] * 1000.0 + future["delay"][0]
        if handover:
            delivery += self.cfg["handover_penalty_ms"]
        lost = bool(future["lost"][0])
        outage = bool(future["outage"][0])
        success = delivery <= deadline_ms and not lost and not outage
        self.handover_prev[state.client_id] = handover
        self.outage_prev[state.client_id] = outage
        return LinkOutcome(success, float(delivery), handover, lost, outage)


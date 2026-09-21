from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Policy:
    name: str
    pruning: float
    bits: int
    aead: bool = True
    secagg: bool = False
    noise: float = 0.0

    @property
    def size_ratio(self) -> float:
        return max(0.01, (1.0 - self.pruning) * self.bits / 32.0)


@dataclass
class ClientState:
    client_id: int
    domain: str
    cpu: float
    memory: float
    energy: float
    rate_mbps: float
    delay_ms: float
    loss_prob: float
    handover_prob: float
    outage_prob: float
    history_risk: float
    contribution: float
    budget: float
    data_weight: float = 1.0
    latent_quality: float = 0.0


@dataclass
class ActionEstimate:
    client_id: int
    policy: Policy
    group_size: int
    utility: float
    reliability: float
    exposure_probability: float
    exposure_cost: float
    bytes_ratio: float
    energy_cost: float
    distortion: float
    delivery_ms: float
    feasible: bool
    reason: str = ""


@dataclass
class Selection:
    method: str
    actions: list[ActionEstimate]
    objective: float
    secagg_size: int
    runtime_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def client_ids(self) -> list[int]:
        return [a.client_id for a in self.actions]


from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ClientData:
    x: np.ndarray
    y: np.ndarray
    diversity: float


class NumpyFederatedTask:
    """Small multiclass softmax FL task used for dependency-light validation."""

    def __init__(self, cfg: dict, n_clients: int, seed: int):
        self.cfg = cfg
        self.n_clients = n_clients
        self.rng = np.random.default_rng(seed)
        self.n_features = int(cfg["features"])
        self.n_classes = int(cfg["classes"])
        self.clients, self.test = self._make_data()
        self.weights = np.zeros((self.n_features + 1, self.n_classes), dtype=np.float64)

    def _make_data(self):
        centers = self.rng.normal(0, 1.25, (self.n_classes, self.n_features))
        alpha = float(self.cfg["dirichlet_alpha"])
        n = int(self.cfg["samples_per_client"])
        clients = []
        for _ in range(self.n_clients):
            probs = self.rng.dirichlet(np.full(self.n_classes, alpha))
            y = self.rng.choice(self.n_classes, n, p=probs)
            x = centers[y] + self.rng.normal(0, 1.75, (n, self.n_features))
            x = np.c_[x, np.ones(n)]
            entropy = -np.sum(probs * np.log(probs + 1e-12)) / np.log(self.n_classes)
            clients.append(ClientData(x, y, float(entropy)))
        y = self.rng.integers(0, self.n_classes, 1200)
        x = centers[y] + self.rng.normal(0, 1.75, (y.size, self.n_features))
        return clients, (np.c_[x, np.ones(y.size)], y)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def local_update(self, client_id: int, round_seed: int) -> np.ndarray:
        data = self.clients[client_id]
        local = self.weights.copy()
        rng = np.random.default_rng(round_seed)
        lr = float(self.cfg["learning_rate"])
        batch = int(self.cfg["batch_size"])
        for _ in range(int(self.cfg["local_epochs"])):
            order = rng.permutation(data.y.size)
            for start in range(0, data.y.size, batch):
                idx = order[start:start + batch]
                probs = self._softmax(data.x[idx] @ local)
                probs[np.arange(idx.size), data.y[idx]] -= 1
                grad = data.x[idx].T @ probs / idx.size
                local -= lr * grad
        return local - self.weights

    def aggregate(self, updates: list[tuple[np.ndarray, float]]) -> None:
        if not updates:
            return
        total = sum(weight for _, weight in updates)
        mean = sum(update * weight for update, weight in updates) / max(total, 1e-12)
        self.weights += mean

    def accuracy(self) -> float:
        x, y = self.test
        return float(np.mean(np.argmax(x @ self.weights, axis=1) == y))

    @property
    def data_weights(self) -> np.ndarray:
        sizes = np.array([c.y.size for c in self.clients], dtype=float)
        return sizes / sizes.sum()

    @property
    def diversities(self) -> np.ndarray:
        return np.array([c.diversity for c in self.clients])

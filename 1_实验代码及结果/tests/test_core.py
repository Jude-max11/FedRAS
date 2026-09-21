from pathlib import Path

import numpy as np

from mprbps.config import load_config
from mprbps.experiment import run_numpy_method
from mprbps.link import CorrelatedLinkModel


ROOT = Path(__file__).parents[1]


def test_config_and_link_reproducibility():
    cfg = load_config(ROOT / "configs/demo.yaml")
    a = CorrelatedLinkModel(cfg["link"], 8, 9)
    b = CorrelatedLinkModel(cfg["link"], 8, 9)
    budgets = np.ones(8)
    risk = np.zeros(8)
    contribution = np.ones(8) / 8
    weight = contribution.copy()
    sa = a.snapshot(budgets, risk, contribution, weight)
    sb = b.snapshot(budgets, risk, contribution, weight)
    assert [(x.domain, x.rate_mbps) for x in sa] == [(x.domain, x.rate_mbps) for x in sb]


def test_one_round_smoke():
    cfg = load_config(ROOT / "configs/demo.yaml")
    cfg["experiment"].update(rounds=1, clients=8, admission_cap=3, min_secagg=2)
    cfg["link"]["monte_carlo_samples"] = 4
    rounds, actions = run_numpy_method(cfg, "gcdp", 11)
    assert len(rounds) == 1
    assert rounds.iloc[0].selected <= 3
    h = int(rounds.iloc[0].secagg_size)
    assert h == 0 or h >= 2
    assert not actions.empty

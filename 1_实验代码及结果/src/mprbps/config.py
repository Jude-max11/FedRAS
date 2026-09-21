from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["_config_path"] = str(path.resolve())
    return cfg


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(clean, handle, sort_keys=False)


def default_policies(cfg: dict[str, Any]):
    from .model import Policy

    return [Policy(**item) for item in cfg["policies"]]


"""Fail-fast checks intended to run before a long remote experiment."""
from __future__ import annotations

import argparse
import platform
import shutil
import sys
from pathlib import Path

from mprbps.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full_cifar10.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    exp, fl = cfg["experiment"], cfg["fl"]
    assert exp["min_secagg"] <= exp["admission_cap"] <= exp["clients"]
    assert exp["rounds"] > 0 and exp["seeds"] and exp["methods"]
    names = {p["name"] for p in cfg["policies"]}
    assert {"q1", "q3", "q4"} <= names
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    print(f"Python: {sys.version.split()[0]} ({platform.platform()})")
    print(f"Disk free: {shutil.disk_usage(out.resolve()).free / 2**30:.1f} GiB")
    if fl["backend"] == "torch":
        import torch, torchvision
        print(f"Torch: {torch.__version__}; torchvision: {torchvision.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if (fl.get("device") == "cuda" or fl.get("require_cuda", False)) and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required by this configuration but unavailable; install a compatible PyTorch CUDA build first")
    print("Configuration and dependencies: OK")


if __name__ == "__main__":
    main()

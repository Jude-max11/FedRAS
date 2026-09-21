#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install -e ".[full]"
python scripts/preflight.py --config configs/full_cifar10.yaml
mprbps verify-gcdp --trials 100
mprbps run --config configs/full_cifar10.yaml

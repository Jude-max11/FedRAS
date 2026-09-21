#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
python scripts/preflight.py --config configs/demo.yaml
python scripts/verify_gcdp.py
python scripts/run_demo.py

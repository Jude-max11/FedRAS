#!/usr/bin/env bash
set -euo pipefail
# Keep the project environment isolated.  If the host already provides a
# driver-compatible CUDA PyTorch, expose it read-only through the venv instead
# of downloading another 700+ MB wheel or modifying the host environment.
if python3 -c 'import torch; assert torch.cuda.is_available()' >/dev/null 2>&1; then
  python3 -m venv --system-site-packages .venv
  USE_HOST_TORCH=1
else
  python3 -m venv .venv
  USE_HOST_TORCH=0
fi
source .venv/bin/activate
python -m pip install --upgrade pip
if [ "$USE_HOST_TORCH" = 1 ]; then
  # Install only this project; torch/torchvision remain supplied by the host
  # site-packages and are not changed by pip in the project venv.
  python -m pip install -e . --no-deps
else
  python -m pip install -e '.[full]'
fi
python scripts/preflight.py --config configs/pilot_cifar10.yaml
python scripts/verify_gcdp.py
echo 'Environment ready. Run the real-data pilot first:'
echo 'python -m mprbps.cli run --config configs/pilot_cifar10.yaml'
echo 'Only after it succeeds, run: bash scripts/run_full.sh'

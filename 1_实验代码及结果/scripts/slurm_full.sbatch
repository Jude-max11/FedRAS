#!/usr/bin/env bash
#SBATCH --job-name=mprbps-fcs
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
source .venv/bin/activate
python scripts/preflight.py --config configs/full_cifar10.yaml
python -m mprbps.cli run --config configs/full_cifar10.yaml

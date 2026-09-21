from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .experiment import run_experiment


def main(argv=None):
    parser = argparse.ArgumentParser(description="MP-RBPS reproducible experiment runner")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run an experiment configuration")
    run.add_argument("--config", required=True)
    verify = sub.add_parser("verify-gcdp", help="compare GCDP with exhaustive search")
    verify.add_argument("--trials", type=int, default=30)
    verify.add_argument("--seed", type=int, default=13)
    args = parser.parse_args(argv)
    if args.command == "run":
        cfg = load_config(args.config)
        outdir = run_experiment(cfg)
        print(f"Results written to: {outdir.resolve()}")
    else:
        from .verification import verify_gcdp
        verify_gcdp(args.trials, args.seed)


if __name__ == "__main__":
    main()


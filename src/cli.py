from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

# Data-driven command definitions
COMMANDS: List[Dict[str, Any]] = [
    {
        "name": "intuition",
        "script": "01_intuition_boosting.py",
        "help": "Run residual-fitting intuition demo",
        "args": [
            ("--n-samples", int, 300),
            ("--noise-std", float, 0.2),
            ("--n-rounds", int, 20),
            ("--learning-rate", float, 0.1),
            ("--max-depth", int, 1),
            ("--random-state", int, 42),
        ],
    },
    {
        "name": "classify",
        "script": "02_xgb_basic_classification.py",
        "help": "Run baseline vs XGBoost classification demo",
        "args": [
            ("--n-samples", int, 3000),
            ("--n-features", int, 12),
            ("--n-informative", int, 6),
            ("--class-sep", float, 1.2),
            ("--test-size", float, 0.2),
            ("--random-state", int, 42),
            ("--max-depth", int, 3),
            ("--n-estimators", int, 300),
            ("--learning-rate", float, 0.05),
            ("--subsample", float, 0.9),
            ("--colsample-bytree", float, 0.9),
            ("--reg-lambda", float, 1.0),
        ],
    },
    {
        "name": "explain",
        "script": "03_xgb_explainability_shap.py",
        "help": "Run SHAP explainability demo",
        "args": [
            ("--n-samples", int, 4000),
            ("--n-features", int, 12),
            ("--n-informative", int, 6),
            ("--class-sep", float, 1.2),
            ("--random-state", int, 42),
            ("--max-depth", int, 3),
            ("--n-estimators", int, 300),
            ("--learning-rate", float, 0.05),
            ("--subsample", float, 0.9),
            ("--colsample-bytree", float, 0.9),
            ("--reg-lambda", float, 1.0),
            ("--top-k", int, 6),
            ("--instance-idx", int, 0),
        ],
    },
    {
        "name": "tune",
        "script": "04_xgb_regularization_tuning.py",
        "help": "Run regularization and tuning demo",
        "args": [
            ("--n-samples", int, 6000),
            ("--n-features", int, 20),
            ("--n-informative", int, 8),
            ("--class-sep", float, 1.0),
            ("--random-state", int, 42),
            ("--early-stopping-rounds", int, 30),
        ],
    },
]


def run_python(script: str, extra_args: list[str]) -> int:
    cmd = [sys.executable, str(SCRIPTS / script), *extra_args]
    # Use run with check=True to surface failures and proper exit codes
    completed = subprocess.run(cmd, check=False)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="XGBoost PoC CLI")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # Register subcommands from COMMANDS
    for cmd_meta in COMMANDS:
        p = subparsers.add_parser(cmd_meta["name"], help=cmd_meta["help"])
        arg_flags: List[str] = []
        for flag, ftype, default in cmd_meta["args"]:
            dest = flag.lstrip("-").replace("-", "_")
            p.add_argument(flag, dest=dest, type=ftype, default=default)
            arg_flags.append(flag)
        p.set_defaults(script=cmd_meta["script"], arg_flags=arg_flags)

    args, extra = parser.parse_known_args()

    # Reconstruct flags/values in declared order
    flags: list[str] = []
    for flag in args.arg_flags:
        dest = flag.lstrip("-").replace("-", "_")
        flags += [flag, str(getattr(args, dest))]

    return run_python(args.script, flags + extra)


if __name__ == "__main__":
    sys.exit(main())

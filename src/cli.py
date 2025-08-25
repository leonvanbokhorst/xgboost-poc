from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

# Common argument definitions to reduce duplication
COMMON_BASIC: List[Tuple[str, type, Any]] = [
    ("--n-samples", int, 300),
    ("--random-state", int, 42),
]

CLASSIF_COMMON: List[Tuple[str, type, Any]] = [
    ("--n-features", int, 12),
    ("--n-informative", int, 6),
    ("--class-sep", float, 1.2),
]

# Data-driven command definitions
COMMANDS: List[Dict[str, Any]] = [
    {
        "name": "intuition",
        "script": "01_intuition_boosting.py",
        "help": "Run residual-fitting intuition demo",
        "args": [
            *COMMON_BASIC,
            ("--noise-std", float, 0.2),
            ("--n-rounds", int, 20),
            ("--learning-rate", float, 0.1),
            ("--max-depth", int, 1),
        ],
    },
    {
        "name": "classify",
        "script": "02_xgb_basic_classification.py",
        "help": "Run baseline vs XGBoost classification demo",
        "args": [
            ("--n-samples", int, 3000),  # override COMMON default
            *CLASSIF_COMMON,
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
            *CLASSIF_COMMON,
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
    script_path = SCRIPTS / script
    if script_path.parent != SCRIPTS or not script_path.exists():
        print(f"Unknown or missing script: {script}", file=sys.stderr)
        return 2
    cmd = [sys.executable, str(script_path), *extra_args]
    try:
        subprocess.run(cmd, check=True)
        return 0
    except subprocess.CalledProcessError as e:
        # Surface the underlying script's exit code
        return e.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="XGBoost PoC CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # Register subcommands from COMMANDS
    for cmd_meta in COMMANDS:
        p = subparsers.add_parser(
            cmd_meta["name"],
            help=cmd_meta["help"],
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
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

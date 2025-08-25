from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

# Common argument definitions to reduce duplication: (flag, type, default[, action])
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
            ("--n-samples", int, 3000),
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
            ("--random-search", int, 0),
        ],
    },
    {
        "name": "advanced",
        "script": "05_xgb_advanced_topics.py",
        "help": "Run advanced topics demo (missing, imbalance, monotonic, optional GPU)",
        "args": [
            ("--n-samples", int, 5000),
            ("--n-features", int, 10),
            ("--n-informative", int, 5),
            ("--imbalance-ratio", float, 0.1),
            ("--missing-rate", float, 0.05),
            ("--pos-weight", float, 1.0),
            ("--monotone", int, 1),
            ("--use-gpu", str, "false"),
            ("--random-state", int, 42),
        ],
    },
]


def normalize(cmd_meta: Dict[str, Any]) -> List[Tuple[str, type, Any, Any | None]]:
    normalized: List[Tuple[str, type, Any, Any | None]] = []
    for spec in cmd_meta["args"]:
        flag, ftype, default, *rest = spec
        action = rest[0] if rest else None
        normalized.append((flag, ftype, default, action))
    return normalized


def build_flags(arg_specs: List[Tuple[str, type, Any, Any | None]], args: argparse.Namespace, overrides: Dict[str, Any]) -> List[str]:
    flags: List[str] = []
    for flag, ftype, default, action in arg_specs:
        dest = flag.lstrip("-").replace("-", "_")
        val = overrides.get(dest, getattr(args, dest))
        if action == "store_true":
            if val:
                flags.append(flag)
            continue
        if action == "store_false":
            if not val:
                flags.append(flag)
            continue
        if ftype is bool:
            flags += [flag, "true" if val else "false"]
            continue
        flags += [flag, str(val)]
    return flags


def run_python(script: str, extra_args: list[str]) -> int:
    script_path = SCRIPTS / script
    if not script_path.exists():
        print(f"Unknown or missing script: {script}", file=sys.stderr)
        return 2
    cmd = [sys.executable, str(script_path), *extra_args]
    try:
        subprocess.run(cmd, check=True)
        return 0
    except subprocess.CalledProcessError as e:
        return e.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="XGBoost PoC CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, help="YAML config to override subcommand args")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    for cmd_meta in COMMANDS:
        p = subparsers.add_parser(
            cmd_meta["name"],
            help=cmd_meta["help"],
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        normalized = normalize(cmd_meta)
        for flag, ftype, default, action in normalized:
            kwargs = dict(dest=flag.lstrip("-").replace("-", "_"), default=default)
            if action:
                p.add_argument(flag, action=action, **kwargs)
            else:
                p.add_argument(flag, type=ftype, **kwargs)
        p.set_defaults(script=cmd_meta["script"], arg_specs=normalized)

    args, extra = parser.parse_known_args()

    overrides: Dict[str, Any] = {}
    if args.config:
        with open(args.config, "r") as f:
            cfg_yaml = yaml.safe_load(f) or {}
        overrides = {k.replace("-", "_"): v for k, v in cfg_yaml.get(args.cmd, {}).items()}

    flags = build_flags(args.arg_specs, args, overrides)
    return run_python(args.script, flags + extra)


if __name__ == "__main__":
    sys.exit(main())

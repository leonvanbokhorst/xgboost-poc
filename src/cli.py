from __future__ import annotations

import argparse
import importlib
import inspect
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import runpy
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
        "runner": "callable",  # Use the direct runner for this script
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
            # output_dir is handled by the runner
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
    scripts_root = SCRIPTS.resolve()
    script_path = (SCRIPTS / script).resolve()
    # Ensure the resolved script stays within the scripts directory and exists
    try:
        script_path.relative_to(scripts_root)
    except ValueError:
        print(f"Refusing to execute script outside scripts/: {script_path}", file=sys.stderr)
        return 2
    if not script_path.exists() or not script_path.is_file():
        print(f"Unknown or missing script: {script}", file=sys.stderr)
        return 2
    # Execute the target script in-process to avoid spawning subprocesses.
    # Simulate command-line args for the script's argparse.
    old_argv = sys.argv
    try:
        sys.argv = [str(script_path), *extra_args]
        runpy.run_path(str(script_path), run_name="__main__")
        return 0
    except SystemExit as e:
        # Propagate script's intended exit code without killing the parent process
        try:
            return int(e.code) if e.code is not None else 0
        except Exception:
            return 1
    finally:
        sys.argv = old_argv


def run_callable(script: str, args: argparse.Namespace, overrides: Dict[str, Any]) -> int:
    """
    Run a script by importing it and calling its `main(cfg)` function.

    This runner assumes the target script has:
    - A `Config` dataclass defining its parameters.
    - A `main(cfg: Config)` function to execute the logic.
    """
    module_name = f"scripts.{script.replace('.py', '')}"
    try:
        mod = importlib.import_module(module_name)
    except ImportError as e:
        print(f"Failed to import script '{module_name}': {e}", file=sys.stderr)
        return 1

    if not hasattr(mod, "main") or not hasattr(mod, "Config"):
        print(f"Script '{module_name}' missing required `main` or `Config`", file=sys.stderr)
        return 1

    # Collect all config fields from the dataclass definition
    config_fields = mod.Config.__dataclass_fields__.keys()

    # Build config, starting with defaults, then applying CLI args, then YAML overrides
    cfg_data = {}
    for dest, value in vars(args).items():
        if dest in config_fields:
            cfg_data[dest] = value
    cfg_data.update(overrides)

    # Add output_dir to config, which is managed by the CLI
    cfg_data["output_dir"] = Path("runs")

    try:
        config = mod.Config(**cfg_data)
        mod.main(config)
        return 0
    except Exception as e:
        print(f"Error running script '{module_name}': {e}", file=sys.stderr)
        return 1


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
        # Attach metadata to the parser object for later retrieval
        p.set_defaults(
            script=cmd_meta["script"],
            arg_specs=normalized,
            runner=cmd_meta.get("runner", "runpy"),
        )

    # Disallow unknown extra args to avoid forwarding arbitrary parameters
    args = parser.parse_args()

    overrides: Dict[str, Any] = {}
    if args.config:
        with open(args.config, "r") as f:
            cfg_yaml = yaml.safe_load(f) or {}
        overrides = {k.replace("-", "_"): v for k, v in cfg_yaml.get(args.cmd, {}).items()}

    # Decide which runner to use based on the command's configuration
    if getattr(args, "runner", "runpy") == "callable":
        # The new direct-callable runner
        return run_callable(args.script, args, overrides)
    else:
        # The legacy runpy runner
        flags = build_flags(args.arg_specs, args, overrides)
        return run_python(args.script, flags)


if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def run_python(script: str, extra_args: list[str]) -> int:
    cmd = [sys.executable, str(SCRIPTS / script), *extra_args]
    return subprocess.call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description="XGBoost PoC CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("intuition", help="Run residual-fitting intuition demo")
    p1.add_argument("--n-samples", type=int, default=300)
    p1.add_argument("--noise-std", type=float, default=0.2)
    p1.add_argument("--n-rounds", type=int, default=20)
    p1.add_argument("--learning-rate", type=float, default=0.1)
    p1.add_argument("--max-depth", type=int, default=1)
    p1.add_argument("--random-state", type=int, default=42)

    p2 = sub.add_parser("classify", help="Run baseline vs XGBoost classification demo")
    p2.add_argument("--n-samples", type=int, default=3000)
    p2.add_argument("--n-features", type=int, default=12)
    p2.add_argument("--n-informative", type=int, default=6)
    p2.add_argument("--class-sep", type=float, default=1.2)
    p2.add_argument("--test-size", type=float, default=0.2)
    p2.add_argument("--random-state", type=int, default=42)
    p2.add_argument("--max-depth", type=int, default=3)
    p2.add_argument("--n-estimators", type=int, default=300)
    p2.add_argument("--learning-rate", type=float, default=0.05)
    p2.add_argument("--subsample", type=float, default=0.9)
    p2.add_argument("--colsample-bytree", type=float, default=0.9)
    p2.add_argument("--reg-lambda", type=float, default=1.0)

    p3 = sub.add_parser("explain", help="Run SHAP explainability demo")
    p3.add_argument("--n-samples", type=int, default=4000)
    p3.add_argument("--n-features", type=int, default=12)
    p3.add_argument("--n-informative", type=int, default=6)
    p3.add_argument("--class-sep", type=float, default=1.2)
    p3.add_argument("--random-state", type=int, default=42)
    p3.add_argument("--max-depth", type=int, default=3)
    p3.add_argument("--n-estimators", type=int, default=300)
    p3.add_argument("--learning-rate", type=float, default=0.05)
    p3.add_argument("--subsample", type=float, default=0.9)
    p3.add_argument("--colsample-bytree", type=float, default=0.9)
    p3.add_argument("--reg-lambda", type=float, default=1.0)
    p3.add_argument("--top-k", type=int, default=6)
    p3.add_argument("--instance-idx", type=int, default=0)

    p4 = sub.add_parser("tune", help="Run regularization and tuning demo")
    p4.add_argument("--n-samples", type=int, default=6000)
    p4.add_argument("--n-features", type=int, default=20)
    p4.add_argument("--n-informative", type=int, default=8)
    p4.add_argument("--class-sep", type=float, default=1.0)
    p4.add_argument("--random-state", type=int, default=42)
    p4.add_argument("--early-stopping-rounds", type=int, default=30)

    args, extra = parser.parse_known_args()

    if args.cmd == "intuition":
        return run_python(
            "01_intuition_boosting.py",
            [
                "--n-samples", str(args.n_samples),
                "--noise-std", str(args.noise_std),
                "--n-rounds", str(args.n_rounds),
                "--learning-rate", str(args.learning_rate),
                "--max-depth", str(args.max_depth),
                "--random-state", str(args.random_state),
            ] + extra,
        )

    if args.cmd == "classify":
        return run_python(
            "02_xgb_basic_classification.py",
            [
                "--n-samples", str(args.n_samples),
                "--n-features", str(args.n_features),
                "--n-informative", str(args.n_informative),
                "--class-sep", str(args.class_sep),
                "--test-size", str(args.test_size),
                "--random-state", str(args.random_state),
                "--max-depth", str(args.max_depth),
                "--n-estimators", str(args.n_estimators),
                "--learning-rate", str(args.learning_rate),
                "--subsample", str(args.subsample),
                "--colsample-bytree", str(args.colsample_bytree),
                "--reg-lambda", str(args.reg_lambda),
            ] + extra,
        )

    if args.cmd == "explain":
        return run_python(
            "03_xgb_explainability_shap.py",
            [
                "--n-samples", str(args.n_samples),
                "--n-features", str(args.n_features),
                "--n-informative", str(args.n_informative),
                "--class-sep", str(args.class_sep),
                "--random-state", str(args.random_state),
                "--max-depth", str(args.max_depth),
                "--n-estimators", str(args.n_estimators),
                "--learning-rate", str(args.learning_rate),
                "--subsample", str(args.subsample),
                "--colsample-bytree", str(args.colsample_bytree),
                "--reg-lambda", str(args.reg_lambda),
                "--top-k", str(args.top_k),
                "--instance-idx", str(args.instance_idx),
            ] + extra,
        )

    if args.cmd == "tune":
        return run_python(
            "04_xgb_regularization_tuning.py",
            [
                "--n-samples", str(args.n_samples),
                "--n-features", str(args.n_features),
                "--n-informative", str(args.n_informative),
                "--class-sep", str(args.class_sep),
                "--random-state", str(args.random_state),
                "--early-stopping-rounds", str(args.early_stopping_rounds),
            ] + extra,
        )

    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

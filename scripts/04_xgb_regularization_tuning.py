#!/usr/bin/env python3
"""
04_xgb_regularization_tuning.py

Demonstrate overfitting and the effect of key XGBoost regularization/tuning parameters.
- Train intentionally overfitted model (deep trees, many rounds) to show gap.
- Run a small search over a few hyperparameters with early stopping.
- Save training/validation curves and a compact results table (CSV).

Outputs:
- curves_overfit.png: training vs validation AUC for an overfit config
- tuning_results.csv: table of configs and metrics
- best_curves.png: curves for the best tuned model

Usage examples:
- uv run python scripts/04_xgb_regularization_tuning.py
- uv run python scripts/04_xgb_regularization_tuning.py --n-samples 8000 --overfit-n-estimators 1000 --overfit-max-depth 10
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost.callback import EarlyStopping


@dataclass
class Config:
    n_samples: int = 6000
    n_features: int = 20
    n_informative: int = 8
    class_sep: float = 1.0
    test_size: float = 0.2
    random_state: int = 42

    # Overfit demo parameters
    overfit_n_estimators: int = 800
    overfit_max_depth: int = 10
    overfit_learning_rate: float = 0.2

    # Tuning search space (small, fast)
    tune_n_estimators: List[int] = None  # filled in parse
    tune_max_depth: List[int] = None
    tune_learning_rate: List[float] = None
    tune_min_child_weight: List[float] = None
    tune_subsample: List[float] = None
    tune_colsample_bytree: List[float] = None
    tune_reg_lambda: List[float] = None

    # General XGB options
    tree_method: str = "hist"
    early_stopping_rounds: int = 30

    # IO
    output_dir: Path = Path("runs")


@dataclass
class Result:
    params: Dict[str, float]
    best_iteration: int
    train_auc: float
    valid_auc: float


def ensure_output_dir(root: Path) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    p = root / ts / "tuning"
    p.mkdir(parents=True, exist_ok=True)
    return p


def make_data(cfg: Config):
    X, y = make_classification(
        n_samples=cfg.n_samples,
        n_features=cfg.n_features,
        n_informative=cfg.n_informative,
        n_redundant=0,
        n_clusters_per_class=2,
        class_sep=cfg.class_sep,
        random_state=cfg.random_state,
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state
    )
    return X_train, X_valid, y_train, y_valid


def plot_curves(evals_result: Dict[str, Dict[str, List[float]]], title: str, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    # For xgboost.train: keys are typically 'train' and 'valid'
    # But be defensive and support 'validation_0' style too
    def get_series(split: str) -> List[float]:
        if split in evals_result:
            return evals_result.get(split, {}).get("auc", [])
        # fallback keys
        if split == "train":
            return evals_result.get("validation_0", {}).get("auc", [])
        if split == "valid":
            return evals_result.get("validation_1", {}).get("auc", [])
        return []

    train_auc = get_series("train")
    valid_auc = get_series("valid")

    if train_auc:
        plt.plot(train_auc, label="train AUC")
    if valid_auc:
        plt.plot(valid_auc, label="valid AUC")

    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("AUC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def train_overfit(cfg: Config, X_train, y_train, X_valid, y_valid, out_dir: Path) -> None:
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": cfg.overfit_max_depth,
        "eta": cfg.overfit_learning_rate,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "lambda": 0.0,
        "tree_method": cfg.tree_method,
        "verbosity": 0,
    }

    evals_result: Dict[str, Dict[str, List[float]]] = {}
    xgb.train(
        params,
        dtrain,
        num_boost_round=cfg.overfit_n_estimators,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        evals_result=evals_result,
        verbose_eval=False,
    )

    plot_curves(
        evals_result,
        title="Overfit demo: training vs validation AUC",
        out_path=out_dir / "curves_overfit.png",
    )


def find_best_iteration_from_history(evals_result: Dict[str, Dict[str, List[float]]]) -> int:
    valid_auc_series = evals_result.get("valid", {}).get("auc", [])
    if not valid_auc_series:
        valid_auc_series = evals_result.get("validation_1", {}).get("auc", [])
    if not valid_auc_series:
        return 0
    return int(np.argmax(valid_auc_series))


def grid_search(cfg: Config, X_train, y_train, X_valid, y_valid) -> Tuple[List[Result], Result]:
    results: List[Result] = []
    best: Result | None = None

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    for n_estimators in cfg.tune_n_estimators:
        for max_depth in cfg.tune_max_depth:
            for learning_rate in cfg.tune_learning_rate:
                for min_child_weight in cfg.tune_min_child_weight:
                    for subsample in cfg.tune_subsample:
                        for colsample_bytree in cfg.tune_colsample_bytree:
                            for reg_lambda in cfg.tune_reg_lambda:
                                params = {
                                    "objective": "binary:logistic",
                                    "eval_metric": "auc",
                                    "max_depth": max_depth,
                                    "eta": learning_rate,
                                    "min_child_weight": min_child_weight,
                                    "subsample": subsample,
                                    "colsample_bytree": colsample_bytree,
                                    "lambda": reg_lambda,
                                    "tree_method": cfg.tree_method,
                                    "verbosity": 0,
                                }

                                evals_result: Dict[str, Dict[str, List[float]]] = {}
                                booster = xgb.train(
                                    params,
                                    dtrain,
                                    num_boost_round=n_estimators,
                                    evals=[(dtrain, "train"), (dvalid, "valid")],
                                    evals_result=evals_result,
                                    callbacks=[
                                        EarlyStopping(
                                            rounds=cfg.early_stopping_rounds,
                                            save_best=True,
                                            data_name="valid",
                                            metric_name="auc",
                                        )
                                    ],
                                    verbose_eval=False,
                                )

                                train_auc_series = evals_result.get("train", {}).get("auc", [])
                                valid_auc_series = evals_result.get("valid", {}).get("auc", [])
                                if not valid_auc_series:
                                    continue

                                best_iter_attr = getattr(booster, "best_iteration", None)
                                if best_iter_attr is None:
                                    best_iter = find_best_iteration_from_history(evals_result)
                                else:
                                    best_iter = int(best_iter_attr)

                                res = Result(
                                    params={
                                        "n_estimators": n_estimators,
                                        "max_depth": max_depth,
                                        "learning_rate": learning_rate,
                                        "min_child_weight": min_child_weight,
                                        "subsample": subsample,
                                        "colsample_bytree": colsample_bytree,
                                        "reg_lambda": reg_lambda,
                                    },
                                    best_iteration=best_iter,
                                    train_auc=float(train_auc_series[best_iter]) if train_auc_series else float("nan"),
                                    valid_auc=float(valid_auc_series[best_iter]),
                                )
                                results.append(res)
                                if best is None or res.valid_auc > best.valid_auc:
                                    best = res

    assert best is not None
    return results, best


def write_results_csv(results: List[Result], out_path: Path) -> None:
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "n_estimators",
            "max_depth",
            "learning_rate",
            "min_child_weight",
            "subsample",
            "colsample_bytree",
            "reg_lambda",
            "best_iteration",
            "train_auc",
            "valid_auc",
        ]
        writer.writerow(header)
        for r in results:
            p = r.params
            writer.writerow(
                [
                    p["n_estimators"],
                    p["max_depth"],
                    p["learning_rate"],
                    p["min_child_weight"],
                    p["subsample"],
                    p["colsample_bytree"],
                    p["reg_lambda"],
                    r.best_iteration,
                    f"{r.train_auc:.4f}",
                    f"{r.valid_auc:.4f}",
                ]
            )


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="XGBoost regularization and tuning demo.")
    p.add_argument("--n-samples", type=int, default=6000)
    p.add_argument("--n-features", type=int, default=20)
    p.add_argument("--n-informative", type=int, default=8)
    p.add_argument("--class-sep", type=float, default=1.0)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)

    p.add_argument("--overfit-n-estimators", type=int, default=800)
    p.add_argument("--overfit-max-depth", type=int, default=10)
    p.add_argument("--overfit-learning-rate", type=float, default=0.2)

    p.add_argument("--early-stopping-rounds", type=int, default=30)

    p.add_argument("--output-dir", type=Path, default=Path("runs"))

    a = p.parse_args()

    cfg = Config(
        n_samples=a.n_samples,
        n_features=a.n_features,
        n_informative=a.n_informative,
        class_sep=a.class_sep,
        test_size=a.test_size,
        random_state=a.random_state,
        overfit_n_estimators=a.overfit_n_estimators,
        overfit_max_depth=a.overfit_max_depth,
        overfit_learning_rate=a.overfit_learning_rate,
        tree_method="hist",
        early_stopping_rounds=a.early_stopping_rounds,
        output_dir=a.output_dir,
    )

    # Small, sensible search space for quick runs
    cfg.tune_n_estimators = [200, 400]
    cfg.tune_max_depth = [3, 4, 5]
    cfg.tune_learning_rate = [0.05, 0.1]
    cfg.tune_min_child_weight = [1.0, 3.0]
    cfg.tune_subsample = [0.8, 1.0]
    cfg.tune_colsample_bytree = [0.8, 1.0]
    cfg.tune_reg_lambda = [0.0, 1.0]

    return cfg


def main() -> None:
    cfg = parse_args()
    out_dir = ensure_output_dir(cfg.output_dir)

    X_train, X_valid, y_train, y_valid = make_data(cfg)

    # 1) Overfitting demonstration
    train_overfit(cfg, X_train, y_train, X_valid, y_valid, out_dir)

    # 2) Tuning with early stopping
    results, best = grid_search(cfg, X_train, y_train, X_valid, y_valid)
    write_results_csv(results, out_dir / "tuning_results.csv")

    # 3) Train best config again to export curves
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    best_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": int(best.params["max_depth"]),
        "eta": float(best.params["learning_rate"]),
        "min_child_weight": float(best.params["min_child_weight"]),
        "subsample": float(best.params["subsample"]),
        "colsample_bytree": float(best.params["colsample_bytree"]),
        "lambda": float(best.params["reg_lambda"]),
        "tree_method": cfg.tree_method,
        "verbosity": 0,
    }

    evals_result: Dict[str, Dict[str, List[float]]] = {}
    xgb.train(
        best_params,
        dtrain,
        num_boost_round=int(best.params["n_estimators"]),
        evals=[(dtrain, "train"), (dvalid, "valid")],
        evals_result=evals_result,
        callbacks=[EarlyStopping(rounds=cfg.early_stopping_rounds, save_best=True, data_name="valid", metric_name="auc")],
        verbose_eval=False,
    )
    plot_curves(evals_result, title="Best model: training vs validation AUC", out_path=out_dir / "best_curves.png")

    print(f"Saved tuning artifacts to: {out_dir}")


if __name__ == "__main__":
    main()

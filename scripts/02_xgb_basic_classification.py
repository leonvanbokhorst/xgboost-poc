#!/usr/bin/env python3
"""
02_xgb_basic_classification.py

A basic classification demo comparing a baseline (logistic regression) with
XGBoost on a synthetic dataset. Saves metrics and simple plots.

Outputs:
- roc_pr_curves.png: ROC and Precision-Recall curves for both models
- calibration.png: Calibration curve comparison
- feature_importance.png: XGBoost feature importance (gain)

Usage examples:
- uv run python scripts/02_xgb_basic_classification.py
- uv run python scripts/02_xgb_basic_classification.py --n-samples 5000 --max-depth 4 --n-estimators 400

May your AUC rise like the twin suns of Tatooine.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier


@dataclass
class Config:
    n_samples: int = 3000
    n_features: int = 12
    n_informative: int = 6
    class_sep: float = 1.2
    test_size: float = 0.2
    random_state: int = 42
    # XGBoost
    max_depth: int = 3
    n_estimators: int = 300
    learning_rate: float = 0.05
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_lambda: float = 1.0
    tree_method: str = "hist"
    # IO
    output_dir: Path = Path("runs")


def ensure_output_dir(root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = root / timestamp / "basic_classification"
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_data(cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=cfg.n_samples,
        n_features=cfg.n_features,
        n_informative=cfg.n_informative,
        n_redundant=0,
        n_clusters_per_class=2,
        class_sep=cfg.class_sep,
        random_state=cfg.random_state,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state
    )
    return X_train, X_test, y_train, y_test


def fit_baseline(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    return model


def fit_xgb(cfg: Config, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_lambda=cfg.reg_lambda,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method=cfg.tree_method,
        random_state=cfg.random_state,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    return model


def plot_roc_pr(
    y_test: np.ndarray,
    lr_prob: np.ndarray,
    xgb_prob: np.ndarray,
    out_dir: Path,
) -> None:
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_prob)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_prob)
    roc_lr = roc_auc_score(y_test, lr_prob)
    roc_xgb = roc_auc_score(y_test, xgb_prob)

    p_lr, r_lr, _ = precision_recall_curve(y_test, lr_prob)
    p_xgb, r_xgb, _ = precision_recall_curve(y_test, xgb_prob)
    ap_lr = average_precision_score(y_test, lr_prob)
    ap_xgb = average_precision_score(y_test, xgb_prob)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(fpr_lr, tpr_lr, label=f"LogReg ROC AUC={roc_lr:.3f}")
    axes[0].plot(fpr_xgb, tpr_xgb, label=f"XGB ROC AUC={roc_xgb:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].legend()

    axes[1].plot(r_lr, p_lr, label=f"LogReg AP={ap_lr:.3f}")
    axes[1].plot(r_xgb, p_xgb, label=f"XGB AP={ap_xgb:.3f}")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "roc_pr_curves.png", dpi=150)
    plt.close(fig)


def plot_calibration(
    y_test: np.ndarray,
    lr_prob: np.ndarray,
    xgb_prob: np.ndarray,
    out_dir: Path,
) -> None:
    prob_true_lr, prob_pred_lr = calibration_curve(y_test, lr_prob, n_bins=10, strategy="quantile")
    prob_true_xgb, prob_pred_xgb = calibration_curve(y_test, xgb_prob, n_bins=10, strategy="quantile")

    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect")
    plt.plot(prob_pred_lr, prob_true_lr, marker="o", label="LogReg")
    plt.plot(prob_pred_xgb, prob_true_xgb, marker="o", label="XGB")
    plt.title("Calibration Curve")
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "calibration.png", dpi=150)
    plt.close()


def plot_feature_importance(model: XGBClassifier, out_dir: Path) -> None:
    importance = model.get_booster().get_score(importance_type="gain")
    if not importance:
        return
    # importance is a dict like {"f0": gain, ...}
    items = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
    labels = [k for k, _ in items]
    values = [v for _, v in items]

    plt.figure(figsize=(8, max(3, len(values) * 0.4)))
    plt.barh(labels[::-1], values[::-1])
    plt.title("XGBoost Feature Importance (Gain)")
    plt.xlabel("Gain")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_importance.png", dpi=150)
    plt.close()


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Baseline vs XGBoost classification demo.")
    parser.add_argument("--n-samples", type=int, default=3000)
    parser.add_argument("--n-features", type=int, default=12)
    parser.add_argument("--n-informative", type=int, default=6)
    parser.add_argument("--class-sep", type=float, default=1.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--tree-method", type=str, default="hist")

    parser.add_argument("--output-dir", type=Path, default=Path("runs"))

    args = parser.parse_args()
    return Config(
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_informative=args.n_informative,
        class_sep=args.class_sep,
        test_size=args.test_size,
        random_state=args.random_state,
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        tree_method=args.tree_method,
        output_dir=args.output_dir,
    )


def main() -> None:
    cfg = parse_args()
    out_dir = ensure_output_dir(cfg.output_dir)

    X_train, X_test, y_train, y_test = make_data(cfg)

    lr = fit_baseline(X_train, y_train)
    xgb = fit_xgb(cfg, X_train, y_train, X_test, y_test)

    # Probabilities for metrics/curves
    lr_prob = lr.predict_proba(X_test)[:, 1]
    xgb_prob = xgb.predict_proba(X_test)[:, 1]

    # Simple accuracy for sanity
    lr_acc = accuracy_score(y_test, (lr_prob >= 0.5).astype(int))
    xgb_acc = accuracy_score(y_test, (xgb_prob >= 0.5).astype(int))

    print(f"LogReg accuracy: {lr_acc:.3f}")
    print(f"XGB accuracy:    {xgb_acc:.3f}")
    print(f"LogReg ROC AUC:  {roc_auc_score(y_test, lr_prob):.3f}")
    print(f"XGB ROC AUC:     {roc_auc_score(y_test, xgb_prob):.3f}")

    plot_roc_pr(y_test, lr_prob, xgb_prob, out_dir)
    plot_calibration(y_test, lr_prob, xgb_prob, out_dir)
    plot_feature_importance(xgb, out_dir)

    print(f"Saved visuals to: {out_dir}")
    print("BB-8 reports gains are nominal, Master Lonn.")


if __name__ == "__main__":
    main()

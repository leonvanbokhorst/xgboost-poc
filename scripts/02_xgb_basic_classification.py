#!/usr/bin/env python3
"""
02_xgb_basic_classification.py

A basic classification demo comparing a baseline (logistic regression) with
XGBoost on a synthetic dataset. Saves metrics and simple plots.

This script is intended to be run via the project's CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from xgboost import XGBClassifier

from src.data import make_synthetic_classification
from src.plotting import plot_calibration, plot_feature_importance, plot_roc_pr
from src.utils import ensure_timestamped_dir


@dataclass
class Config:
    """Configuration object for the classification demo."""
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
    """Create a timestamped subdirectory for the output."""
    return ensure_timestamped_dir(root, "basic_classification")


def make_data(cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate and split synthetic classification data."""
    return make_synthetic_classification(
        n_samples=cfg.n_samples,
        n_features=cfg.n_features,
        n_informative=cfg.n_informative,
        class_sep=cfg.class_sep,
        random_state=cfg.random_state,
        test_size=cfg.test_size,
    )


def fit_baseline(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Fit a logistic regression baseline model."""
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    return model


def fit_xgb(
    cfg: Config, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> XGBClassifier:
    """Fit an XGBoost classification model."""
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


def main(cfg: Config) -> None:
    """
    Main function for the classification demo.

    Args:
        cfg: A configuration object with all necessary parameters.
    """
    out_dir = ensure_output_dir(cfg.output_dir)

    X_train, X_test, y_train, y_test = make_data(cfg)

    lr = fit_baseline(X_train, y_train)
    xgb = fit_xgb(cfg, X_train, y_train, X_test, y_test)

    lr_prob = lr.predict_proba(X_test)[:, 1]
    xgb_prob = xgb.predict_proba(X_test)[:, 1]

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

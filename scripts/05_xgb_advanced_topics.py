#!/usr/bin/env python3
"""
05_xgb_advanced_topics.py

Advanced XGBoost topics on synthetic datasets:
- Missing values handling (automatic default direction)
- Class imbalance handling via scale_pos_weight
- Monotonic constraints demonstration
- Optional GPU training (if tree_method supports and GPU available)

Outputs:
- pr_curves_imbalance.png
- monotonic_dependence.png

Usage examples:
- uv run python scripts/05_xgb_advanced_topics.py --use-gpu false
- uv run python scripts/05_xgb_advanced_topics.py --pos-weight 5.0
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.utils import ensure_timestamped_dir


@dataclass
class Config:
    n_samples: int = 5000
    n_features: int = 10
    n_informative: int = 5
    class_sep: float = 1.2
    imbalance_ratio: float = 0.1  # positive class proportion
    missing_rate: float = 0.05
    pos_weight: float = 1.0
    monotone: int = 1  # apply +1 constraint to the first feature
    use_gpu: bool = False
    random_state: int = 42
    output_dir: Path = Path("runs")


def make_imbalanced_data(cfg: Config):
    rng = np.random.default_rng(cfg.random_state)
    n_pos = int(cfg.n_samples * cfg.imbalance_ratio)
    n_neg = cfg.n_samples - n_pos

    X_pos = rng.normal(loc=1.0, scale=1.0, size=(n_pos, cfg.n_features))
    X_neg = rng.normal(loc=0.0, scale=1.0, size=(n_neg, cfg.n_features))
    X = np.vstack([X_pos, X_neg])
    y = np.hstack([np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)])

    # Add signal on first informative features
    for i in range(cfg.n_informative):
        X[:n_pos, i] += 1.5  # shift positives

    # Introduce missing values at random
    mask = rng.uniform(size=X.shape) < cfg.missing_rate
    X[mask] = np.nan

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=cfg.random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def train_with_options(cfg: Config, X_train, y_train, X_test, y_test):
    tree_method = "hist"
    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=cfg.pos_weight,
        tree_method=tree_method,
        random_state=cfg.random_state,
    )
    # Device selection for GPU if requested
    if cfg.use_gpu:
        model.set_params(device="cuda")
    else:
        model.set_params(device="cpu")

    # Monotonic constraint applied to first feature only
    constraints = [cfg.monotone] + [0] * (cfg.n_features - 1)
    constraints_str = "(" + ",".join(str(v) for v in constraints) + ")"
    model.set_params(monotone_constraints=constraints_str)

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    prob = model.predict_proba(X_test)[:, 1]
    return model, prob


def plot_pr_curves(y_test, prob, out_dir: Path):
    p, r, _ = precision_recall_curve(y_test, prob)
    ap = average_precision_score(y_test, prob)
    plt.figure(figsize=(6, 5))
    plt.plot(r, p, label=f"XGB AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Imbalanced)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "pr_curves_imbalance.png", dpi=150)
    plt.close()


def plot_monotonic_dependence(model: XGBClassifier, cfg: Config, out_dir: Path):
    # Sweep first feature to visualize monotone effect on predictions
    x_vals = np.linspace(-3, 3, 200)
    X = np.zeros((200, cfg.n_features))
    X[:, 0] = x_vals
    preds = model.predict_proba(X)[:, 1]
    plt.figure(figsize=(6, 4))
    plt.plot(x_vals, preds)
    plt.title("Monotonic constraint effect (feature 0)")
    plt.xlabel("feature 0 value")
    plt.ylabel("P(y=1)")
    plt.tight_layout()
    plt.savefig(out_dir / "monotonic_dependence.png", dpi=150)
    plt.close()


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="XGBoost advanced topics demo")
    p.add_argument("--n-samples", type=int, default=5000)
    p.add_argument("--n-features", type=int, default=10)
    p.add_argument("--n-informative", type=int, default=5)
    p.add_argument("--imbalance-ratio", type=float, default=0.1)
    p.add_argument("--missing-rate", type=float, default=0.05)
    p.add_argument("--pos-weight", type=float, default=1.0)
    p.add_argument("--monotone", type=int, default=1)
    p.add_argument("--use-gpu", type=lambda s: s.lower() in {"1","true","yes"}, default=False)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=Path("runs"))
    a = p.parse_args()
    return Config(
        n_samples=a.n_samples,
        n_features=a.n_features,
        n_informative=a.n_informative,
        imbalance_ratio=a.imbalance_ratio,
        missing_rate=a.missing_rate,
        pos_weight=a.pos_weight,
        monotone=a.monotone,
        use_gpu=bool(a.use_gpu),
        random_state=a.random_state,
        output_dir=a.output_dir,
    )


def main() -> None:
    cfg = parse_args()
    out_dir = ensure_timestamped_dir(cfg.output_dir, "advanced")
    X_train, X_test, y_train, y_test = make_imbalanced_data(cfg)
    model, prob = train_with_options(cfg, X_train, y_train, X_test, y_test)
    plot_pr_curves(y_test, prob, out_dir)
    plot_monotonic_dependence(model, cfg, out_dir)
    print(f"Saved advanced topics artifacts to: {out_dir}")


if __name__ == "__main__":
    main()

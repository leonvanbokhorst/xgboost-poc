#!/usr/bin/env python3
"""
03_xgb_explainability_shap.py

XGBoost explainability with SHAP on a synthetic classification dataset.
Generates global and local explanations:
- SHAP summary (beeswarm) plot
- SHAP bar plot (mean |SHAP|)
- SHAP dependence plots for top-K features
- Local explanation for a specific test instance

Outputs are saved under runs/<timestamp>/explainability/.

Usage examples:
- uv run python scripts/03_xgb_explainability_shap.py
- uv run python scripts/03_xgb_explainability_shap.py --n-samples 6000 --top-k 8 --instance-idx 1

Note: requires installing the optional extras group: `uv sync --extra explain`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


@dataclass
class Config:
    n_samples: int = 4000
    n_features: int = 12
    n_informative: int = 6
    class_sep: float = 1.2
    test_size: float = 0.2
    random_state: int = 42
    # XGB params
    max_depth: int = 3
    n_estimators: int = 300
    learning_rate: float = 0.05
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_lambda: float = 1.0
    tree_method: str = "hist"
    # Explainability
    top_k: int = 6
    instance_idx: int = 0
    # IO
    output_dir: Path = Path("runs")


def ensure_output_dir(root: Path) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    p = root / ts / "explainability"
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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state
    )
    feature_names = [f"f{i}" for i in range(cfg.n_features)]
    return X_train, X_test, y_train, y_test, feature_names


def train_xgb(cfg: Config, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> XGBClassifier:
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


def compute_shap(model: XGBClassifier, X_train: np.ndarray, X_test: np.ndarray):
    # Use TreeExplainer for tree models (fast exact/approx)
    explainer = shap.TreeExplainer(model, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_test)
    # For binary classification, shap_values is array of shape (n_samples, n_features)
    # Ensure we handle potential list output format from older versions
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, tuple)):
        expected_value = expected_value[0]
    return explainer, shap_values, expected_value


def plot_summary(shap_values: np.ndarray, X_test: np.ndarray, feature_names: list[str], out_dir: Path) -> None:
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary_bar.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_dependence(
    shap_values: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    top_k: int,
    out_dir: Path,
) -> None:
    # Rank by mean absolute SHAP
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:top_k]

    for idx in top_idx:
        shap.dependence_plot(
            ind=idx,
            shap_values=shap_values,
            features=X_test,
            feature_names=feature_names,
            show=False,
            interaction_index="auto",
        )
        plt.tight_layout()
        plt.savefig(out_dir / f"dependence_{feature_names[idx]}.png", dpi=150, bbox_inches="tight")
        plt.close()


def plot_local(
    explainer: shap.TreeExplainer,
    expected_value: float,
    shap_values: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    instance_idx: int,
    out_dir: Path,
) -> None:
    idx = int(np.clip(instance_idx, 0, len(X_test) - 1))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[idx],
            base_values=expected_value,
            data=X_test[idx],
            feature_names=feature_names,
        ),
        show=False,
        max_display=14,
    )
    plt.tight_layout()
    plt.savefig(out_dir / f"local_waterfall_idx_{idx}.png", dpi=150, bbox_inches="tight")
    plt.close()


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="XGBoost explainability with SHAP.")
    p.add_argument("--n-samples", type=int, default=4000)
    p.add_argument("--n-features", type=int, default=12)
    p.add_argument("--n-informative", type=int, default=6)
    p.add_argument("--class-sep", type=float, default=1.2)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)

    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--subsample", type=float, default=0.9)
    p.add_argument("--colsample-bytree", type=float, default=0.9)
    p.add_argument("--reg-lambda", type=float, default=1.0)
    p.add_argument("--tree-method", type=str, default="hist")

    p.add_argument("--top-k", type=int, default=6, help="Top features for dependence plots.")
    p.add_argument("--instance-idx", type=int, default=0, help="Test instance index for local plot.")

    p.add_argument("--output-dir", type=Path, default=Path("runs"))

    a = p.parse_args()
    return Config(
        n_samples=a.n_samples,
        n_features=a.n_features,
        n_informative=a.n_informative,
        class_sep=a.class_sep,
        test_size=a.test_size,
        random_state=a.random_state,
        max_depth=a.max_depth,
        n_estimators=a.n_estimators,
        learning_rate=a.learning_rate,
        subsample=a.subsample,
        colsample_bytree=a.colsample_bytree,
        reg_lambda=a.reg_lambda,
        tree_method=a.tree_method,
        top_k=a.top_k,
        instance_idx=a.instance_idx,
        output_dir=a.output_dir,
    )


def main() -> None:
    cfg = parse_args()
    out_dir = ensure_output_dir(cfg.output_dir)

    X_train, X_test, y_train, y_test, feature_names = make_data(cfg)
    model = train_xgb(cfg, X_train, y_train, X_test, y_test)

    explainer, shap_values, expected_value = compute_shap(model, X_train, X_test)

    plot_summary(shap_values, X_test, feature_names, out_dir)
    plot_dependence(shap_values, X_test, feature_names, cfg.top_k, out_dir)
    plot_local(explainer, expected_value, shap_values, X_test, feature_names, cfg.instance_idx, out_dir)

    print(f"Saved SHAP visuals to: {out_dir}")


if __name__ == "__main__":
    main()

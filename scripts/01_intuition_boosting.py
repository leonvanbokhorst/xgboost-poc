#!/usr/bin/env python3
"""
01_intuition_boosting.py

A hands-on, visual intuition for gradient boosting via a simple 1D regression
problem using decision stumps (depth-1 trees). We explicitly fit residuals like
correcting errors from the previous step.

Outputs:
- progress.png: overlay of model predictions after several boosting rounds
- residuals_round_*.png: residual scatter plots per selected rounds

Usage examples:
- uv run python scripts/01_intuition_boosting.py
- uv run python scripts/01_intuition_boosting.py --n-samples 400 --n-rounds 30 --learning-rate 0.2
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor

from src.data import make_sine_regression
from src.utils import ensure_timestamped_dir


@dataclass
class BoostingConfig:
    n_samples: int = 300
    noise_std: float = 0.2
    n_rounds: int = 20
    learning_rate: float = 0.1
    max_depth: int = 1
    random_state: int = 42
    output_dir: Path = Path("runs")


def fit_boosting_regression(
    X: np.ndarray,
    y: np.ndarray,
    n_rounds: int,
    learning_rate: float,
    max_depth: int,
    random_state: int,
) -> Tuple[List[DecisionTreeRegressor], List[np.ndarray]]:
    rng = np.random.default_rng(seed=random_state)
    residual_seed_seq = rng.integers(low=0, high=1_000_000, size=n_rounds)

    trees: List[DecisionTreeRegressor] = []
    predictions_per_round: List[np.ndarray] = []

    base_pred_value = float(np.mean(y))
    y_pred = np.full_like(y, fill_value=base_pred_value, dtype=float)

    for round_idx in range(n_rounds):
        residuals = y - y_pred
        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=int(residual_seed_seq[round_idx]))
        tree.fit(X, residuals)
        update = tree.predict(X)
        y_pred = y_pred + learning_rate * update
        trees.append(tree)
        predictions_per_round.append(y_pred.copy())

    return trees, predictions_per_round


def predict_boosted(
    X_new: np.ndarray,
    base_value: float,
    trees: List[DecisionTreeRegressor],
    learning_rate: float,
) -> np.ndarray:
    y_hat = np.full(shape=(X_new.shape[0],), fill_value=base_value, dtype=float)
    for tree in trees:
        y_hat += learning_rate * tree.predict(X_new)
    return y_hat


def plot_progress(
    X: np.ndarray,
    y: np.ndarray,
    x_grid: np.ndarray,
    y_grid_clean: np.ndarray,
    base_value: float,
    trees: List[DecisionTreeRegressor],
    predictions_per_round: List[np.ndarray],
    learning_rate: float,
    out_dir: Path,
) -> None:
    highlight_rounds = []
    if len(predictions_per_round) >= 5:
        highlight_rounds = [0, 2, 4, len(predictions_per_round) - 1]
    else:
        highlight_rounds = list(range(len(predictions_per_round)))

    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], y, s=16, alpha=0.5, label="data (noisy)")
    plt.plot(x_grid[:, 0], y_grid_clean, "k--", lw=2, label="true sin(x)")

    for idx in highlight_rounds:
        grid_pred = predict_boosted(x_grid, base_value, trees[: idx + 1], learning_rate)
        plt.plot(x_grid[:, 0], grid_pred, lw=2, label=f"round {idx+1}")

    plt.title("Manual Gradient Boosting with Decision Stumps (Regression)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "progress.png", dpi=150)
    plt.close()


def plot_residuals(
    X: np.ndarray,
    y: np.ndarray,
    predictions_per_round: List[np.ndarray],
    out_dir: Path,
) -> None:
    if len(predictions_per_round) == 0:
        return

    selected = []
    if len(predictions_per_round) >= 4:
        selected = [0, len(predictions_per_round) // 2, len(predictions_per_round) - 1]
    else:
        selected = list(range(len(predictions_per_round)))

    for idx in selected:
        residuals = y - predictions_per_round[idx]
        plt.figure(figsize=(8, 4))
        plt.scatter(X[:, 0], residuals, s=18, alpha=0.6)
        plt.axhline(0.0, color="k", linestyle=":")
        plt.title(f"Residuals after round {idx+1}")
        plt.xlabel("x")
        plt.ylabel("residual = y - y_hat")
        plt.tight_layout()
        plt.savefig(out_dir / f"residuals_round_{idx+1:02d}.png", dpi=140)
        plt.close()


def parse_args() -> BoostingConfig:
    parser = argparse.ArgumentParser(
        description="Visual intuition for gradient boosting via residual fitting (regression)."
    )
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--noise-std", type=float, default=0.2)
    parser.add_argument("--n-rounds", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))

    args = parser.parse_args()
    return BoostingConfig(
        n_samples=args.n_samples,
        noise_std=args.noise_std,
        n_rounds=args.n_rounds,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        random_state=args.random_state,
        output_dir=args.output_dir,
    )


def main() -> None:
    cfg = parse_args()
    out_dir = ensure_timestamped_dir(cfg.output_dir, "intuition")

    X, y, x_grid, y_grid_clean = make_sine_regression(
        n_samples=cfg.n_samples, noise_std=cfg.noise_std, random_state=cfg.random_state
    )

    trees, predictions_per_round = fit_boosting_regression(
        X=X,
        y=y,
        n_rounds=cfg.n_rounds,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        random_state=cfg.random_state,
    )

    base_value = float(np.mean(y))

    plot_progress(
        X=X,
        y=y,
        x_grid=x_grid,
        y_grid_clean=y_grid_clean,
        base_value=base_value,
        trees=trees,
        predictions_per_round=predictions_per_round,
        learning_rate=cfg.learning_rate,
        out_dir=out_dir,
    )
    plot_residuals(X=X, y=y, predictions_per_round=predictions_per_round, out_dir=out_dir)

    print(f"Saved visuals to: {out_dir}")


if __name__ == "__main__":
    main()

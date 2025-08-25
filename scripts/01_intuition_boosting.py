#!/usr/bin/env python3
"""
01_intuition_boosting.py

A hands-on, visual intuition for gradient boosting via a simple 1D regression
problem using decision stumps (depth-1 trees). We explicitly fit residuals like
a Padawan learning to correct errors from the previous step.

Outputs:
- progress.png: overlay of model predictions after several boosting rounds
- residuals_round_*.png: residual scatter plots per selected rounds

Usage examples:
- uv run python scripts/01_intuition_boosting.py
- uv run python scripts/01_intuition_boosting.py --n-samples 400 --n-rounds 30 --learning-rate 0.2

May the residuals shrink with you.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

# Use a non-interactive backend for script execution
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor


@dataclass
class BoostingConfig:
    n_samples: int = 300
    noise_std: float = 0.2
    n_rounds: int = 20
    learning_rate: float = 0.1
    max_depth: int = 1
    random_state: int = 42
    output_dir: Path = Path("runs")


def generate_regression_data(
    n_samples: int, noise_std: float, random_state: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a simple 1D regression dataset to visualize boosting behavior.

    y = sin(x) + noise
    """
    rng = np.random.default_rng(seed=random_state)
    x = rng.uniform(-3.0, 3.0, size=n_samples)
    y_clean = np.sin(x)
    y = y_clean + rng.normal(0.0, noise_std, size=n_samples)

    # For smooth plotting of ground truth and predictions
    x_grid = np.linspace(-3.0, 3.0, 600)
    y_grid_clean = np.sin(x_grid)

    return x.reshape(-1, 1), y, x_grid.reshape(-1, 1), y_grid_clean


def fit_boosting_regression(
    X: np.ndarray,
    y: np.ndarray,
    n_rounds: int,
    learning_rate: float,
    max_depth: int,
    random_state: int,
) -> Tuple[List[DecisionTreeRegressor], List[np.ndarray]]:
    """
    Fit a simple gradient boosting regressor by hand using decision trees that learn
    to predict residuals at each round. We store the prediction curve after each round.
    """
    rng = np.random.default_rng(seed=random_state)
    residual_seed_seq = rng.integers(low=0, high=1_000_000, size=n_rounds)

    trees: List[DecisionTreeRegressor] = []
    predictions_per_round: List[np.ndarray] = []

    # Start with a constant prediction (mean of y)
    base_pred_value = float(np.mean(y))
    y_pred = np.full_like(y, fill_value=base_pred_value, dtype=float)

    for round_idx in range(n_rounds):
        residuals = y - y_pred

        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=int(residual_seed_seq[round_idx]))
        tree.fit(X, residuals)
        update = tree.predict(X)

        # Shrinkage/learning rate controls how much we trust each step
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
    """Predict using the manually boosted ensemble."""
    y_hat = np.full(shape=(X_new.shape[0],), fill_value=base_value, dtype=float)
    for tree in trees:
        y_hat += learning_rate * tree.predict(X_new)
    return y_hat


def ensure_output_dir(root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = root / timestamp / "intuition"
    path.mkdir(parents=True, exist_ok=True)
    return path


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
    """
    Create a single figure showing the ground truth and model predictions after
    several rounds, to visualize the ensemble taking shape.
    """
    # Choose a few representative rounds to overlay
    highlight_rounds = []
    if len(predictions_per_round) >= 5:
        highlight_rounds = [0, 2, 4, len(predictions_per_round) - 1]
    else:
        highlight_rounds = list(range(len(predictions_per_round)))

    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], y, s=16, alpha=0.5, label="data (noisy)")
    plt.plot(x_grid[:, 0], y_grid_clean, "k--", lw=2, label="true sin(x)")

    # Plot predictions after selected rounds
    for idx in highlight_rounds:
        grid_pred = predict_boosted(x_grid, base_value, trees[: idx + 1], learning_rate)
        plt.plot(
            x_grid[:, 0],
            grid_pred,
            lw=2,
            label=f"round {idx+1}",
        )

    plt.title("Manual Gradient Boosting with Decision Stumps (Regression)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    out_file = out_dir / "progress.png"
    plt.savefig(out_file, dpi=150)
    plt.close()


def plot_residuals(
    X: np.ndarray,
    y: np.ndarray,
    predictions_per_round: List[np.ndarray],
    out_dir: Path,
) -> None:
    """
    Plot residuals at a couple of rounds to visualize how the model keeps
    correcting mistakes.
    """
    if len(predictions_per_round) == 0:
        return

    # Select a couple of intermediate rounds and the final
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
        out_file = out_dir / f"residuals_round_{idx+1:02d}.png"
        plt.savefig(out_file, dpi=140)
        plt.close()


def parse_args() -> BoostingConfig:
    parser = argparse.ArgumentParser(
        description="Visual intuition for gradient boosting via residual fitting (regression)."
    )
    parser.add_argument("--n-samples", type=int, default=300, help="Number of data points.")
    parser.add_argument("--noise-std", type=float, default=0.2, help="Gaussian noise std dev.")
    parser.add_argument("--n-rounds", type=int, default=20, help="Boosting rounds (trees).")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Shrinkage parameter.")
    parser.add_argument("--max-depth", type=int, default=1, help="Tree depth (use 1 for stumps).")
    parser.add_argument("--random-state", type=int, default=42, help="RNG seed.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs"), help="Root directory for outputs."
    )

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
    out_dir = ensure_output_dir(cfg.output_dir)

    X, y, x_grid, y_grid_clean = generate_regression_data(
        n_samples=cfg.n_samples, noise_std=cfg.noise_std, random_state=cfg.random_state
    )

    # Train manual boosting
    trees, predictions_per_round = fit_boosting_regression(
        X=X,
        y=y,
        n_rounds=cfg.n_rounds,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        random_state=cfg.random_state,
    )

    # Base value used during predict: mean of target
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
    print("As Master Lonn says: small steps, strong gradients.")


if __name__ == "__main__":
    main()

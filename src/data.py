from __future__ import annotations

from typing import Tuple
import numpy as np
from sklearn.datasets import make_classification, make_regression


def make_synthetic_classification(
    n_samples: int,
    n_features: int,
    n_informative: int,
    class_sep: float,
    random_state: int,
    test_size: float = 0.2,
):
    from sklearn.model_selection import train_test_split

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_clusters_per_class=2,
        class_sep=class_sep,
        random_state=random_state,
    )
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def make_train_val_test_classification(
    n_samples: int,
    n_features: int,
    n_informative: int,
    class_sep: float,
    random_state: int,
    holdout: float = 0.2,
):
    from sklearn.model_selection import train_test_split

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_clusters_per_class=2,
        class_sep=class_sep,
        random_state=random_state,
    )
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=holdout, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def make_sine_regression(
    n_samples: int,
    noise_std: float,
    random_state: int,
):
    rng = np.random.default_rng(seed=random_state)
    x = rng.uniform(-3.0, 3.0, size=n_samples)
    y_clean = np.sin(x)
    y = y_clean + rng.normal(0.0, noise_std, size=n_samples)
    x_grid = np.linspace(-3.0, 3.0, 600)
    y_grid_clean = np.sin(x_grid)
    return x.reshape(-1, 1), y, x_grid.reshape(-1, 1), y_grid_clean

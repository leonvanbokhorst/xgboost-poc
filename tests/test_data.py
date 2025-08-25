from __future__ import annotations

import numpy as np
import pytest

from src.data import make_synthetic_classification


def test_make_synthetic_classification():
    """
    Test the shape and properties of the synthetic classification data.
    """
    n_samples = 100
    n_features = 10
    test_size = 0.25

    X_train, X_test, y_train, y_test = make_synthetic_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=5,
        class_sep=1.0,
        test_size=test_size,
        random_state=42,
    )

    # Check types
    assert isinstance(X_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(y_test, np.ndarray)

    # Check shapes
    assert X_train.shape[0] == n_samples * (1 - test_size)
    assert X_test.shape[0] == n_samples * test_size
    assert X_train.shape[1] == n_features
    assert X_test.shape[1] == n_features
    assert y_train.shape[0] == n_samples * (1 - test_size)
    assert y_test.shape[0] == n_samples * test_size

    # Check total number of samples
    assert X_train.shape[0] + X_test.shape[0] == n_samples

#!/usr/bin/env python3
"""
make_synthetic.py

Generate small synthetic datasets (classification and regression) and persist as CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression


def make_classification_csv(out_dir: Path, n_samples: int, n_features: int, n_informative: int, class_sep: float, random_state: int) -> Path:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_clusters_per_class=2,
        class_sep=class_sep,
        random_state=random_state,
    )
    cols = [f"f{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y
    out = out_dir / f"synthetic_classification_{n_samples}x{n_features}.csv"
    df.to_csv(out, index=False)
    return out


def make_regression_csv(out_dir: Path, n_samples: int, n_features: int, noise: float, random_state: int) -> Path:
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state,
    )
    cols = [f"f{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y
    out = out_dir / f"synthetic_regression_{n_samples}x{n_features}.csv"
    df.to_csv(out, index=False)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic datasets and save to data/")
    p.add_argument("--out-dir", type=Path, default=Path("data"))
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    a.out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        make_classification_csv(a.out_dir, 3000, 12, 6, 1.2, a.seed),
        make_regression_csv(a.out_dir, 2000, 10, 10.0, a.seed),
    ]
    print("Saved:")
    for pth in paths:
        print("-", pth)


if __name__ == "__main__":
    main()

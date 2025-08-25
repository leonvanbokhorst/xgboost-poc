#!/usr/bin/env python3
"""
fetch_data.py

Download small public datasets and persist to CSV under data/.
Currently supports: Titanic (via seaborn), California Housing (sklearn).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_california_housing
import seaborn as sns


def save_titanic(out_dir: Path) -> Path:
    try:
        df = sns.load_dataset("titanic")
    except Exception as e:
        print(f"Error loading Titanic dataset: {e}")
        raise RuntimeError(
            "Failed to load Titanic dataset. Please check your network connection or seaborn installation."
        ) from e
    out = out_dir / "titanic.csv"
    df.to_csv(out, index=False)
    return out


def save_california_housing(out_dir: Path) -> Path:
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    df["MedHouseVal"] = data.target
    out = out_dir / "california_housing.csv"
    df.to_csv(out, index=False)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch and persist public datasets to data/")
    p.add_argument("--out-dir", type=Path, default=Path("data"))
    a = p.parse_args()

    try:
        a.out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating output directory {a.out_dir}: {e}")
        raise SystemExit(1)

    paths = [
        save_titanic(a.out_dir),
        save_california_housing(a.out_dir),
    ]
    print("Saved:")
    for pth in paths:
        print("-", pth)


if __name__ == "__main__":
    main()

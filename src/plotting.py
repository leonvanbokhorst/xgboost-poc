from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_auc_curves(evals_result: Dict[str, Dict[str, List[float]]], title: str, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))

    def get_series(split: str) -> List[float]:
        if split in evals_result:
            return evals_result.get(split, {}).get("auc", [])
        if split == "train":
            return evals_result.get("validation_0", {}).get("auc", [])
        if split == "valid":
            return evals_result.get("validation_1", {}).get("auc", [])
        return []

    train_auc = get_series("train")
    valid_auc = get_series("valid")

    if train_auc:
        plt.plot(train_auc, label="train AUC")
    if valid_auc:
        plt.plot(valid_auc, label="valid AUC")

    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("AUC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap


def compute_tree_shap(model, X_test):
    """Return (explainer, shap_values, expected_value) for tree models.

    Handles SHAP API variations and multi-class list outputs by selecting the
    first class and warning the user via print (kept minimal to avoid logger deps).
    """
    explainer = shap.TreeExplainer(model, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_test)
    if isinstance(shap_values, list):
        if len(shap_values) > 1:
            print(f"Warning: multi-class SHAP output (len={len(shap_values)}); using class 0.")
        shap_values = shap_values[0]
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, tuple)):
        expected_value = expected_value[0]
    return explainer, shap_values, expected_value


def shap_summary(shap_values, X, feature_names: List[str], out_dir: Path) -> None:
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary_bar.png", dpi=150, bbox_inches="tight")
    plt.close()


def shap_dependence(
    shap_values,
    X,
    feature_names: List[str],
    top_k: int,
    out_dir: Path,
) -> None:
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    n_features = len(feature_names)
    k = max(0, min(top_k, n_features))
    top_idx = np.argsort(mean_abs)[::-1][:k]

    for idx in top_idx:
        shap.dependence_plot(
            ind=idx,
            shap_values=shap_values,
            features=X,
            feature_names=feature_names,
            show=False,
            interaction_index="auto",
        )
        plt.tight_layout()
        plt.savefig(out_dir / f"dependence_{feature_names[idx]}.png", dpi=150, bbox_inches="tight")
        plt.close()


def shap_local_waterfall(
    explainer: shap.TreeExplainer,
    expected_value: float,
    shap_values,
    X,
    feature_names: List[str],
    instance_idx: int,
    out_dir: Path,
) -> None:
    idx = int(instance_idx)
    if idx < 0 or idx >= len(X):
        raise IndexError(f"instance-idx {instance_idx} out of bounds for test set size {len(X)}")
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[idx],
            base_values=expected_value,
            data=X[idx],
            feature_names=feature_names,
        ),
        show=False,
        max_display=14,
    )
    plt.tight_layout()
    plt.savefig(out_dir / f"local_waterfall_idx_{idx}.png", dpi=150, bbox_inches="tight")
    plt.close()

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier

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


def plot_roc_pr(
    y_test: np.ndarray,
    lr_prob: np.ndarray,
    xgb_prob: np.ndarray,
    out_dir: Path,
) -> None:
    """Plot and save ROC and Precision-Recall curves."""
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_prob)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_prob)
    roc_lr = roc_auc_score(y_test, lr_prob)
    roc_xgb = roc_auc_score(y_test, xgb_prob)

    p_lr, r_lr, _ = precision_recall_curve(y_test, lr_prob)
    p_xgb, r_xgb, _ = precision_recall_curve(y_test, xgb_prob)
    ap_lr = average_precision_score(y_test, lr_prob)
    ap_xgb = average_precision_score(y_test, xgb_prob)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(fpr_lr, tpr_lr, label=f"LogReg ROC AUC={roc_lr:.3f}")
    axes[0].plot(fpr_xgb, tpr_xgb, label=f"XGB ROC AUC={roc_xgb:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].legend()

    axes[1].plot(r_lr, p_lr, label=f"LogReg AP={ap_lr:.3f}")
    axes[1].plot(r_xgb, p_xgb, label=f"XGB AP={ap_xgb:.3f}")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "roc_pr_curves.png", dpi=150)
    plt.close(fig)


def plot_calibration(
    y_test: np.ndarray,
    lr_prob: np.ndarray,
    xgb_prob: np.ndarray,
    out_dir: Path,
) -> None:
    """Plot and save calibration curves."""
    prob_true_lr, prob_pred_lr = calibration_curve(y_test, lr_prob, n_bins=10, strategy="quantile")
    prob_true_xgb, prob_pred_xgb = calibration_curve(y_test, xgb_prob, n_bins=10, strategy="quantile")

    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect")
    plt.plot(prob_pred_lr, prob_true_lr, marker="o", label="LogReg")
    plt.plot(prob_pred_xgb, prob_true_xgb, marker="o", label="XGB")
    plt.title("Calibration Curve")
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "calibration.png", dpi=150)
    plt.close()


def plot_feature_importance(model: XGBClassifier, out_dir: Path) -> None:
    """Plot and save XGBoost feature importance."""
    importance = model.get_booster().get_score(importance_type="gain")
    if not importance:
        return
    items = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
    labels = [k for k, _ in items]
    values = [v for _, v in items]

    plt.figure(figsize=(8, max(3, len(values) * 0.4)))
    plt.barh(labels[::-1], values[::-1])
    plt.title("XGBoost Feature Importance (Gain)")
    plt.xlabel("Gain")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_importance.png", dpi=150)
    plt.close()

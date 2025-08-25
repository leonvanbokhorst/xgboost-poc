## XGBoost PoC — Practical Guide

This repository provides a structured, script-based proof of concept to explain how XGBoost works. It focuses on building intuition first and then validating with metrics, visualizations, and explainability.

---

## TL;DR

- Script-based, step-by-step demos for gradient boosting and XGBoost.
- Visualize training dynamics and feature effects (SHAP optional extra).
- Compare baselines, tune key hyperparameters, and avoid common pitfalls.
- Reproducible runs and simple CLI-style arguments.

---

## Objectives

- Explain gradient boosting and decision trees with clear visuals and small datasets.
- Show how XGBoost learns: residuals, learning rate, depth, regularization.
- Provide practical patterns: classification, regression, imbalanced classes, missing data.
- Keep modules short, focused, and easy to run from the command line.

---

## Repository Structure

```
.
├─ data/                         # Small sample datasets (downloaded/created by scripts)
├─ scripts/                      # Script-based, heavily commented demos
│  ├─ 01_intuition_boosting.py
│  ├─ 02_xgb_basic_classification.py
│  ├─ 03_xgb_explainability_shap.py          # (added when SHAP is installed)
│  ├─ 04_xgb_regularization_tuning.py        # (planned)
│  ├─ 05_xgb_advanced_topics.py              # (planned)
│  └─ 99_playground.py
├─ src/
│  ├─ data.py                    # dataset loaders, synthetic generators
│  ├─ train.py                   # train/eval utilities, early stopping
│  ├─ explain.py                 # SHAP, feature importance, partial dependence
│  └─ cli.py                     # simple CLI wrapper (planned)
├─ scripts_util/                 # optional helpers used by scripts
│  └─ plotting.py
├─ pyproject.toml                # managed by uv
├─ Makefile                      # optional shortcuts (uv run ...)
└─ README.md
```

---

## Datasets

- Titanic (classification): familiar, small, easy to explain.
- California Housing (regression): continuous target, useful for SHAP plots.
- Credit Default / Fraud (classification, imbalanced): demonstrate class weighting.
- Synthetic moons/blobs/regression: visualize decision boundaries and residuals.

---

## Environment Setup (uv)

Install `uv` (see `https://docs.astral.sh/uv/` for options). On macOS/Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create and install core dependencies:

```bash
uv venv -p 3.11 .venv
uv sync
```

Install SHAP extras (optional, for explainability script):

```bash
uv sync -E explain
```

Run scripts without activating the venv:

```bash
uv run python --version
```

---

## Running the Demos

```bash
# Intuition: manual residual fitting with decision stumps (regression)
uv run python scripts/01_intuition_boosting.py --n-samples 400 --n-rounds 30 --learning-rate 0.2

# Baseline vs XGBoost (classification) with metrics and plots
uv run python scripts/02_xgb_basic_classification.py --n-samples 4000 --max-depth 4 --n-estimators 400

# SHAP explainability (install extras first: uv sync -E explain)
uv run python scripts/03_xgb_explainability_shap.py --n-samples 4000 --top-k 6
```

Outputs are written under `runs/<timestamp>/<demo>/`.

---

## High-Level Plan

1. Intuition: Trees, residuals, and boosting (`scripts/01_intuition_boosting.py`)

- Fit decision stumps on residuals; visualize progressive improvement.
- Control rounds and learning rate via CLI.

2. Baseline vs XGBoost (`scripts/02_xgb_basic_classification.py`)

- Compare Logistic Regression vs XGBoost.
- Metrics: accuracy, ROC AUC, PR AUC; calibration; feature importance.

3. Explainability with SHAP (`scripts/03_xgb_explainability_shap.py`)

- Global: SHAP summary/bar; Local: example explanation; Dependence plots.

4. Regularization and tuning (`scripts/04_xgb_regularization_tuning.py`)

- Demonstrate overfitting; explore `eta`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`.
- Early stopping and compact comparison tables.

5. Advanced topics (`scripts/05_xgb_advanced_topics.py`)

- Missing values handling; imbalanced classes (`scale_pos_weight`); monotonic constraints; GPU training; custom objectives.

6. CLI + reproducibility (`src/`)

- `src/cli.py` to run training with config files; save models, metrics, and plots under `runs/`.

---

## Minimal Example

```python
from xgboost import XGBClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

X, y = make_classification(n_samples=2000, n_features=10, n_informative=5, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist"
)

model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

proba = model.predict_proba(X_val)[:, 1]
print("AUC:", roc_auc_score(y_val, proba))
```

---

## Success Criteria

- Short, visual scripts that clarify gradient boosting and XGBoost.
- Reproducible runs with saved models, metrics, and plots.
- Clear improvements vs baselines and evidence of effective tuning.
- Explainability artifacts for global and local understanding.

---

## Risks and Pitfalls

- Overfitting demonstrations require clear validation plots.
- SHAP can be slow on large datasets; use sampling.
- Too many hyperparameters can overwhelm; keep defaults sensible.

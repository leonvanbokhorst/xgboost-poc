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
│  ├─ 03_xgb_explainability_shap.py
│  ├─ 04_xgb_regularization_tuning.py
│  └─ 99_playground.py
├─ src/
│  ├─ data.py                    # dataset loaders, synthetic generators
│  ├─ plotting.py                # shared plotting helpers
│  └─ utils.py                   # timestamped dirs, grid helpers
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
uv sync --extra explain
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

# SHAP explainability (install extras first: uv sync --extra explain)
uv run python scripts/03_xgb_explainability_shap.py --n-samples 4000 --top-k 6

# Regularization and tuning (overfitting demo + early-stopping grid)
uv run python scripts/04_xgb_regularization_tuning.py
```

Or via the simple CLI:

```bash
# Intuition
uv run python -m src.cli intuition --n-samples 400 --n-rounds 30 --learning-rate 0.2
# Classification
uv run python -m src.cli classify --n-samples 4000 --max-depth 4 --n-estimators 400
# Explainability
uv run python -m src.cli explain --n-samples 4000 --top-k 6
# Tuning
uv run python -m src.cli tune
# Advanced topics
uv run python -m src.cli advanced --n-samples 4000 --pos-weight 3.0 --use-gpu false
```

Makefile shortcuts:

```bash
make sync
make intuition
make classify
make explain
make tune
make advanced
make clean-runs
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

5. Advanced topics (`scripts/05_xgb_advanced_topics.py`) — planned

- Missing values handling; imbalanced classes (`scale_pos_weight`); monotonic constraints; GPU training; custom objectives.

6. CLI + reproducibility (`src/cli.py`) — planned

- Orchestrate training with config files; save models, metrics, and plots under `runs/`.

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

---

## Learning Guide: Script-by-Script Explanations

### 1) 01_intuition_boosting.py — Residual Fitting Intuition (Regression)

- Purpose: Build mental model of gradient boosting as iterative error-correction.
- Core idea: Each new shallow tree fits current residuals; learning_rate controls how much we trust each correction.
- What it does:
  - Generates y = sin(x) + noise; trains depth-1 trees iteratively.
  - Visualizes prediction curves at several rounds and residual plots.
- What to look for:
  - Early rounds quickly reduce large errors; later rounds make smaller refinements.
  - Too many rounds or large learning_rate can start chasing noise.
- Key parameters:
  - n_rounds: more trees → more capacity; combine with small learning_rate.
  - learning_rate (eta): shrinkage; lower values need more rounds but generalize better.
  - max_depth: use 1 (stumps) for pure intuition; higher depth increases local fit.
- Pitfalls:
  - Overfitting with high depth/rounds and no validation.
  - Misinterpreting residuals: they should shrink towards zero over rounds.
- Exercises:
  - Compare learning_rate = 0.3 vs 0.05 at fixed n_rounds; inspect residual plots.
  - Increase noise_std and observe how predictions behave.

### 2) 02_xgb_basic_classification.py — Baseline vs XGBoost (Classification)

- Purpose: Show practical gains over simple baselines and how to evaluate them.
- What it does:
  - Compares Logistic Regression and XGBoost on synthetic classification.
  - Saves ROC, PR, calibration plots and feature importance (gain).
- What to look for:
  - ROC AUC vs PR AUC: PR is more sensitive in class-imbalanced settings.
  - Calibration: boosted models can be overconfident; consider calibration for production.
  - Feature importance: sanity-check top features match how data was generated.
- Key parameters:
  - max_depth, n_estimators, learning_rate: main capacity/speed knobs.
  - subsample, colsample_bytree: stochasticity that regularizes and speeds up.
  - tree_method: use "hist" for fast CPU training.
- Pitfalls:
  - Comparing only accuracy; prefer ROC AUC/PR AUC.
  - Reading feature importance as causality; it’s association within the learned model.
- Exercises:
  - Lower class_sep and compare baseline vs XGB gaps.
  - Change subsample/colsample_bytree to see stability-speed tradeoffs.

### 3) 03_xgb_explainability_shap.py — Global and Local Explainability (SHAP)

- Purpose: Build trust by explaining predictions globally (which features matter) and locally (why this prediction?).
- What it does:
  - Trains with proper train/val split; computes TreeSHAP on test data.
  - Saves SHAP summary (beeswarm + bar), dependence plots (top_k), and a local waterfall.
- What to look for:
  - Beeswarm: spread (impact) and color (feature value) patterns; interactions show as color gradients.
  - Dependence plots: monotonic or non-linear relationships; interaction hints via color.
  - Local waterfall: how base value + contributions sum to the prediction.
- Key parameters:
  - top_k: pick a small number to focus on clearest signals.
  - instance_idx: probe different examples (typical vs edge cases).
- Pitfalls:
  - Running SHAP on huge test sets; prefer sampling.
  - Multi-class SHAP: ensure the class dimension is handled correctly.
- Exercises:
  - Compare summaries on different random_state; does feature ranking remain stable?
  - Choose instances near the decision boundary for local plots.

### 4) 04_xgb_regularization_tuning.py — Overfitting, Early Stopping, Tuning

- Purpose: Demonstrate overfitting, then tune regularization to improve generalization.
- What it does:
  - Overfit demo: deep trees with many rounds; saves train vs valid AUC curves.
  - Grid/random search with early stopping; saves results CSV and curves for best model.
- What to look for:
  - Divergence of train vs valid curves in overfit settings.
  - Early stopping stabilizes valid metrics; best_iteration marks the sweet spot.
- Key parameters (regularization):
  - learning_rate (eta), max_depth, min_child_weight, subsample, colsample_bytree, reg_lambda.
  - early_stopping_rounds: patience; avoid setting too low.
  - random_search: sample N combos quickly for a fast sweep.
- Pitfalls:
  - Comparing models at different boosting rounds without early stopping is misleading.
  - Too small learning_rate without enough n_estimators underfits.
- Exercises:
  - Run tune with random_search=5 then =20; compare best valid AUC.
  - Vary min_child_weight and subsample together; inspect stability.

### 5) 05_xgb_advanced_topics.py — Missing, Imbalance, Monotonic, (Optional) GPU

- Purpose: Cover common production realities and constraints.
- What it does:
  - Introduces missing values; XGBoost routes NaNs via learned default directions.
  - Handles class imbalance via scale_pos_weight and evaluates with PR curves.
  - Enforces monotonic constraints (e.g., feature 0 increasing shouldn’t decrease risk).
  - Optional GPU via device="cuda" (requires CUDA-enabled environment).
- What to look for:
  - Precision-Recall improvement when using appropriate pos_weight.
  - Monotonic dependence curve: predicted probability should be non-decreasing/non-increasing per constraint.
- Key parameters:
  - pos_weight ≈ (negatives/positives) as a first heuristic.
  - monotone constraints vector; start small (one feature) and validate behavior.
  - use-gpu: true/false depending on environment.
- Pitfalls:
  - Overusing pos_weight can hurt calibration; verify PR and calibration.
  - Monotonic constraints can reduce accuracy if incorrectly specified.
- Exercises:
  - Sweep pos_weight from 1 to 10 and plot AP vs pos_weight.
  - Apply a negative monotone constraint and verify the direction in dependence plot.

---

## Study Tips and Mental Models

- Gradient boosting = repeated residual fixing: start simple, add nuance.
- Depth controls local expressiveness; learning_rate controls step size.
- Early stopping is your friend: compare at best_iteration, not at fixed n_estimators.
- Trust but verify with explainability: global patterns + local checks.
- Prefer PR AUC when positives are rare; use calibration when probabilities matter.

## Suggested Learning Path

1. Run intuition (01) to lock the mental model.
2. Compare baseline vs XGB (02) and learn the metrics.
3. Explain your model (03) to connect features with outcomes.
4. Tune (04) with early stopping; understand trade-offs.
5. Explore advanced constraints and imbalance (05).

## Quick Commands

- Intuition: `make intuition`
- Classification: `make classify`
- Explainability (requires extras): `make explain`
- Tuning: `make tune`
- Advanced topics: `make advanced`

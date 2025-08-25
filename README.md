## XGBoost PoC — A Jedi Padawan’s Field Guide

Welcome, Padawan. Under the watchful eye of Master Jedi Lonn (and a beeping BB‑8), you’ll learn the ways of gradient boosting until XGBoost bends to your will. May your AUC be high and your overfitting low.

---

## TL;DR

- Build a hands-on, story-driven PoC that explains how XGBoost works.
- Start with intuition (residuals as “errors the next tree must fix”), then code.
- Visualize trees, training dynamics, and feature effects with SHAP.
- Compare baselines, tune hyperparameters, and show common pitfalls.
- Deliver script-based demos (no notebooks required), a clean CLI, and reproducible runs.

---

## Objectives

- Explain gradient boosting and decision trees in clear, visual, bite-sized steps.
- Show exactly how XGBoost trains: residuals, learning rate, depth, regularization.
- Build intuition using tiny synthetic datasets before scaling up.
- Provide practical recipes: classification, regression, imbalanced classes, missing data.
- Make it delightful and ADHD-friendly: short modules, visual feedback, interactive sliders.

---

## Proposed Repo Structure

```
.
├─ data/                         # Small sample datasets (downloaded/created by scripts)
├─ scripts/                      # Script-based, heavily commented "notebooks"
│  ├─ 01_intuition_boosting.py
│  ├─ 02_xgb_basic_classification.py
│  ├─ 03_xgb_explainability_shap.py
│  ├─ 04_xgb_regularization_tuning.py
│  ├─ 05_xgb_advanced_topics.py
│  └─ 99_playground.py
├─ src/
│  ├─ data.py                    # dataset loaders, synthetic generators
│  ├─ train.py                   # train/eval utilities, early stopping
│  ├─ explain.py                 # SHAP, feature importance, partial dependence
│  └─ cli.py                     # simple CLI wrapper
├─ scripts_util/                 # optional helpers used by scripts
│  └─ plotting.py
├─ pyproject.toml                # managed by uv
├─ Makefile                      # optional shortcuts (uv run ...)
└─ README.md
```

---

## Datasets

- Titanic (classification): familiar, small, easy to explain.
- California Housing (regression): continuous target, nice for SHAP plots.
- Credit Default / Fraud (classification, imbalanced): demonstrate scale_pos_weight.
- Synthetic moons/blobs/regression: perfect for visualizing decision boundaries and residuals.

---

## Environment Setup (uv)

First, install `uv` (fast Python package manager):

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# then restart your shell or source your profile as prompted
```

Project setup using `uv` and `pyproject.toml`:

```bash
# Create a virtual environment (optional; uv can also run without activation)
uv venv -p 3.11 .venv

# Install project dependencies from pyproject.toml
uv sync

# Option A: activate the venv
source .venv/bin/activate

# Option B: run without activating
uv run python --version
```

Running scripts:

```bash
# Intuition demo
uv run python scripts/01_intuition_boosting.py
# Classification baseline vs XGBoost
uv run python scripts/02_xgb_basic_classification.py
# SHAP explainability
uv run python scripts/03_xgb_explainability_shap.py
```

Managing deps:

```bash
# Add a new dependency (writes to pyproject.toml)
uv add lightgbm
# Add a dev-only dependency
uv add --dev ruff
```

---

## High-Level Plan (Milestones)

### 1) Intuition: Trees, Residuals, and Boosting (script: scripts/01_intuition_boosting.py)

- Build a tiny synthetic dataset.
- Fit a decision stump; plot decision regions.
- Compute residuals; fit the next stump to fix the errors.
- Repeat a few rounds; visualize how ensemble improves.
- Add simple CLI args for num_rounds and learning_rate.

### 2) Baseline vs XGBoost (script: scripts/02_xgb_basic_classification.py)

- Baseline models: logistic regression, single DecisionTree.
- Train XGBoost classifier with early stopping.
- Show metrics: accuracy, ROC AUC, PR AUC; plot curves and calibration.
- Print a compact tree dump to see splits, gains, and cover.

### 3) Explainability with SHAP (script: scripts/03_xgb_explainability_shap.py)

- Global: SHAP summary beeswarm; feature importance comparison.
- Local: force plots for individual predictions (saved as images).
- Partial dependence / ICE for key features.
- Discuss trust: “why did this predict positive?” vs aggregates.

### 4) Regularization and Tuning (script: scripts/04_xgb_regularization_tuning.py)

- Demonstrate overfitting by intentionally cranking depth/rounds.
- Introduce eta, max_depth, min_child_weight, subsample, colsample_bytree, reg_alpha/lambda.
- Simple grid/random search + early stopping; compare via a compact table.
- Save training/validation curves and discuss bias-variance tradeoff.

### 5) Advanced Topics (script: scripts/05_xgb_advanced_topics.py)

- Missing values handling (XGBoost’s default direction).
- Imbalanced classes: scale_pos_weight, eval metrics beyond accuracy.
- Monotonic constraints (business logic imbued into the force).
- GPU training (if available) and training speed notes.
- Custom objectives/metrics: when and why to consider.

### 6) CLI + Reproducibility (src/ + scripts/)

- Simple `src/cli.py` to run training from the command line with JSON/YAML config.
- Save model, metrics, and plots into `runs/<timestamp>/`.
- `Makefile` shortcuts: `make data`, `make train`, `make explain`, `make demos` (via uv run).

### 7) Final Story and Slides

- A short slide deck that mirrors the narrative and includes key visuals.
- A 2–3 minute video/GIF capturing the demos and SHAP visuals.

---

## Quickstart: Minimal Example

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

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

proba = model.predict_proba(X_val)[:, 1]
print("AUC:", roc_auc_score(y_val, proba))
```

---

## Makefile Targets (planned)

```Makefile
# Install/refresh environment
sync:
	uv sync

# Run demos
intuition:
	uv run python scripts/01_intuition_boosting.py

classify:
	uv run python scripts/02_xgb_basic_classification.py

explain:
	uv run python scripts/03_xgb_explainability_shap.py
```

---

## Success Criteria

- Short, visual scripts that make boosting “click.”
- Reproducible training with saved models, metrics, and plots.
- Clear evidence of: baseline → boosted improvement; tuning → better generalization.
- Trust via SHAP and simple narratives that stakeholders can repeat.

---

## Risks and Pitfalls

- Overfitting demos may confuse without clear validation plots.
- SHAP on very large datasets can be slow; sample appropriately.
- Too many knobs can overwhelm; use sane defaults and progressive disclosure.

---

## Roadmap Checklist

- [ ] Create `pyproject.toml` and initialize uv environment
- [ ] Implement `scripts/fetch_data.py` and `scripts/make_synthetic.py`
- [ ] 01: Intuition script with residual visuals
- [ ] 02: Baseline vs XGBoost script with metrics and curves
- [ ] 03: SHAP global/local explainability script
- [ ] 04: Regularization/tuning script with compact comparison table
- [ ] 05: Advanced topics (missing, imbalance, monotonic, GPU) script
- [ ] CLI training pipeline with saved runs
- [ ] Slides + short demo video/GIF

---

## Credits

- Master Jedi Lonn for ancient gradient-boosting wisdom. BB‑8 for beeps of moral support.
- XGBoost: Chen & Guestrin (KDD’16) and the XGBoost maintainers.

May the gradient be with you.

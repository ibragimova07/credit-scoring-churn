# Bank Customer Churn — End-to-End ML Project

A lean, end-to-end binary classification project: EDA → preprocessing → training →
evaluation in notebooks, plus a Streamlit UI that scores customers using the saved
model. Built from the actual dataset in `data/raw/` (the original brief described
loan repayment, but the data only contains the churn target `Exited` — see
[`VISION.md`](VISION.md)).

## Structure

```
data/raw/                         # immutable source CSVs
data/processed/                   # stratified train/val splits (generated)
notebooks/
  01_eda.ipynb                    # distributions, imbalance, segment churn rates
  02_preprocessing.ipynb          # stratified split + train-only preprocessing
  03_training.ipynb               # 9-model CV benchmark (LogReg, trees, bagging, LightGBM, CatBoost, XGBoost, …)
  04_evaluation.ipynb             # held-out metrics, charts, importance, submission
models/
  pipeline.pkl                    # fitted preprocessor + best classifier
  metadata.json                   # metrics, threshold, feature order, confusion matrix
  leaderboard.csv                 # cross-validation comparison
  figures/                        # PNG charts consumed by the UI
app/streamlit_app.py              # single-app Streamlit UI
submission.csv                    # churn probabilities for the unlabelled test set
```

## Setup

Dependencies are managed with [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Run the notebooks

Execute in order (each depends on artifacts from the previous one). From the
project root:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_preprocessing.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/03_training.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/04_evaluation.ipynb
```

Or open them interactively:

```bash
uv run jupyter notebook
```

## Run the Streamlit app

After the notebooks have produced `models/`:

```bash
uv run streamlit run app/streamlit_app.py
```

The app has four sections: **Overview**, **Single prediction**, **Batch prediction**
(CSV upload) and **Performance**. All displayed metrics come from
`models/metadata.json`; the app never hardcodes numbers.

## Design system — Powder Blue Minimalism

The UI is styled as a light, minimalist analytics product:

- `app/design.py` holds the design tokens (CSS variables), the injected global
  stylesheet and small presentational helpers (`page_header`, `section_title`,
  `card_html`, `card_container`, `pill_html`, `fact_grid`).
- `.streamlit/config.toml` sets the matching native Streamlit light theme
  (background, surface, text, accent, Inter-compatible sans-serif).
- All four sections share the same heroes, cards, metrics, forms, tables,
  uploader, alerts, navigation and status pills. Interactions use 150–180ms
  transitions and respect `prefers-reduced-motion`.
- Text uses the specified palette. Small secondary text uses a slightly darker
  shade of `#718096` (`#5F6E82`) so it meets WCAG AA contrast on light surfaces;
  the original token is retained in the palette.

## Key modelling decisions

- **Target:** `Exited` — `1` = customer left the bank, `0` = stayed. ~20% positive.
- **Identifier columns dropped:** `id`, `CustomerId`, `Surname` (leakage / no signal).
- **No missing values** in `train.csv`; no duplicate rows.
- **Stratified 80/20 split** (seed 42); preprocessing fitted on train only via a
  `ColumnTransformer` (median impute + scale for numeric; mode impute + one-hot for
  categorical, `handle_unknown="ignore"`).
- **Candidate models:** Dummy baseline, Logistic Regression, Decision Tree, Random
  Forest, Extra Trees, HistGradientBoosting, LightGBM, CatBoost and XGBoost — all
  compared under the same 5-fold CV / preprocessing contract.
- **Selection** by 5-fold CV ROC-AUC with churn recall reported as the risk metric.
- **Evaluation** on the held-out validation split only; `test.csv` is unlabelled and
  used solely to produce `submission.csv`. No performance is claimed on it.
- **Churn recall is emphasised** — missing a churner costs the bank more than a false
  alarm. Precision, F1, ROC-AUC and PR-AUC are reported alongside.

## Integrity notes (carried over from `prodev-skill.md`)

- `data/raw/` is never modified.
- No target/future leakage: identifiers excluded; transforms fitted on train only.
- The whole fitted `Pipeline` is persisted, guaranteeing training/serving parity.
- UI metrics are loaded from real saved artifacts — no placeholder numbers.
- Results reflect the data that exists; no loan-repayment label was invented.

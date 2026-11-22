# Cirrhosis Patient Survival Prediction

An end-to-end multiclass clinical analytics project: immutable raw data → audited EDA →
leakage-aware preprocessing → cross-validated model benchmark → calibrated probabilities →
competition submission → interactive Streamlit dashboard.

**Contract:** predict `Status_C`, `Status_CL`, `Status_D` probabilities for each patient.
**Primary metric:** multiclass log loss. Accuracy and macro F1 are diagnostic only.

> Outputs are statistical estimates derived from synthetic clinical data. They are not a
> medical diagnosis, prognosis, or treatment recommendation.

## Structure

```
data/raw/multiclassificationtask (1)/   # immutable source CSVs (train, test, sample_submission)
data/processed/                         # cleaned stratified train/validation splits
notebooks/
  01_data_audit.ipynb                   # schema, integrity findings, N_Days leakage probe
  02_eda.ipynb                          # distributions, missingness, correlations, outcomes
  03_preprocessing.ipynb                # cleaning audit, stratified split, train-only preprocessing
  04_training.ipynb                     # 9-model 5-fold CV benchmark, OOF, learning curves
  05_calibration_selection.ipynb        # N_Days ablation, tuning, calibration, selection, submission
src/cirrhosis/                          # reusable, typed modules (data, features, preprocessing,
                                        # models, evaluation, inference, config, paths)
models/
  final_pipeline.joblib                 # end-to-end fitted pipeline (features → prep → classifier)
  metadata.json                         # real measured metrics and all design decisions
  leaderboard.csv, feature_importance.csv, oof_predictions.csv
  figures/                              # PNG figures reused by the dashboard
reports/                                # data quality, cleaning, experiments, final evaluation
app/
  design.py                             # light-theme design tokens, CSS and helpers
  streamlit_app.py                      # 4-section, plain-language dashboard
tests/                                  # unit, leakage, pipeline and UI smoke tests
submission.csv                          # competition predictions in exact template format
```

## Setup

Dependencies are managed with [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Reproduce the workflow

Run the notebooks in order from the project root (each consumes the previous one's artifacts):

```bash
for nb in 01_data_audit 02_eda 03_preprocessing 04_training 05_calibration_selection; do
  uv run jupyter nbconvert --to notebook --execute --inplace notebooks/$nb.ipynb
done
uv run streamlit run app/streamlit_app.py
```

Alternatively, `python tools/run_notebook.py 01_data_audit.ipynb` executes a single notebook
in place with `nbclient` (one file per invocation).

Verify the platform:

```bash
uv run python -m compileall -q src app tests
uv run pytest -v
```

## Application design

The interface follows a maximum-minimalism, universal-usability standard:

- **Four clearly named sections only:** Home, Patient Prediction, Data Insights,
  Model Information. Advanced tools (batch scoring, competition file, technical
  metrics) live in optional expanders, never in the main navigation.
- **Plain language everywhere.** ML terms and clinical abbreviations are explained;
  results use everyday descriptions of the three outcomes.
- **One primary action.** Patient Prediction has a single large
  **Generate Prediction** button and shows horizontal probability bars with
  written labels and percentages (never colour alone).
- **Accessible by default.** 16–18px body text, large click targets, strong
  contrast, visible labels, no icons-only controls, no time-sensitive steps,
  clear error messages and a light, muted palette that respects browser zoom.
- **Honest framing.** Every estimate page repeats that the output is an
  experimental statistical estimate and not a medical diagnosis.

## Measured results

All numbers below are read directly from `models/metadata.json` and the saved experiment
tables; the UI never hardcodes them.

| Item | Value |
|---|---|
| Labelled rows used (after removing 1 invalid label) | 14,999 |
| Best CV log loss (XGBoost, 5-fold, with `N_Days`) | 0.3829 |
| Selected model | LightGBM (tuned) |
| Selected model CV log loss | 0.3886 |
| Holdout log loss (out-of-sample, trained on 80% only) | 0.3727 |
| Holdout accuracy / macro F1 | 0.8507 / 0.6555 |
| Calibration winner | uncalibrated (isotonic and sigmoid both increased holdout log loss) |
| Submission | 10,000 rows, normalized probabilities, exact template column order |

**`N_Days` decision.** With `N_Days`, the top three models improved CV log loss by
approximately 0.009–0.016 (e.g. XGBoost 0.3829 → 0.3985 without it). The feature was therefore
retained for the competition submission, but this is documented as a *target-proxy* trade-off:
in the source clinical study `N_Days` is time-to-event and would not be available for a genuine
pre-registration prediction. A strictly clinical deployment should use the ablated contract
(available in `src/cirrhosis/features.py` via `include_n_days=False`) and should not interpret
these probabilities as patient-specific prognosis.

Permutation importance confirms clinically plausible signals (Bilirubin, Prothrombin, `N_Days`,
Age and Copper lead); see `models/feature_importance.csv`.

## Key modelling decisions

- **Target contract:** one invalid `Y` label (id 13789) is removed with a before/after audit;
  raw files are never modified. Implausible ages (<18 or >100 years) are flagged and clipped.
- **Identifiers:** `id` is sequential and disjoint between train/test → dropped.
- **Stratification:** 80/20 stratified holdout (seed 42). Five-fold stratified CV selects the
  model; the holdout is used once for calibration comparison and final evaluation. `test.csv`
  is used only to generate submission probabilities, never for selection.
- **Preprocessing:** median imputation + missing indicators + scaling (numeric), mode imputation
  + one-hot with `handle_unknown="ignore"` (categorical), all fitted on training rows only.
  The structured ~43% missingness is additionally captured as a `missing_count` feature.
- **Feature engineering:** age in days → years with a clipping flag, Bilirubin/Albumin ratio,
  log transforms of skewed analytes. Implemented as a sklearn transformer inside the persisted
  pipeline so training and serving apply an identical contract.
- **Candidates:** dummy prior, logistic regression, decision tree, random forest, extra trees,
  HistGradientBoosting, XGBoost, LightGBM and CatBoost — all under one preprocessing contract.
- **Selection:** lowest validation log loss; ties within 0.005 broken by smaller/faster artifact.
  Probability calibration (sigmoid/isotonic) is compared on the holdout, not assumed.
- **`N_Days` ablation:** `N_Days` is a time-to-event variable in the source study and is strongly
  associated with the outcome. The project runs a documented with/without comparison and keeps
  the feature only if it measurably improves log loss; the decision is stored in `metadata.json`.

## Integrity notes (carried over from `prodev-skill.md`)

- Raw data is read-only; cleaned data is written to `data/processed/`.
- No target, future, identifier or test-set information is used in feature fitting or selection.
- All displayed metrics, tables and charts come from saved artifacts — no hardcoded numbers.
- The full fitted pipeline is persisted, guaranteeing training–serving parity.
- Submission rows are checked for exact column order, preserved IDs, finite values and
  per-row normalization before being written.

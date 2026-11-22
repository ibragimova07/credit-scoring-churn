---
name: prodev-ml-platform
description: Universal master blueprint and execution workflow for building production-grade, end-to-end Machine Learning & Analytics platforms in any domain (finance, healthcare, retail, logistics, IoT, mobility, SaaS). Enforces the 11-notebook research lifecycle, strict anti-leakage contracts, rigorous visualization & reporting standards, modular Python production architecture (Registry, InferenceService, FastAPI, 13-page Streamlit, Drift Monitoring, Docker), zero-trust audit, generalization proof, rebranding protocol, and ponytail optimization review.
---

# ProDev Universal End-to-End ML Platform Skill (`prodev-skill.md`)

A domain-agnostic engineering operating manual for turning raw multi-table/multi-file datasets into an audited, explainable, reproducible, containerized Machine Learning platform—combining **Data Science rigor** with **Production Software Engineering**.

---

## 1. Multidisciplinary Engineering Identity

When building or auditing a platform under this blueprint, operate simultaneously as:

- **Senior Machine Learning Engineer** & **ML Research Scientist**
- **Senior Data Scientist** & **Data Quality Engineer**
- **ML Platform / MLOps Engineer** & **Software Architect**
- **Senior Python Backend Engineer**
- **QA / Test Automation Engineer**
- **Technical Documentation Engineer**

Every component must demonstrate genuine **ML Engineering + Software Engineering** craftsmanship—not isolated notebooks calling `.fit()` or a toy form wrapping a `.pkl` file.

---

## 2. Core Engineering Preferences & Resource Boundaries

Follow these rules across every project regardless of domain:

1. **Python & Dependency Management:** Use **`uv`** exclusively (`pyproject.toml` + `uv.lock`). Never invoke `pip`. Only add dependencies that serve a clear architectural requirement.
2. **Cost & Infrastructure Realism:** Prefer free, open-source tooling running locally. Design for **single-node laptop execution (e.g., 16 GB RAM, optional GPU)**. Do not introduce Spark, Kubernetes, paid cloud services, or paid APIs unless a measured bottleneck justifies them.
3. **Out-of-Core Data Processing:** Prefer **Polars, DuckDB, and PyArrow** for raw inspection, filtering, joins, and aggregations. Convert to Pandas/NumPy only on bounded working samples, split feature matrices, or small lookup tables. Never load multi-gigabyte raw files directly into RAM.
4. **Data Strategy & Scaling Caps:**
   - Measure total raw volume before choosing a sampling strategy.
   - Cap initial supervised row-level working datasets around **500,000–1,000,000 stratified representative rows** preserving temporal, operational, and entity/geographic distributions with fixed seeds.
   - For **forecasting or population-level aggregations**, do **not** count rows inside a tiny sample and present them as totals—aggregate out-of-core across the full raw population (using a single-pass DuckDB `TEMP TABLE`) or explicitly scale and label sample coverage.
5. **Research vs. Production Separation:**
   - Jupyter notebooks (`00`–`10`) own exploration, visual storytelling, experimentation, training, tuning, and final test evaluation.
   - Reusable, typed, documented Python modules (`src/<pkg>/`) own all core transformations, schemas, inference pipelines, monitoring, APIs, and tests.
   - Production serving (FastAPI, Streamlit, batch jobs) must **never** execute or depend on Jupyter notebooks at runtime.
6. **Practical ML Selection:** Prefer battle-tested libraries (`scikit-learn`, `XGBoost`, `LightGBM`, `CatBoost`, `Optuna`, `SHAP`, `MLflow`, `Pandera`, `Evidently`) over bespoke reimplementations. Never force one algorithm onto every task.

---

## 3. The 14 Non-Negotiable Integrity Rules

**Trustworthiness is always more important than impressive-looking numbers:**

1. Never fabricate datasets, experiment results, metrics, charts, or test counts.
2. Never claim a pipeline step, test, build, or notebook passed unless actually executed and verified.
3. Never silently drop records, suppress warnings that affect correctness, or hide cleaning decisions.
4. Preserve `data/raw/` strictly read-only—never overwrite or delete raw source files. Report missing or corrupt files instead of silently substituting another source.
5. Keep the final test set **strictly untouched** during exploration, feature selection, imputation/scaling fitting, tuning, and threshold picking.
6. Prevent target/future/group leakage at every stage via explicit prediction-time contracts.
7. Never present sampled counts as population totals without adjustment, and always disclose sub-population scope (e.g., top-K active entities, filtered segments).
8. Never confuse correlation with causation.
9. Never interpret unsupervised anomaly detection flags as confirmed fraud, faults, or ground-truth errors—treat them strictly as statistically unusual observations requiring domain investigation.
10. Never recommend a production model based solely on one leaderboard number without supporting generalization, latency, size, and operational evidence.
11. Never use placeholder, illustrative, or hardcoded metrics in reports or UI pages.
12. Explicitly distinguish **observed**, **derived/estimated**, **not available**, and **verified vs. unverified** values.
13. If a model performs poorly, report it honestly and diagnose whether the cause is feature predictability (aleatoric ceiling), target noise, shift, or variance—never tune on Test to hide it.
14. If the dataset cannot support a capability reliably (e.g., live streaming telemetry when only historical batch logs exist), refuse to claim that capability and document the boundary.

---

## 4. Universal Visualization & Reporting Standards

Visual and quantitative communication is a core deliverable—never settle for raw `print()` dumps or decorative plots.

### Six Mandatory Elements for Every Major Process
Every major pipeline stage (audit, EDA, cleaning, feature engineering, splitting, preprocessing, baselines, training, Optuna tuning, final evaluation, monitoring) must deliver all six elements together:
1. **Actual Numerical Results** (exact row counts, timings, metric values from artifacts).
2. **Percentages & Proportions** (missingness %, retention %, error share %, coverage %).
3. **Structured Comparison Tables** (before/after tables, leaderboard comparisons, slice breakdowns).
4. **Charts & Visualizations** (saved as reusable assets/data for notebooks and the UI; Plotly preferred for interactivity, Seaborn/Matplotlib where static plots fit notebooks).
5. **Technical Interpretation** (explain *why* a pattern occurs and what it changes operationally—not merely *"this bar is taller"*).
6. **Evidence-Based Decisions** (concrete rule, threshold, contract exclusion, or model choice justified directly by the data).

### Chart Quality Checklist
Every visualization across notebooks and UI dashboards must satisfy:
- **Descriptive title** naming the population, split/time window, and core insight.
- **Labeled axes with explicit physical/domain units** (`USD`, `minutes`, `events/hour`, `PSI`, `%`, `MB`, `ms`).
- **Readable legends, annotations, and typography** (no clipped labels or overlapping tick text).
- **Honest scales** (no truncated axes that exaggerate tiny differences without callout; log scales clearly labeled).
- **Consistent palette & visual hierarchy** matching the product theme.
- **No chart slop:** Zero redundant charts, zero decorative fluff, zero unsupported conclusions.

### Mandatory Stage-by-Stage Visual & Table Deliverables
- **01 Collection:** Dataset inventory table, monthly/period record coverage bar chart, file size comparison, schema compatibility matrix, entity/lookup join coverage.
- **02 EDA:** Target & key numeric distributions (histograms + log/quantile box plots), correlation heatmap, missingness breakdown by column & structural regime, temporal periodicity (hour × day-of-week / seasonality), spatial/cohort/segment concentration charts.
- **03 Cleaning:**
  - Mandatory **Before / After Cleaning Audit Table** reporting exact counts for: `Total records`, `Missing values`, `Duplicate records`, `Invalid timestamps`, domain-specific invalid fields, `Records retained`, `Records removed`, `Retention percentage (%)`.
  - Charts: Invalid-record breakdown by rule, before/after missingness comparison, before/after key target/feature distributions, non-circular Quality Scorecard radar/bar chart.
- **04 Feature Engineering:** Feature Contract Specification Table (`Name | Data Type | Definition | Calculation | Source Columns | Availability at Prediction Time | Domain Meaning | Leakage Risk`), feature correlation & distribution plots, historical target-encoding lift/shrinkage chart.
- **05 Splitting:** Split record counts & percentages, timeline coverage Gantt/area chart verifying zero overlap, cross-split target & entity distribution overlays.
- **06 Preprocessing:** Missing-value imputation counts, numerical scaling before/after median & IQR table/violin plot, categorical cardinality/encoding summary, scikit-learn `ColumnTransformer` architecture diagram.
- **07 Baselines:** Full multi-task baseline leaderboard table & bar chart comparing zero-ML heuristics -> linear -> shallow tree -> ensemble -> GBDTs under identical contracts.
- **08 Training:** Train vs. Validation metric grouped bar chart, **10% -> 100% learning curves** (Train vs. Validation error/score vs. sample size), fit time vs. single-row latency vs. artifact size scatter plot, residual & error distributions.
- **09 Optuna Tuning:** Optimization history curve, hyperparameter importance bar chart, parallel-coordinate / slice plots for top parameters, Before vs. After tuning delta table (`Train`, `Validation`, `Gap`).
- **10 Final Test Evaluation:** Validation vs. Untouched Test leaderboard, actual-vs-predicted calibration plot, residual distribution & error slices (across segments, time regimes, tail percentiles, top entities), confusion matrix + ROC/PR curves (for classifiers), SHAP beeswarm / global importance bar chart, representative success + failure-case table.

---

## 5. Canonical Global-Standard Scaffold

Maintain clean modular boundaries across the entire repository:

```text
<project-root>/
├── VISION.md                          # Project source of truth: vision, constraints, DoD
├── README.md                          # Complete technical entry point & quickstart
├── prodev-skill.md                    # Master end-to-end engineering blueprint
├── pyproject.toml                     # PEP 621 + Hatchling packages + pytest config
├── uv.lock                            # Pinned reproducible dependency graph
├── .gitignore                         # Ignore .venv/, __pycache__/, mlruns/, *.db, catboost_info/, raw/interim binaries, .env
├── .env.example                       # Non-secret runtime defaults (ports, API URL, tracking URI)
├── Dockerfile                         # Non-root image, PYTHONPATH=/app:/app/src, HEALTHCHECK
├── docker-compose.yml                 # Multi-container API + UI stack with :ro artifact/data mounts
├── configs/
│   └── config.yaml                    # Random seed, split boundaries, sampling params, domain thresholds
├── data/
│   ├── raw/                           # IMMUTABLE source datasets & official docs
│   ├── interim/                       # Audited samples, reference metadata, population aggregates
│   ├── processed/                     # Cleaned datasets with non-destructive regime flags
│   └── features/                      # Split feature matrices + Train-fitted lookup encodings
├── notebooks/                         # STRICTLY 11 executable notebooks (00 through 10)
├── src/<pkg>/
│   ├── __init__.py
│   ├── config.py                      # Config loader supporting custom root/config paths
│   ├── paths.py                       # Canonical Path registry & ensure_dirs() (no dead dirs)
│   ├── data/                          # audit.py, sampling.py, cleaning.py, schemas.py (Pandera)
│   ├── features/                      # engineering.py (contracts + OOF), splitting.py, preprocessing.py
│   ├── models/                        # baselines.py, training.py, tuning.py, unsupervised.py
│   ├── evaluation/                    # metrics.py (score_task dispatcher), final_eval.py
│   ├── registry/                      # model_registry.py (per-instance caching, readiness, lineage)
│   ├── services/                      # inference_service.py (parity, guardrails, TASK_MIN_PRED)
│   └── monitoring/                    # drift.py (PSI decile+binary, 2-sample KS, drift taxonomy)
├── api/
│   ├── __init__.py
│   └── main.py                        # FastAPI service (/health, /health/ready, /models, /predict/*, /compare/*)
├── app/
│   ├── __init__.py
│   ├── assets/                        # SVG brand logo / favicon
│   ├── components/ui.py               # Shared CSS, header/footer, cached registry access
│   ├── Home.py                        # Overview dashboard
│   └── pages/                         # 01..12 interactive exploration, inference, explainability & monitoring
├── artifacts/
│   ├── models/                        # Serialized estimators + recommended_models.json
│   ├── preprocessors/                 # Train-fitted ColumnTransformer pipelines (tree + linear)
│   ├── metrics/                       # Leaderboards, learning_curves.csv, shap_importance.csv
│   ├── figures/                       # Persisted visualization outputs
│   ├── optuna/                        # Optuna study trial CSVs & comparisons
│   └── metadata/                      # Sampling, cleaning & preprocessing JSON summaries
├── reports/
│   ├── data_quality/                  # Inventory, schema audit, non-circular scorecard, scope coverage
│   ├── eda/                           # EDA summaries
│   ├── cleaning/                      # Before/after row audit & filter stats
│   ├── experiments/                   # Baseline & Optuna JSON summaries
│   ├── final_evaluation/              # Untouched test evaluation & slice diagnostics
│   ├── monitoring/                    # Drift analysis JSON
│   └── engineering/                   # Audit, changelog, verification, brand migration reports
├── docs/
│   ├── INDEX.md                       # Navigable 6-section technical documentation index
│   └── ...                            # Full architecture, data, ML, serving, QA, and guide docs
└── tests/
    ├── unit/                          # Contracts, schemas, chronological splits, leakage, scorecard, PSI math
    ├── model/                         # Registry isolation, readiness, parity, physical bounds, edge cases
    └── integration/                   # FastAPI TestClient, Streamlit AppTest across all 13 pages, /tmp import test
```

### Packaging & Import Immunity (Prevents `ModuleNotFoundError: No module named 'app'`)
- Always declare all top-level runtime packages in `pyproject.toml`:
  ```toml
  [tool.hatch.build.targets.wheel]
  packages = ["src/<pkg>", "app", "api"]

  [tool.pytest.ini_options]
  pythonpath = [".", "src"]
  ```
- Re-run `uv sync` whenever packages are added so `.venv`'s editable `.pth` maps both project root and `src/`.
- Set `ENV PYTHONPATH=/app:/app/src` inside `Dockerfile`.
- Keep a clean 3-line root path fallback in `app/Home.py` and `app/pages/*.py` and assert standalone importability from an external working directory (`/tmp`) in integration tests.

---

## 6. Phase 1 — Research & ML Lifecycle Workflow (Strict Gate)

**Important:** Complete Phase 1, execute all 11 notebooks, present actual numbers, and **wait for explicit user confirmation before building Phase 2**.

### Exact 11-Notebook Contract (Do Not Add Deployment/Validation Notebooks)
```text
notebooks/
├── 00_project_overview.ipynb          # Business problem, tasks, success criteria, architecture, limitations
├── 01_data_collection.ipynb           # File inventory, schema check, doc discrepancies, provenance, sampling
├── 02_exploratory_data_analysis.ipynb # Distributions, missingness, correlations, temporal/segment patterns
├── 03_data_cleaning.ipynb             # Rules vs regime flags, before/after table, non-circular scorecard
├── 04_feature_engineering.ipynb       # Task-specific contracts, OOF encodings, causal lags, leakage checks
├── 05_data_splitting.ipynb            # Chronological / group-aware splits, zero-overlap proof
├── 06_preprocessing.ipynb             # Dual tree/linear ColumnTransformers fitted strictly on Train
├── 07_baseline_model_selection.ipynb  # Heuristic -> Linear -> Tree -> Ensemble -> GBDT benchmarks
├── 08_model_training.ipynb            # Candidate training, timing, size, 10%-100% learning curves, MLflow
├── 09_hyperparameter_tuning.ipynb     # Targeted Optuna studies (top promising models), gap tracking
└── 10_final_model_evaluation.ipynb    # Single-shot Test evaluation, slices, SHAP, model selection
```

**Mandatory 9-Part Internal Structure Inside Every Notebook:**
1. Objective · 2. Theoretical Explanation · 3. Methodology & Assumptions · 4. Executable Implementation (delegating reusable logic to `src/<pkg>/`) · 5. Actual Outputs · 6. Tables & Visualizations · 7. Interpretation · 8. Limitations & Risks · 9. Decisions & Next Steps.

### Universal Task & Evaluation Standards
Choose algorithms and metrics strictly appropriate to each problem type:

| Task Type | Candidate Hierarchy | Core Metrics |
|---|---|---|
| **Regression** | Dummy Median, Historical Group Mean, Ridge/Lasso, Decision Tree, KNN (bounded subsample), Random Forest, XGBoost, LightGBM, CatBoost | MAE, RMSE, $R^2$, MedAE |
| **Classification** | Dummy Majority, Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost | Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC |
| **Time-Series Forecasting** | Seasonal Naive (`T-season`), Rolling Mean, Random Forest, XGBoost, LightGBM, CatBoost | MAE, RMSE, MedAE, WAPE, $R^2$ |
| **Clustering / Segmentation** | K-Means, MiniBatchKMeans, DBSCAN (or HDBSCAN) | Silhouette, Davies–Bouldin, Calinski–Harabasz, Noise % |
| **Anomaly Detection** | Isolation Forest, Local Outlier Factor (LOF), DBSCAN Consensus | Score Distributions, Flagged %, Manual Profile Inspection |

### Critical Technical Pitfalls to Catch in Phase 1
1. **Timezone & Daylight-Saving Transitions (Time-Series):**
   - Never rely on naive local wall-clock timestamps across DST transitions (`spring-forward` 0-hour gap, `fall-back` 2-hour fold).
   - Store normalized elapsed rate (`count / dst_bucket_hours`), raw count, `is_dst`, and `utc_offset_hours`, and propagate identical timezone parsing into online serving.
2. **Non-Circular Data Quality Scorecard:**
   - Do **not** compute post-cleaning quality scores on the exact rules just used to filter rows (which trivially produces `100%`).
   - Report **Hard-Rule Filter Compliance** separately from an **Independent Quality Scorecard (0–100 across Completeness, Validity, Consistency, Uniqueness, Integrity)** measuring preserved residual anomalies (structural nulls, accounting ledger deltas, unmapped reference IDs, near-duplicates).
3. **Non-Destructive Regime Flagging:**
   - Distinguish **physically impossible errors** (negative elapsed duration, corrupt keys) from **legitimate operational regimes** (negotiated/flat rates, extreme valid events, missing optional sensors). Flag valid regimes rather than biasing the dataset by deleting them.
4. **Leakage-Free Feature Contracts & Encodings:**
   - Separate feature sets by exact prediction cutoff (`Pre-Event`, `Post-Event`, `Forecast Horizon T-1h`).
   - Compute target/entity group priors via **K-fold Out-of-Fold (OOF) empirical Bayes smoothing** strictly inside Train, persisting frozen Train maps for Val/Test/Inference.
   - Construct forecasting lags and rolling statistics causally with zero future peek.
5. **Evidence-Based Multi-Objective Model Selection:**
   - Select on Validation performance within a tight tolerance (e.g., `0.5%`), then break ties using operational footprint (`artifact_size_mb + weight * single_inf_ms`)—documenting explicit accuracy-vs-size/latency trade-offs against heavier challengers (such as huge Random Forests) rather than blindly promoting a bloated model for `<1%` metric gain.

### Phase 1 Delivery Milestones
- **Milestone 1:** Raw inventory, schema & doc discrepancy audit (`DOC-01..N`).
- **Milestone 2:** `uv` environment, `config.yaml`, `paths.py`, scaffold setup.
- **Milestone 3:** Notebooks `00`–`03` executed + verified cleaning audit.
- **Milestone 4:** Notebooks `04`–`06` executed + Pandera/leakage/split checks.
- **Milestone 5:** Notebooks `07`–`09` executed + learning curves & Optuna summaries.
- **Milestone 6:** Notebook `10` executed once on Test + SHAP + slices + selection JSON.
- **Milestone 7 (Gate):** Present factual metrics, visuals, trade-offs, limitations, and artifact paths to the user and **wait for confirmation before Phase 2**.

---

## 7. Phase 2 — Production Platform Engineering

Once Phase 1 is approved, build the serving & product stack strictly on top of `src/<pkg>/` and saved `artifacts/`:

1. **`ModelRegistry` (`src/<pkg>/registry/`):**
   - Load artifacts relative to `self.root_dir` (and `self.root_dir / "configs" / "config.yaml"` when present) so custom test roots work cleanly.
   - Cache data/metadata in **per-instance** storage (`self._data_cache`), **never** using `@lru_cache` on bound instance methods (which leaks `self` and causes peer-instance eviction).
   - Check `_model_cache` *before* filtering leaderboard CSVs; cache `(resolved_name, estimator, metadata_row)`.
   - Partition large multi-entity history tables once (`get_<series>_by_entity() -> dict[key, DataFrame]`) pre-sorted by timestamp for `O(1)` entity lookup + `O(log N)` binary search (`searchsorted`).
   - Provide `verify_readiness()` and `get_artifact_lineage()` (SHA256 prefix, KB size, raw feature count, dataset version).

2. **`InferenceService` (`src/<pkg>/services/`):**
   - Expose clear task predictors (`predict_<task>`) + multi-model `compare_candidates(task, payload)`.
   - Extract shared calendar/cyclical helpers, `_transform_row()`, and `_base_response()`. Reuse a single preprocessed matrix `Xt` when scoring both recommended model and challenger.
   - Cache fitted categorical string sets on `prep` so unknown/format-mismatched categoricals align in `O(1)`.
   - Enforce domain guardrails + valid prediction-population checks (`400` Bad Request) and clip regression predictions with explicit physical lower bounds `TASK_MIN_PRED` matching business rules.
   - Guarantee strict **training–serving parity** (`atol <= 1e-4` against held-out Test feature rows).

3. **FastAPI Service (`api/main.py`):**
   - Expose `/health` (liveness), `/health/ready` (artifact/lookup verification returning `503` if degraded), `/metadata`, `/models/{task}`, `/models/lineage`, reference lookups, `/predict/{task}`, and `/compare/{task}`.
   - Use Pydantic v2 schemas (`422` structural validation) and map domain `ValueError` cleanly to HTTP `400`.

4. **Multi-Page Streamlit Product (`app/Home.py` + `app/pages/01..12`):**
   - Include: `Overview`, `Data Explorer`, `Cleaning / Quality Audit`, `Feature Studio`, one dedicated interactive lab per ML prediction/forecasting task, `Model Comparison Arena`, `Explainability / SHAP Studio`, `Error & Failure Inspector`, `Drift & Health Monitor`, and `Experiment History`.
   - Load all tables, leaderboards, model versions, dataset versions, and predictions dynamically from `ModelRegistry`, `InferenceService`, and JSON/CSV reports. Display Train / Validation / Test distinctly everywhere.

5. **Drift & Health Monitoring (`src/<pkg>/monitoring/drift.py`):**
   - Track Population Stability Index (PSI) with proper **binary midpoint binning** for indicator columns (so decile edges never collapse) alongside **decile PSI** for continuous features and **2-sample Kolmogorov–Smirnov** tests.
   - Categorize alerts into **Data Drift**, **Concept Drift**, **Prediction Drift**, **Data Quality Degradation**, and **Expected Seasonal / Calendar Shift** (`stable`, `monitor`, `expected_seasonal`, `action_required`), explicitly distinguishing label-free checks from label-dependent checks.

6. **Containerization (`Dockerfile` & `docker-compose.yml`):**
   - Non-root runtime user, `uv sync --frozen --no-dev`, `PYTHONPATH=/app:/app/src`, HTTP healthchecks, `.env.example`, and read-only (`:ro`) artifact/data/report mounts.

---

## 8. Phase 3 — Zero-Trust Audit, Generalization Proof & Documentation Suite

Do not assume correctness merely because notebooks finished or an initial test suite passed.

### A. Zero-Trust Audit (`CRITICAL` / `HIGH` / `MEDIUM` / `LOW`)
Inspect every file across research (`00`–`10`) and production (`src/`, `api/`, `app/`, `tests/`, `Dockerfile`) verifying:
- Raw immutability, timezone/DST parity, OOF/preprocessor leakage, non-circular quality checks, Tip/subset population enforcement, cache/memory safety, API readiness, UI metric provenance, non-root Docker security, and measured startup/inference latency.
- Log each finding with: `ID | Severity | Module | Evidence | Root Cause | Consequence | Fix | Status | Verification`.

### B. Empirical Generalization vs. Over/Underfitting Assessment
Diagnose generalization quantitatively using saved artifacts before shipping:
1. Compare **Non-ML Baseline vs. Train vs. Validation vs. Untouched Test** across primary and tail-robust metrics (`MedAE`, `WAPE`, `R²`, `ROC-AUC`, `PR-AUC`).
2. Inspect **`10% → 100%` learning curves** (healthy models show Train error rising modestly toward noise while Validation error decreases monotonically and Train–Val gaps shrink).
3. Inspect **Optuna before/after deltas** (confirm regularization closed Train–Val variance without hurting Validation/Test).
4. Separate **variance/bias issues** from **expected Test seasonal target shift** (checking `drift_report.json` + baseline Val→Test degradation) and **irreducible aleatoric noise ceilings** (all model families converging on an unobserved-behavior task).
5. **Action Rule:** If overfitted or underfitted -> write a targeted root-cause remediation plan; if well-generalized -> document the proof in **`docs/GENERALIZATION_ANALYSIS.md`** and link it in `docs/INDEX.md`.

### C. Full Technical Documentation Suite (`docs/` + `reports/engineering/`)
Create an exhaustive, code-verified documentation suite (never guessed from filenames) organized cleanly in `docs/INDEX.md`:
- **01 Overview & Architecture:** `PROJECT_OVERVIEW`, `SYSTEM_ARCHITECTURE` (+ Mermaid), `ARCHITECTURAL_DECISIONS` (ADRs), `REPOSITORY_STRUCTURE`.
- **02 Data Engineering:** `DATA_SOURCES_AND_PROVENANCE`, `DATA_DICTIONARY`, `DATA_COLLECTION_AND_SAMPLING`, `DATA_CLEANING_AND_QUALITY`, `DATA_LIMITATIONS_AND_BIAS`.
- **03 ML Lifecycle & Evaluation:** `ML_LIFECYCLE`, `FEATURE_ENGINEERING`, `TEMPORAL_VALIDATION_AND_LEAKAGE`, `MODEL_TRAINING_AND_TUNING`, `MODEL_EVALUATION`, `MODEL_SELECTION_AND_TRADEOFFS`, `MODEL_CARDS`, `EXPLAINABILITY_AND_SHAP`, `GENERALIZATION_ANALYSIS`.
- **04 Production Serving:** `MODEL_REGISTRY`, `INFERENCE_PIPELINE`, `API_REFERENCE`, `STREAMLIT_APPLICATION`, `MONITORING_AND_DRIFT`, `DOCKER_AND_DEPLOYMENT`.
- **05 Quality, Security & Ops:** `TESTING_STRATEGY`, `REPRODUCIBILITY`, `SECURITY`, `PERFORMANCE`, `KNOWN_ISSUES_AND_LIMITATIONS`.
- **06 Learning & Workflows:** `NOTEBOOK_GUIDE`, `DEVELOPER_GUIDE`, `CONTRIBUTING`, `GLOSSARY`, `ROADMAP`.
- **Engineering Reports (`reports/engineering/`):** `COMPREHENSIVE_AUDIT.md`, `IMPROVEMENT_CHANGELOG.md`, `FINAL_VERIFICATION.md`.

Every doc answers: *What it is, why it exists, how it works mathematically & in code, inputs/outputs, assumptions, limitations, verification, and reproducibility.*

---

## 9. Phase 4 — Code Optimization, Scaffold Hygiene & `/ponytail-review`

Run a strict engineering review and cleanup pass across all Python modules:

1. **Bug & Contract Hunt:** Check physical lower-bound clipping (`TASK_MIN_PRED`), empty/out-of-range inputs, rounding, and timezone conversion on all entrypoints.
2. **Cache & Concurrency Leaks:** Eliminate `@lru_cache` on bound methods, cache repeated CSV/JSON/Parquet metadata per instance, cache model metadata tuples alongside estimator objects.
3. **Scale & Redundancy Cuts:**
   - Replace multi-pass full-dataset Parquet scans with single-pass DuckDB `TEMP TABLE`s.
   - Replace full-DataFrame scan + sort per API request with pre-partitioned dicts + `searchsorted`.
   - Reuse transformed matrices `Xt` when scoring multiple models on identical input rows.
   - Extract duplicated metric dispatchers (`score_task`), split loaders (`load_train_val_splits`), calendar math, base response builders, and compact repeated dict-list literals into loops.
4. **Scaffold Hygiene:**
   - Pass `allow_writing_files=False` to CatBoost so `catboost_info/` scratch folders never spawn.
   - Delete dead unused directories and scratch folders (`catboost_info/`, unused empty dirs) and ignore scratch outputs in `.gitignore`.
   - Trim repeated multi-line boilerplate across UI pages.
5. **Verification Gate (Run After Final Edit):**
   - `uv run python -m compileall -q src api app tests` -> exit `0`
   - `uv run pytest -v` (covering unit, model/parity/bounds/cache-isolation, API, Streamlit `AppTest` across all 13 pages, and `/tmp` external-cwd subprocess import check) -> **100% pass, zero failures**.

---

## 10. Phase 5 — Cohesive Brand Identity Migration (Optional / When Renaming)

When migrating an existing ML platform to a final brand identity (name, tagline, SVG wordmark):
1. Audit every occurrence across `pyproject.toml`, `uv.lock`, Python package folder (`src/<new_pkg>/`), imports, `configs/`, `.env.example`, `Dockerfile`, `docker-compose.yml`, FastAPI title/OpenAPI, Streamlit titles/headers/footer/CSS/SVG, `mlflow.db` experiment metadata, all 11 notebooks, `tests/`, `README.md`, and every file in `docs/`.
2. Preserve persisted artifact/schema keys deliberately where external compatibility requires it and document compatibility decisions explicitly.
3. Verify zero broken imports, run `uv sync`, `compileall`, `pytest`, and publish `reports/engineering/BRAND_MIGRATION_REPORT.md` indexed in `docs/INDEX.md`.

---

## 11. Reusable Kickoff Prompts (Drop Into Any New Project)

### Prompt 1 — Start Phase 1 (Research Foundation)
> Read `prodev-skill.md` and `VISION.md` completely. Inspect `data/raw/` recursively without modifying raw files. Audit schemas, coverage, missingness, and documentation discrepancies using DuckDB/Polars out-of-core. Configure `pyproject.toml` + `uv.lock`, build reusable code under `src/<pkg>/`, and implement & execute the exact 11-notebook pipeline (`00`–`10`) adhering to all integrity, anti-leakage, DST/timezone, non-circular quality scorecard, and visualization rules. Present actual results, charts/tables summary, model trade-offs, and limitations, and **wait for my explicit approval before starting Phase 2**.

### Prompt 2 — Start Phase 2 (Production Platform)
> Phase 1 gates are verified and approved. Implement `ModelRegistry`, `InferenceService`, `api/main.py`, the 13-page Streamlit application (`app/Home.py` + `app/pages/01..12`), `monitoring/drift.py`, `Dockerfile` + `docker-compose.yml`, and full `tests/{unit,model,integration}` coverage including Streamlit `AppTest` and `/tmp` subprocess import checks.

### Prompt 3 — Zero-Trust Audit, Generalization Proof & Documentation
> Perform a comprehensive zero-trust audit of Phase 1 and Phase 2, verify whether the platform generalizes or is over/underfitted (producing `docs/GENERALIZATION_ANALYSIS.md` or a remediation plan based strictly on evidence), build/synchronize the full `docs/` suite with `docs/INDEX.md` + `README.md`, and write `reports/engineering/{COMPREHENSIVE_AUDIT,IMPROVEMENT_CHANGELOG,FINAL_VERIFICATION}.md`.

### Prompt 4 — Optimization & Scaffold Hardening (`/ponytail-review`)
> Run `/ponytail-review` across the entire repository: fix subtle bugs/bounds, eliminate cache leaks and redundant scans/transforms, remove AI boilerplate and scratch directories, verify a global-standard scaffold, and prove correctness via `compileall` and full `pytest`.

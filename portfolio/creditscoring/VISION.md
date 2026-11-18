# Project Vision — Bank Customer Churn Prediction

## 1. Project Overview

Banks lose revenue when customers close their accounts ("churn"). Predicting which
customers are likely to leave lets a bank intervene early with targeted retention
offers. This project builds a supervised binary classifier that estimates the
probability a customer has churned, using the actual dataset in `data/raw/`.

> **Data-vs-brief note:** the original task brief described *loan repayment*
> prediction (`1 = repaid`, `0 = defaulted`). The dataset in this repository does
> **not** contain a repayment label. Its target is `Exited`, which marks whether a
> customer left the bank (`1 = churn`, `0 = stayed`). This vision therefore
> describes **churn prediction**, the only outcome the data can actually support.
> No target or metric has been invented to fit the original brief.

## 2. Problem Statement

A bank must decide which customers to invest retention resources in. Reaching out
to every customer is expensive; waiting until a customer has already left is too
late. The challenge is to identify at-risk customers from attributes available at
score time (credit score, age, tenure, balance, product usage, activity, geography,
gender) while dealing with an imbalanced target (~20% churn), where missing a
churner (false negative) is costlier than an unnecessary retention offer.

## 3. Project Objectives

- Predict the probability that a customer has churned (`Exited = 1`).
- Identify the factors most strongly associated with churn.
- Prioritise **recall for the churn class** so at-risk customers are not missed.
- Provide a simple Streamlit UI for single and batch scoring plus model transparency.

## 4. Dataset Understanding

Source: a public "bank churn / churn modelling" style dataset provided here as three
CSV files under `data/raw/binaryclassificationwithabankchurndataset/`.

| File | Rows | Columns | Notes |
|---|---|---|---|
| `train.csv` | 15,000 | 14 | Labelled; includes target `Exited` |
| `test.csv` | 10,000 | 13 | Unlabelled (no `Exited`) |
| `sample_submission.csv` | 10,000 | 2 | `id`, `Exited` placeholder probabilities |

**Target:** `Exited` — `1 = customer left the bank (churn)`, `0 = stayed`.
Class balance: ~19.9% positive (3,000 / 15,000) — moderately imbalanced.

**Features (13 in train):**

| Column | Type | Meaning |
|---|---|---|
| `id` | int | Row identifier — **dropped** (no predictive meaning) |
| `CustomerId` | float | Customer identifier — **dropped** (identifier / leakage risk) |
| `Surname` | string | Surname (758 unique) — **dropped** (high-cardinality identifier) |
| `CreditScore` | float | Customer credit score (~431–850) |
| `Geography` | categorical | France / Spain / Germany |
| `Gender` | categorical | Female / Male |
| `Age` | float | Customer age (~18–72) |
| `Tenure` | float | Years with the bank (0–10) |
| `Balance` | float | Account balance (many zeros) |
| `NumOfProducts` | float | Number of bank products (1–5) |
| `HasCrCard` | float | Has a credit card (0/1) |
| `IsActiveMember` | float | Active membership (0/1) |
| `EstimatedSalary` | float | Estimated salary |

**Data quality:** no missing values in `train.csv`; no duplicate rows. `CustomerId`
repeats across rows, suggesting the source was augmented — identifiers are excluded
from modelling. `test.csv` has no label, so all evaluation uses a held-out split of
`train.csv`, and `test.csv` is used only to produce a submission file.

## 5. Machine Learning Approach

1. Exploratory data analysis (distributions, imbalance, segment churn rates).
2. Drop identifier columns; keep 10 genuine features.
3. Stratified train/validation split (80/20, fixed seed) from `train.csv`.
4. Preprocessing in a `ColumnTransformer` fitted on train only:
   numeric imputation + scaling; categorical imputation + one-hot encoding.
5. Train baseline (DummyClassifier), Logistic Regression, Random Forest, XGBoost.
6. Compare with ROC-AUC, PR-AUC, F1, and churn-class recall under stratified CV.
7. Select the best model, evaluate once on the held-out validation split.
8. Explain with SHAP / permutation importance.
9. Persist a single fitted `Pipeline` (preprocessor + model) for serving.

## 6. Candidate Models

- **DummyClassifier** — sanity baseline for imbalanced data.
- **Logistic Regression** — interpretable linear baseline.
- **Decision Tree / Random Forest / Extra Trees** — capture non-linear interactions.
- **HistGradientBoosting / LightGBM / CatBoost / XGBoost** — gradient boosting,
  consistently strong on tabular data (CatBoost also handles categoricals natively).

All are appropriate for a small (~15k row) tabular binary task; the best is chosen
empirically via cross-validation, not assumed.

## 7. Model Evaluation

Metrics: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC, PR-AUC.

Because the positive class is churn, **recall on the churn class** is emphasised: a
missed churner costs a bank more than a false alarm. Precision/F1 and ROC-AUC are
reported alongside so the recall/precision trade-off is visible. PR-AUC is included
because of class imbalance. No performance is claimed on `test.csv` (unlabelled).

## 8. Explainability

SHAP values (or permutation importance as a fallback) reveal which features drive
predictions, supporting trust and giving the bank actionable insight into churn
drivers. The Streamlit app surfaces a global importance chart.

## 9. Data Leakage and Reliability

- `id`, `CustomerId`, `Surname` are excluded — identifiers carry no legitimate
  signal and can encode leakage.
- All imputation, scaling, and encoding are fitted on **training data only**, inside
  a scikit-learn `Pipeline`, so validation and inference see identical transforms.
- The validation split is stratified and never used for fitting or tuning.
- The final `Pipeline` is saved whole, guaranteeing training/serving parity.

## 10. Ethical Considerations

Churn scores must support, not replace, human retention decisions. Geography and
gender are protected-ish attributes; their inclusion can encode historical bias.
Predictions should be audited for disparate impact before operational use, and used
to *offer* retention support rather than to penalise customers.

## 11. Expected Outcome

A trained, evaluated binary classifier saved as `models/pipeline.pkl`, with a
Streamlit app that scores individual customers and uploaded CSV batches, displays
performance charts, and explains the main churn drivers. A `submission.csv` for the
unlabelled test set is produced as a convenience.

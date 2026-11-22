"""Project-wide constants. All experiments share these contracts."""

from pathlib import Path

# Reproducibility
RANDOM_SEED = 42
TEST_SIZE = 0.20
N_SPLITS = 5
TUNING_N_ITER = 8
TUNING_CV = 3

# Target contract
TARGET = "Status"
CLASS_LABELS = ["C", "CL", "D"]  # alphabetical == model predict_proba order
PROB_COLUMNS = [f"Status_{label}" for label in CLASS_LABELS]
CLASS_DESCRIPTIONS = {
    "C": "Censored — alive at end of observation",
    "CL": "Censored — liver transplantation",
    "D": "Death during observation",
}

# Identifiers / raw layout
ID_COLUMN = "id"
RAW_SUBDIR = "multiclassificationtask (1)"
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
SUBMISSION_FILE = "sample_submission.csv"

# Raw feature groups (everything in train except the identifier and target)
RAW_NUMERIC = [
    "N_Days", "Age", "Bilirubin", "Cholesterol", "Albumin", "Copper",
    "Alk_Phos", "SGOT", "Tryglicerides", "Platelets", "Prothrombin", "Stage",
]
RAW_CATEGORICAL = [
    "Drug", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema",
]
RAW_FEATURES = RAW_NUMERIC + RAW_CATEGORICAL

# Numeric columns kept after feature engineering (N_Days is optional / ablated)
ENGINEERED_NUMERIC_BASE = [
    "Bilirubin", "Cholesterol", "Albumin", "Copper", "Alk_Phos", "SGOT",
    "Tryglicerides", "Platelets", "Prothrombin", "Stage",
    "Age_years", "age_imputed_flag", "missing_count", "Bilirubin_Albumin_ratio",
]
SKEWED_NUMERIC = [
    "Bilirubin", "Cholesterol", "Copper", "Alk_Phos", "SGOT", "Tryglicerides",
]
LOG_NUMERIC = [f"log_{column}" for column in SKEWED_NUMERIC]

# Human-readable form grouping for the Streamlit input form
INPUT_GROUPS = {
    "Demographics & study": ["Drug", "Age", "Sex", "N_Days", "Stage"],
    "Clinical findings": ["Ascites", "Hepatomegaly", "Spiders", "Edema"],
    "Liver function": ["Bilirubin", "Albumin", "Alk_Phos", "Prothrombin"],
    "Blood chemistry & cells": [
        "Cholesterol", "Copper", "SGOT", "Tryglicerides", "Platelets",
    ],
}

NUMERIC_RANGES = {
    "N_Days": (0.0, 50000.0, 1800.0),
    "Age": (0.0, 50000.0, 19500.0),
    "Bilirubin": (0.0, 30.0, 1.0),
    "Cholesterol": (0.0, 2500.0, 280.0),
    "Albumin": (0.0, 6.0, 3.5),
    "Copper": (0.0, 800.0, 50.0),
    "Alk_Phos": (0.0, 15000.0, 1000.0),
    "SGOT": (0.0, 3000.0, 100.0),
    "Tryglicerides": (0.0, 2000.0, 100.0),
    "Platelets": (0.0, 1200.0, 250.0),
    "Prothrombin": (5.0, 20.0, 10.5),
    "Stage": (1.0, 4.0, 3.0),
}
CATEGORICAL_OPTIONS = {
    "Drug": ["D-penicillamine", "Placebo"],
    "Sex": ["F", "M"],
    "Ascites": ["N", "Y"],
    "Hepatomegaly": ["N", "Y"],
    "Spiders": ["N", "Y"],
    "Edema": ["N", "S", "Y"],
}

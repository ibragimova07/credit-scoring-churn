"""Raw data loading, schema auditing and documented cleaning rules.

The raw CSVs in ``data/raw/`` are treated as immutable: cleaning produces new
DataFrames and an auditable before/after report, never an overwrite.
"""

from __future__ import annotations

import pandas as pd

from . import config, paths


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three immutable source files."""
    train = pd.read_csv(paths.SOURCE_DIR / config.TRAIN_FILE)
    test = pd.read_csv(paths.SOURCE_DIR / config.TEST_FILE)
    submission = pd.read_csv(paths.SOURCE_DIR / config.SUBMISSION_FILE)
    return train, test, submission


def schema_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a compact dtype / missingness / cardinality table."""
    rows = []
    for column in frame.columns:
        series = frame[column]
        missing = int(series.isna().sum())
        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "non_null": int(series.notna().sum()),
                "missing": missing,
                "missing_pct": round(100 * missing / len(series), 2),
                "n_unique": int(series.nunique(dropna=True)),
                "example": str(series.dropna().iloc[0]) if series.notna().any() else "",
            }
        )
    return pd.DataFrame(rows)


def integrity_audit(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Collect concrete integrity findings without modifying any input."""
    invalid = train[~train[config.TARGET].isin(config.CLASS_LABELS)]
    age_years = train["Age"] / 365.25
    return {
        "train_rows": len(train),
        "test_rows": len(test),
        "duplicate_train_rows": int(train.duplicated().sum()),
        "id_overlap": len(set(train[config.ID_COLUMN]) & set(test[config.ID_COLUMN])),
        "invalid_label_rows": invalid[config.ID_COLUMN].tolist(),
        "invalid_label_count": len(invalid),
        "age_below_18": int((age_years < 18).sum()),
        "age_above_100": int((age_years > 100).sum()),
        "n_days_median_by_status": train.groupby(config.TARGET)["N_Days"]
        .median()
        .round(1)
        .to_dict(),
        "class_counts": train[config.TARGET].value_counts().to_dict(),
    }


def clean_training_data(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the documented, non-destructive cleaning rules.

    Rules
    -----
    1. Remove rows whose ``Status`` is outside the declared C/CL/D contract.
    2. Flag implausible ages (<18 or >100 years) and clip ``Age`` in days to
       the same physical range. The original values stay in ``data/raw/``.

    Returns the cleaned frame and a before/after audit table.
    """
    before_rows = len(train)
    before_missing = int(train.isna().sum().sum())
    before_invalid = int((~train[config.TARGET].isin(config.CLASS_LABELS)).sum())

    clean = train[train[config.TARGET].isin(config.CLASS_LABELS)].copy()
    age_years = clean["Age"] / 365.25
    invalid_age = (age_years < 18) | (age_years > 100)
    clean["age_imputed_flag"] = invalid_age.astype(int)
    clean["Age"] = age_years.clip(18, 100) * 365.25

    audit = pd.DataFrame(
        [
            {"check": "Total records", "before": before_rows, "after": len(clean)},
            {
                "check": "Missing values",
                "before": before_missing,
                "after": int(clean.isna().sum().sum()),
            },
            {"check": "Invalid target labels", "before": before_invalid, "after": 0},
            {
                "check": "Out-of-range ages",
                "before": int(((train["Age"] / 365.25) < 18).sum()
                              + ((train["Age"] / 365.25) > 100).sum()),
                "after": int(invalid_age.sum()),  # flagged, not deleted
            },
            {
                "check": "Records retained (%)",
                "before": 100.0,
                "after": round(100 * len(clean) / before_rows, 4),
            },
        ]
    )
    return clean, audit


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only the raw model input columns (no identifier, no target)."""
    return frame[config.RAW_FEATURES].copy()

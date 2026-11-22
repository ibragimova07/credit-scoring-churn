"""Loading persisted artifacts and scoring single or batch patient records."""

from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd

from . import config, paths


def load_artifacts() -> tuple[Any, dict]:
    """Load the fitted pipeline and its metadata from disk."""
    if not paths.PIPELINE_PATH.exists() or not paths.METADATA_PATH.exists():
        raise FileNotFoundError("Run notebooks 03-05 first to create model artifacts")
    pipeline = joblib.load(paths.PIPELINE_PATH)
    with paths.METADATA_PATH.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    return pipeline, metadata


def validate_input(frame: pd.DataFrame) -> list[str]:
    """Return a list of missing / invalid raw feature columns."""
    missing = [c for c in config.RAW_FEATURES if c not in frame.columns]
    invalid_types = []
    for column in config.RAW_NUMERIC:
        if column in frame.columns:
            values = frame[column].dropna()  # missing values are legitimate input
            if len(values) and pd.to_numeric(values, errors="coerce").isna().all():
                invalid_types.append(column)
    return missing + invalid_types


def predict_dataframe(frame: pd.DataFrame, pipeline: Any) -> pd.DataFrame:
    """Return class probabilities and the most likely outcome for each row."""
    problems = validate_input(frame)
    if problems:
        raise ValueError(f"Invalid input columns: {problems}")
    X = frame[config.RAW_FEATURES].copy()
    for column in config.RAW_NUMERIC:
        X[column] = pd.to_numeric(X[column], errors="coerce")
    proba = pipeline.predict_proba(X)
    # Probabilities follow the alphabetical C / CL / D contract regardless of
    # whether the underlying classifier stores integer codes or string labels.
    result = pd.DataFrame(
        {
            f"Status_{label}": proba[:, config.CLASS_LABELS.index(label)]
            for label in config.CLASS_LABELS
        }
    )
    result[config.PROB_COLUMNS] = (
        result[config.PROB_COLUMNS]
        .div(result[config.PROB_COLUMNS].sum(axis=1), axis=0)
        .clip(lower=0.0, upper=1.0)
    )
    result["predicted_status"] = np.asarray(config.CLASS_LABELS)[
        result[config.PROB_COLUMNS].to_numpy().argmax(axis=1)
    ]
    result["confidence"] = result[config.PROB_COLUMNS].max(axis=1)
    return result

"""Leakage-free feature engineering, packaged as a sklearn transformer.

Putting the transformation inside the persisted Pipeline guarantees that
training, cross-validation and serving all apply exactly the same contract.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import config


def numeric_columns(include_n_days: bool = True) -> list[str]:
    """Numeric columns present after :func:`FeatureEngineeringTransformer`."""
    columns = (["N_Days"] if include_n_days else []) + config.ENGINEERED_NUMERIC_BASE
    return columns + config.LOG_NUMERIC


def categorical_columns() -> list[str]:
    return list(config.RAW_CATEGORICAL)


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """Deterministic, target-free feature engineering.

    Parameters
    ----------
    include_n_days:
        When ``False`` the leakage-prone ``N_Days`` column is removed. Used
        for the documented ablation study.
    """

    def __init__(self, include_n_days: bool = True):
        self.include_n_days = include_n_days

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineeringTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        # Age is stored in days in the source data. Clip to a plausible human
        # lifespan; the clipping flag is itself a feature.
        age_years = out["Age"] / 365.25
        out["Age_years"] = age_years.clip(18, 100)
        out["age_imputed_flag"] = ((age_years < 18) | (age_years > 100)).astype(float)
        out = out.drop(columns=["Age"])

        # The competition data has two structured missingness regimes; keep an
        # explicit count rather than silently discarding that information.
        out["missing_count"] = X[config.RAW_FEATURES].isna().sum(axis=1).astype(float)

        # A small set of clinically interpretable, monotone transforms.
        out["Bilirubin_Albumin_ratio"] = out["Bilirubin"] / out["Albumin"].clip(lower=0.1)
        for column in config.SKEWED_NUMERIC:
            out[f"log_{column}"] = np.log1p(out[column].clip(lower=0))

        if not self.include_n_days:
            out = out.drop(columns=["N_Days"])

        ordered = numeric_columns(self.include_n_days) + categorical_columns()
        return out[ordered]

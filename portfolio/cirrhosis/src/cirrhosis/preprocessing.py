"""Train-only preprocessing built with ColumnTransformer."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import features


def build_preprocessor(include_n_days: bool = True) -> ColumnTransformer:
    """Median/impute/scale numeric, mode/impute/one-hot categorical."""
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, features.numeric_columns(include_n_days)),
            ("cat", categorical, features.categorical_columns()),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

"""Cross-validation, tuning, calibration and competition submission helpers."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline

from . import config, features, models, preprocessing


def make_pipeline(model: Any, include_n_days: bool = True) -> Pipeline:
    """Feature engineering + preprocessing + classifier as one picklable unit."""
    return Pipeline(
        [
            ("features", features.FeatureEngineeringTransformer(include_n_days)),
            ("preprocessor", preprocessing.build_preprocessor(include_n_days)),
            ("classifier", model),
        ]
    )


def score_predictions(
    y_true: pd.Series, proba: np.ndarray, classes: list[str] = config.CLASS_LABELS
) -> dict[str, float]:
    """Log loss is primary; accuracy and macro F1 are reported for context."""
    # Accept either string labels or fitted integer codes (XGBoost requires codes).
    labels = np.asarray(
        [classes[int(value)] if isinstance(value, (int, np.integer)) else value
         for value in y_true]
    )
    predicted = np.asarray(classes)[np.argmax(proba, axis=1)]
    return {
        "log_loss": float(log_loss(labels, proba, labels=classes)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "f1_macro": float(f1_score(labels, predicted, average="macro")),
    }


def benchmark(
    X: pd.DataFrame,
    y: pd.Series,
    candidates: dict[str, Any],
    include_n_days: bool = True,
    n_splits: int = config.N_SPLITS,
    seed: int = config.RANDOM_SEED,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Stratified K-fold OOF benchmark under an identical preprocessing contract."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows: list[dict[str, Any]] = []
    oof: dict[str, np.ndarray] = {}
    for name, estimator in candidates.items():
        start = time.perf_counter()
        proba = cross_val_predict(
            make_pipeline(estimator, include_n_days), X, y,
            cv=cv, method="predict_proba", n_jobs=1,
        )
        scores = score_predictions(y, proba)
        rows.append(
            {
                "model": name,
                "include_n_days": include_n_days,
                "log_loss": scores["log_loss"],
                "accuracy": scores["accuracy"],
                "f1_macro": scores["f1_macro"],
                "fit_seconds": round(time.perf_counter() - start, 2),
            }
        )
        oof[name] = proba
    leaderboard = pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
    return leaderboard, oof


def tune_candidate(
    X: pd.DataFrame,
    y: pd.Series,
    name: str,
    include_n_days: bool = True,
    seed: int = config.RANDOM_SEED,
) -> tuple[Any, dict, float]:
    """Small randomized hyperparameter search scored by multiclass log loss."""
    space = models.tuning_spaces(seed).get(name)
    if not space:
        raise ValueError(f"No tuning space defined for {name}")
    cv = StratifiedKFold(n_splits=config.TUNING_CV, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        make_pipeline(models.get_candidates(seed)[name], include_n_days),
        param_distributions=space,
        n_iter=config.TUNING_N_ITER,
        scoring="neg_log_loss",
        cv=cv,
        random_state=seed,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    search.fit(X, y)
    # Return an unfitted clone: some boosters cannot be refit with string labels.
    return clone(search.best_estimator_), search.best_params_, float(-search.best_score_)


def calibrate(
    fitted_pipeline: Any,
    X: pd.DataFrame,
    y: pd.Series,
    method: str = "sigmoid",
    seed: int = config.RANDOM_SEED,
) -> Any:
    """Wrap a pipeline in cross-fitted probability calibration."""
    cal = CalibratedClassifierCV(
        fitted_pipeline,
        method=method,
        cv=StratifiedKFold(
            n_splits=config.TUNING_CV, shuffle=True, random_state=seed
        ),
    )
    cal.fit(X, y)
    return cal


def build_submission(
    test_raw: pd.DataFrame,
    template: pd.DataFrame,
    pipeline: Any,
) -> pd.DataFrame:
    """Produce a competition-ready submission from the exact template."""
    proba = pipeline.predict_proba(test_raw[config.RAW_FEATURES])
    # Class columns always follow the declared alphabetical contract, even when
    # the fitted classifier stores integer codes internally.
    by_class = {
        label: proba[:, config.CLASS_LABELS.index(label)] for label in config.CLASS_LABELS
    }
    submission = template[[config.ID_COLUMN]].copy()
    for label in config.CLASS_LABELS:
        submission[f"Status_{label}"] = by_class[label]
    submission = submission[list(template.columns)]

    values = submission[config.PROB_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Submission contains invalid probabilities")
    submission[config.PROB_COLUMNS] = values / values.sum(axis=1, keepdims=True)
    return submission


def validate_submission(
    submission: pd.DataFrame, template: pd.DataFrame
) -> dict[str, Any]:
    """Explicit format / probability checks used by tests and the UI."""
    probabilities = submission[config.PROB_COLUMNS].to_numpy(dtype=float)
    row_sums = probabilities.sum(axis=1)
    checks = {
        "columns_exact": list(submission.columns) == list(template.columns),
        "row_count": len(submission) == len(template),
        "ids_preserved": submission[config.ID_COLUMN].equals(template[config.ID_COLUMN]),
        "finite": bool(np.isfinite(probabilities).all()),
        "non_negative": bool((probabilities >= 0).all()),
        "normalized": bool(np.allclose(row_sums, 1.0, atol=1e-6)),
    }
    checks["all_passed"] = all(checks.values())
    return checks

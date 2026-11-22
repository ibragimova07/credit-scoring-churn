"""Candidate model zoo and focused tuning spaces (all seeded)."""

from __future__ import annotations

from typing import Any

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from . import config


def get_candidates(seed: int = config.RANDOM_SEED) -> dict[str, Any]:
    """Nine classifiers under one identical preprocessing contract."""
    return {
        "DummyPrior": DummyClassifier(strategy="prior"),
        "LogisticRegression": LogisticRegression(
            max_iter=2000, C=1.0, random_state=seed
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=20, random_state=seed
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=seed
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=seed
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, random_state=seed
        ),
        "XGBoost": XGBClassifier(
            n_estimators=350, learning_rate=0.05, max_depth=4, subsample=0.85,
            colsample_bytree=0.85, reg_lambda=1.0, eval_metric="mlogloss",
            n_jobs=-1, random_state=seed,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=350, learning_rate=0.05, num_leaves=15,
            min_child_samples=30, reg_lambda=1.0, verbosity=-1,
            n_jobs=-1, random_state=seed,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=350, learning_rate=0.05, depth=5, l2_leaf_reg=3.0,
            loss_function="MultiClass", verbose=False,
            allow_writing_files=False, thread_count=-1, random_seed=seed,
        ),
    }


def tuning_spaces(seed: int = config.RANDOM_SEED) -> dict[str, dict]:
    """Small, honest search spaces for the top candidates."""
    return {
        "LogisticRegression": {
            "classifier__C": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        },
        "HistGradientBoosting": {
            "classifier__max_iter": [200, 350, 500],
            "classifier__learning_rate": [0.03, 0.05, 0.1],
            "classifier__max_leaf_nodes": [15, 31, 63],
            "classifier__l2_regularization": [0.0, 1.0, 5.0],
        },
        "XGBoost": {
            "classifier__n_estimators": [200, 350, 500],
            "classifier__learning_rate": [0.03, 0.05, 0.1],
            "classifier__max_depth": [3, 4, 5, 6],
            "classifier__subsample": [0.7, 0.85, 1.0],
            "classifier__reg_lambda": [0.5, 1.0, 5.0],
        },
        "LightGBM": {
            "classifier__n_estimators": [200, 350, 500],
            "classifier__learning_rate": [0.03, 0.05, 0.1],
            "classifier__num_leaves": [7, 15, 31],
            "classifier__min_child_samples": [20, 30, 50],
        },
        "CatBoost": {
            "classifier__iterations": [200, 350, 500],
            "classifier__learning_rate": [0.03, 0.05, 0.1],
            "classifier__depth": [4, 5, 6],
            "classifier__l2_leaf_reg": [1.0, 3.0, 8.0],
        },
    }

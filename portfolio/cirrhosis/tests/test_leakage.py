"""Anti-leakage contracts: train-only fitting, no future/target peeking."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.cirrhosis import config, data, evaluation, features, preprocessing


def test_feature_transformer_is_target_free(splits):
    train_df, _ = splits
    X = data.feature_matrix(train_df)
    transformer = features.FeatureEngineeringTransformer(include_n_days=True)
    a = transformer.transform(X)
    b = transformer.transform(X.iloc[::-1])
    pd.testing.assert_frame_equal(a.sort_index(), b.sort_index())


def test_ndays_ablation_contract(splits):
    train_df, _ = splits
    X = data.feature_matrix(train_df)
    with_days = features.FeatureEngineeringTransformer(True).transform(X)
    without = features.FeatureEngineeringTransformer(False).transform(X)
    assert "N_Days" in with_days.columns
    assert "N_Days" not in without.columns
    assert with_days.shape[1] == without.shape[1] + 1


def test_preprocessor_statistics_come_from_train_only(splits):
    train_df, val_df = splits
    Xtr = data.feature_matrix(train_df)
    Xval = data.feature_matrix(val_df)
    Xval_modified = Xval.copy()
    Xval_modified["Bilirubin"] = 9999.0

    from sklearn.pipeline import Pipeline

    def build():
        return Pipeline([
            ("features", features.FeatureEngineeringTransformer(True)),
            ("preprocessor", preprocessing.build_preprocessor(True)),
        ])

    stack_a = build().fit(Xtr)
    stack_b = build().fit(Xtr)
    out_a = stack_a.transform(Xval).astype(float)
    out_b = stack_b.transform(Xval_modified).astype(float)
    prep_a = stack_a.named_steps["preprocessor"]
    prep_b = stack_b.named_steps["preprocessor"]
    # Fitted statistics are identical, and validation data never influences them.
    np.testing.assert_allclose(
        prep_a.named_transformers_["num"].named_steps["impute"].statistics_,
        prep_b.named_transformers_["num"].named_steps["impute"].statistics_,
    )
    engineered = features.FeatureEngineeringTransformer(True).transform(Xtr)
    learned = prep_a.named_transformers_["num"].named_steps["impute"].statistics_
    expected = engineered[features.numeric_columns(True)].median().to_numpy()
    np.testing.assert_allclose(learned[:len(expected)], expected, equal_nan=True)
    assert out_a.shape == out_b.shape
    assert np.isfinite(out_a).all()


def test_stratified_cv_preserves_class_balance(splits):
    train_df, _ = splits
    y = train_df[config.TARGET]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
    overall = y.value_counts(normalize=True).sort_index()
    for _, idx in cv.split(np.zeros(len(y)), y):
        fold = y.iloc[idx].value_counts(normalize=True).sort_index()
        assert np.allclose(fold.values, overall.values, atol=0.01)


def test_validation_rows_are_not_in_training_split(splits):
    train_df, val_df = splits
    assert set(train_df[config.ID_COLUMN]).isdisjoint(set(val_df[config.ID_COLUMN]))

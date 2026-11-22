"""Tests for persisted artifacts, inference parity and submission format."""

import json

import numpy as np
import pandas as pd

from src.cirrhosis import config, data, evaluation, inference, paths


def test_artifacts_exist():
    for path in (paths.PIPELINE_PATH, paths.METADATA_PATH, paths.LEADERBOARD_PATH,
                 paths.IMPORTANCE_PATH, paths.SUBMISSION_PATH):
        assert path.exists(), f"missing artifact: {path}"


def test_metadata_has_real_measured_values():
    metadata = json.loads(paths.METADATA_PATH.read_text())
    assert metadata["selected_model"] in {
        "LogisticRegression", "DecisionTree", "RandomForest", "ExtraTrees",
        "HistGradientBoosting", "XGBoost", "LightGBM", "CatBoost",
    }
    assert 0 < metadata["validation_log_loss"] < 5
    assert metadata["n_labelled_rows"] == 14999
    assert metadata["classes"] == config.CLASS_LABELS
    assert set(metadata["class_distribution"]) == set(config.CLASS_LABELS)


def test_single_prediction_contract():
    pipeline, _ = inference.load_artifacts()
    _, test_raw, _ = data.load_raw()
    row = test_raw[config.RAW_FEATURES].head(1)
    result = inference.predict_dataframe(row, pipeline)
    assert list(result.columns) == config.PROB_COLUMNS + ["predicted_status", "confidence"]
    assert np.allclose(result[config.PROB_COLUMNS].sum(axis=1), 1.0, atol=1e-6)
    assert result["predicted_status"].iloc[0] in config.CLASS_LABELS


def test_batch_prediction_matches_rowwise():
    pipeline, _ = inference.load_artifacts()
    _, test_raw, _ = data.load_raw()
    batch = test_raw[config.RAW_FEATURES].head(12)
    result = inference.predict_dataframe(batch, pipeline)
    single = pd.concat(
        [inference.predict_dataframe(batch.iloc[[i]], pipeline) for i in range(len(batch))]
    ).reset_index(drop=True)
    np.testing.assert_allclose(
        result[config.PROB_COLUMNS].to_numpy(),
        single[config.PROB_COLUMNS].to_numpy(),
        atol=1e-8,
    )


def test_submission_format_is_exact():
    submission = pd.read_csv(paths.SUBMISSION_PATH)
    _, _, template = data.load_raw()
    checks = evaluation.validate_submission(submission, template)
    assert checks["all_passed"], checks
    assert list(submission.columns) == list(template.columns)


def test_submission_probabilities_are_finite_and_normalized():
    submission = pd.read_csv(paths.SUBMISSION_PATH)
    values = submission[config.PROB_COLUMNS].to_numpy()
    assert np.isfinite(values).all()
    assert (values >= 0).all() and (values <= 1).all()
    assert np.allclose(values.sum(axis=1), 1.0, atol=1e-6)


def test_invalid_input_is_rejected():
    pipeline, _ = inference.load_artifacts()
    try:
        inference.predict_dataframe(pd.DataFrame({"Age": [1000.0]}), pipeline)
    except ValueError as exc:
        assert "Invalid input columns" in str(exc)
    else:
        raise AssertionError("Expected a validation error")

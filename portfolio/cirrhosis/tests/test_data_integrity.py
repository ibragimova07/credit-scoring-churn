"""Tests for schema, cleaning rules and raw-file immutability."""

import json

import numpy as np
import pandas as pd

from src.cirrhosis import config, data, paths


def test_raw_schema_contract(raw):
    train, test, submission = raw
    assert set(train.columns) == set(config.RAW_FEATURES + [config.ID_COLUMN, config.TARGET])
    assert set(test.columns) == set(config.RAW_FEATURES + [config.ID_COLUMN])
    assert list(submission.columns) == [config.ID_COLUMN] + config.PROB_COLUMNS
    assert len(train) == 15000 and len(test) == 10000


def test_no_identifier_overlap_or_duplicates(raw):
    train, test, _ = raw
    assert set(train[config.ID_COLUMN]).isdisjoint(set(test[config.ID_COLUMN]))
    assert train[config.ID_COLUMN].is_unique
    assert train.duplicated().sum() == 0


def test_invalid_label_audit_is_detected(raw):
    train, test = raw[0], raw[1]
    audit = data.integrity_audit(train, test)
    assert audit["invalid_label_count"] == 1
    assert audit["invalid_label_rows"] == [13789]
    assert audit["n_days_median_by_status"]["C"] > audit["n_days_median_by_status"]["D"]


def test_cleaning_removes_invalid_label_and_flags_ages(raw):
    train = raw[0]
    clean, audit_table = data.clean_training_data(train)
    assert set(clean[config.TARGET]) == set(config.CLASS_LABELS)
    assert len(clean) == len(train) - 1
    assert clean["age_imputed_flag"].sum() >= 2
    assert (clean["Age"] / 365.25).between(18, 100).all()
    assert "Records retained (%)" in set(audit_table["check"])


def test_saved_audit_matches_executed_rules():
    audit_path = paths.REPORTS_DIR / "cleaning" / "before_after_audit.csv"
    if audit_path.exists():
        table = pd.read_csv(audit_path)
        checks = {str(value).lower(): value for value in table["check"]}
        assert {"total records", "invalid labels"} <= set(checks)
        assert int(table.loc[table["check"] == checks["invalid labels"], "after"].iloc[0]) == 0


def test_integrity_report_is_real_json():
    report = paths.REPORTS_DIR / "data_quality" / "integrity_audit.json"
    if report.exists():
        content = json.loads(report.read_text())
        assert content["invalid_label_rows"] == [13789]
        assert isinstance(content["class_counts"], dict)

"""Canonical path registry. Every directory is created on demand (no dead dirs)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SOURCE_DIR = RAW_DIR / "multiclassificationtask (1)"
PROCESSED_DIR = DATA_DIR / "processed"
NOTEBOOKS_DIR = ROOT / "notebooks"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = MODELS_DIR / "figures"
REPORTS_DIR = ROOT / "reports"
TESTS_DIR = ROOT / "tests"
APP_DIR = ROOT / "app"
SUBMISSION_PATH = ROOT / "submission.csv"

PIPELINE_PATH = MODELS_DIR / "final_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"
LEADERBOARD_PATH = MODELS_DIR / "leaderboard.csv"
IMPORTANCE_PATH = MODELS_DIR / "feature_importance.csv"
OOF_PATH = MODELS_DIR / "oof_predictions.csv"


def ensure_dirs() -> None:
    for directory in (
        PROCESSED_DIR, NOTEBOOKS_DIR, FIGURES_DIR, REPORTS_DIR, TESTS_DIR, APP_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

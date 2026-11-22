"""Shared pytest fixtures. Paths are resolved independent of the caller's cwd."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


@pytest.fixture(scope="session")
def raw():
    from src.cirrhosis.data import load_raw

    return load_raw()


@pytest.fixture(scope="session")
def splits():
    import pandas as pd
    from src.cirrhosis.paths import PROCESSED_DIR

    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    val = pd.read_csv(PROCESSED_DIR / "val.csv")
    return train, val

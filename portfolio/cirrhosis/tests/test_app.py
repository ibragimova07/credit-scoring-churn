"""Streamlit UI smoke tests across all sections plus external-cwd import check."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _app():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120)
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def test_app_renders_all_sections():
    app = _app()
    radio = app.sidebar.radio[0]
    assert radio.options == [
        "Home", "Patient Prediction", "Data Insights", "Model Information",
    ]
    for section in radio.options:
        radio.set_value(section).run()
        assert not app.exception, (section, [e.value for e in app.exception])


def test_single_primary_prediction_action_scores_a_record():
    app = _app()
    app.sidebar.radio[0].set_value("Patient Prediction").run()
    buttons = [b for b in app.button if b.label == "Generate Prediction"]
    assert len(buttons) == 1
    buttons[0].click().run()
    assert not app.exception, [e.value for e in app.exception]
    assert any("Most likely outcome" in md.value for md in app.markdown)
    assert any("not a medical diagnosis" in md.value for md in app.markdown)


def test_model_information_page_shows_plain_language_summary():
    app = _app()
    app.sidebar.radio[0].set_value("Model Information").run()
    assert not app.exception, [e.value for e in app.exception]
    assert any("How accurate is it?" in md.value for md in app.markdown)


def test_modules_import_from_external_working_directory():
    code = (
        f"import sys; sys.path.insert(0, {str(ROOT)!r}); "
        "from src.cirrhosis import config, data, features, preprocessing, "
        "models, evaluation, inference; from app import design; "
        "print('import-ok', config.RANDOM_SEED)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd="/tmp", capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert "import-ok 42" in result.stdout

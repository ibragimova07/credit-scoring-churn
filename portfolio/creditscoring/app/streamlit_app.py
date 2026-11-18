"""Bank Customer Churn — Streamlit app.

Loads the fitted pipeline and evaluation artifacts produced by the notebooks.
All metrics are read from models/metadata.json — nothing is hardcoded here.

Run:  uv run streamlit run app/streamlit_app.py
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:  # works both with `streamlit run app/streamlit_app.py` and as a package
    from app.design import (
        card_container,
        card_html,
        fact_grid,
        inject_design_system,
        page_header,
        pill_html,
        section_title,
    )
except ModuleNotFoundError:  # pragma: no cover - Streamlit adds app/ to sys.path
    from design import (  # type: ignore[no-redef]
        card_container,
        card_html,
        fact_grid,
        inject_design_system,
        page_header,
        pill_html,
        section_title,
    )

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
FIG = MODELS / "figures"

# Powder Blue Minimalism — a small line icon, no decorative emoji.
_ICON = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E"
    "%3Crect width='64' height='64' rx='16' fill='%23A9C7E8'/%3E"
    "%3Cpath d='M17 45V29m10 16V21m10 24V35m10 10V26' stroke='%2326364A' "
    "stroke-width='5' stroke-linecap='round'/%3E%3C/svg%3E"
)
st.set_page_config(
    page_title="Bank Churn Predictor",
    page_icon=_ICON,
    layout="wide",
)

# Inject the global design system before any other UI element.
inject_design_system()


def render_wide(obj, kind="image"):
    """Render an image/dataframe full-width across Streamlit versions."""
    renderer = st.image if kind == "image" else st.dataframe
    try:
        renderer(obj, width="stretch")
    except (TypeError, ValueError):  # older Streamlit < 1.60
        renderer(obj, use_container_width=True)


@st.cache_resource
def load_artifacts():
    pipe = joblib.load(MODELS / "pipeline.pkl")
    with open(MODELS / "metadata.json") as f:
        meta = json.load(f)
    return pipe, meta


try:
    pipe, meta = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model artifacts are missing. Run the notebooks in order first "
        "(`01_eda` → `04_evaluation`) so that `models/pipeline.pkl` and "
        "`models/metadata.json` exist."
    )
    st.stop()

FEATURES = meta["feature_order"]
THRESHOLD = meta["threshold"]
NUMERIC = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
           "HasCrCard", "IsActiveMember", "EstimatedSalary"]
CATEGORICAL = ["Geography", "Gender"]

# ------------------------------------------------------------------- Sidebar
st.sidebar.markdown(
    """
    <div class="pb-brand">
      <div class="pb-brand-mark">C</div>
      <div>
        <div class="pb-brand-name">Churn Analytics</div>
        <div class="pb-brand-sub">Model workspace</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

section = st.sidebar.radio(
    "Navigate",
    ["Overview", "Single prediction", "Batch prediction", "Performance"],
)

st.sidebar.markdown("---")
st.sidebar.caption("Bank Churn · notebooks → artifacts → Streamlit")

# ------------------------------------------------------------------ Overview
if section == "Overview":
    page_header(
        "Model overview",
        "Bank Customer Churn Predictor",
        "Binary classification of whether a customer has left the bank "
        "(Exited = 1). The model and every metric shown are produced by the "
        "notebooks in this repository.",
    )

    section_title("At a glance")
    card_html(
        fact_grid([
            ("Task", "Binary classification"),
            ("Target", "Exited = 1 (churn)"),
            ("Train rows", f"{meta['train_rows']:,}"),
            ("Validation rows", f"{meta['val_rows']:,}"),
            ("Churn rate", f"{meta['churn_rate']:.1%}"),
            ("Decision threshold", f"{THRESHOLD:.2f}"),
        ])
        + '<p class="pb-caption-note">Identifier columns (id, CustomerId, '
          "Surname) were excluded as leakage risks. Metrics are computed on a "
          "held-out validation split; the unlabelled test.csv is not used "
          "for evaluation.</p>"
    )

    section_title(
        "Held-out validation metrics",
        "Every value is loaded from the saved evaluation artifacts.",
    )
    m = meta["metrics"]
    cols = st.columns(5)
    labels = [
        ("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"), ("F1", "f1"),
        ("Churn recall", "recall_churn"), ("Accuracy", "accuracy"),
    ]
    for col, (label, key) in zip(cols, labels):
        col.metric(label, f"{m[key]:.3f}")

    fig = FIG / "importance.png"
    if fig.exists():
        section_title("What drives churn predictions")
        with card_container():
            render_wide(str(fig), "image")

# ---------------------------------------------------------- Single prediction
elif section == "Single prediction":
    page_header(
        "Scoring · single customer",
        "Score one customer",
        "Enter the customer's attributes to estimate churn probability.",
    )

    with st.form("single_form"):
        c1, c2 = st.columns(2)
        values = {}
        with c1:
            values["CreditScore"] = st.number_input("Credit score", 300, 900, 650)
            values["Age"] = st.number_input("Age", 18, 100, 37)
            values["Tenure"] = st.number_input("Tenure (years)", 0, 10, 5)
            values["Balance"] = st.number_input("Balance", 0.0, 300000.0, 0.0, step=1000.0)
            values["NumOfProducts"] = st.number_input("Number of products", 1, 5, 1)
        with c2:
            values["HasCrCard"] = st.selectbox("Has credit card", [0, 1], index=1)
            values["IsActiveMember"] = st.selectbox("Is active member", [0, 1], index=0)
            values["EstimatedSalary"] = st.number_input(
                "Estimated salary", 0.0, 1000000.0, 120000.0, step=1000.0
            )
            values["Geography"] = st.selectbox("Geography", ["France", "Spain", "Germany"])
            values["Gender"] = st.selectbox("Gender", ["Female", "Male"])
        submitted = st.form_submit_button("Predict churn")

    if submitted:
        row = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
        proba = float(pipe.predict_proba(row)[0, 1])
        churn = proba >= THRESHOLD

        section_title("Prediction result")
        col1, col2 = st.columns(2)
        with col1:
            with card_container():
                st.markdown(
                    '<div class="pb-kicker">CHURN PROBABILITY</div>'
                    f'<div class="pb-big-value">{proba:.1%}</div>',
                    unsafe_allow_html=True,
                )
        with col2:
            with card_container():
                tone = "error" if churn else "success"
                label = (
                    "HIGH RISK · likely to churn" if churn
                    else "LOW RISK · likely to stay"
                )
                st.markdown(
                    '<div class="pb-kicker">RISK ASSESSMENT</div>'
                    + pill_html(label, tone)
                    + f'<p class="pb-subtle" style="margin:12px 0 0">'
                      f'Decision threshold: {THRESHOLD:.2f}</p>',
                    unsafe_allow_html=True,
                )
        st.markdown(
            '<p class="pb-subtle" style="margin:16px 0 6px">'
            "Estimated probability of churn</p>",
            unsafe_allow_html=True,
        )
        st.progress(proba)

# ----------------------------------------------------------- Batch prediction
elif section == "Batch prediction":
    page_header(
        "Scoring · batch upload",
        "Score a batch of customers",
        "Upload a CSV containing the model features to score many customers "
        "at once. Identifier columns are ignored if present.",
    )

    section_title("Required columns")
    with card_container():
        st.markdown(
            '<p class="pb-subtle" style="margin:0">Your CSV must include these '
            f"{len(FEATURES)} columns:</p>"
            '<div class="pb-chips">'
            + "".join(f'<span class="pb-chip">{c}</span>' for c in FEATURES)
            + "</div>",
            unsafe_allow_html=True,
        )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
            missing = [c for c in FEATURES if c not in batch.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                proba = pipe.predict_proba(batch[FEATURES])[:, 1]
                out = batch.copy()
                out["churn_probability"] = np.round(proba, 4)
                out["churn_prediction"] = (proba >= THRESHOLD).astype(int)
                st.success(
                    f"Scored {len(out):,} customers. "
                    f"{out['churn_prediction'].sum():,} flagged at risk."
                )
                section_title("Scored customers", "First 50 rows of the result.")
                with card_container():
                    render_wide(out.head(50), "dataframe")
                csv_bytes = out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download predictions", csv_bytes,
                    "churn_predictions.csv", "text/csv",
                )
        except Exception as exc:  # noqa: BLE001 - surface parse errors to the user
            st.error(f"Could not score file: {exc}")

# ----------------------------------------------------------------- Performance
elif section == "Performance":
    page_header(
        "Model evaluation",
        "Model performance",
        "Held-out validation metrics, cross-validated model comparison and "
        "diagnostic charts — all read from the saved evaluation artifacts.",
    )

    m = meta["metrics"]
    cols = st.columns(5)
    for col, (label, key) in zip(
        cols,
        [("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"), ("F1", "f1"),
         ("Churn recall", "recall_churn"), ("Accuracy", "accuracy")],
    ):
        col.metric(label, f"{m[key]:.3f}")

    lb_path = MODELS / "leaderboard.csv"
    if lb_path.exists():
        section_title(
            "Cross-validated model comparison",
            "Candidate models benchmarked on the training split.",
        )
        with card_container():
            render_wide(pd.read_csv(lb_path).round(4), "dataframe")

    for name, caption in [
        ("confusion_matrix.png", "Confusion matrix — validation split"),
        ("roc_pr.png", "ROC and Precision–Recall curves — validation split"),
        ("importance.png", "Permutation importance (validation ROC-AUC drop)"),
    ]:
        p = FIG / name
        if p.exists():
            section_title(caption)
            with card_container():
                render_wide(str(p), "image")

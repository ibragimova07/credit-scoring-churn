"""Simple, accessible Streamlit dashboard for cirrhosis outcome estimates.

Four clearly named sections. Plain language first; technical detail is optional
and kept out of the main workflow. All displayed values come from saved artifacts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from src.cirrhosis import config, data, evaluation, inference, paths  # noqa: E402
from app import design  # noqa: E402

st.set_page_config(
    page_title="Liver Cirrhosis — Outcome Estimates",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)
design.apply_theme()

SECTIONS = ["Home", "Patient Prediction", "Data Insights", "Model Information"]

OUTCOMES = {
    "C": ("Alive at the end of the study", "still living when the study ended"),
    "CL": ("Received a liver transplant", "had a liver transplant during the study"),
    "D": ("Death during the study", "died during the study period"),
}

# Labels, units and allowed ranges shown to the user. Values come from the study data.
NUMERIC_FIELDS = {
    "Age": ("Age", "years", 0.0, 120.0),
    "Bilirubin": ("Bilirubin", "mg/dL", 0.0, 30.0),
    "Cholesterol": ("Cholesterol", "mg/dL", 0.0, 2500.0),
    "Albumin": ("Albumin", "g/dL", 0.0, 6.0),
    "Copper": ("Copper", "µg/dL", 0.0, 800.0),
    "Alk_Phos": ("Alkaline phosphatase", "U/L", 0.0, 15000.0),
    "SGOT": ("SGOT (liver enzyme)", "U/L", 0.0, 3000.0),
    "Tryglicerides": ("Triglycerides", "mg/dL", 0.0, 2000.0),
    "Platelets": ("Platelets", "×10⁹/L", 0.0, 1200.0),
    "Prothrombin": ("Prothrombin time", "seconds", 5.0, 20.0),
    "Stage": ("Liver scarring stage", "stage 1 to 4", 1.0, 4.0),
    "N_Days": ("Days in the study", "days", 0.0, 50000.0),
}

CATEGORICAL_FIELDS = {
    "Sex": ("Sex", ["Female", "Male"]),
    "Drug": ("Study drug", ["D-penicillamine", "Placebo"]),
    "Ascites": ("Fluid in the abdomen (ascites)", ["No", "Yes"]),
    "Hepatomegaly": ("Enlarged liver (hepatomegaly)", ["No", "Yes"]),
    "Spiders": ("Spider-like skin vessels", ["No", "Yes"]),
    "Edema": ("Swelling (edema)", ["None", "Slight", "Yes"]),
}

TO_RAW = {
    "Sex": {"Female": "F", "Male": "M"},
    "Drug": {"D-penicillamine": "D-penicillamine", "Placebo": "Placebo"},
    "Ascites": {"No": "N", "Yes": "Y"},
    "Hepatomegaly": {"No": "N", "Yes": "Y"},
    "Spiders": {"No": "N", "Yes": "Y"},
    "Edema": {"None": "N", "Slight": "S", "Yes": "Y"},
}

DISCLAIMER = (
    "This estimate is an experimental statistical result based on past patient records. "
    "It is not a medical diagnosis, a prediction of any individual's future, or medical advice. "
    "Please talk with a qualified healthcare professional about health decisions."
)


@st.cache_resource(show_spinner=False)
def get_pipeline():
    return inference.load_artifacts()


@st.cache_data(show_spinner=False)
def get_labelled_data() -> pd.DataFrame:
    train = pd.read_csv(paths.PROCESSED_DIR / "train.csv")
    val = pd.read_csv(paths.PROCESSED_DIR / "val.csv")
    return pd.concat([train, val], ignore_index=True)


@st.cache_data(show_spinner=False)
def get_table(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if Path(path).exists() else None


def show_image(name: str, caption: str) -> None:
    path = paths.FIGURES_DIR / name
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")


def fact_row(items: list[tuple[str, str, str]]) -> None:
    cells = st.columns(len(items))
    for cell, (label, value, note_text) in zip(cells, items):
        with cell:
            design.fact(label, value, note_text)


# ------------------------------------------------------------------- pages ---
def page_home(metadata: dict) -> None:
    design.page_header(
        "Welcome",
        "This tool gives a simple estimate of three possible outcomes for a person with liver "
        "cirrhosis. It learns patterns from past patient records. It is designed to help you "
        "understand data, not to make health decisions.",
    )
    design.note(
        "**How to use this tool:** 1) Choose **Patient Prediction** in the menu on the left. "
        "2) Enter the details you have. 3) Select **Generate Prediction** and read the results. "
        "You can change any value and try again."
    )

    fact_row(
        [
            ("People in the study", f"{metadata['n_labelled_rows']:,}", "records used to build the tool"),
            ("Details used per person", str(metadata["n_features_raw"]), "age, test results and study details"),
            ("Checked on new people", "Yes", "tested on records it had not seen before"),
        ]
    )

    st.markdown("## What outcomes does the tool estimate?")
    distribution = metadata.get("class_distribution", {})
    total = sum(distribution.values()) or 1
    for label in config.CLASS_LABELS:
        name, description = OUTCOMES[label]
        share = distribution.get(label, 0) / total
        st.markdown(
            design.probability_row(name, description, share, False),
            unsafe_allow_html=True,
        )
    st.caption("The bars show how common each outcome was in the study records.")

    with st.expander("Technical details (optional)"):
        st.markdown(
            f"- Model used: **{metadata['selected_model']}**  \n"
            f"- Cross-validation log loss: **{metadata['cv_log_loss']:.3f}** (lower is better)  \n"
            f"- Holdout log loss: **{metadata['validation_log_loss']:.3f}**  \n"
            f"- Holdout accuracy: **{metadata['validation_accuracy']:.1%}**"
        )
    design.warning(DISCLAIMER)

    with st.expander("For organizations: get the ready-to-submit results file"):
        if not paths.SUBMISSION_PATH.exists():
            st.info("The results file has not been generated yet.")
        else:
            submission = pd.read_csv(paths.SUBMISSION_PATH)
            _, template, _ = data.load_raw()
            checks = evaluation.validate_submission(submission, template)
            st.markdown(
                f"File ready: **{len(submission):,}** patient records. "
                f"All format checks passed: **{checks['all_passed']}**."
            )
            st.dataframe(submission.head(5), width="stretch", hide_index=True)
            st.download_button(
                "Download results file (CSV)",
                data=submission.to_csv(index=False).encode("utf-8"),
                file_name="submission.csv",
                mime="text/csv",
            )


def _numeric_input(column: str, defaults: pd.Series) -> float:
    label, unit, low, high = NUMERIC_FIELDS[column]
    default = float(defaults[column])
    if column == "Age":
        default = round(default / 365.25)
    return st.number_input(
        f"{label} ({unit})",
        min_value=low, max_value=high, value=float(default), step=1.0,
        key=f"number_{column}",
    )


def _categorical_input(column: str) -> object:
    label, options = CATEGORICAL_FIELDS[column]
    choice = st.selectbox(label, ["Not provided"] + options, index=0, key=f"choice_{column}")
    return np.nan if choice == "Not provided" else TO_RAW[column][choice]


def page_prediction(pipeline) -> None:
    design.page_header(
        "Patient Prediction",
        "Enter the details you have for one patient. Fields already contain typical values "
        "from the study data, so you only need to change the ones you know.",
    )
    labelled = get_labelled_data()
    defaults = labelled[config.RAW_NUMERIC].median()

    values: dict[str, object] = {}
    design.step("Step 1 — Enter patient information")
    design.note("You cannot enter a value outside the allowed range. If you do not have a result, leave the default value or choose **Not provided**.")

    st.markdown("**About the person**")
    a, b = st.columns(2)
    with a:
        age_years = _numeric_input("Age", defaults)
        values["Age"] = age_years * 365.25  # source data stores age in days
    with b:
        values["Sex"] = _categorical_input("Sex")

    with st.expander("Examination findings (optional)"):
        c, d = st.columns(2)
        with c:
            values["Ascites"] = _categorical_input("Ascites")
            values["Hepatomegaly"] = _categorical_input("Hepatomegaly")
        with d:
            values["Spiders"] = _categorical_input("Spiders")
            values["Edema"] = _categorical_input("Edema")

    st.markdown("**Blood test results**")
    groups = [
        ["Bilirubin", "Albumin", "Prothrombin"],
        ["Cholesterol", "Copper", "Alk_Phos"],
        ["SGOT", "Tryglicerides", "Platelets"],
    ]
    for group in groups:
        cells = st.columns(3)
        for cell, column in zip(cells, group):
            with cell:
                values[column] = _numeric_input(column, defaults)

    with st.expander("Study details (optional)"):
        e, f = st.columns(2)
        with e:
            values["Drug"] = _categorical_input("Drug")
            values["Stage"] = _numeric_input("Stage", defaults)
        with f:
            values["N_Days"] = _numeric_input("N_Days", defaults)
            st.caption("Days the person was followed in the study. Leave the default if unknown.")

    st.markdown("")  # spacing before the single primary action
    generate = st.button("Generate Prediction", type="primary", width="stretch")

    if generate:
        design.step("Step 2 — Your result")
        try:
            row = pd.DataFrame([values])[config.RAW_FEATURES]
            result = inference.predict_dataframe(row, pipeline).iloc[0]
        except Exception as exc:
            st.error(f"We could not calculate an estimate: {exc}. Please check the values and try again.")
            return

        top = str(result["predicted_status"])
        st.markdown(
            f'<div class="outcome">Most likely outcome: {OUTCOMES[top][0]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("**How likely is each outcome?**")
        for label in config.CLASS_LABELS:
            name, description = OUTCOMES[label]
            value = float(result[f"Status_{label}"])
            st.markdown(
                design.probability_row(name, description, value, label == top),
                unsafe_allow_html=True,
            )
        design.note(
            "Each bar shows how often people with similar details had that outcome in the data "
            "used to build this tool. The percentages add up to 100%, but they are estimates, "
            "not certainties. Change any value above and select **Generate Prediction** again to see what changes."
        )
        design.warning(DISCLAIMER)

    with st.expander("Score many patients at once (CSV file)"):
        uploaded = st.file_uploader("Choose a CSV file with the same columns as the study data", type=["csv"])
        if uploaded is not None:
            try:
                frame = pd.read_csv(uploaded)
            except Exception as exc:
                st.error(f"We could not read that file: {exc}")
            else:
                problems = inference.validate_input(frame)
                if problems:
                    st.error("The file is missing these columns: " + ", ".join(map(str, problems)))
                else:
                    predictions = inference.predict_dataframe(frame, pipeline)
                    st.dataframe(pd.concat([frame.reset_index(drop=True), predictions], axis=1),
                                 width="stretch")
                    st.download_button(
                        "Download results (CSV)",
                        data=predictions.to_csv(index=False).encode("utf-8"),
                        file_name="patient_predictions.csv",
                        mime="text/csv",
                    )
                    design.warning(DISCLAIMER)


def page_insights() -> None:
    design.page_header(
        "Data Insights",
        "Explore the information collected in the study. These charts show patterns in the "
        "records used to build the tool.",
    )
    df = get_labelled_data()
    label_names = {label: OUTCOMES[label][0] for label in config.CLASS_LABELS}
    plot_df = df.copy()
    plot_df["Outcome"] = plot_df[config.TARGET].map(label_names)

    design.note("Choose a measurement to see how its values were distributed across the three outcomes.")
    numeric = st.selectbox("Measurement", config.RAW_NUMERIC, index=2)
    fig = px.histogram(
        plot_df, x=numeric, color="Outcome", barmode="overlay", opacity=0.65,
        nbins=40, labels={numeric: NUMERIC_FIELDS[numeric][0], "count": "Number of people"},
        title=f"{NUMERIC_FIELDS[numeric][0]} by outcome",
    )
    st.plotly_chart(fig, width="stretch")

    categorical = st.selectbox("Other patient details", list(CATEGORICAL_FIELDS), index=0)
    counts = plot_df.groupby(categorical, dropna=False).size().reset_index(name="Number of people")
    counts[categorical] = counts[categorical].fillna("Not provided").astype(str)
    fig = px.bar(
        counts, x=categorical, y="Number of people",
        title=f"{CATEGORICAL_FIELDS[categorical][0]} — how many people",
        labels={categorical: CATEGORICAL_FIELDS[categorical][0]},
    )
    st.plotly_chart(fig, width="stretch")

    with st.expander("More charts (optional)"):
        show_image("02_missingness.png", "Which details were often missing in the study records")
        show_image("02_correlation.png", "How measurements move together")
        show_image("02_target_relationships.png", "Outcomes by liver scarring stage")
    design.warning("The study records are synthetic. Patterns here describe this dataset, not real patients.")


def page_model_info(metadata: dict) -> None:
    design.page_header(
        "Model Information",
        "This page explains how the tool was checked. You do not need to read it to use "
        "Patient Prediction.",
    )
    design.step("How accurate is it?")
    accuracy = metadata["validation_accuracy"]
    design.note(
        f"When tested on **{3000:,}** people whose records it had never seen, about "
        f"**{accuracy * 100:.0f} out of 100** predictions matched the recorded outcome. "
        "The tool also gives a probability for each outcome, which is more informative than a single label."
    )
    fact_row(
        [
            ("Model used", metadata["selected_model"], "the method chosen after comparison"),
            ("Holdout log loss", f"{metadata['validation_log_loss']:.3f}", "lower is better"),
            ("Probability adjustment", metadata["calibration_method"], "chosen by comparison"),
        ]
    )
    design.note(
        "The estimates were checked against real records from the dataset, then the final "
        "version was trained on all available records. The source data is synthetic, so these "
        "results may not transfer to other hospitals or real patients."
    )

    with st.expander("Technical details (optional)"):
        st.markdown("**Comparison of methods** (log loss is the main measure; lower is better)")
        leaderboard = get_table(paths.LEADERBOARD_PATH)
        if leaderboard is not None:
            st.dataframe(leaderboard, width="stretch", hide_index=True)
        ablation = get_table(paths.REPORTS_DIR / "experiments" / "ndays_ablation.csv")
        if ablation is not None:
            st.markdown("**Days-in-study field: with and without**")
            st.dataframe(ablation, width="stretch", hide_index=True)
            st.caption(
                "The field improved the score but acts as a stand-in for the outcome. "
                "A real clinical tool should leave it out."
            )
        calibration = get_table(paths.REPORTS_DIR / "final_evaluation" / "calibration_comparison.csv")
        if calibration is not None:
            st.markdown("**Probability adjustment comparison**")
            st.dataframe(calibration, width="stretch", hide_index=True)
        importance = get_table(paths.IMPORTANCE_PATH)
        if importance is not None:
            st.markdown("**Which details mattered most**")
            st.dataframe(importance.head(10), width="stretch", hide_index=True)
        show_image("05_confusion_calibration.png", "Prediction mistakes and probability reliability")
    design.warning(DISCLAIMER)


# ------------------------------------------------------------------- router --
def main() -> None:
    if not paths.PIPELINE_PATH.exists() or not paths.METADATA_PATH.exists():
        design.page_header("Welcome", "The tool is not ready yet.")
        st.error("The saved results are missing. Run the notebooks in the README first.")
        return
    pipeline, metadata = get_pipeline()
    choice = st.sidebar.radio("Where would you like to go?", SECTIONS)
    st.sidebar.markdown("---")
    st.sidebar.caption("For learning and research only · not a medical device")
    if choice == "Home":
        page_home(metadata)
    elif choice == "Patient Prediction":
        page_prediction(pipeline)
    elif choice == "Data Insights":
        page_insights()
    else:
        page_model_info(metadata)


if __name__ == "__main__":
    main()

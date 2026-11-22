# VISION.md

# Cirrhosis Patient Survival Prediction

**Project Type:** Machine Learning & Predictive Analytics  
**Application:** Interactive Streamlit Web Application  
**Domain:** Healthcare Analytics  
**Primary Objective:** Multiclass Patient Outcome Prediction

---

## 1. Project Vision

The goal of this project is to develop a Machine Learning-based predictive system that analyzes clinical and demographic patient data to estimate the probability of different outcomes for patients diagnosed with liver cirrhosis.

The system will use historical patient data to identify patterns and relationships between clinical measurements, patient characteristics, and observed outcomes.

The final product will be an interactive Streamlit application where users can enter patient information, receive probabilistic predictions, explore model performance, and analyze relevant dataset patterns.

The project aims to demonstrate how Machine Learning can support healthcare-related data analysis through predictive modeling, statistical evaluation, and accessible data visualization.

**Core principle:** Build a reliable, interpretable, reproducible, and user-friendly prediction system rather than simply maximizing classification accuracy.

## 2. Problem Statement

Liver cirrhosis is a chronic liver disease characterized by progressive scarring and deterioration of liver function.

Analyzing clinical data can reveal patterns associated with different patient outcomes.

The challenge is to develop a multiclass classification model that estimates the probability of three possible outcomes:

- **C:** Censored. The patient was alive at the end of the observation period.
- **CL:** Censored due to liver transplantation.
- **D:** Death. The patient died during the observation period.

The system must predict probabilities for all three classes rather than producing only a single class label.

## 3. Project Objectives

The project has the following objectives:

1. Explore and understand the available cirrhosis patient dataset.
2. Identify meaningful patterns in clinical and demographic variables.
3. Develop effective data preprocessing and feature engineering strategies.
4. Train and compare multiple Machine Learning classification algorithms.
5. Optimize model performance using cross-validation and hyperparameter tuning.
6. Evaluate predicted probability quality using multiclass Log Loss.
7. Build an interactive Streamlit application for individual and batch predictions.
8. Provide meaningful visualizations and model performance insights.
9. Generate competition-ready predictions in the required submission format.
10. Maintain reproducible experiments and modular project architecture.

## 4. Dataset Overview

The project uses a competition dataset generated from a deep learning model trained on the original Cirrhosis Patient Survival Prediction dataset.

The dataset contains three primary files:

| File | Description |
|---|---|
| train.csv | Training data containing patient features and Status |
| test.csv | Unlabeled patient data requiring predictions |
| sample_submission.csv | Reference submission format |

The dataset contains clinical measurements, demographic information, and other patient-related variables.

The actual feature names, data types, distributions, missing values, and relationships must be discovered through dataset exploration rather than assumed in advance.

### Target Variable

`Status`

Possible values:

- C
- CL
- D

### Prediction Output

For each patient, the model must produce:

- `Status_C`
- `Status_CL`
- `Status_D`

Each value represents the predicted probability of the corresponding outcome.

## 5. Evaluation Strategy

The primary evaluation metric is **Multiclass Log Loss**.

The model must optimize the quality of its predicted probabilities rather than focusing exclusively on classification accuracy.

The evaluation strategy should prioritize:

- Multiclass Log Loss
- Stratified Cross-Validation
- Out-of-Fold Predictions
- Probability Calibration
- Generalization Performance

Secondary evaluation metrics include:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Confusion Matrix

A model with higher accuracy but significantly worse Log Loss should not automatically be selected as the final model.

## 6. Machine Learning Strategy

The project should investigate and compare several algorithms suitable for structured tabular data.

Potential models include:

- Logistic Regression
- Decision Tree
- Random Forest
- Extra Trees
- HistGradientBoosting
- XGBoost
- LightGBM
- CatBoost

Model selection must be based on experimental results rather than assumptions.

The project should support:

- Appropriate missing-value treatment
- Categorical feature encoding
- Numerical feature scaling where necessary
- Feature engineering
- Hyperparameter optimization
- Stratified cross-validation
- Probability calibration
- Model comparison

The final model must provide reliable probabilities for all three target classes.

All preprocessing and model selection procedures must avoid data leakage.

## 7. Application Vision

The final application will be built using Streamlit.

It should provide a clean, minimalist, modern, and professional interface.

The visual design should prioritize:

- Light theme
- Consistent colors
- Clear typography
- Logical information hierarchy
- Simple navigation
- Responsive layouts
- Readable charts and tables
- Minimal unnecessary visual elements

The application should feel like a professional healthcare analytics dashboard rather than a basic Machine Learning demonstration.

### 7.1 Dashboard Overview

The main dashboard should introduce the project and display:

- Project description
- Dataset overview
- Number of training observations
- Number of available features
- Target class distribution
- Selected model
- Validation Log Loss

### 7.2 Individual Patient Prediction

Users should be able to enter patient information through an organized input form.

Input fields must be generated based on the actual dataset features.

The interface should:

- Group related clinical variables.
- Use suitable widgets for different data types.
- Validate user inputs.
- Apply the saved preprocessing pipeline.
- Generate probabilities for C, CL, and D.
- Display the most probable outcome.
- Visualize the predicted probabilities.

The results must be presented clearly and must not be described as a confirmed medical diagnosis.

### 7.3 Model Performance

Provide an analytical dashboard showing:

- Model comparison
- Cross-validation results
- Log Loss comparison
- Classification metrics
- Confusion matrix
- Feature importance
- Probability calibration results

### 7.4 Exploratory Data Analysis

Allow users to explore:

- Numerical feature distributions
- Categorical feature distributions
- Missing-value patterns
- Correlation relationships
- Target class distribution
- Relationships between features and patient outcomes

### 7.5 Batch Prediction

Users should be able to upload a CSV containing multiple patient records.

The application should:

1. Validate the uploaded file.
2. Check required columns.
3. Apply the saved preprocessing pipeline.
4. Generate class probabilities.
5. Display the prediction results.
6. Allow the user to download the results as CSV.

### 7.6 Competition Submission

The project must support generating a competition-ready `submission.csv`.

The file must:

- Preserve test dataset IDs.
- Contain `Status_C`, `Status_CL`, and `Status_D`.
- Follow the exact column order of `sample_submission.csv`.
- Contain valid finite probabilities.
- Have normalized probabilities for each observation.

## 8. Technical Architecture

The project should follow a modular architecture that separates data processing, model development, inference, and user interface responsibilities.

### Main Technologies

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Scikit-learn | Preprocessing, modeling, evaluation |
| XGBoost | Gradient boosting |
| LightGBM | Gradient boosting |
| CatBoost | Categorical boosting |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Plotly | Interactive charts |
| Streamlit | Web application |
| Joblib | Model serialization |

### Architectural Components

**Data Layer**
- Dataset loading
- Data validation
- Exploratory analysis

**Preprocessing Layer**
- Missing-value handling
- Encoding
- Scaling
- Feature engineering

**Model Layer**
- Model training
- Cross-validation
- Hyperparameter tuning
- Model selection
- Probability calibration

**Inference Layer**
- Single-patient prediction
- Batch prediction
- Competition submission generation

**Presentation Layer**
- Streamlit dashboard
- Prediction interface
- Analytics and visualizations

## 9. Project Structure

The codebase should maintain a clear separation of responsibilities.

Suggested structure:

```text
cirrhosis-prediction/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── notebooks/
│   └── eda.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   ├── best_model.pkl
│   ├── preprocessor.pkl
│   └── model_metadata.json
│
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── submission.csv
│
├── app.py
├── requirements.txt
├── README.md
├── VISION.md
└── .gitignore
```

The structure may evolve as the project develops, provided that responsibilities remain clearly separated.

## 10. Important Constraints and Considerations

The project must follow these principles:

**Data Integrity**
- Never modify the original dataset unnecessarily.
- Do not assume that synthetic competition data exactly matches the original dataset.
- Investigate differences between original and generated datasets when the original data is available.

**Data Leakage Prevention**
- Fit preprocessing transformations only on training data within each validation fold.
- Do not use test data for model selection.
- Investigate potentially leakage-prone variables, especially `N_Days`.
- Ensure that every feature is appropriate for the intended prediction setting.

**Model Reliability**
- Prioritize Log Loss over accuracy.
- Investigate class imbalance.
- Ensure correct probability column ordering.
- Evaluate probability calibration.
- Maintain reproducible random seeds.

**Healthcare Responsibility**
- Predictions represent statistical estimates.
- The application must not claim to diagnose cirrhosis or determine an individual's actual survival.
- Clearly communicate model limitations and uncertainty.

## 11. Success Criteria

The project will be considered successful when:

1. The dataset is thoroughly explored and understood.
2. Multiple Machine Learning models are trained and compared.
3. Model selection is supported by cross-validation results.
4. The final model is optimized for multiclass Log Loss.
5. Probability calibration is investigated.
6. The trained model can be saved and loaded independently.
7. Individual patient predictions work correctly.
8. Batch predictions work correctly.
9. The Streamlit application runs without errors.
10. Competition submission files are generated in the required format.
11. The application provides clear visualizations and meaningful insights.
12. The entire workflow is reproducible and documented.

## 12. Future Improvements

Potential future improvements include:

- SHAP-based model explainability.
- More advanced ensemble methods.
- Improved probability calibration.
- Comparison with the original competition dataset.
- Model monitoring and prediction diagnostics.
- Deployment to a cloud platform.
- Enhanced interactive clinical data visualizations.

## 13. Final Product Definition

The final deliverable is a complete Machine Learning application that combines predictive modeling, clinical data exploration, probability estimation, and interactive visualization.

The system should demonstrate a complete applied Machine Learning workflow, from raw patient data to validated predictive models and an accessible web-based interface.

**The ultimate goal is not simply to predict a patient's outcome, but to build a technically sound system that demonstrates how clinical data can be transformed into meaningful probabilistic insights.**
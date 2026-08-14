"""
StrokeGuard AI — Stroke Risk Assessment App
Run with: streamlit run app.py   (from inside the app/ folder)
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import streamlit as st

# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="StrokeGuard AI",
    page_icon="🧠",
    layout="centered"
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_stroke_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")
BACKGROUND_PATH = os.path.join(MODEL_DIR, "shap_background.pkl")


@st.cache_resource
def load_artifacts():
    if not all(os.path.exists(p) for p in [MODEL_PATH, PREPROCESSOR_PATH, BACKGROUND_PATH]):
        return None, None, None
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    background = joblib.load(BACKGROUND_PATH)
    return model, preprocessor, background


@st.cache_resource
def get_explainer(_model, _background):
    # LinearExplainer is fast and exact for Logistic Regression — safe to build fresh here
    return shap.LinearExplainer(_model, _background)


model, preprocessor, background = load_artifacts()

st.title("🧠 StrokeGuard AI")
st.caption("An Explainable Machine Learning System for Early Stroke Risk Assessment")

if model is None or preprocessor is None or background is None:
    st.error(
        "Model files not found. Please run notebooks 02, 03, and 04 first — "
        "they generate the files this app needs (including the SHAP background sample)."
    )
    st.stop()

explainer = get_explainer(model, background)

st.write("Enter patient health details below to get an instant stroke risk assessment.")
st.divider()

# ---------------------------------------------------------
# Input form
# ---------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=100, value=50)
    gender = st.selectbox("Gender", ["Male", "Female"])
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
    ever_married = st.selectbox("Ever Married", ["Yes", "No"])

with col2:
    avg_glucose_level = st.number_input("Average Glucose Level (mg/dL)", min_value=50.0, max_value=300.0, value=100.0)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

predict_clicked = st.button("Assess Stroke Risk", type="primary", use_container_width=True)

st.divider()

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if predict_clicked:
    patient = pd.DataFrame([{
        "gender": gender,
        "age": float(age),
        "hypertension": 1 if hypertension == "Yes" else 0,
        "heart_disease": 1 if heart_disease == "Yes" else 0,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": residence_type,
        "avg_glucose_level": float(avg_glucose_level),
        "bmi": float(bmi),
        "smoking_status": smoking_status
    }])

    patient_processed = preprocessor.transform(patient)
    patient_dense = patient_processed.toarray() if hasattr(patient_processed, "toarray") else patient_processed

    probability = model.predict_proba(patient_dense)[0][1]
    prediction = model.predict(patient_dense)[0]

    st.subheader("Result")

    if prediction == 1:
        st.error(f"⚠️ Higher Stroke Risk — Estimated probability: {probability * 100:.1f}%")
    else:
        st.success(f"✅ Lower Stroke Risk — Estimated probability: {probability * 100:.1f}%")

    # Simple visual risk gauge
    st.progress(min(float(probability), 1.0))

    st.caption(
        "This is a screening estimate based on a machine learning model trained on historical "
        "patient data. It is not a medical diagnosis — please consult a healthcare professional."
    )

    # -------------------------------------------------------
    # Real SHAP explanation for this specific patient
    # -------------------------------------------------------
    st.subheader("What influenced this result (SHAP)")

    shap_values = explainer(patient_dense)
    feature_names = preprocessor.get_feature_names_out()
    clean_names = [f.split("__")[-1].replace("_", " ") for f in feature_names]

    fig, ax = plt.subplots(figsize=(7, 4))
    shap.plots.bar(
        shap.Explanation(
            values=shap_values.values[0],
            base_values=shap_values.base_values[0],
            data=patient_dense[0],
            feature_names=clean_names
        ),
        max_display=6,
        show=False
    )
    plt.tight_layout()
    st.pyplot(fig)

    st.caption(
        "This chart is generated live from SHAP values calculated for this exact patient profile "
        "using the trained model — not a fixed or generic explanation."
    )

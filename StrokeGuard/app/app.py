"""
StrokeGuard AI — Stroke Risk Assessment App
Run with: streamlit run app.py   (from inside the app/ folder)
"""

import os
import joblib
import numpy as np
import pandas as pd
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


@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


model, preprocessor = load_artifacts()

st.title("🧠 StrokeGuard AI")
st.caption("An Explainable Machine Learning System for Early Stroke Risk Assessment")

if model is None or preprocessor is None:
    st.error(
        "Model files not found. Please run notebooks 02_Preprocessing.ipynb and "
        "03_Model_Training.ipynb first — they generate the files this app needs."
    )
    st.stop()

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
    probability = model.predict_proba(patient_processed)[0][1]
    prediction = model.predict(patient_processed)[0]

    st.subheader("Result")

    if prediction == 1:
        st.error(f"⚠️ Higher Stroke Risk — Estimated probability: {probability * 100:.1f}%")
    else:
        st.success(f"✅ Lower Stroke Risk — Estimated probability: {probability * 100:.1f}%")

    st.caption(
        "This is a screening estimate based on a machine learning model trained on historical "
        "patient data. It is not a medical diagnosis — please consult a healthcare professional."
    )

    # -------------------------------------------------------
    # Explainability: real coefficients from the trained model
    # (Only works cleanly for Logistic Regression, which is what
    # this project selected. If you swap to a different model,
    # this section will need a different explainability method,
    # e.g. feature_importances_ for Random Forest/XGBoost.)
    # -------------------------------------------------------
    if hasattr(model, "coef_"):
        st.subheader("What influenced this result")

        feature_names = preprocessor.get_feature_names_out()
        coefficients = model.coef_[0]
        patient_values = patient_processed.toarray()[0] if hasattr(patient_processed, "toarray") else patient_processed[0]

        contributions = coefficients * patient_values
        contrib_df = pd.DataFrame({
            "feature": feature_names,
            "contribution": contributions
        }).sort_values("contribution", ascending=False)

        top_risk_factors = contrib_df[contrib_df["contribution"] > 0].head(3)

        if len(top_risk_factors) > 0:
            st.write("Factors that pushed this prediction toward higher risk:")
            for _, row in top_risk_factors.iterrows():
                clean_name = row["feature"].split("__")[-1].replace("_", " ")
                st.write(f"- **{clean_name}**")
        else:
            st.write("No strong individual risk-increasing factors were detected for this profile.")

        st.caption(
            "These factors are calculated directly from the trained model's learned coefficients "
            "for this specific patient — not a fixed or generic list."
        )

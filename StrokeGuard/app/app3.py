"""
StrokeGuard AI — Cozy Healthcare AI Dashboard
Run with: streamlit run app.py

UI/UX redesigned only.
The existing ML artifacts, preprocessing, prediction logic,
SHAP calculation, metadata logic, and disclaimer are preserved.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import streamlit as st

# =========================================================
# PAGE SETUP
# =========================================================

st.set_page_config(
    page_title="StrokeGuard AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# COZY BLUE HEALTHCARE THEME
# =========================================================

st.markdown(
    """
    <style>
    /* ---------- Global ---------- */
    .stApp {
        background: #F5FAFF;
    }

    .main .block-container {
        max-width: 1380px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #EAF4FF 0%, #F5FAFF 100%);
        border-right: 1px solid #C7DDF3;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }

    /* ---------- Typography ---------- */
    h1, h2, h3 {
        color: #183B5B !important;
    }

    p, label, span, div {
        font-family: Inter, Arial, sans-serif;
    }

    .muted {
        color: #668099;
    }

    /* ---------- Header ---------- */
    .hero {
        background: linear-gradient(135deg, #E5F2FF 0%, #F8FCFF 70%);
        border: 1px solid #B9D5EF;
        border-radius: 24px;
        padding: 1.7rem 2rem;
        margin-bottom: 1.3rem;
        box-shadow: 0 8px 24px rgba(64, 112, 151, 0.08);
    }

    .hero-title {
        font-size: 2.35rem;
        font-weight: 800;
        color: #315B83;
        margin: 0;
        letter-spacing: -1px;
    }

    .hero-title .accent {
        color: #6D9FD0;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: #607C96;
        margin-top: 0.35rem;
    }

    .badge {
        display: inline-block;
        background: #DCEEFF;
        color: #356A96;
        border: 1px solid #B7D6F2;
        border-radius: 999px;
        padding: 0.28rem 0.75rem;
        font-size: 0.78rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
    }

    /* ---------- Cards ---------- */
    .soft-card {
        background: rgba(255,255,255,0.92);
        border: 1px solid #C8DDF0;
        border-radius: 22px;
        padding: 1.35rem;
        box-shadow: 0 7px 20px rgba(61, 108, 145, 0.08);
    }

    .section-title {
        color: #244E70;
        font-size: 1.35rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .section-subtitle {
        color: #7590A7;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }

    /* ---------- Risk card ---------- */
    .risk-low {
        background: linear-gradient(135deg, #E9F8F0, #F7FCF9);
        border: 1px solid #A9D9BC;
        border-radius: 22px;
        padding: 1.35rem 1.5rem;
        box-shadow: 0 8px 22px rgba(76, 139, 99, 0.08);
    }

    .risk-high {
        background: linear-gradient(135deg, #FFF0EF, #FFF9F8);
        border: 1px solid #E9B4AF;
        border-radius: 22px;
        padding: 1.35rem 1.5rem;
        box-shadow: 0 8px 22px rgba(170, 85, 76, 0.08);
    }

    .risk-label {
        font-size: 0.85rem;
        color: #6A7D8E;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .risk-value {
        font-size: 2rem;
        font-weight: 850;
        color: #254C69;
        margin: 0.2rem 0;
    }

    .risk-prob {
        font-size: 2.8rem;
        font-weight: 850;
        color: #315B83;
        line-height: 1;
    }

    /* ---------- Metrics ---------- */
    div[data-testid="stMetric"] {
        background: #EFF7FF;
        border: 1px solid #C5DCF0;
        border-radius: 18px;
        padding: 1rem;
        box-shadow: 0 5px 15px rgba(61, 108, 145, 0.06);
    }

    div[data-testid="stMetricLabel"] {
        color: #66819A !important;
    }

    div[data-testid="stMetricValue"] {
        color: #285778 !important;
    }

    /* ---------- Inputs ---------- */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        border-radius: 12px;
        border-color: #C7DDF0;
        background: white;
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        width: 100%;
        border: 1px solid #78A9D4 !important;
        border-radius: 16px !important;
        background: linear-gradient(180deg, #A9D0F3, #8EBBE3) !important;
        color: #214C70 !important;
        font-size: 1.05rem !important;
        font-weight: 800 !important;
        padding: 0.8rem 1rem !important;
        box-shadow: 0 5px 0 #6E9BC2, 0 10px 20px rgba(70, 116, 153, 0.12);
    }

    .stButton > button:hover {
        border-color: #5C91BF !important;
        background: linear-gradient(180deg, #B9D9F6, #98C5EB) !important;
    }

    /* ---------- Divider ---------- */
    hr {
        border-color: #D6E5F2 !important;
    }

    /* ---------- Footer ---------- */
    .footer {
        text-align: center;
        color: #7891A6;
        font-size: 0.82rem;
        padding: 1.5rem 0 0.5rem;
    }

    /* Hide Streamlit decoration */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# EXISTING MODEL PATHS — UNCHANGED
# =========================================================

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_stroke_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")
BACKGROUND_PATH = os.path.join(MODEL_DIR, "shap_background.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")


# =========================================================
# EXISTING ARTIFACT LOADING — UNCHANGED
# =========================================================

@st.cache_resource
def load_artifacts():
    required = [MODEL_PATH, PREPROCESSOR_PATH, BACKGROUND_PATH]

    if not all(os.path.exists(p) for p in required):
        return None, None, None, None

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    background = joblib.load(BACKGROUND_PATH)

    metadata = None

    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)

    return model, preprocessor, background, metadata


# =========================================================
# EXISTING SHAP EXPLAINER — UNCHANGED
# =========================================================

@st.cache_resource
def get_explainer(_model, _background):
    return shap.LinearExplainer(_model, _background)


model, preprocessor, background, metadata = load_artifacts()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="badge">✦ ML-POWERED SCREENING</div>
        <div class="hero-title">
            <span class="accent">StrokeGuard AI:</span> Early Stroke Risk Assessment
        </div>
        <div class="hero-subtitle">
            Explainable machine learning designed to make stroke-risk predictions
            easier to understand.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Preserve existing missing-artifact behavior
if model is None or preprocessor is None or background is None:
    st.error(
        "Model files not found. Please run notebooks 02, 03, and 04 first — "
        "they generate the files this app needs."
    )
    st.stop()

explainer = get_explainer(model, background)


# =========================================================
# SIDEBAR — PATIENT INPUTS
# =========================================================

st.sidebar.markdown("## 🫀 Patient Profile")
st.sidebar.caption("Enter the patient's information below.")

st.sidebar.markdown("### Demographics")

age = st.sidebar.slider(
    "Age",
    min_value=1,
    max_value=100,
    value=50,
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"],
)

ever_married = st.sidebar.selectbox(
    "Ever Married",
    ["Yes", "No"],
)

residence_type = st.sidebar.selectbox(
    "Residence Type",
    ["Urban", "Rural"],
)

work_type = st.sidebar.selectbox(
    "Work Type",
    ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
)

st.sidebar.markdown("### Health Factors")

hypertension = st.sidebar.radio(
    "Hypertension",
    ["No", "Yes"],
    horizontal=True,
)

heart_disease = st.sidebar.radio(
    "Heart Disease",
    ["No", "Yes"],
    horizontal=True,
)

avg_glucose_level = st.sidebar.slider(
    "Average Glucose Level (mg/dL)",
    min_value=50.0,
    max_value=300.0,
    value=100.0,
)

bmi = st.sidebar.slider(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0,
)

st.sidebar.markdown("### Lifestyle")

smoking_status = st.sidebar.selectbox(
    "Smoking Status",
    ["never smoked", "formerly smoked", "smokes", "Unknown"],
)

predict_clicked = st.sidebar.button(
    "🧠  Assess Stroke Risk",
    type="primary",
    use_container_width=True,
)


# =========================================================
# EMPTY STATE
# =========================================================

if not predict_clicked:
    left, right = st.columns([1.35, 1], gap="large")

    with left:
        st.markdown(
            """
            <div class="soft-card">
                <div class="section-title">Welcome to StrokeGuard AI 🧠</div>
                <div class="section-subtitle">
                    A simple, explainable way to explore stroke-risk predictions.
                </div>
                <p style="color:#5F7890; line-height:1.7;">
                    Complete the patient profile in the sidebar and click
                    <b>Assess Stroke Risk</b>. The model will estimate the probability
                    of stroke and show which patient features influenced the result.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="soft-card">
                <div class="section-title">What you'll see</div>
                <div style="color:#627C92; line-height:1.8; margin-top:0.6rem;">
                    ✦ Stroke-risk probability<br>
                    ✦ Visual risk indicator<br>
                    ✦ Live SHAP explanation<br>
                    ✦ Precision, Recall, F1 & ROC-AUC<br>
                    ✦ Clear model-selection reasoning
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="footer">
            StrokeGuard AI · Explainable Machine Learning for Stroke Risk Assessment
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()


# =========================================================
# EXISTING PREDICTION LOGIC — PRESERVED
# =========================================================

with st.spinner("Analyzing patient profile..."):

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
        "smoking_status": smoking_status,
    }])

    patient_processed = preprocessor.transform(patient)

    patient_dense = (
        patient_processed.toarray()
        if hasattr(patient_processed, "toarray")
        else patient_processed
    )

    probability = model.predict_proba(patient_dense)[0][1]
    prediction = model.predict(patient_dense)[0]

    # Existing live per-patient SHAP calculation
    shap_values = explainer(patient_dense)

    feature_names = preprocessor.get_feature_names_out()

    clean_names = [
        f.split("__")[-1].replace("_", " ")
        for f in feature_names
    ]


# =========================================================
# RISK RESULT
# =========================================================

st.markdown(
    '<div class="section-title">Stroke Risk Assessment</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-subtitle">Prediction generated from the patient profile</div>',
    unsafe_allow_html=True,
)

if prediction == 1:
    risk_class = "risk-high"
    risk_title = "Higher Stroke Risk"
    risk_icon = "⚠"
else:
    risk_class = "risk-low"
    risk_title = "Lower Stroke Risk"
    risk_icon = "✓"

risk_col, gauge_col = st.columns([1.15, 1], gap="large")

with risk_col:
    st.markdown(
        f"""
        <div class="{risk_class}">
            <div class="risk-label">Risk Level</div>
            <div class="risk-value">{risk_icon} {risk_title}</div>
            <div style="margin-top:1rem;">
                <div class="risk-label">Estimated Probability</div>
                <div class="risk-prob">{probability * 100:.1f}%</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Existing disclaimer preserved exactly
    st.caption(
        "This is a screening estimate based on a machine learning model trained on "
        "historical patient data. It is not a medical diagnosis — please consult a "
        "healthcare professional."
    )

with gauge_col:
    # Visual-only probability indicator.
    # Does not alter the model prediction or probability.
    fig_gauge, ax_gauge = plt.subplots(
        figsize=(5.2, 2.8),
        subplot_kw={"projection": "polar"},
    )

    ax_gauge.set_theta_zero_location("W")
    ax_gauge.set_theta_direction(-1)

    theta = np.linspace(np.pi, 2 * np.pi, 200)
    probability_clipped = min(max(float(probability), 0.0), 1.0)

    # Soft blue background arc
    ax_gauge.plot(
        theta,
        np.ones_like(theta),
        linewidth=18,
        color="#DCECFB",
        solid_capstyle="round",
    )

    # Actual probability arc
    actual_theta = np.linspace(
        np.pi,
        np.pi + np.pi * probability_clipped,
        100,
    )

    gauge_color = "#D9827C" if prediction == 1 else "#78B894"

    ax_gauge.plot(
        actual_theta,
        np.ones_like(actual_theta),
        linewidth=18,
        color=gauge_color,
        solid_capstyle="round",
    )

    ax_gauge.text(
        0.5,
        0.12,
        f"{probability * 100:.1f}%",
        transform=ax_gauge.transAxes,
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        color="#315B83",
    )

    ax_gauge.text(
        0.5,
        -0.01,
        "Estimated stroke probability",
        transform=ax_gauge.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        color="#71889C",
    )

    ax_gauge.set_ylim(0, 1.15)
    ax_gauge.set_axis_off()

    plt.tight_layout()
    st.pyplot(fig_gauge, use_container_width=True)
    plt.close(fig_gauge)


# =========================================================
# SHAP EXPLANATION
# =========================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">Why did the model make this prediction?</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-subtitle">Live SHAP-based explanation for this exact patient</div>',
    unsafe_allow_html=True,
)

shap_col1, shap_col2 = st.columns([1.5, 1], gap="large")

with shap_col1:
    fig, ax = plt.subplots(figsize=(8, 4.8))

    shap.plots.bar(
        shap.Explanation(
            values=shap_values.values[0],
            base_values=shap_values.base_values[0],
            data=patient_dense[0],
            feature_names=clean_names,
        ),
        max_display=6,
        show=False,
    )

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with shap_col2:
    st.markdown(
        """
        <div class="soft-card">
            <div class="section-title" style="font-size:1.1rem;">
                🧠 Model explanation
            </div>
            <p style="color:#627C92; line-height:1.7;">
                SHAP shows which features influenced the model's prediction for
                <b>this exact patient profile</b>.
            </p>
            <p style="color:#627C92; line-height:1.7;">
                Larger SHAP contributions indicate features that had a stronger
                influence on the model's output.
            </p>
            <p style="color:#7891A6; font-size:0.84rem;">
                This explanation is generated live from the trained model.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# MODEL PERFORMANCE — EXISTING METADATA LOGIC PRESERVED
# =========================================================

if metadata and "metrics_summary" in metadata:

    st.markdown("---")

    st.markdown(
        '<div class="section-title">Model Performance</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">Evaluation metrics from the selected model</div>',
        unsafe_allow_html=True,
    )

    best_name = metadata.get("selected_best_model", "Model")
    metrics = metadata["metrics_summary"].get(best_name, {})

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Precision",
        f"{metrics.get('Precision', 0):.2f}",
    )

    m2.metric(
        "Recall",
        f"{metrics.get('Recall', 0):.2f}",
    )

    m3.metric(
        "F1-Score",
        f"{metrics.get('F1-Score', 0):.2f}",
    )

    m4.metric(
        "ROC-AUC",
        f"{metrics.get('ROC-AUC', 0):.2f}",
    )

    st.info(
        f"**{best_name}** was selected because it achieved the highest recall — "
        "meaning it catches the most real stroke cases. In a medical screening "
        "context, missing a true stroke case is more costly than a false alarm, "
        "so recall was prioritized over raw accuracy or precision."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        <b>StrokeGuard AI</b> · Explainable Machine Learning for Stroke Risk Assessment
        <br>
        Research / educational screening tool — not a medical diagnostic system.
    </div>
    """,
    unsafe_allow_html=True,
)

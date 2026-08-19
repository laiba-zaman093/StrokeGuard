"""
StrokeGuard AI — Enhanced Healthcare AI Dashboard
Run with: streamlit run app_enhanced.py

✨ UI/UX Significantly Redesigned
✓ All existing ML logic, preprocessing, prediction, SHAP, and metadata preserved
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
    initial_sidebar_state="collapsed",
)

# =========================================================
# ENHANCED PROFESSIONAL HEALTHCARE THEME
# =========================================================

st.markdown(
    """
    <style>
    /* ---------- Global ---------- */
    * {
        margin: 0;
        padding: 0;
    }
    
    .stApp {
        background: linear-gradient(135deg, #F5FAFF 0%, #F0F8FF 100%);
    }

    .main .block-container {
        max-width: 1400px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* ---------- Typography ---------- */
    h1, h2, h3 {
        color: #1A3A52 !important;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    p, label, span, div {
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ---------- Header/Hero ---------- */
    .hero-banner {
        background: linear-gradient(135deg, #1A3A52 0%, #2D5A7E 50%, #3D6B95 100%);
        border-radius: 28px;
        padding: 2.5rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 12px 40px rgba(26, 58, 82, 0.15);
        color: white;
    }

    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.25);
        color: #E8F4FF;
        border: 1px solid rgba(255,255,255,0.4);
        border-radius: 999px;
        padding: 0.35rem 0.9rem;
        font-size: 0.8rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
        letter-spacing: 0.05em;
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0.5rem 0;
        letter-spacing: -0.5px;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: rgba(255,255,255,0.85);
        margin-top: 0.5rem;
        line-height: 1.5;
    }

    /* ---------- Metrics Row ---------- */
    .metrics-row {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: white;
        border: 1px solid #D4E7F7;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(26, 58, 82, 0.06);
        text-align: center;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        box-shadow: 0 8px 25px rgba(26, 58, 82, 0.12);
        transform: translateY(-2px);
    }

    .metric-label {
        font-size: 0.85rem;
        color: #6B8BA0;
        font-weight: 600;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #1A3A52;
        margin: 0;
    }

    .metric-change {
        font-size: 0.75rem;
        color: #6B8BA0;
        margin-top: 0.5rem;
    }

    /* ---------- Section Styling ---------- */
    .section-header {
        margin-bottom: 1.8rem;
    }

    .section-title {
        font-size: 1.5rem;
        font-weight: 800;
        color: #1A3A52;
        margin-bottom: 0.4rem;
    }

    .section-subtitle {
        font-size: 0.95rem;
        color: #7A95AD;
        margin: 0;
    }

    /* ---------- Cards ---------- */
    .card {
        background: white;
        border: 1px solid #D4E7F7;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 4px 16px rgba(26, 58, 82, 0.08);
    }

    .card-compact {
        background: white;
        border: 1px solid #D4E7F7;
        border-radius: 18px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(26, 58, 82, 0.06);
    }

    /* ---------- Input Fields ---------- */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        border-radius: 12px !important;
        border: 1.5px solid #D4E7F7 !important;
        background: white !important;
    }

    div[data-baseweb="select"] > div:hover,
    div[data-baseweb="input"] > div:hover {
        border-color: #A9D0F3 !important;
    }

    div[data-baseweb="select"] > div:focus,
    div[data-baseweb="input"] > div:focus {
        border-color: #78A9D4 !important;
    }

    /* ---------- Sliders ---------- */
    div[data-testid="stSlider"] > div {
        padding: 1rem 0;
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        width: 100%;
        border: none !important;
        border-radius: 14px !important;
        background: linear-gradient(135deg, #2D5A7E 0%, #1A3A52 100%) !important;
        color: white !important;
        font-size: 1.1rem !important;
        font-weight: 800 !important;
        padding: 1rem 1.5rem !important;
        box-shadow: 0 6px 20px rgba(26, 58, 82, 0.2);
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #3D6B95 0%, #2D5A7E 100%) !important;
        box-shadow: 0 8px 28px rgba(26, 58, 82, 0.3) !important;
        transform: translateY(-1px);
    }

    /* ---------- Risk Result Cards ---------- */
    .risk-result {
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
    }

    .risk-high {
        background: linear-gradient(135deg, #FFEBEB 0%, #FFE8E8 100%);
        border: 2px solid #F3A8A8;
        box-shadow: 0 8px 25px rgba(243, 115, 115, 0.12);
    }

    .risk-low {
        background: linear-gradient(135deg, #E8F7F0 0%, #E0F5EB 100%);
        border: 2px solid #91D4B8;
        box-shadow: 0 8px 25px rgba(145, 212, 184, 0.12);
    }

    .risk-label {
        font-size: 0.8rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
    }

    .risk-high .risk-label {
        color: #C85555;
    }

    .risk-low .risk-label {
        color: #4A9B7F;
    }

    .risk-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }

    .risk-high .risk-title {
        color: #A64D4D;
    }

    .risk-low .risk-title {
        color: #2D7557;
    }

    .risk-probability {
        font-size: 3.2rem;
        font-weight: 900;
        margin-top: 1rem;
        font-family: 'Courier New', monospace;
    }

    .risk-high .risk-probability {
        color: #C85555;
    }

    .risk-low .risk-probability {
        color: #4A9B7F;
    }

    /* ---------- Info Boxes ---------- */
    .info-box {
        background: rgba(64, 112, 151, 0.06);
        border-left: 4px solid #3D6B95;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 1rem 0;
        color: #3A5571;
        line-height: 1.6;
    }

    /* ---------- Divider ---------- */
    hr {
        border: 0 !important;
        height: 1px !important;
        background: linear-gradient(to right, #E5EFF8, #D4E7F7, #E5EFF8) !important;
        margin: 2rem 0 !important;
    }

    /* ---------- Footer ---------- */
    .footer {
        text-align: center;
        color: #7A95AD;
        font-size: 0.85rem;
        padding: 2rem 0 1rem;
        border-top: 1px solid #E5EFF8;
        margin-top: 3rem;
    }

    /* Hide Streamlit UI elements */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stDeployButton { visibility: hidden; }

    /* ---------- Custom Metrics ---------- */
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #D4E7F7;
        border-radius: 16px;
        padding: 1.3rem;
        box-shadow: 0 3px 10px rgba(26, 58, 82, 0.05);
    }

    div[data-testid="stMetricLabel"] {
        color: #7A95AD !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }

    div[data-testid="stMetricValue"] {
        color: #1A3A52 !important;
        font-weight: 800 !important;
    }

    /* ---------- Layout spacing ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        border: 1px solid #D4E7F7;
        background: white;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #E8F2FF, #F0F8FF) !important;
        border-color: #78A9D4 !important;
    }
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
    <div class="hero-banner">
        <div class="hero-badge">✦ ML-POWERED SCREENING SYSTEM</div>
        <div class="hero-title">🧠 StrokeGuard AI</div>
        <div class="hero-subtitle">
            Advanced machine learning for early stroke risk detection. Evidence-based assessment with explainable AI insights.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Preserve existing missing-artifact behavior
if model is None or preprocessor is None or background is None:
    st.error(
        "⚠️ Model artifacts not found. Please ensure the following files exist in `../models/`:\n\n"
        "• `best_stroke_model.pkl`\n"
        "• `preprocessor.pkl`\n"
        "• `shap_background.pkl`"
    )

    st.info(
        "📖 **About StrokeGuard AI:**\n\n"
        "This system combines multiple machine learning algorithms to predict stroke risk based on patient health indicators. Each prediction includes:\n\n"
        "✦ Real-time SHAP explanation\n"
        "✦ Model performance metrics\n"
        "✦ Clear clinical interpretation"
    )

    st.markdown(
        """
        <div class="footer">
            StrokeGuard AI · ML-powered Stroke Risk Assessment
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()


# =========================================================
# PATIENT ASSESSMENT FORM
# =========================================================

st.markdown(
    """
    <div class="section-header">
        <div class="section-title">📋 Patient Assessment</div>
        <div class="section-subtitle">Enter patient information to assess stroke risk</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# FORM SECTION — Better organized with columns
form_container = st.container()

with form_container:
    with st.form("patient_form"):
        
        # Row 1: Demographics
        col1, col2, col3 = st.columns(3, gap="large")
        
        with col1:
            age = st.slider(
                "Age (years)",
                min_value=18,
                max_value=100,
                value=45,
                step=1,
            )
        
        with col2:
            gender = st.selectbox(
                "Gender",
                ["Male", "Female"],
                index=0,
            )
        
        with col3:
            ever_married = st.selectbox(
                "Marital Status",
                ["Married", "Single", "Divorced", "Widowed"],
                index=0,
            )

        # Row 2: Health Indicators
        col4, col5, col6 = st.columns(3, gap="large")
        
        with col4:
            hypertension = st.selectbox(
                "Hypertension",
                ["No", "Yes"],
                index=0,
            )
        
        with col5:
            heart_disease = st.selectbox(
                "Heart Disease",
                ["No", "Yes"],
                index=0,
            )
        
        with col6:
            avg_glucose_level = st.slider(
                "Glucose Level (mg/dL)",
                min_value=50,
                max_value=300,
                value=120,
                step=5,
            )

        # Row 3: Physical & Lifestyle
        col7, col8, col9 = st.columns(3, gap="large")
        
        with col7:
            bmi = st.slider(
                "BMI (kg/m²)",
                min_value=10.0,
                max_value=50.0,
                value=25.0,
                step=0.5,
            )
        
        with col8:
            work_type = st.selectbox(
                "Work Type",
                ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
                index=0,
            )
        
        with col9:
            smoking_status = st.selectbox(
                "Smoking Status",
                ["never smoked", "formerly smoked", "smokes", "Unknown"],
                index=0,
            )

        # Row 4: Additional Info
        col10, col11 = st.columns(2, gap="large")
        
        with col10:
            residence_type = st.selectbox(
                "Residence Type",
                ["Urban", "Rural"],
                index=0,
            )

        # Submit Button
        st.markdown("")
        submit_button = st.form_submit_button(
            "🔬 Assess Stroke Risk",
            use_container_width=True,
        )

    if not submit_button:
        st.stop()


# =========================================================
# EXISTING PREDICTION LOGIC — PRESERVED
# =========================================================

with st.spinner("🔄 Analyzing patient profile..."):

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
    explainer = get_explainer(model, background)
    shap_values = explainer(patient_dense)

    feature_names = preprocessor.get_feature_names_out()

    clean_names = [
        f.split("__")[-1].replace("_", " ").title()
        for f in feature_names
    ]


# =========================================================
# DIVIDER
# =========================================================

st.markdown("---")


# =========================================================
# RISK ASSESSMENT RESULT
# =========================================================

st.markdown(
    """
    <div class="section-header">
        <div class="section-title">🎯 Risk Assessment Result</div>
        <div class="section-subtitle">Prediction and clinical interpretation</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Risk result layout
col_risk, col_gauge = st.columns([1.2, 1], gap="large")

with col_risk:
    if prediction == 1:
        risk_class = "risk-high"
        risk_label_text = "Higher Risk"
        risk_icon = "⚠️"
        risk_color = "#C85555"
    else:
        risk_class = "risk-low"
        risk_label_text = "Lower Risk"
        risk_icon = "✓"
        risk_color = "#4A9B7F"

    st.markdown(
        f"""
        <div class="risk-result {risk_class}">
            <div class="risk-label">{risk_icon} Risk Level</div>
            <div class="risk-title">{risk_label_text} of Stroke</div>
            <div style="margin-top: 1rem;">
                <div class="risk-label">Probability Score</div>
                <div class="risk-probability">{probability * 100:.1f}%</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("📖 What does this mean?"):
        if prediction == 1:
            st.markdown("""
            **This patient shows indicators associated with higher stroke risk.**
            
            - Recommend clinical consultation and further evaluation
            - Consider preventive measures and lifestyle modifications
            - Regular monitoring of risk factors is advised
            """)
        else:
            st.markdown("""
            **This patient shows indicators associated with lower stroke risk.**
            
            - Continue healthy lifestyle practices
            - Regular health check-ups are still recommended
            - Monitor for any changes in health status
            """)

    st.caption(
        "⚕️ **Disclaimer:** This is a screening estimate based on machine learning trained on historical data. "
        "It is NOT a medical diagnosis. Always consult a healthcare professional for clinical decisions."
    )

with col_gauge:
    # Probability gauge visualization
    fig_gauge, ax_gauge = plt.subplots(
        figsize=(5.5, 3),
        subplot_kw={"projection": "polar"},
    )

    ax_gauge.set_theta_zero_location("W")
    ax_gauge.set_theta_direction(-1)

    theta = np.linspace(np.pi, 2 * np.pi, 200)
    probability_clipped = min(max(float(probability), 0.0), 1.0)

    # Background arc
    ax_gauge.plot(
        theta,
        np.ones_like(theta),
        linewidth=20,
        color="#E5EFF8",
        solid_capstyle="round",
    )

    # Probability arc
    actual_theta = np.linspace(
        np.pi,
        np.pi + np.pi * probability_clipped,
        100,
    )

    gauge_color = "#E07A7A" if prediction == 1 else "#7AC89D"

    ax_gauge.plot(
        actual_theta,
        np.ones_like(actual_theta),
        linewidth=20,
        color=gauge_color,
        solid_capstyle="round",
    )

    ax_gauge.text(
        0.5,
        0.15,
        f"{probability * 100:.1f}%",
        transform=ax_gauge.transAxes,
        ha="center",
        va="center",
        fontsize=28,
        fontweight="bold",
        color="#1A3A52",
        family="monospace",
    )

    ax_gauge.text(
        0.5,
        -0.05,
        "Stroke Probability",
        transform=ax_gauge.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color="#7A95AD",
        fontweight=600,
    )

    ax_gauge.set_ylim(0, 1.15)
    ax_gauge.set_axis_off()

    plt.tight_layout()
    st.pyplot(fig_gauge, use_container_width=True)
    plt.close(fig_gauge)


# =========================================================
# DIVIDER
# =========================================================

st.markdown("---")


# =========================================================
# SHAP EXPLANATION
# =========================================================

st.markdown(
    """
    <div class="section-header">
        <div class="section-title">🔍 Feature Importance Analysis</div>
        <div class="section-subtitle">Which factors influenced this prediction?</div>
    </div>
    """,
    unsafe_allow_html=True,
)

shap_col1, shap_col2 = st.columns([1.6, 1], gap="large")

with shap_col1:
    st.markdown("**Top factors affecting the prediction:**")
    
    fig, ax = plt.subplots(figsize=(8.5, 5))

    shap.plots.bar(
        shap.Explanation(
            values=shap_values.values[0],
            base_values=shap_values.base_values[0],
            data=patient_dense[0],
            feature_names=clean_names,
        ),
        max_display=8,
        show=False,
    )

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with shap_col2:
    st.markdown(
        """
        <div class="card-compact">
            <div style="font-size: 1.1rem; font-weight: 800; color: #1A3A52; margin-bottom: 1rem;">
                💡 How to read this
            </div>
            <p style="color: #5A7E96; line-height: 1.7; font-size: 0.95rem;">
                <b>SHAP values</b> show which patient features most influenced the model's prediction for <b>this exact profile</b>.
            </p>
            <p style="color: #5A7E96; line-height: 1.7; font-size: 0.95rem;">
                Longer bars = stronger influence on the risk score.
            </p>
            <p style="color: #7A95AD; font-size: 0.85rem; margin-top: 1rem;">
                ✓ Updated in real-time based on patient data
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# DIVIDER
# =========================================================

st.markdown("---")


# =========================================================
# MODEL PERFORMANCE METRICS
# =========================================================

if metadata and "metrics_summary" in metadata:

    st.markdown(
        """
        <div class="section-header">
            <div class="section-title">📊 Model Performance</div>
            <div class="section-subtitle">Evaluation metrics from the selected model</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    best_name = metadata.get("selected_best_model", "Selected Model")
    metrics = metadata["metrics_summary"].get(best_name, {})

    # Metrics in a clean row
    m1, m2, m3, m4 = st.columns(4, gap="medium")

    with m1:
        st.metric(
            "Precision",
            f"{metrics.get('Precision', 0):.3f}",
            help="True positives / All predicted positives",
        )

    with m2:
        st.metric(
            "Recall",
            f"{metrics.get('Recall', 0):.3f}",
            help="True positives / All actual positives",
        )

    with m3:
        st.metric(
            "F1-Score",
            f"{metrics.get('F1-Score', 0):.3f}",
            help="Harmonic mean of precision & recall",
        )

    with m4:
        st.metric(
            "ROC-AUC",
            f"{metrics.get('ROC-AUC', 0):.3f}",
            help="Area under receiver operating curve",
        )

    # Model selection rationale
    st.markdown(
        f"""
        <div class="info-box">
            <b>Why {best_name}?</b><br>
            This model was selected for highest <b>Recall</b> ({metrics.get('Recall', 0):.3f}). 
            In medical screening, missing true stroke cases is riskier than false alarms. 
            Recall prioritizes catching real positives.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        <b>🧠 StrokeGuard AI</b> · Explainable ML for Stroke Risk Screening<br>
        <span style="font-size: 0.8rem;">
            Educational & Research Tool · Not a Medical Diagnostic System
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

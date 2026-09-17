"""
Farmer Crop Advisory System — Main Application Entry Point
Modular Streamlit Dashboard for Leaf Disease Diagnosis & Crop/Soil Suitability Analysis.
"""

from __future__ import annotations

import pathlib
import pandas as pd
import streamlit as st
from PIL import Image

import config
from src.utils import setup_device, pretty_crop
from src.data_loader import load_models, load_profile_data
from src.analysis import classify_leaf, predict_soil, diagnose_soil, generate_advisory_summary
from src.visualization import plot_soil_parameters_bar, format_status_badge

# ============================================================
# PAGE CONFIGURATION & STYLING
# ============================================================

st.set_page_config(
    page_title="Farmer Crop Advisory System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: #1e3d29;
        margin-bottom: 2px;
    }
    .subtitle {
        color: #555;
        font-size: 17px;
        margin-bottom: 20px;
    }
    .status-card {
        padding: 16px;
        border-radius: 8px;
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown('<div class="main-title">🌱 Farmer Crop Advisory System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Intelligent Leaf Pathology Diagnosis & Crop-Conditioned Soil Suitability Advisory</div>',
    unsafe_allow_html=True,
)

# ============================================================
# HARDWARE INITIALIZATION
# ============================================================

gpu_active, device_msg = setup_device()
if gpu_active:
    st.success(device_msg)
else:
    st.info(device_msg)

# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

try:
    image_model, image_classes, soil_model = load_models(
        image_model_path=config.IMAGE_MODEL_PATH,
        image_classes_path=config.IMAGE_CLASSES_PATH,
        soil_model_path=config.SOIL_MODEL_PATH,
    )
except Exception as exc:
    st.error(f"Error loading model artifacts: {exc}")
    st.stop()

# ============================================================
# SIDEBAR / NAVIGATION
# ============================================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/wheat.png", width=64)
    st.title("System Controls")
    st.markdown("---")
    st.markdown("**Pipeline Mode**: Dual-Input Neural Network")
    st.markdown(f"**Vision Classes**: {len(image_classes)} pathology categories")
    st.markdown(f"**Supported Crops**: {len(config.SOIL_CROPS)} agricultural profiles")
    st.markdown("---")
    st.caption("AI-Powered Agricultural Decision Support Architecture")

# ============================================================
# 1. LEAF PATHOLOGY ANALYSIS
# ============================================================

st.header("1. Leaf Image Analysis")

uploaded_file = st.file_uploader(
    "Upload a crop leaf image for disease diagnosis (JPG, PNG, WebP)",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is None:
    st.info("Please upload a leaf photograph to initiate disease identification.")
    st.stop()

leaf_image = Image.open(uploaded_file)
col_img, col_diag = st.columns([1, 1])

with col_img:
    st.image(leaf_image, caption="Uploaded Leaf Specimen", use_container_width=True)

try:
    leaf_result = classify_leaf(leaf_image, image_model, image_classes)
except Exception as exc:
    st.error(f"Leaf image analysis failed: {exc}")
    st.stop()

with col_diag:
    st.subheader("Diagnostic Detection")
    m1, m2 = st.columns(2)
    with m1:
        st.metric("Detected Crop", leaf_result["crop_display"])
    with m2:
        st.metric("Condition / Disease", leaf_result["disease"])

    st.metric("Model Confidence", f"{leaf_result['confidence'] * 100:.2f}%")
    st.caption(f"PlantVillage Class: `{leaf_result['class_name']}`")

    if leaf_result["reliable"]:
        st.success("High confidence detection. Suitable for automated crop advisory.")
    else:
        st.warning(
            "Confidence is below threshold (60%). Please review detection manually before applying treatments."
        )

# ============================================================
# 2. SOIL & ENVIRONMENTAL CONDITIONS
# ============================================================

st.divider()
st.header("2. Soil & Environmental Parameters")
st.write(f"Target Crop Profile: **{leaf_result['crop_display']}**")

col_n, col_p, col_k = st.columns(3)

with col_n:
    nitrogen = st.number_input("Nitrogen (N) - mg/kg", min_value=0.0, max_value=500.0, value=50.0, step=1.0)
    temperature = st.number_input("Temperature (°C)", min_value=-20.0, max_value=70.0, value=25.0, step=0.5)

with col_p:
    phosphorus = st.number_input("Phosphorus (P) - mg/kg", min_value=0.0, max_value=500.0, value=40.0, step=1.0)
    humidity = st.number_input("Relative Humidity (%)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)

with col_k:
    potassium = st.number_input("Potassium (K) - mg/kg", min_value=0.0, max_value=500.0, value=40.0, step=1.0)
    ph = st.number_input("Soil pH Level", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
    rainfall = st.number_input("Annual Rainfall (mm)", min_value=0.0, max_value=5000.0, value=100.0, step=5.0)

# Submit button
analyze_btn = st.button("Generate Farm Advisory Analysis", type="primary", use_container_width=True)

if not analyze_btn:
    st.stop()

soil_readings = {
    "N": nitrogen,
    "P": phosphorus,
    "K": potassium,
    "temperature": temperature,
    "humidity": humidity,
    "ph": ph,
    "rainfall": rainfall,
}

# Run neural network prediction
try:
    with st.spinner("Evaluating soil suitability using neural network..."):
        soil_result = predict_soil(soil_readings, leaf_result["crop"], soil_model)
except Exception as exc:
    st.error(f"Soil suitability evaluation failed: {exc}")
    st.stop()

# Load profile data for empirical diagnostics
profile_df = load_profile_data()
diagnostics = diagnose_soil(soil_readings, leaf_result["crop"], profile_df)
advisory = generate_advisory_summary(leaf_result, soil_result, diagnostics)

# ============================================================
# 3. AGRICULTURAL ADVISORY & RESULTS
# ============================================================

st.divider()
st.header("3. Comprehensive Agricultural Advisory")

score_col, status_col, conf_col = st.columns(3)
with score_col:
    st.metric("Soil Suitability Score", f"{soil_result['score']:.1f} / 100")
with status_col:
    st.metric("Suitability Status", soil_result["status"])
with conf_col:
    st.metric("Vision Confidence", f"{leaf_result['confidence'] * 100:.1f}%")

if soil_result["status"] == "GOOD":
    st.success(f"Optimal Conditions: The field parameters are well-suited for cultivating {leaf_result['crop_display']}.")
elif soil_result["status"] == "ACCEPTABLE":
    st.info(f"Moderate Conditions: Cultivation of {leaf_result['crop_display']} is feasible with routine monitoring.")
elif soil_result["status"] == "NEEDS ATTENTION":
    st.warning(f"Caution: Important parameter adjustments are advised for {leaf_result['crop_display']}.")
else:
    st.error(f"Suboptimal Conditions: Current environment presents significant risks for {leaf_result['crop_display']}.")

# Diagnostic Breakdown Table & Visualization
st.subheader("Diagnostic Breakdown")
col_chart, col_table = st.columns([1, 1])

with col_chart:
    fig = plot_soil_parameters_bar(diagnostics)
    if fig:
        st.pyplot(fig)

with col_table:
    if diagnostics:
        table_df = pd.DataFrame(diagnostics)
        st.dataframe(
            table_df[["Parameter", "Value", "Optimal_Range", "Status"]],
            use_container_width=True,
            hide_index=True,
        )

# Deviations list
deviations = advisory["deviations"]
if deviations:
    st.subheader("Identified Soil/Environmental Deviations")
    for dev in deviations:
        st.markdown(f"- ⚠️ {dev}")
else:
    st.success("All soil and environmental metrics are within the recommended empirical range.")

# Actionable recommendations
st.subheader("Actionable Agronomic Recommendations")
for rec in advisory["recommendations"]:
    st.markdown(f"- 📋 **{rec}**")

# Disclaimer
st.caption(
    "Disclaimer: This advisory system provides machine-learning decision support. "
    "Agricultural and disease treatments should be validated by agronomists or local agricultural extension services."
)

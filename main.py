"""
Farmer Crop Advisory System — Unified Single-Page Application
Fully automated: Crop species and disease are automatically detected from the leaf photo.
No manual crop selection required. Integrates soil suitability modeling and targeted agronomic prescriptions.
"""

from __future__ import annotations

import io
import pathlib
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st

import config
from src.analysis import (
    classify_leaf,
    diagnose_soil,
    generate_advisory_summary,
    predict_soil,
    recommend_crops,
)
from src.data_loader import load_crop_data, load_models, load_soil_profiles
from src.utils import canonical_crop, pretty_crop, setup_device
from src.visualization import (
    format_status_badge,
    plot_crop_recommendations,
    plot_soil_parameters_bar,
    render_advisory,
    render_device_status,
    render_header,
)

# ============================================================
# 1. PAGE CONFIGURATION & STYLING
# ============================================================

st.set_page_config(
    page_title="Farmer Crop & Soil Advisory AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, #f8fbf9 0%, #e9f5ed 100%);
        border: 1px solid #cce3d5;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1b4332;
        margin-bottom: 8px;
        border-bottom: 2px solid #52b788;
        padding-bottom: 4px;
    }
    div[data-testid="stExpander"] {
        border-radius: 8px;
        border: 1px solid #d8e2dc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Render Banner Header
render_header()

# ============================================================
# 2. HARDWARE & CACHED MODEL INITIALIZATION
# ============================================================

gpu_active, device_msg = setup_device()
render_device_status(gpu_active, device_msg)

if "models_ready" not in st.session_state:
    with st.spinner("Initializing neural network models & caching weights..."):
        try:
            image_model, image_classes, soil_model = load_models(
                image_model_path=config.DISEASE_MODEL,
                image_classes_path=config.DISEASE_CLASSES,
                soil_model_path=config.SOIL_MODEL,
            )
        except Exception as exc:
            st.error(f"Failed to load neural network artifacts: {exc}")
            st.stop()

        profile_df = load_crop_data(config.CROP_DATA)
        profiles_dict = load_soil_profiles(config.SOIL_PROFILES)
        st.session_state.models_ready = True
else:
    image_model, image_classes, soil_model = load_models(
        image_model_path=config.DISEASE_MODEL,
        image_classes_path=config.DISEASE_CLASSES,
        soil_model_path=config.SOIL_MODEL,
    )
    profile_df = load_crop_data(config.CROP_DATA)
    profiles_dict = load_soil_profiles(config.SOIL_PROFILES)

# ============================================================
# 3. SIDEBAR CONTROLS (SOIL PRESETS ONLY)
# ============================================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/wheat.png", width=64)
    st.markdown("### 🌾 Quick Field Presets")
    st.caption("Auto-populate field metrics or adjust sliders manually:")

    preset_choice = st.selectbox(
        "Apply Soil Environment Preset:",
        options=["Custom Inputs", "Tomato Garden", "Corn Field", "Apple Orchard", "Grape Vineyard"],
        index=1,
    )

    preset_values = {
        "Tomato Garden": {"N": 60.0, "P": 50.0, "K": 60.0, "temperature": 25.0, "humidity": 78.0, "ph": 6.2, "rainfall": 120.0},
        "Corn Field": {"N": 80.0, "P": 45.0, "K": 40.0, "temperature": 24.0, "humidity": 65.0, "ph": 6.5, "rainfall": 85.0},
        "Apple Orchard": {"N": 25.0, "P": 125.0, "K": 150.0, "temperature": 18.0, "humidity": 70.0, "ph": 6.0, "rainfall": 110.0},
        "Grape Vineyard": {"N": 20.0, "P": 130.0, "K": 200.0, "temperature": 22.0, "humidity": 80.0, "ph": 6.0, "rainfall": 70.0},
    }

    current_preset = preset_values.get(preset_choice, {})
    st.markdown("---")
    st.caption("Farmer Advisory System v2.2 • Fully Automated AI Engine")

# ============================================================
# 4. UNIFIED INPUT WORKSPACE: TWO INTUITIVE STEPS
# ============================================================

st.markdown(
    """
    <div style="background-color: #f1f8f5; border-left: 5px solid #2d6a4f; padding: 12px 18px; border-radius: 6px; margin-bottom: 20px;">
        <strong>How it works:</strong> Take or upload a photo of your crop's leaf. The AI <strong>automatically identifies the crop species and diagnoses any disease</strong>.
        It then pairs this with your soil readings to uncover why the disease occurred and prescribe the exact soil adjustments needed.
    </div>
    """,
    unsafe_allow_html=True,
)

col_leaf, col_soil = st.columns([1, 1], gap="large")

# ------------------------------------------------------------
# STEP 1: CROP LEAF SPECIMEN (AUTO-DETECTS CROP & DISEASE)
# ------------------------------------------------------------
leaf_img: Optional[Image.Image] = None
leaf_source_name = ""

def load_image_safely(file_or_path: Any) -> Optional[Image.Image]:
    """Safely reads, corrects EXIF orientation, and converts any uploaded image to RGB."""
    try:
        if hasattr(file_or_path, "getvalue"):
            raw_bytes = file_or_path.getvalue()
        elif isinstance(file_or_path, (bytes, bytearray)):
            raw_bytes = file_or_path
        elif isinstance(file_or_path, (str, pathlib.Path)):
            with open(file_or_path, "rb") as f:
                raw_bytes = f.read()
        else:
            return None

        img = Image.open(io.BytesIO(raw_bytes))
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")
    except Exception as err:
        st.error(f"Error reading image: {err}")
        return None

with col_leaf:
    st.markdown('<div class="section-title">📸 Step 1: Crop Leaf Specimen</div>', unsafe_allow_html=True)
    st.caption("Upload a leaf photo, pick a demo sample, or snap with your camera. The AI auto-detects the crop:")

    uploaded_file = st.file_uploader(
        "Upload or drag & drop leaf photo (any image format):",
        type=None,
        key="file_uploader_specimen",
    )

    sample_dict = {
        "None (Upload My Own)": None,
        "🍅 Tomato — Early Blight": config.TEST_DIR / "Tomato - Early Blight",
        "🌽 Corn — Common Rust": config.TEST_DIR / "Corn (Maize) - Common Rust",
        "🫑 Bell Pepper — Bacterial Spot": config.TEST_DIR / "Bell Pepper - Bacterial Spot",
        "🍏 Apple — Apple Scab": config.TEST_DIR / "Apple - Apple Scab",
        "🥔 Potato — Healthy": config.TEST_DIR / "Potato - Healthy",
        "🍇 Grape — Black Rot": config.TEST_DIR / "Grape - Black Rot",
        "🍑 Peach — Bacterial Spot": config.TEST_DIR / "Peach - Bacterial Spot",
        "🍓 Strawberry — Leaf Scorch": config.TEST_DIR / "Strawberry - Leaf Scorch",
    }

    selected_sample_key = st.selectbox(
        "Or pick an instant verified field specimen to test:",
        options=list(sample_dict.keys()),
        index=0,
        key="sample_leaf_select",
    )

    with st.expander("📷 Or take a photo using camera / webcam", expanded=False):
        camera_file = st.camera_input("Snap picture of crop leaf:", key="camera_specimen")

    # Determine active leaf image source
    if uploaded_file is not None:
        leaf_img = load_image_safely(uploaded_file)
        leaf_source_name = uploaded_file.name
    elif camera_file is not None:
        leaf_img = load_image_safely(camera_file)
        leaf_source_name = "Camera Snapshot"
    elif selected_sample_key != "None (Upload My Own)" and sample_dict.get(selected_sample_key):
        sample_folder = sample_dict[selected_sample_key]
        if sample_folder.exists():
            sample_files = list(sample_folder.glob("*.jpg")) + list(sample_folder.glob("*.JPG"))
            if sample_files:
                leaf_img = load_image_safely(sample_files[0])
                leaf_source_name = selected_sample_key

    # Execute Vision AI Classification & Auto-Detection
    leaf_res: Optional[Dict[str, Any]] = None
    detected_crop_key: Optional[str] = None
    detected_crop_display: str = "Awaiting Crop Leaf"

    if leaf_img is not None:
        st.image(leaf_img, caption=f"Specimen: {leaf_source_name}", width=400)

        with st.spinner("AI analyzing foliar pathology and auto-identifying crop..."):
            leaf_res = classify_leaf(leaf_img, image_model, image_classes)

        detected_crop_key = leaf_res["crop"]
        detected_crop_display = leaf_res["crop_display"]
        detected_disease = leaf_res["disease"]
        confidence_pct = leaf_res["confidence"] * 100.0

        is_healthy = detected_disease.lower() == "healthy"
        status_color = "#2d6a4f" if is_healthy else "#d62828"
        status_icon = "✅" if is_healthy else "⚠️"

        st.markdown(
            f"""
            <div style="background-color: #ffffff; border: 2px solid {status_color}; border-radius: 8px; padding: 14px 18px; margin-top: 10px;">
                <div style="font-size: 1.18rem; font-weight: 700; color: #1b4332;">
                    🌾 Auto-Identified Crop: <span style="color: #2d6a4f;">{detected_crop_display}</span>
                </div>
                <div style="font-size: 1.05rem; font-weight: 600; color: {status_color}; margin-top: 4px;">
                    {status_icon} Condition: {detected_disease} ({confidence_pct:.1f}% confidence)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        treatment = leaf_res.get("treatment", {})
        if treatment and not is_healthy:
            with st.expander("📖 Pathology Symptoms & Management Protocols", expanded=False):
                st.markdown(f"**Observed Symptoms:** {treatment.get('symptoms', 'N/A')}")
                st.markdown(f"**Cultural Sanitation:** {treatment.get('cultural', 'N/A')}")
                st.markdown(f"**Targeted Chemical / Biological Spray:** {treatment.get('chemical', 'N/A')}")
    else:
        st.info("💡 **No crop leaf loaded yet.** Upload a photo, take a picture with your camera, or pick a demo sample above. The AI will automatically identify your crop.")

# ------------------------------------------------------------
# STEP 2: FIELD SOIL & ENVIRONMENTAL CONDITIONS
# ------------------------------------------------------------
with col_soil:
    title_suffix = f"for Auto-Detected {detected_crop_display}" if detected_crop_key else "(Enter Field Measurements)"
    st.markdown(
        f'<div class="section-title">🧪 Step 2: Soil & Weather {title_suffix}</div>',
        unsafe_allow_html=True,
    )
    st.caption("Enter soil test metrics and climate parameters to evaluate suitability & diagnose deficiencies.")

    c1, c2, c3 = st.columns(3)
    with c1:
        n_val = st.number_input(
            "Nitrogen (N) mg/kg",
            min_value=0.0, max_value=500.0,
            value=float(current_preset.get("N", 60.0)),
            step=2.0, key="soil_n",
        )
        temp_val = st.number_input(
            "Temperature (°C)",
            min_value=-20.0, max_value=70.0,
            value=float(current_preset.get("temperature", 25.0)),
            step=0.5, key="soil_temp",
        )

    with c2:
        p_val = st.number_input(
            "Phosphorus (P) mg/kg",
            min_value=0.0, max_value=500.0,
            value=float(current_preset.get("P", 50.0)),
            step=2.0, key="soil_p",
        )
        humidity_val = st.number_input(
            "Humidity (%)",
            min_value=0.0, max_value=100.0,
            value=float(current_preset.get("humidity", 78.0)),
            step=1.0, key="soil_humidity",
        )

    with c3:
        k_val = st.number_input(
            "Potassium (K) mg/kg",
            min_value=0.0, max_value=500.0,
            value=float(current_preset.get("K", 60.0)),
            step=2.0, key="soil_k",
        )
        ph_val = st.number_input(
            "Soil pH (0-14)",
            min_value=0.0, max_value=14.0,
            value=float(current_preset.get("ph", 6.2)),
            step=0.1, key="soil_ph",
        )
        rain_val = st.number_input(
            "Rainfall (mm)",
            min_value=0.0, max_value=5000.0,
            value=float(current_preset.get("rainfall", 120.0)),
            step=5.0, key="soil_rain",
        )

    soil_readings = {
        "N": n_val,
        "P": p_val,
        "K": k_val,
        "temperature": temp_val,
        "humidity": humidity_val,
        "ph": ph_val,
        "rainfall": rain_val,
    }

# ============================================================
# 5. INTEGRATED AI ADVISORY & PRESCRIPTIONS
# ============================================================

st.markdown("---")

if detected_crop_key and leaf_res:
    # Target crop is AUTOMATICALLY the detected crop
    target_crop = detected_crop_key if detected_crop_key in config.SOIL_CROPS else "tomato"

    soil_res = predict_soil(soil_readings, target_crop, soil_model)
    diagnostics = diagnose_soil(soil_readings, target_crop, profile_df, profiles_dict)
    top_crops = recommend_crops(soil_readings, soil_model, top_k=4)

    advisory = generate_advisory_summary(
        leaf_result=leaf_res,
        soil_result=soil_res,
        diagnostics=diagnostics,
        top_crops=top_crops,
        values=soil_readings,
    )

    st.markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px;">
            <h2 style="margin: 0; color: #1b4332;">🚜 Complete Advisory & Soil Prescription: {pretty_crop(target_crop)}</h2>
            <div>{format_status_badge(soil_res['status'])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3 Primary KPI Cards
    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        is_h = leaf_res["disease"].lower() == "healthy"
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-size: 0.85rem; color: #6c757d; font-weight: 600; text-transform: uppercase;">Auto-Detected Crop & Pathology</div>
                <div style="font-size: 1.4rem; font-weight: 700; color: #1b4332;">{pretty_crop(target_crop)}</div>
                <div style="font-size: 0.95rem; font-weight: 600; margin-top: 4px; color: {'#2d6a4f' if is_h else '#d62828'};">
                    {leaf_res['disease']} ({leaf_res['confidence']*100:.1f}%)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-size: 0.85rem; color: #6c757d; font-weight: 600; text-transform: uppercase;">Soil Suitability Score</div>
                <div style="font-size: 1.4rem; font-weight: 700; color: #1b4332;">{soil_res['score']:.1f} / 100</div>
                <div style="font-size: 0.95rem; font-weight: 600; margin-top: 4px; color: #2d6a4f;">
                    Rating: {soil_res['status']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi3:
        deviated_count = len([d for d in diagnostics if d["Status"] != "NORMAL"])
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-size: 0.85rem; color: #6c757d; font-weight: 600; text-transform: uppercase;">Prescription Status</div>
                <div style="font-size: 1.4rem; font-weight: 700; color: #1b4332;">{deviated_count} Metric Deviations</div>
                <div style="font-size: 0.95rem; font-weight: 600; margin-top: 4px; color: {'#2d6a4f' if deviated_count == 0 else '#e76f51'};">
                    {'Balanced Soil Profile' if deviated_count == 0 else 'Nutrient Amendments Needed'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Render Root Causes, Target Soil Concentrations, and Field Actions
    render_advisory(advisory)

    # Visual Comparison Chart & Alternative Crops
    res_col1, res_col2 = st.columns([1.1, 0.9], gap="medium")

    with res_col1:
        st.markdown(f"#### 📊 Measured Readings vs Ideal Range for {pretty_crop(target_crop)}")
        fig_bar = plot_soil_parameters_bar(diagnostics)
        if fig_bar:
            st.pyplot(fig_bar, clear_figure=True)

    with res_col2:
        st.markdown("#### 🔄 Alternative Best Crops for this Soil")
        st.caption("If you plan crop rotation, the neural network ranks these crops as naturally best suited:")
        fig_rec = plot_crop_recommendations(top_crops)
        if fig_rec:
            st.pyplot(fig_rec, clear_figure=True)

else:
    # Clean waiting state when no photo has been supplied yet
    st.markdown(
        """
        <div style="text-align: center; padding: 30px; background-color: #f8fbf9; border: 2px dashed #b7e4c7; border-radius: 12px; margin-bottom: 24px;">
            <h3 style="color: #2d6a4f; margin-bottom: 8px;">📸 Awaiting Crop Leaf Specimen</h3>
            <p style="color: #555; font-size: 1.05rem; max-width: 600px; margin: 0 auto 16px auto;">
                Please upload a photo of your crop's leaf, snap one with your camera, or pick a sample specimen above.
                <strong>The AI will automatically identify your crop</strong>, diagnose any disease, and generate the tailored soil prescription.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # In the meantime, show the farmer what crops naturally fit their soil
    st.markdown("#### 🔄 General Soil Suitability (Based on Entered Field Metrics)")
    st.caption("Here is how your current soil and weather metrics match candidate crops before leaf identification:")
    top_general_crops = recommend_crops(soil_readings, soil_model, top_k=5)
    fig_rec_gen = plot_crop_recommendations(top_general_crops)
    if fig_rec_gen:
        st.pyplot(fig_rec_gen, clear_figure=True)

st.markdown("---")
st.caption(
    "🌱 Farmer Crop Advisory System | Multi-Modal Deep Learning Agricultural Platform | "
    "Models: EfficientNetB0 (Vision Pathology) + Dual-Input Dense Neural Network (Soil Suitability & Quantile Calibration)"
)

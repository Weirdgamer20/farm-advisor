"""
Farmer Crop Advisory System — Unified Single-Page Application
Automated crop species detection, foliar disease diagnosis,
soil suitability prediction, and agronomic prescription generator.
"""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional
from PIL import Image, ImageOps
import streamlit as st

from src.engine import (
    TEST_DIR,
    canonical_crop,
    classify_leaf,
    evaluate_soil_and_crops,
    generate_advisory,
    load_artifacts,
    plot_crop_bars,
    plot_soil_bars,
    pretty_crop,
    setup_device,
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

# Banner Header
st.markdown(
    """
    <div style="padding: 18px 24px; background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 100%); border-radius: 12px; margin-bottom: 24px; color: white;">
        <h1 style="color: #d8f3dc; margin: 0; font-size: 2.2rem; font-weight: 700;">🌱 Farmer Crop Advisory System</h1>
        <p style="color: #b7e4c7; margin: 6px 0 0 0; font-size: 1.05rem;">
            Production Decision Support: Leaf Disease Diagnosis & Crop-Conditioned Soil Suitability Analysis
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 2. HARDWARE & CACHED MODEL INITIALIZATION
# ============================================================

gpu_active, device_msg = setup_device()
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/wheat.png", width=64)
    st.markdown("### ⚙️ Hardware Status")
    if gpu_active:
        st.success(f"🚀 {device_msg}")
    else:
        st.info(f"💻 {device_msg}")
    st.caption("Inference runs seamlessly across CPU and GPU hardware.")

try:
    with st.spinner("Initializing neural network models & caching weights..."):
        image_model, image_classes, soil_model, soil_profiles = load_artifacts()
except Exception as exc:
    st.error(f"Failed to load neural network artifacts: {exc}")
    st.stop()

# ============================================================
# 3. SIDEBAR CONTROLS (SOIL PRESETS)
# ============================================================

with st.sidebar:
    st.markdown("---")
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
    st.caption("Farmer Advisory System v3.0 • Optimized AI Engine")

# ============================================================
# 4. UNIFIED WORKSPACE (LEAF & SOIL)
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

def load_image_safely(file_or_path: Any) -> Optional[Image.Image]:
    try:
        if hasattr(file_or_path, "getvalue"): raw_bytes = file_or_path.getvalue()
        elif isinstance(file_or_path, (bytes, bytearray)): raw_bytes = file_or_path
        else:
            with open(file_or_path, "rb") as f: raw_bytes = f.read()
        return ImageOps.exif_transpose(Image.open(io.BytesIO(raw_bytes))).convert("RGB")
    except Exception as err:
        st.error(f"Error reading image: {err}")
        return None

# STEP 1: CROP LEAF SPECIMEN
leaf_img: Optional[Image.Image] = None
leaf_source_name = ""
leaf_cache_key: Optional[str] = None

with col_leaf:
    st.markdown('<div class="section-title">📸 Step 1: Crop Leaf Specimen</div>', unsafe_allow_html=True)
    st.caption("Upload a leaf photo, pick a demo sample, or snap with your camera:")

    uploaded_file = st.file_uploader("Upload leaf photo (any image format):", type=None, key="leaf_file")

    sample_dict = {
        "None (Upload My Own)": None,
        "🍅 Tomato — Early Blight": TEST_DIR / "Tomato - Early Blight",
        "🌽 Corn — Common Rust": TEST_DIR / "Corn (Maize) - Common Rust",
        "🫑 Bell Pepper — Bacterial Spot": TEST_DIR / "Bell Pepper - Bacterial Spot",
        "🍏 Apple — Apple Scab": TEST_DIR / "Apple - Apple Scab",
        "🥔 Potato — Healthy": TEST_DIR / "Potato - Healthy",
        "🍇 Grape — Black Rot": TEST_DIR / "Grape - Black Rot",
        "🍑 Peach — Bacterial Spot": TEST_DIR / "Peach - Bacterial Spot",
        "🍓 Strawberry — Leaf Scorch": TEST_DIR / "Strawberry - Leaf Scorch",
    }

    selected_sample = st.selectbox("Or pick an instant verified field specimen to test:", options=list(sample_dict.keys()), index=0)
    with st.expander("📷 Or take a photo using camera / webcam", expanded=False):
        camera_file = st.camera_input("Snap picture of crop leaf:", key="leaf_cam")

    if uploaded_file is not None:
        leaf_img = load_image_safely(uploaded_file)
        leaf_source_name = uploaded_file.name
        leaf_cache_key = f"upload_{uploaded_file.name}_{getattr(uploaded_file, 'size', 0)}"
    elif camera_file is not None:
        leaf_img = load_image_safely(camera_file)
        leaf_source_name = "Camera Snapshot"
        leaf_cache_key = f"camera_{camera_file.name}_{getattr(camera_file, 'size', 0)}"
    elif selected_sample != "None (Upload My Own)" and sample_dict.get(selected_sample):
        sample_folder = sample_dict[selected_sample]
        if sample_folder.exists():
            files = list(sample_folder.glob("*.jpg")) + list(sample_folder.glob("*.JPG"))
            if files:
                leaf_img = load_image_safely(files[0])
                leaf_source_name = selected_sample
                leaf_cache_key = f"sample_{selected_sample}"

    leaf_res: Optional[Dict[str, Any]] = None
    detected_crop_key: Optional[str] = None
    detected_crop_display: str = "Awaiting Crop Leaf"

    if leaf_img is not None:
        st.image(leaf_img, caption=f"Specimen: {leaf_source_name}", width=400)

        # Fast Session Cache: Avoids re-running 45MB Vision CNN when user tunes soil sliders
        if st.session_state.get("last_leaf_key") == leaf_cache_key and "cached_leaf_res" in st.session_state:
            leaf_res = st.session_state.cached_leaf_res
        else:
            with st.spinner("AI analyzing foliar pathology and auto-identifying crop..."):
                leaf_res = classify_leaf(leaf_img, image_model, image_classes)
            st.session_state.last_leaf_key = leaf_cache_key
            st.session_state.cached_leaf_res = leaf_res

        detected_crop_key = leaf_res["crop"]
        detected_crop_display = leaf_res["crop_display"]
        disease = leaf_res["disease"]
        conf_pct = leaf_res["confidence"] * 100.0

        is_h = disease.lower() == "healthy"
        color = "#2d6a4f" if is_h else "#d62828"
        icon = "✅" if is_h else "⚠️"

        st.markdown(
            f"""
            <div style="background-color: #ffffff; border: 2px solid {color}; border-radius: 8px; padding: 14px 18px; margin-top: 10px;">
                <div style="font-size: 1.18rem; font-weight: 700; color: #1b4332;">🌾 Auto-Identified Crop: <span style="color: #2d6a4f;">{detected_crop_display}</span></div>
                <div style="font-size: 1.05rem; font-weight: 600; color: {color}; margin-top: 4px;">{icon} Condition: {disease} ({conf_pct:.1f}% confidence)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        treatment = leaf_res.get("treatment", {})
        if treatment and not is_h:
            with st.expander("📖 Pathology Symptoms & Management Protocols", expanded=False):
                st.markdown(f"**Observed Symptoms:** {treatment.get('symptoms', 'N/A')}")
                st.markdown(f"**Cultural Sanitation:** {treatment.get('cultural', 'N/A')}")
                st.markdown(f"**Targeted Chemical / Biological Spray:** {treatment.get('chemical', 'N/A')}")
    else:
        st.session_state.last_leaf_key = None
        st.session_state.pop("cached_leaf_res", None)
        st.info("💡 **No crop leaf loaded yet.** Upload a photo, take a picture with your camera, or pick a demo sample above. The AI will automatically identify your crop.")

# STEP 2: FIELD SOIL CONDITIONS
with col_soil:
    title_sfx = f"for Auto-Detected {detected_crop_display}" if detected_crop_key else "(Enter Field Measurements)"
    st.markdown(f'<div class="section-title">🧪 Step 2: Soil & Weather {title_sfx}</div>', unsafe_allow_html=True)
    st.caption("Enter soil test metrics and climate parameters to evaluate suitability & diagnose deficiencies.")

    c1, c2, c3 = st.columns(3)
    with c1:
        n_val = st.number_input("Nitrogen (N) mg/kg", 0.0, 500.0, float(current_preset.get("N", 60.0)), 2.0, key="soil_n")
        temp_val = st.number_input("Temperature (°C)", -20.0, 70.0, float(current_preset.get("temperature", 25.0)), 0.5, key="soil_temp")
    with c2:
        p_val = st.number_input("Phosphorus (P) mg/kg", 0.0, 500.0, float(current_preset.get("P", 50.0)), 2.0, key="soil_p")
        hum_val = st.number_input("Humidity (%)", 0.0, 100.0, float(current_preset.get("humidity", 78.0)), 1.0, key="soil_hum")
    with c3:
        k_val = st.number_input("Potassium (K) mg/kg", 0.0, 500.0, float(current_preset.get("K", 60.0)), 2.0, key="soil_k")
        ph_val = st.number_input("Soil pH (0-14)", 0.0, 14.0, float(current_preset.get("ph", 6.2)), 0.1, key="soil_ph")
        rain_val = st.number_input("Rainfall (mm)", 0.0, 5000.0, float(current_preset.get("rainfall", 120.0)), 5.0, key="soil_rain")

    soil_readings = {"N": n_val, "P": p_val, "K": k_val, "temperature": temp_val, "humidity": hum_val, "ph": ph_val, "rainfall": rain_val}

# ============================================================
# 5. INTEGRATED ADVISORY & PRESCRIPTIONS
# ============================================================

st.markdown("---")

badge_colors = {"GOOD": "#2d6a4f", "ACCEPTABLE": "#1d3557", "NEEDS ATTENTION": "#e76f51", "NOT SUITABLE": "#d62828"}

if detected_crop_key and leaf_res:
    target_crop = detected_crop_key
    soil_res, ranked_crops, diagnostics = evaluate_soil_and_crops(soil_readings, target_crop, soil_model, soil_profiles)
    advisory = generate_advisory(leaf_res, soil_res, diagnostics, ranked_crops, soil_readings)

    badge_color = badge_colors.get(soil_res["status"], "#6c757d")
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px;">
            <h2 style="margin: 0; color: #1b4332;">🚜 Complete Advisory & Soil Prescription: {pretty_crop(target_crop)}</h2>
            <span style="background-color: {badge_color}; color: #ffffff; padding: 4px 12px; border-radius: 6px; font-weight: 700;">{soil_res['status']}</span>
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
                <div style="font-size: 0.85rem; color: #6c757d; font-weight: 600; text-transform: uppercase;">Auto-Detected Pathology</div>
                <div style="font-size: 1.35rem; font-weight: 700; color: #1b4332;">{pretty_crop(target_crop)}</div>
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
                <div style="font-size: 1.35rem; font-weight: 700; color: #1b4332;">{soil_res['score']:.1f} / 100</div>
                <div style="font-size: 0.95rem; font-weight: 600; margin-top: 4px; color: #2d6a4f;">Rating: {soil_res['status']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi3:
        deviated = len([d for d in diagnostics if d["Status"] != "NORMAL"])
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-size: 0.85rem; color: #6c757d; font-weight: 600; text-transform: uppercase;">Nutrient Status</div>
                <div style="font-size: 1.35rem; font-weight: 700; color: #1b4332;">{deviated} Parameter Deviations</div>
                <div style="font-size: 0.95rem; font-weight: 600; margin-top: 4px; color: {'#2d6a4f' if deviated == 0 else '#e76f51'};">
                    {'Optimal Profile' if deviated == 0 else 'Amendments Needed'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Root causes & Prescriptions
    if advisory.get("disease_causes"):
        st.markdown("#### 🔬 Root Cause: Environmental & Soil Drivers")
        for cause in advisory["disease_causes"]:
            st.markdown(f"- ⚠️ **{cause}**")

    if advisory.get("prescriptions"):
        st.markdown(f"#### 🧪 Target Soil Concentrations & Prescription for {pretty_crop(target_crop)}")
        import pandas as pd
        st.dataframe(pd.DataFrame(advisory["prescriptions"]), use_container_width=True, hide_index=True)

    if advisory.get("recommendations"):
        st.markdown("#### 📋 Actionable Field Guidelines")
        for rec in advisory["recommendations"]:
            st.markdown(f"- **{rec}**")

    # Analytical Charts
    res_col1, res_col2 = st.columns([1.1, 0.9], gap="medium")
    with res_col1:
        st.markdown(f"#### 📊 Measured Readings vs Ideal Range for {pretty_crop(target_crop)}")
        fig_bars = plot_soil_bars(diagnostics)
        if fig_bars: st.pyplot(fig_bars, clear_figure=True)
    with res_col2:
        st.markdown("#### 🔄 Alternative Best Crops for this Soil")
        fig_crops = plot_crop_bars(ranked_crops)
        if fig_crops: st.pyplot(fig_crops, clear_figure=True)
else:
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

    st.markdown("#### 🔄 General Soil Suitability (Based on Entered Field Metrics)")
    st.caption("Here is how your current soil and weather metrics match candidate crops before leaf identification:")
    _, general_crops, _ = evaluate_soil_and_crops(soil_readings, "tomato", soil_model, soil_profiles)
    fig_gen = plot_crop_bars(general_crops)
    if fig_gen: st.pyplot(fig_gen, clear_figure=True)

st.markdown("---")
st.caption("🌱 Farmer Crop Advisory System | EfficientNetB0 Vision Pathology + Dual-Input Neural Suitability Engine")

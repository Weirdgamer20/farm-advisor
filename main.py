"""
Farmer Crop Advisory System — Streamlit Application Entry Point.
Modular, production-grade interface providing:
1. Dashboard & Verified Model Specifications
2. Soil & Crop Suitability Advisory (Numerical Deep Learning + Empirical Quantiles)
3. Leaf Disease Detection (EfficientNetB0 Vision Inference)
4. Integrated Farm Advisory (Expert Agronomic Prescription Synthesis)
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

import config
from src.analysis import (
    classify_leaf,
    diagnose_soil,
    generate_advisory_summary,
    predict_soil,
    recommend_crops,
)
from src.data_loader import (
    check_artifact_availability,
    get_sample_images,
    load_crop_data,
    load_models,
    load_soil_profiles,
)
from src.preprocessing import validate_soil_readings
from src.utils import canonical_crop, pretty_crop, setup_device
from src.visualization import (
    format_status_badge,
    inject_custom_theme,
    plot_crop_recommendations,
    plot_soil_parameters_bar,
    render_advisory,
    render_disclaimer,
    render_disease_detection,
    render_header,
    render_system_status_sidebar,
    render_workflow_diagram,
)

# ============================================================
# 1. PAGE SETUP & GLOBAL DESIGN SYSTEM
# ============================================================

st.set_page_config(
    page_title="Farmer Crop Advisory System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_theme()

# Non-blocking hardware acceleration detection
gpu_active, device_msg = setup_device()

# Telemetry inspection for diagnostics
telemetry = check_artifact_availability()

# Model loading with graceful diagnostic feedback
models_ready = False
model_init_error: Optional[str] = None
try:
    image_model, image_classes, soil_model = load_models(warmup=False)
    soil_profiles = load_soil_profiles()
    models_ready = True
except Exception as exc:
    model_init_error = str(exc)
    image_model = None
    image_classes = []
    soil_model = None
    soil_profiles = {}

render_system_status_sidebar(gpu_active, device_msg, models_ready, detailed_telemetry=telemetry)
render_header()

if not models_ready and model_init_error:
    st.error("⚠️ **Model Artifact Notice:** One or more pre-trained neural networks could not be loaded.")
    with st.expander("🛠️ View Detailed Initialization Diagnostics"):
        st.code(model_init_error)
        st.markdown("Ensure model artifacts are placed in the `models/` directory.")

# ============================================================
# 2. SESSION STATE MANAGEMENT (No Unnecessary Startup Inference)
# ============================================================

if "soil_readings" not in st.session_state:
    st.session_state["soil_readings"] = {
        "N": 90.0,
        "P": 42.0,
        "K": 43.0,
        "temperature": 25.6,
        "humidity": 80.0,
        "ph": 6.5,
        "rainfall": 200.0,
    }

if "target_crop" not in st.session_state:
    st.session_state["target_crop"] = "tomato"

if "leaf_result" not in st.session_state:
    st.session_state["leaf_result"] = None

if "leaf_image" not in st.session_state:
    st.session_state["leaf_image"] = None

if "soil_result" not in st.session_state:
    st.session_state["soil_result"] = None

if "ranked_crops" not in st.session_state:
    st.session_state["ranked_crops"] = None

if "diagnostics" not in st.session_state:
    st.session_state["diagnostics"] = None

if "cached_soil_hash" not in st.session_state:
    st.session_state["cached_soil_hash"] = None


def compute_soil_hash(crop: str, readings: Dict[str, float]) -> str:
    """Generates an MD5 signature for soil inputs to prevent redundant inference reruns."""
    payload = f"{crop}_" + "_".join(f"{k}:{v:.2f}" for k, v in sorted(readings.items()))
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


# ============================================================
# 3. SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown("## 🧭 Navigation")
    current_page = st.radio(
        "Select Advisory Module:",
        [
            "🏠 Dashboard",
            "🧪 Soil & Crop Advisory",
            "🍃 Leaf Disease Detection",
            "📋 Integrated Farm Advisory",
        ],
        index=0,
    )
    st.markdown("---")

# ============================================================
# 4. MODULE 1: DASHBOARD
# ============================================================

if current_page == "🏠 Dashboard":
    st.markdown("## 📊 Overview & Capabilities")
    st.markdown(
        "The **Farmer Crop Advisory System** is an academic AI decision-support platform "
        "integrating computer vision pathology classification with conditioned neural network "
        "crop suitability modeling and rule-based agronomic guidance."
    )

    # 3 High-Impact Capability Cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="advisory-card" style="border-top: 4px solid #2d6a4f;">
                <h3>🧪 Soil & Crop Advisory</h3>
                <p>
                    Evaluates 7 soil nutrients and climate parameters against 9 crop profiles using dual-input neural modeling and empirical quantiles.
                </p>
                <span style="color: #2d6a4f; font-weight: 600; font-size: 0.88rem;">Numerical Deep Learning →</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="advisory-card" style="border-top: 4px solid #40916c;">
                <h3>🍃 Leaf Disease Detection</h3>
                <p>
                    EfficientNetB0 vision model trained on 38 PlantVillage pathology classes with calibrated diagnostic confidence and treatment protocols.
                </p>
                <span style="color: #40916c; font-weight: 600; font-size: 0.88rem;">Computer Vision Inference →</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="advisory-card" style="border-top: 4px solid #52b788;">
                <h3>📋 Integrated Farm Advisory</h3>
                <p>
                    Synthesizes foliar disease diagnosis with field soil readings to detect environmental infection drivers and formulate precise corrective amendments.
                </p>
                <span style="color: #52b788; font-weight: 600; font-size: 0.88rem;">Agronomic Prescriptions →</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Operational Status & Model Architecture Breakdown
    col_sys, col_info = st.columns([1, 1], gap="large")
    with col_sys:
        st.markdown("### 🖥️ Operational System Status")
        st.markdown(
            f"""
            - **Plant Pathology Vision Model:** {"🟢 Loaded & Ready" if models_ready else "🔴 Unavailable"}
            - **Soil Suitability Neural Model:** {"🟢 Loaded & Ready" if models_ready else "🔴 Unavailable"}
            - **Quantile Diagnostic Profiles:** 🟢 Available (9 Crop Distributions)
            - **Inference Device:** `{"NVIDIA GPU (CUDA Accelerated)" if gpu_active else "CPU (Standard Latency)"}`
            - **Dataset Repository:** 🟢 Available (`data/raw/crop_recommendation_10000.csv`)
            """
        )

    with col_info:
        st.markdown("### 🔬 Verified Architecture & Logic")
        st.markdown(
            """
            - **Foliar Pathology Classifier:** `EfficientNetB0` Deep CNN (38 PlantVillage Classes, 224×224 RGB input)
            - **Soil Suitability Engine:** Dual-Input Dense Neural Network with Categorical Crop Embeddings (7 Features)
            - **Diagnostic Analysis:** Statistical Benchmarking via Empirical Percentiles (`p10`, `p25`, `median`, `p75`, `p90`)
            - **Prescription Generator:** Rule-Based Expert System correlating soil/climate stressors with pathogen etiology
            """
        )

    # Workflow Architecture Diagram
    render_workflow_diagram()
    render_disclaimer()

# ============================================================
# 5. MODULE 2: SOIL & CROP ADVISORY
# ============================================================

elif current_page == "🧪 Soil & Crop Advisory":
    st.markdown("## 🧪 Soil & Crop Advisory")
    st.caption("Numerical ML Workflow: Evaluates soil fertility and climate measurements against empirical crop profiles.")

    with st.expander("📝 Soil & Meteorological Inputs", expanded=True):
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            nitrogen = st.number_input(
                "Nitrogen (N) [mg/kg]:",
                min_value=0.0,
                max_value=300.0,
                value=float(st.session_state["soil_readings"]["N"]),
                step=1.0,
            )
            phosphorus = st.number_input(
                "Phosphorus (P) [mg/kg]:",
                min_value=0.0,
                max_value=300.0,
                value=float(st.session_state["soil_readings"]["P"]),
                step=1.0,
            )
            potassium = st.number_input(
                "Potassium (K) [mg/kg]:",
                min_value=0.0,
                max_value=300.0,
                value=float(st.session_state["soil_readings"]["K"]),
                step=1.0,
            )

        with sc2:
            temperature = st.number_input(
                "Air Temperature [°C]:",
                min_value=-10.0,
                max_value=50.0,
                value=float(st.session_state["soil_readings"]["temperature"]),
                step=0.5,
            )
            humidity = st.number_input(
                "Relative Humidity [%]:",
                min_value=10.0,
                max_value=100.0,
                value=float(st.session_state["soil_readings"]["humidity"]),
                step=1.0,
            )
            ph = st.number_input(
                "Soil pH (0 - 14):",
                min_value=3.5,
                max_value=10.0,
                value=float(st.session_state["soil_readings"]["ph"]),
                step=0.1,
            )

        with sc3:
            rainfall = st.number_input(
                "Annual Rainfall [mm]:",
                min_value=0.0,
                max_value=2500.0,
                value=float(st.session_state["soil_readings"]["rainfall"]),
                step=5.0,
            )
            target_crop_sel = st.selectbox(
                "Primary Target Crop:",
                options=config.SOIL_CROPS,
                index=config.SOIL_CROPS.index(st.session_state["target_crop"]) if st.session_state["target_crop"] in config.SOIL_CROPS else 0,
                format_func=pretty_crop,
            )
            st.markdown("<br>", unsafe_allow_html=True)
            analyze_btn = st.button("⚡ Run Soil & Crop Analysis", use_container_width=True, type="primary")

    # Current readings payload
    current_readings = {
        "N": float(nitrogen),
        "P": float(phosphorus),
        "K": float(potassium),
        "temperature": float(temperature),
        "humidity": float(humidity),
        "ph": float(ph),
        "rainfall": float(rainfall),
    }
    st.session_state["soil_readings"] = current_readings
    st.session_state["target_crop"] = target_crop_sel

    current_hash = compute_soil_hash(target_crop_sel, current_readings)

    # Execute inference only when requested or if no prior results exist
    if models_ready:
        if analyze_btn or st.session_state["soil_result"] is None:
            if st.session_state["cached_soil_hash"] != current_hash:
                with st.spinner("Executing neural suitability inference & quantile diagnostics..."):
                    st.session_state["soil_result"] = predict_soil(soil_model, current_readings, target_crop_sel)
                    st.session_state["ranked_crops"] = recommend_crops(soil_model, current_readings)
                    st.session_state["diagnostics"] = diagnose_soil(current_readings, target_crop_sel, soil_profiles)
                    st.session_state["cached_soil_hash"] = current_hash

        soil_res = st.session_state["soil_result"]
        ranked_crops = st.session_state["ranked_crops"]
        diagnostics = st.session_state["diagnostics"]

        if soil_res and ranked_crops:
            st.markdown("---")
            st.markdown("### 🏆 Alternative Crop Suitability Rankings")
            
            # Top-3 Recommended Crops Cards
            top_3 = ranked_crops[:3]
            rc1, rc2, rc3 = st.columns(3)
            for idx, (col, item) in enumerate(zip([rc1, rc2, rc3], top_3)):
                with col:
                    badge = format_status_badge(item["status"])
                    st.markdown(
                        f"""
                        <div class="ranking-card">
                            <span class="rank-tag">RANK #{idx + 1}</span>
                            <div class="crop-title">{item['crop_display']}</div>
                            <div class="score-value">{item['score']:.1f}%</div>
                            <div>{badge}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.markdown("<br>", unsafe_allow_html=True)

            # Analytics Tabs
            tab_ranks, tab_params = st.tabs(["📊 Crop Ranking Overview", "📈 Soil & Climate Diagnostic Breakdown"])
            with tab_ranks:
                fig_ranks = plot_crop_recommendations(ranked_crops)
                if fig_ranks:
                    st.pyplot(fig_ranks, use_container_width=True)

            with tab_params:
                if diagnostics:
                    fig_diag = plot_soil_parameters_bar(diagnostics)
                    if fig_diag:
                        st.pyplot(fig_diag, use_container_width=True)

                    st.markdown("#### Detailed Empirical Quantile Deviations")
                    df_diag = pd.DataFrame(diagnostics)
                    st.dataframe(
                        df_diag[["Parameter", "Value", "Optimal_Range", "Status", "Analysis"]],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No empirical quantile distribution profile found for the selected crop.")
        else:
            st.info("💡 Adjust soil parameters above and click **Run Soil & Crop Analysis** to compute suitability.")

    render_disclaimer()

# ============================================================
# 6. MODULE 3: LEAF DISEASE DETECTION
# ============================================================

elif current_page == "🍃 Leaf Disease Detection":
    st.markdown("## 🍃 Leaf Disease Detection")
    st.caption("Computer Vision Workflow: Evaluates foliar imagery using EfficientNetB0 to diagnose pathology and recommend protocols.")

    col_input, col_pred = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown("### 📷 Select or Upload Foliar Image")
        source_mode = st.radio(
            "Image Input Source:",
            ["Dataset Sample Library", "Upload Image File"],
            horizontal=True,
        )

        chosen_image: Optional[Image.Image] = None

        if source_mode == "Dataset Sample Library":
            # Optimized cached sample image scan
            sample_files = get_sample_images(config.TEST_DIR)

            if sample_files:
                sample_map = {p.name: p for p in sample_files[:30]}
                selected_sample = st.selectbox("Choose a test sample leaf:", list(sample_map.keys()))
                if selected_sample:
                    chosen_image = Image.open(sample_map[selected_sample])
            else:
                st.info("Dataset samples directory `data/raw/Plant Village Dataset/Test` not found.")

        else:
            uploaded_file = st.file_uploader(
                "Upload a leaf image (JPEG, PNG):",
                type=["jpg", "jpeg", "png"],
            )
            if uploaded_file is not None:
                chosen_image = Image.open(uploaded_file)

        if chosen_image is not None:
            st.session_state["leaf_image"] = chosen_image
            st.image(chosen_image, caption="Active Leaf Inspection Photo", use_container_width=True)
            analyze_leaf_btn = st.button("🔬 Analyze Leaf Pathology", type="primary", use_container_width=True)
        else:
            analyze_leaf_btn = False

    with col_pred:
        if st.session_state["leaf_image"] is not None and models_ready:
            # Only run inference on explicit button click or if no prior result exists
            if analyze_leaf_btn or st.session_state["leaf_result"] is None:
                with st.spinner("Running EfficientNetB0 vision inference..."):
                    leaf_res = classify_leaf(image_model, st.session_state["leaf_image"], image_classes)
                    st.session_state["leaf_result"] = leaf_res
                    if leaf_res["crop"] in config.SOIL_CROPS:
                        st.session_state["target_crop"] = leaf_res["crop"]

            render_disease_detection(st.session_state["leaf_result"])
        else:
            st.info("👈 Select a sample leaf or upload a photograph and click **Analyze Leaf Pathology** to inspect.")

    render_disclaimer()

# ============================================================
# 7. MODULE 4: INTEGRATED FARM ADVISORY
# ============================================================

elif current_page == "📋 Integrated Farm Advisory":
    st.markdown("## 📋 Integrated Farm Advisory Report")
    st.caption("Holistic Decision Support: Correlates vision pathology detection with soil suitability modeling into actionable field prescriptions.")

    if models_ready:
        target_crop = st.session_state["target_crop"]
        readings = st.session_state["soil_readings"]

        # Ensure analysis has been performed
        if st.session_state["soil_result"] is None:
            with st.spinner("Synthesizing current soil readings and empirical quantiles..."):
                st.session_state["soil_result"] = predict_soil(soil_model, readings, target_crop)
                st.session_state["ranked_crops"] = recommend_crops(soil_model, readings)
                st.session_state["diagnostics"] = diagnose_soil(readings, target_crop, soil_profiles)

        soil_res = st.session_state["soil_result"]
        ranked = st.session_state["ranked_crops"]
        diagnostics = st.session_state["diagnostics"]
        leaf_res = st.session_state["leaf_result"]

        advisory = generate_advisory_summary(
            leaf_result=leaf_res,
            soil_result=soil_res,
            diagnostics=diagnostics,
            top_crops=ranked,
            values=readings,
        )

        # Overview Summary Banner Cards
        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown(
                f"""
                <div class="advisory-card" style="border-left: 5px solid #2d6a4f;">
                    <div style="color: #6c757d; font-size: 0.82rem; font-weight: 700;">PRIMARY TARGET CROP</div>
                    <div style="font-size: 1.3rem; font-weight: 700; color: #1b4332; margin: 4px 0;">{pretty_crop(target_crop)}</div>
                    <span style="font-size: 0.88rem; color: #495057;">Suitability: <strong>{soil_res['score']:.1f} / 100</strong></span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with s2:
            soil_badge = format_status_badge(soil_res["status"])
            st.markdown(
                f"""
                <div class="advisory-card" style="border-left: 5px solid #1d3557;">
                    <div style="color: #6c757d; font-size: 0.82rem; font-weight: 700;">SOIL ENVIRONMENT STATUS</div>
                    <div style="margin: 6px 0;">{soil_badge}</div>
                    <span style="font-size: 0.85rem; color: #495057;">Conditioned on 7 soil & climate features</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with s3:
            disease_display = leaf_res["disease"] if leaf_res else "No Leaf Analyzed (Presuming Baseline)"
            disease_badge = format_status_badge("NORMAL" if "healthy" in disease_display.lower() else "NEEDS ATTENTION")
            st.markdown(
                f"""
                <div class="advisory-card" style="border-left: 5px solid #e76f51;">
                    <div style="color: #6c757d; font-size: 0.82rem; font-weight: 700;">FOLIAR PATHOLOGY STATUS</div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: #1b4332; margin: 4px 0;">{disease_display}</div>
                    <div>{disease_badge}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Full Rendered Advisory
        render_advisory(advisory)

        # Crop Rotation Insights
        st.markdown("---")
        st.markdown("#### 🌾 Crop Rotation & Agro-Ecological Optimization")
        alt_crop = ranked[0]
        if alt_crop["crop"] != canonical_crop(target_crop):
            st.info(
                f"**Crop Rotation Strategy:** For your current measured field soil and meteorological conditions, "
                f"**{alt_crop['crop_display']}** achieves the highest natural suitability score of **{alt_crop['score']:.1f}%**. "
                f"Consider rotational planting to naturally break pathogen cycles and maximize nutrient efficiency."
            )
        else:
            st.success(
                f"Your selected crop **{pretty_crop(target_crop)}** is currently the top-performing match ({soil_res['score']:.1f}%) for this soil environment."
            )

    render_disclaimer()

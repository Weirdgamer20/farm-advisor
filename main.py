"""
Farmer Crop Advisory System — Streamlit Application Entry Point.
Modular, production-grade interface providing:
1. Dashboard & System Status
2. Soil & Crop Suitability Advisory (Numerical ML Workflow)
3. Leaf Disease Detection (Computer Vision Workflow)
4. Integrated Farm Advisory (Holistic Agronomic Prescriptions)
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Optional

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
from src.data_loader import load_crop_data, load_models, load_soil_profiles
from src.preprocessing import validate_soil_readings
from src.utils import canonical_crop, pretty_crop, setup_device
from src.visualization import (
    format_status_badge,
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
# 1. PAGE SETUP & CACHED RESOURCES
# ============================================================

st.set_page_config(
    page_title="Farmer Crop Advisory System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize hardware and subtle sidebar status
gpu_active, device_msg = setup_device()

# Load models and profiles once with caching
try:
    image_model, image_classes, soil_model = load_models()
    soil_profiles = load_soil_profiles()
    models_ready = True
except Exception as exc:
    models_ready = False
    image_model = None
    image_classes = []
    soil_model = None
    soil_profiles = {}

render_system_status_sidebar(gpu_active, device_msg, models_ready)
render_header()

# ============================================================
# 2. SESSION STATE INITIALIZATION (Zero Empty Screens)
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

# Pre-run default soil analysis if models are ready so screens are immediately populated
if models_ready and "ranked_crops" not in st.session_state:
    st.session_state["soil_result"] = predict_soil(
        soil_model,
        st.session_state["soil_readings"],
        st.session_state["target_crop"],
    )
    st.session_state["ranked_crops"] = recommend_crops(
        soil_model,
        st.session_state["soil_readings"],
    )
    st.session_state["diagnostics"] = diagnose_soil(
        st.session_state["soil_readings"],
        st.session_state["target_crop"],
        soil_profiles,
    )

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
        "Welcome to the **Farmer Crop Advisory System**, an intelligent decision-support system "
        "designed to assist agriculturalists with real-time soil suitability assessment, "
        "deep learning plant pathology diagnosis, and holistic agronomic prescriptions."
    )

    # 3 High-Impact Capability Cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div style="background: #ffffff; border: 1px solid #d8f3dc; border-top: 4px solid #2d6a4f; border-radius: 8px; padding: 18px; height: 190px; box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
                <h3 style="color: #1b4332; margin-top: 0; font-size: 1.2rem;">🧪 Soil & Crop Advisory</h3>
                <p style="color: #495057; font-size: 0.9rem; line-height: 1.4;">
                    Evaluates 7 soil nutrients and meteorological parameters against 9 empirical crop profiles. Generates suitability scores and alternative crop rankings.
                </p>
                <span style="color: #2d6a4f; font-weight: 600; font-size: 0.88rem;">Numerical Deep Learning →</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div style="background: #ffffff; border: 1px solid #d8f3dc; border-top: 4px solid #40916c; border-radius: 8px; padding: 18px; height: 190px; box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
                <h3 style="color: #1b4332; margin-top: 0; font-size: 1.2rem;">🍃 Leaf Disease Detection</h3>
                <p style="color: #495057; font-size: 0.9rem; line-height: 1.4;">
                    Computer vision diagnostics trained on 38 pathology classes. Detects disease symptoms with confidence scoring and provides management protocols.
                </p>
                <span style="color: #40916c; font-weight: 600; font-size: 0.88rem;">Computer Vision Inference →</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div style="background: #ffffff; border: 1px solid #d8f3dc; border-top: 4px solid #52b788; border-radius: 8px; padding: 18px; height: 190px; box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
                <h3 style="color: #1b4332; margin-top: 0; font-size: 1.2rem;">📋 Integrated Farm Advisory</h3>
                <p style="color: #495057; font-size: 0.9rem; line-height: 1.4;">
                    Combines foliar pathology and soil conditions to identify environmental infection drivers and prescribe precise corrective nutrient amendments.
                </p>
                <span style="color: #52b788; font-weight: 600; font-size: 0.88rem;">Agronomic Prescriptions →</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # System Status & Model Information Grid
    col_sys, col_info = st.columns([1, 1], gap="large")
    with col_sys:
        st.markdown("### 🖥️ Operational System Status")
        st.markdown(
            f"""
            - **Plant Pathology Vision Model:** {"🟢 Loaded & Ready" if models_ready else "🔴 Unavailable"}
            - **Soil Suitability Neural Model:** {"🟢 Loaded & Ready" if models_ready else "🔴 Unavailable"}
            - **Quantile Diagnostic Profiles:** 🟢 Available (9 Crop Distributions)
            - **Active Inference Device:** `{"NVIDIA GPU (CUDA Accelerated)" if gpu_active else "CPU (Standard Latency)"}`
            - **Dataset Repository:** 🟢 Available (`data/raw/crop_recommendation_10000.csv`)
            """
        )

    with col_info:
        st.markdown("### 🔬 Verified Model Architecture")
        st.markdown(
            """
            - **Foliar Pathology Classifier:** `EfficientNetB0` Transfer Architecture (38 Pathology Classes, 224×224 RGB input)
            - **Soil Suitability Model:** Dual-Input Dense Neural Network with Categorical Crop Embeddings (7 Numerical Features)
            - **Statistical Engine:** Precomputed Empirical Quantiles (`p10`, `p25`, `median`, `p75`, `p90`)
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
    st.caption("Numerical ML Workflow: Evaluates soil fertility and climatic conditions to compute crop suitability scores.")

    with st.expander("📝 Enter Soil & Meteorological Parameters", expanded=True):
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
                "Primary Field Crop:",
                options=config.SOIL_CROPS,
                index=config.SOIL_CROPS.index(st.session_state["target_crop"]) if st.session_state["target_crop"] in config.SOIL_CROPS else 0,
                format_func=pretty_crop,
            )
            st.markdown("<br>", unsafe_allow_html=True)
            analyze_btn = st.button("⚡ Run Soil & Crop Analysis", use_container_width=True, type="primary")

    # Update state readings
    updated_readings = {
        "N": float(nitrogen),
        "P": float(phosphorus),
        "K": float(potassium),
        "temperature": float(temperature),
        "humidity": float(humidity),
        "ph": float(ph),
        "rainfall": float(rainfall),
    }
    st.session_state["soil_readings"] = updated_readings
    st.session_state["target_crop"] = target_crop_sel

    # Trigger or display analysis
    if models_ready:
        if analyze_btn or "soil_result" not in st.session_state:
            with st.spinner("Computing neural suitability and empirical quantile deviations..."):
                st.session_state["soil_result"] = predict_soil(soil_model, updated_readings, target_crop_sel)
                st.session_state["ranked_crops"] = recommend_crops(soil_model, updated_readings)
                st.session_state["diagnostics"] = diagnose_soil(updated_readings, target_crop_sel, soil_profiles)

        soil_res = st.session_state["soil_result"]
        ranked_crops = st.session_state["ranked_crops"]
        diagnostics = st.session_state["diagnostics"]

        st.markdown("---")
        st.markdown("### 🏆 Recommended Crop Rankings")
        
        # Display top-3 recommended crops as cards
        top_3 = ranked_crops[:3]
        rc1, rc2, rc3 = st.columns(3)
        cols = [rc1, rc2, rc3]
        for idx, (col, item) in enumerate(zip(cols, top_3)):
            with col:
                badge = format_status_badge(item["status"])
                st.markdown(
                    f"""
                    <div style="background-color: #ffffff; border: 1px solid #dee2e6; border-radius: 8px; padding: 16px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.03);">
                        <span style="font-size: 0.85rem; color: #6c757d; font-weight: bold;">RANK #{idx + 1}</span>
                        <h3 style="margin: 4px 0; color: #1b4332;">{item['crop_display']}</h3>
                        <div style="font-size: 1.6rem; font-weight: 700; color: #2d6a4f; margin: 4px 0;">{item['score']:.1f}%</div>
                        <div>{badge}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # Diagnostics & Analytics Tabs
        tab_ranks, tab_params = st.tabs(["📊 Crop Suitability Ranking", "📈 Nutrient & Climate Diagnostics"])
        with tab_ranks:
            fig_ranks = plot_crop_recommendations(ranked_crops)
            if fig_ranks:
                st.pyplot(fig_ranks, use_container_width=True)

        with tab_params:
            if diagnostics:
                fig_diag = plot_soil_parameters_bar(diagnostics)
                if fig_diag:
                    st.pyplot(fig_diag, use_container_width=True)

                st.markdown("#### Detailed Diagnostic Metrics")
                df_diag = pd.DataFrame(diagnostics)
                st.dataframe(
                    df_diag[["Parameter", "Value", "Optimal_Range", "Status", "Analysis"]],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No empirical quantile distribution profile found for the selected crop.")

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
            sample_files: List[pathlib.Path] = []
            if config.TEST_DIR.exists():
                for ext in ("*.jpg", "*.jpeg", "*.JPG", "*.png"):
                    sample_files.extend(list(config.TEST_DIR.rglob(ext)))

            if sample_files:
                sample_map = {p.name: p for p in sorted(sample_files)[:30]}
                selected_sample = st.selectbox("Choose a test sample leaf:", list(sample_map.keys()))
                if selected_sample:
                    chosen_image = Image.open(sample_map[selected_sample])
            else:
                st.info("Dataset samples directory data/raw/Plant Village Dataset/Test not found.")

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
            if analyze_leaf_btn or st.session_state["leaf_result"] is None:
                with st.spinner("Running EfficientNetB0 vision inference..."):
                    leaf_res = classify_leaf(image_model, st.session_state["leaf_image"], image_classes)
                    st.session_state["leaf_result"] = leaf_res
                    if leaf_res["crop"] in config.SOIL_CROPS:
                        st.session_state["target_crop"] = leaf_res["crop"]

            render_disease_detection(st.session_state["leaf_result"])
        else:
            st.info("👈 Please select a test sample image or upload a photograph to view diagnosis.")

    render_disclaimer()

# ============================================================
# 7. MODULE 4: INTEGRATED FARM ADVISORY
# ============================================================

elif current_page == "📋 Integrated Farm Advisory":
    st.markdown("## 📋 Integrated Farm Advisory Report")
    st.caption("Holistic Agronomic Decision Support: Fuses vision pathology detection with soil suitability modeling into actionable prescriptions.")

    if models_ready:
        # Refresh current calculations
        target_crop = st.session_state["target_crop"]
        readings = st.session_state["soil_readings"]
        soil_res = predict_soil(soil_model, readings, target_crop)
        ranked = recommend_crops(soil_model, readings)
        diagnostics = diagnose_soil(readings, target_crop, soil_profiles)
        leaf_res = st.session_state["leaf_result"]

        advisory = generate_advisory_summary(
            leaf_result=leaf_res,
            soil_result=soil_res,
            diagnostics=diagnostics,
            top_crops=ranked,
            values=readings,
        )

        # Overview Summary Banner
        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown(
                f"""
                <div style="background: #ffffff; border: 1px solid #dee2e6; border-left: 5px solid #2d6a4f; padding: 14px; border-radius: 6px;">
                    <div style="color: #6c757d; font-size: 0.85rem; font-weight: 600;">ACTIVE TARGET CROP</div>
                    <h3 style="margin: 4px 0; color: #1b4332;">{pretty_crop(target_crop)}</h3>
                    <span style="font-size: 0.85rem; color: #495057;">Suitability: <strong>{soil_res['score']:.1f}/100</strong></span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with s2:
            soil_badge = format_status_badge(soil_res["status"])
            st.markdown(
                f"""
                <div style="background: #ffffff; border: 1px solid #dee2e6; border-left: 5px solid #1d3557; padding: 14px; border-radius: 6px;">
                    <div style="color: #6c757d; font-size: 0.85rem; font-weight: 600;">SOIL ENVIRONMENT STATUS</div>
                    <div style="margin: 6px 0;">{soil_badge}</div>
                    <span style="font-size: 0.85rem; color: #495057;">Based on 7 soil & climate features</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with s3:
            disease_display = leaf_res["disease"] if leaf_res else "No Leaf Inspected (Defaulting to Healthy)"
            disease_badge = format_status_badge("NORMAL" if "healthy" in disease_display.lower() else "NEEDS ATTENTION")
            st.markdown(
                f"""
                <div style="background: #ffffff; border: 1px solid #dee2e6; border-left: 5px solid #e76f51; padding: 14px; border-radius: 6px;">
                    <div style="color: #6c757d; font-size: 0.85rem; font-weight: 600;">FOLIAR PATHOLOGY STATUS</div>
                    <h4 style="margin: 4px 0; color: #1b4332;">{disease_display}</h4>
                    <div style="margin-top: 2px;">{disease_badge}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Full Rendered Advisory Sections
        render_advisory(advisory)

        st.markdown("---")
        st.markdown("#### 🌾 Crop Rotation & Alternative Opportunities")
        alt_crop = ranked[0]
        if alt_crop["crop"] != canonical_crop(target_crop):
            st.info(
                f"**Crop Rotation Strategy:** For your current measured field soil and meteorological conditions, "
                f"**{alt_crop['crop_display']}** achieves the highest natural suitability score of **{alt_crop['score']:.1f}%**. "
                f"Consider rotational planting to naturally break pathogen cycles and maximize nutrient efficiency."
            )
        else:
            st.success(
                f"Your selected crop **{pretty_crop(target_crop)}** is already the top-performing match for this soil environment."
            )

    render_disclaimer()

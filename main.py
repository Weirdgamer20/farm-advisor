"""
Farmer Crop Advisory System — Streamlit Application Entry Point.
Modular, production-grade interface with asynchronous background runtime:
1. Dashboard & Verified Model Specifications
2. Soil & Crop Suitability Advisory (Numerical Deep Learning + Empirical Quantiles)
3. Leaf Disease Detection (EfficientNetB0 Vision Inference)
4. Integrated Farm Advisory (Expert Agronomic Prescription Synthesis)
"""

from __future__ import annotations

import hashlib
import logging
from typing import Dict, Optional

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
from src.application_runtime import ApplicationRuntime
from src.data_loader import check_artifact_availability, get_sample_images
from src.utils import canonical_crop, pretty_crop
from src.visualization import (
    format_status_badge,
    inject_custom_theme,
    plot_crop_recommendations,
    plot_soil_parameters_bar,
    render_advisory,
    render_ai_engine_status_card,
    render_disclaimer,
    render_disease_detection,
    render_header,
    render_system_status_sidebar,
    render_workflow_diagram,
)

logger = logging.getLogger(__name__)

# 1. Page Configuration & Header — Renders immediately without blocking
st.set_page_config(
    page_title="Farmer Crop Advisory System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_custom_theme()
render_header()

# 2. Asynchronous Background Initialization (Non-blocking)
runtime = ApplicationRuntime.get_instance()
runtime.start_initialization(warmup=True)
snapshot = runtime.get_snapshot()

# 3. Transparent Sidebar Telemetry (Honest GPU & TensorFlow Runtime Status)
render_system_status_sidebar(
    gpu_active=snapshot["hardware"].get("tf_cuda_active", False),
    device_msg=snapshot["hardware"].get("tf_device_summary", ""),
    models_ready=runtime.is_ready(),
    detailed_telemetry=check_artifact_availability(),
    runtime_snapshot=snapshot,
)

# 4. Session State Management
session_defaults = {
    "soil_readings": {"N": 90.0, "P": 42.0, "K": 43.0, "temperature": 25.6, "humidity": 80.0, "ph": 6.5, "rainfall": 200.0},
    "target_crop": "tomato",
    "leaf_result": None,
    "leaf_image": None,
    "soil_result": None,
    "ranked_crops": None,
    "diagnostics": None,
    "cached_soil_hash": None,
    "cached_leaf_hash": None,
}
for key, val in session_defaults.items():
    st.session_state.setdefault(key, val)


def compute_soil_hash(crop: str, readings: Dict[str, float]) -> str:
    """Generates an MD5 signature for soil inputs to prevent redundant inference reruns."""
    payload = f"{crop}_" + "_".join(f"{k}:{v:.2f}" for k, v in sorted(readings.items()))
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


# 5. Sidebar Navigation
with st.sidebar:
    st.markdown("## 🧭 Navigation")
    current_page = st.radio(
        "Select Advisory Module:",
        ["🏠 Dashboard", "🧪 Soil & Crop Advisory", "🍃 Leaf Disease Detection", "📋 Integrated Farm Advisory"],
        index=0,
    )
    st.markdown("---")

# 6. Module 1: Dashboard
if current_page == "🏠 Dashboard":
    st.markdown("## 📊 Overview & Capabilities")
    render_ai_engine_status_card(snapshot)

    st.markdown(
        "The **Farmer Crop Advisory System** is an academic AI decision-support platform "
        "integrating computer vision pathology classification with conditioned neural network "
        "crop suitability modeling and rule-based agronomic guidance."
    )

    overview_cards = [
        ("🧪 Soil & Crop Advisory", "Evaluates 7 soil nutrients and climate parameters against 9 crop profiles using dual-input neural modeling and empirical quantiles.", "Numerical Deep Learning →", "#2d6a4f"),
        ("🍃 Leaf Disease Detection", "EfficientNetB0 vision model trained on 38 PlantVillage pathology classes with calibrated diagnostic confidence and treatment protocols.", "Computer Vision Inference →", "#40916c"),
        ("📋 Integrated Farm Advisory", "Synthesizes foliar disease diagnosis with field soil readings to detect environmental infection drivers and formulate precise corrective amendments.", "Agronomic Prescriptions →", "#52b788"),
    ]
    for col, (title, desc, tag, border_color) in zip(st.columns(3), overview_cards):
        with col:
            st.markdown(
                f"""<div class="advisory-card" style="border-top: 4px solid {border_color};">
                    <h3>{title}</h3><p>{desc}</p>
                    <span style="color: {border_color}; font-weight: 600; font-size: 0.88rem;">{tag}</span>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    col_sys, col_info = st.columns([1, 1], gap="large")
    with col_sys:
        st.markdown("### 🖥️ Hardware & AI Engine Status")
        hw = snapshot["hardware"]
        st.markdown(
            f"""
            - **Physical Hardware:** `{hw.get('hardware_label', 'Host Processor')}`
            - **TensorFlow Execution Backend:** `{hw.get('tf_device_summary', 'CPU Inference')}`
            - **CUDA Device Acceleration:** `{hw.get('cuda_status', 'Standard')}`
            - **AI Runtime Lifecycle:** `{"● READY (Pre-warmed)" if runtime.is_ready() else "⟳ INITIALIZING (Background)"}`
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

    render_workflow_diagram()
    render_disclaimer()

# 7. Module 2: Soil & Crop Advisory
elif current_page == "🧪 Soil & Crop Advisory":
    st.markdown("## 🧪 Soil & Crop Advisory")
    st.caption("Numerical ML Workflow: Evaluates soil fertility and climate measurements against empirical crop profiles.")

    with st.expander("📝 Soil & Meteorological Inputs", expanded=True):
        cols = st.columns(3)
        specs = [
            ("N", "Nitrogen (N) [mg/kg]:", 0.0, 300.0, 1.0, 0),
            ("P", "Phosphorus (P) [mg/kg]:", 0.0, 300.0, 1.0, 0),
            ("K", "Potassium (K) [mg/kg]:", 0.0, 300.0, 1.0, 0),
            ("temperature", "Air Temperature [°C]:", -10.0, 50.0, 0.5, 1),
            ("humidity", "Relative Humidity [%]:", 10.0, 100.0, 1.0, 1),
            ("ph", "Soil pH (0 - 14):", 3.5, 10.0, 0.1, 1),
            ("rainfall", "Annual Rainfall [mm]:", 0.0, 2500.0, 5.0, 2),
        ]
        current_readings = {
            f: float(cols[c].number_input(lbl, min_value=lo, max_value=hi, value=float(st.session_state["soil_readings"][f]), step=s))
            for f, lbl, lo, hi, s, c in specs
        }
        with cols[2]:
            target_crop_sel = st.selectbox(
                "Primary Target Crop:",
                options=config.SOIL_CROPS,
                index=config.SOIL_CROPS.index(st.session_state["target_crop"]) if st.session_state["target_crop"] in config.SOIL_CROPS else 0,
                format_func=pretty_crop,
            )
            st.markdown("<br>", unsafe_allow_html=True)
            analyze_btn = st.button("⚡ Run Soil & Crop Analysis", use_container_width=True, type="primary")

    st.session_state["soil_readings"] = current_readings
    st.session_state["target_crop"] = target_crop_sel
    current_hash = compute_soil_hash(target_crop_sel, current_readings)

    # Strictly on-demand inference with background initialization gate
    if analyze_btn:
        if not runtime.is_ready():
            with st.spinner("⏳ AI Engine is finishing background initialization..."):
                runtime.wait_until_ready(timeout=25.0)

        if runtime.is_ready():
            if st.session_state["cached_soil_hash"] != current_hash:
                with st.spinner("Executing neural suitability inference & quantile diagnostics..."):
                    st.session_state["soil_result"] = predict_soil(runtime.soil_model, current_readings, target_crop_sel)
                    st.session_state["ranked_crops"] = recommend_crops(runtime.soil_model, current_readings)
                    st.session_state["diagnostics"] = diagnose_soil(current_readings, target_crop_sel, runtime.soil_profiles)
                    st.session_state["cached_soil_hash"] = current_hash
        else:
            st.warning("⚠️ Neural models are still initializing. Please wait a moment and click analyze again.")

    soil_res = st.session_state["soil_result"]
    ranked_crops = st.session_state["ranked_crops"]
    diagnostics = st.session_state["diagnostics"]

    if soil_res and ranked_crops:
        st.markdown("---")
        st.markdown("### 🏆 Alternative Crop Suitability Rankings")
        for idx, (col, item) in enumerate(zip(st.columns(3), ranked_crops[:3])):
            with col:
                st.markdown(
                    f"""<div class="ranking-card">
                        <span class="rank-tag">RANK #{idx + 1}</span>
                        <div class="crop-title">{item['crop_display']}</div>
                        <div class="score-value">{item['score']:.1f}%</div>
                        <div>{format_status_badge(item["status"])}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)
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
                st.dataframe(
                    pd.DataFrame(diagnostics)[["Parameter", "Value", "Optimal_Range", "Status", "Analysis"]],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No empirical quantile distribution profile found for the selected crop.")
    else:
        st.info("💡 Adjust soil parameters above and click **⚡ Run Soil & Crop Analysis** to compute suitability.")

    render_disclaimer()

# 8. Module 3: Leaf Disease Detection
elif current_page == "🍃 Leaf Disease Detection":
    st.markdown("## 🍃 Leaf Disease Detection")
    st.caption("Computer Vision Workflow: Evaluates foliar imagery using EfficientNetB0 to diagnose pathology and recommend protocols.")

    col_input, col_pred = st.columns([1, 1], gap="large")
    with col_input:
        st.markdown("### 📷 Select or Upload Foliar Image")
        source_mode = st.radio("Image Input Source:", ["Dataset Sample Library", "Upload Image File"], horizontal=True)
        chosen_image: Optional[Image.Image] = None

        if source_mode == "Dataset Sample Library":
            sample_files = get_sample_images(config.TEST_DIR)
            if sample_files:
                sample_map = {p.name: p for p in sample_files[:30]}
                selected_sample = st.selectbox("Choose a test sample leaf:", list(sample_map.keys()))
                if selected_sample:
                    chosen_image = Image.open(sample_map[selected_sample])
            else:
                st.info("Dataset samples directory `data/raw/Plant Village Dataset/Test` not found.")
        else:
            uploaded_file = st.file_uploader("Upload a leaf image (JPEG, PNG):", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                chosen_image = Image.open(uploaded_file)

        if chosen_image is not None:
            st.session_state["leaf_image"] = chosen_image
            st.image(chosen_image, caption="Active Leaf Inspection Photo", use_container_width=True)
            analyze_leaf_btn = st.button("🔬 Analyze Leaf Pathology", type="primary", use_container_width=True)
            img_hash = hashlib.md5(chosen_image.tobytes()).hexdigest()
        else:
            analyze_leaf_btn = False
            img_hash = None

    with col_pred:
        if st.session_state["leaf_image"] is not None:
            if analyze_leaf_btn:
                if not runtime.is_ready():
                    with st.spinner("⏳ AI Vision Engine is finishing background initialization..."):
                        runtime.wait_until_ready(timeout=25.0)

                if runtime.is_ready():
                    if st.session_state.get("cached_leaf_hash") != img_hash:
                        with st.spinner("Running EfficientNetB0 vision inference..."):
                            leaf_res = classify_leaf(runtime.image_model, st.session_state["leaf_image"], runtime.image_classes)
                            st.session_state["leaf_result"] = leaf_res
                            st.session_state["cached_leaf_hash"] = img_hash
                            if leaf_res["crop"] in config.SOIL_CROPS:
                                st.session_state["target_crop"] = leaf_res["crop"]
                else:
                    st.warning("⚠️ Vision model is still initializing. Please wait a moment and try again.")

            if st.session_state["leaf_result"]:
                render_disease_detection(st.session_state["leaf_result"])
            else:
                st.info("👆 Click **🔬 Analyze Leaf Pathology** to evaluate foliar health.")
        else:
            st.info("👈 Select a sample leaf or upload a photograph and click **Analyze Leaf Pathology** to inspect.")

    render_disclaimer()

# 9. Module 4: Integrated Farm Advisory
elif current_page == "📋 Integrated Farm Advisory":
    st.markdown("## 📋 Integrated Farm Advisory Report")
    st.caption("Holistic Decision Support: Correlates vision pathology detection with soil suitability modeling into actionable field prescriptions.")

    target_crop = st.session_state["target_crop"]
    readings = st.session_state["soil_readings"]

    if st.session_state["soil_result"] is None:
        st.info("💡 No soil analysis has been generated yet for this session.")
        col_act, _ = st.columns([1, 1])
        with col_act:
            generate_now = st.button("⚡ Synthesize Field Data & Generate Report", type="primary")
        if generate_now:
            if not runtime.is_ready():
                with st.spinner("⏳ AI Engine is finishing background initialization..."):
                    runtime.wait_until_ready(timeout=25.0)

            if runtime.is_ready():
                with st.spinner("Synthesizing field soil readings and empirical quantiles..."):
                    st.session_state["soil_result"] = predict_soil(runtime.soil_model, readings, target_crop)
                    st.session_state["ranked_crops"] = recommend_crops(runtime.soil_model, readings)
                    st.session_state["diagnostics"] = diagnose_soil(readings, target_crop, runtime.soil_profiles)
                    st.rerun()
            else:
                st.warning("⚠️ Neural models are still initializing. Please wait a moment and try again.")

    soil_res = st.session_state["soil_result"]
    ranked = st.session_state["ranked_crops"]
    diagnostics = st.session_state["diagnostics"]
    leaf_res = st.session_state["leaf_result"]

    if soil_res and ranked and diagnostics:
        advisory = generate_advisory_summary(
            leaf_result=leaf_res,
            soil_result=soil_res,
            diagnostics=diagnostics,
            top_crops=ranked,
            values=readings,
        )

        disease_display = leaf_res["disease"] if leaf_res else "No Leaf Analyzed (Presuming Baseline)"
        disease_badge = format_status_badge("NORMAL" if "healthy" in disease_display.lower() else "NEEDS ATTENTION")

        summary_cards = [
            ("PRIMARY TARGET CROP", pretty_crop(target_crop), f"Suitability: <strong>{soil_res['score']:.1f} / 100</strong>", "#2d6a4f"),
            ("SOIL ENVIRONMENT STATUS", format_status_badge(soil_res["status"]), "Conditioned on 7 soil & climate features", "#1d3557"),
            ("FOLIAR PATHOLOGY STATUS", f"<div style='font-size: 1.1rem; font-weight: 700; color: #1b4332; margin: 4px 0;'>{disease_display}</div>{disease_badge}", "Pathology assessment", "#e76f51"),
        ]

        for col, (label, content, footer, border_color) in zip(st.columns(3), summary_cards):
            with col:
                st.markdown(
                    f"""<div class="advisory-card" style="border-left: 5px solid {border_color};">
                        <div style="color: #6c757d; font-size: 0.82rem; font-weight: 700;">{label}</div>
                        <div style="font-size: 1.3rem; font-weight: 700; color: #1b4332; margin: 4px 0;">{content}</div>
                        <span style="font-size: 0.85rem; color: #495057;">{footer}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)
        render_advisory(advisory)

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
            st.success(f"Your selected crop **{pretty_crop(target_crop)}** is currently the top-performing match ({soil_res['score']:.1f}%) for this soil environment.")

    render_disclaimer()

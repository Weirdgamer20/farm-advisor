"""
Farmer Crop Advisory System — Main Application Entry Point
Production-grade modular Streamlit dashboard for leaf pathology diagnosis,
crop suitability evaluation, and soil nutrient advisory.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from PIL import Image

import config
from src.data_loader import load_crop_data, load_models, load_soil_profiles
from src.analysis import (
    classify_leaf,
    diagnose_soil,
    generate_advisory_summary,
    predict_soil,
    recommend_crops,
)
from src.utils import pretty_crop, setup_device
from src.visualization import (
    format_status_badge,
    plot_crop_recommendations,
    plot_soil_parameters_bar,
    render_advisory,
    render_device_status,
    render_disease_detection,
    render_header,
)

# ============================================================
# 1. PAGE CONFIGURATION & STYLING
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
    .metric-container {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 18px;
        border-radius: 6px 6px 0 0;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Render Banner
render_header()

# ============================================================
# 2. HARDWARE & MODEL INITIALIZATION (NON-BLOCKING)
# ============================================================

gpu_active, device_msg = setup_device()
render_device_status(gpu_active, device_msg)

with st.spinner("Loading agricultural neural network models and profiles..."):
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

# ============================================================
# 3. SIDEBAR NAVIGATION & SYSTEM METRICS
# ============================================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/wheat.png", width=64)
    st.markdown("### 🌾 Advisory Controls")
    st.markdown(f"**Vision Classes**: {len(image_classes)} pathology labels")
    st.markdown(f"**Soil Profiles**: {len(config.SOIL_CROPS)} crop varieties")
    st.markdown("---")
    st.markdown("### 📋 Soil Parameter Presets")

    preset_choice = st.selectbox(
        "Apply Soil Preset (Optional)",
        options=["Custom Inputs", "Apple Orchard", "Corn Field", "Tomato Garden", "Grape Vineyard"],
        index=0,
    )

    preset_values = {
        "Apple Orchard": {"N": 25.0, "P": 125.0, "K": 150.0, "temperature": 18.0, "humidity": 70.0, "ph": 6.0, "rainfall": 110.0},
        "Corn Field": {"N": 80.0, "P": 45.0, "K": 40.0, "temperature": 24.0, "humidity": 65.0, "ph": 6.5, "rainfall": 85.0},
        "Tomato Garden": {"N": 60.0, "P": 50.0, "K": 60.0, "temperature": 25.0, "humidity": 68.0, "ph": 6.2, "rainfall": 100.0},
        "Grape Vineyard": {"N": 20.0, "P": 130.0, "K": 200.0, "temperature": 22.0, "humidity": 80.0, "ph": 6.0, "rainfall": 70.0},
    }

    current_preset = preset_values.get(preset_choice, {})

    st.markdown("---")
    st.caption("AI-Powered Agricultural Decision Support System v2.0")

# ============================================================
# 4. MAIN WORKSPACE WITH 3 PRODUCTION WORKFLOWS
# ============================================================

tab_soil, tab_leaf, tab_full = st.tabs([
    "🌾 Soil Suitability & Crop Recommendation",
    "🍃 Leaf Pathology Diagnosis",
    "🚜 Integrated Farm Advisory",
])

# ------------------------------------------------------------
# TAB 1: SOIL ANALYSIS & CROP RECOMMENDATION
# ------------------------------------------------------------
with tab_soil:
    st.markdown("### 1. Soil & Environmental Parameter Inputs")
    st.write("Enter laboratory soil test readings and local climate conditions to assess land suitability.")

    col1, col2, col3 = st.columns(3)
    with col1:
        n_val = st.number_input(
            "Nitrogen (N) - mg/kg",
            min_value=0.0, max_value=500.0,
            value=float(current_preset.get("N", 50.0)),
            step=1.0, key="soil_n"
        )
        temp_val = st.number_input(
            "Temperature (°C)",
            min_value=-20.0, max_value=70.0,
            value=float(current_preset.get("temperature", 24.0)),
            step=0.5, key="soil_temp"
        )

    with col2:
        p_val = st.number_input(
            "Phosphorus (P) - mg/kg",
            min_value=0.0, max_value=500.0,
            value=float(current_preset.get("P", 40.0)),
            step=1.0, key="soil_p"
        )
        humidity_val = st.number_input(
            "Relative Humidity (%)",
            min_value=0.0, max_value=100.0,
            value=float(current_preset.get("humidity", 70.0)),
            step=1.0, key="soil_humidity"
        )

    with col3:
        k_val = st.number_input(
            "Potassium (K) - mg/kg",
            min_value=0.0, max_value=500.0,
            value=float(current_preset.get("K", 45.0)),
            step=1.0, key="soil_k"
        )
        ph_val = st.number_input(
            "Soil pH Level",
            min_value=0.0, max_value=14.0,
            value=float(current_preset.get("ph", 6.5)),
            step=0.1, key="soil_ph"
        )
        rain_val = st.number_input(
            "Annual Rainfall (mm)",
            min_value=0.0, max_value=5000.0,
            value=float(current_preset.get("rainfall", 100.0)),
            step=5.0, key="soil_rain"
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

    st.markdown("---")
    eval_mode = st.radio(
        "Analysis Goal:",
        options=["Evaluate Target Crop", "Recommend Best Crops for this Soil"],
        horizontal=True,
    )

    if eval_mode == "Evaluate Target Crop":
        col_crop, col_btn = st.columns([3, 1])
        with col_crop:
            selected_crop = st.selectbox(
                "Select crop to evaluate:",
                options=config.SOIL_CROPS,
                format_func=pretty_crop,
                index=config.SOIL_CROPS.index("tomato") if "tomato" in config.SOIL_CROPS else 0,
            )
        with col_btn:
            st.write("")
            st.write("")
            eval_btn = st.button("Evaluate Suitability", type="primary", use_container_width=True)

        if eval_btn:
            with st.spinner(f"Evaluating soil suitability for {pretty_crop(selected_crop)}..."):
                soil_res = predict_soil(soil_readings, selected_crop, soil_model)
                diagnostics = diagnose_soil(soil_readings, selected_crop, profile_df, profiles_dict)
                advisory = generate_advisory_summary(soil_result=soil_res, diagnostics=diagnostics)

            st.markdown("### 📊 Evaluation Results")
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.metric("Suitability Score", f"{soil_res['score']:.1f} / 100")
            with sc2:
                st.markdown(f"**Status**: {format_status_badge(soil_res['status'])}", unsafe_allow_html=True)
            with sc3:
                st.metric("Evaluated Crop", pretty_crop(selected_crop))

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

            render_advisory(advisory)

    else:
        rec_btn = st.button("Find Top Recommended Crops", type="primary", use_container_width=True)
        if rec_btn:
            with st.spinner("Computing suitability across all agricultural profiles..."):
                recommendations = recommend_crops(soil_readings, soil_model, top_k=len(config.SOIL_CROPS))

            top_crop = recommendations[0]
            st.success(
                f"🌟 **Top Recommendation: {top_crop['crop_display']}** "
                f"(Suitability Score: {top_crop['score']:.1f}/100 — {top_crop['status']})"
            )

            col_rec_chart, col_rec_table = st.columns([1.2, 0.8])
            with col_rec_chart:
                fig_rec = plot_crop_recommendations(recommendations)
                if fig_rec:
                    st.pyplot(fig_rec)

            with col_rec_table:
                rec_table_data = [
                    {"Rank": idx + 1, "Crop": r["crop_display"], "Score": f"{r['score']:.1f}%", "Status": r["status"]}
                    for idx, r in enumerate(recommendations)
                ]
                st.dataframe(pd.DataFrame(rec_table_data), use_container_width=True, hide_index=True)


# ------------------------------------------------------------
# TAB 2: LEAF PATHOLOGY DIAGNOSIS
# ------------------------------------------------------------
with tab_leaf:
    st.markdown("### 2. Leaf Image Pathology Diagnosis")
    st.write("Upload a photographic specimen of crop foliage to detect bacterial, fungal, or viral infections.")

    uploaded_leaf = st.file_uploader(
        "Upload Leaf Image (JPG, PNG, WebP)",
        type=["jpg", "jpeg", "png", "webp"],
        key="leaf_uploader_tab",
    )

    if uploaded_leaf is not None:
        leaf_img = Image.open(uploaded_leaf)
        col_img, col_pred = st.columns([1, 1.2])

        with col_img:
            st.image(leaf_img, caption="Uploaded Leaf Specimen", use_container_width=True)

        with col_pred:
            with st.spinner("Executing EfficientNetB0 Pathology Vision Model..."):
                leaf_res = classify_leaf(leaf_img, image_model, image_classes)

            render_disease_detection(leaf_res)
    else:
        st.info("💡 Please upload an image of an affected leaf to run automated disease identification.")
        st.markdown(
            """
            **Supported Crops for Disease Detection:**
            - **Apple**: Apple Scab, Black Rot, Cedar Apple Rust, Healthy
            - **Bell Pepper**: Bacterial Spot, Healthy
            - **Cherry**: Powdery Mildew, Healthy
            - **Corn (Maize)**: Common Rust, Northern Leaf Blight, Cercospora Spot, Healthy
            - **Grape**: Black Rot, Esca (Black Measles), Leaf Blight, Healthy
            - **Peach**: Bacterial Spot, Healthy
            - **Potato**: Early Blight, Late Blight, Healthy
            - **Strawberry**: Leaf Scorch, Healthy
            - **Tomato**: Early Blight, Late Blight, Septoria Spot, Yellow Leaf Curl, Bacterial Spot, Healthy
            """
        )


# ------------------------------------------------------------
# TAB 3: INTEGRATED ADVISORY (DUAL-MODAL DECISION SUPPORT)
# ------------------------------------------------------------
with tab_full:
    st.markdown("### 3. Integrated Dual-Modal Farm Advisory")
    st.write("Harmonizes foliar disease diagnosis with field soil parameters to provide comprehensive decision support.")

    col_full_img, col_full_soil = st.columns([1, 1])

    with col_full_img:
        st.markdown("#### Step A: Leaf Image Specimen")
        integrated_leaf = st.file_uploader(
            "Upload leaf image for integrated analysis",
            type=["jpg", "jpeg", "png", "webp"],
            key="integrated_leaf_uploader",
        )

    with col_full_soil:
        st.markdown("#### Step B: Target Crop Selection")
        if integrated_leaf is not None:
            temp_img = Image.open(integrated_leaf)
            with st.spinner("Detecting crop identity from specimen..."):
                auto_leaf_res = classify_leaf(temp_img, image_model, image_classes)
            detected_crop_key = auto_leaf_res["crop"]
            st.success(f"Auto-detected Target Crop: **{auto_leaf_res['crop_display']}**")
            integrated_crop = detected_crop_key if detected_crop_key in config.SOIL_CROPS else config.SOIL_CROPS[0]
        else:
            integrated_crop = st.selectbox(
                "Or select target crop manually:",
                options=config.SOIL_CROPS,
                format_func=pretty_crop,
                key="integrated_crop_select",
            )
            auto_leaf_res = None

    generate_full_btn = st.button("Generate Complete Agronomic Advisory", type="primary", use_container_width=True)

    if generate_full_btn:
        with st.spinner("Synthesizing multi-modal agricultural analysis..."):
            soil_res = predict_soil(soil_readings, integrated_crop, soil_model)
            diagnostics = diagnose_soil(soil_readings, integrated_crop, profile_df, profiles_dict)
            top_crops = recommend_crops(soil_readings, soil_model, top_k=3)
            integrated_advisory = generate_advisory_summary(
                leaf_result=auto_leaf_res,
                soil_result=soil_res,
                diagnostics=diagnostics,
                top_crops=top_crops,
            )

        st.markdown("---")
        st.markdown(f"### 📋 Comprehensive Farm Report: {pretty_crop(integrated_crop)}")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Soil Suitability Score", f"{soil_res['score']:.1f} / 100")
        with m2:
            st.markdown(f"**Soil Condition**: {format_status_badge(soil_res['status'])}", unsafe_allow_html=True)
        with m3:
            if auto_leaf_res:
                st.metric("Pathology Condition", auto_leaf_res["disease"])
            else:
                st.metric("Pathology Condition", "No image provided (Soil-only)")

        if auto_leaf_res:
            render_disease_detection(auto_leaf_res)

        col_c, col_t = st.columns([1, 1])
        with col_c:
            fig_bar = plot_soil_parameters_bar(diagnostics)
            if fig_bar:
                st.pyplot(fig_bar)
        with col_t:
            if diagnostics:
                st.dataframe(
                    pd.DataFrame(diagnostics)[["Parameter", "Value", "Optimal_Range", "Status"]],
                    use_container_width=True,
                    hide_index=True,
                )

        render_advisory(integrated_advisory)

st.markdown("---")
st.caption(
    "Farmer Crop Advisory System | Multi-Modal Deep Learning Agricultural Platform | "
    "Models: EfficientNetB0 (Vision) + Dual-Input Dense Neural Network (Soil Suitability)"
)

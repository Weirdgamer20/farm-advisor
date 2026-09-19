"""
Visualization module for Farmer Crop Advisory System.
Contains Streamlit presentation components, status badges, diagnostic cards,
workflow architecture diagrams, and publication-quality Matplotlib chart generators.
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

try:
    import streamlit as st
except ImportError:
    st = None


def render_header() -> None:
    """
    Renders the agricultural-themed application banner.
    """
    if st is None:
        return
    st.markdown(
        """
        <div style="padding: 20px 24px; background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 100%); border-radius: 12px; margin-bottom: 24px; color: white; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <h1 style="color: #d8f3dc; margin: 0; font-size: 2.1rem; font-weight: 700; letter-spacing: -0.5px;">🌱 Farmer Crop Advisory System</h1>
                    <p style="color: #b7e4c7; margin: 6px 0 0 0; font-size: 1.05rem;">
                        AI-Powered Agricultural Decision Support: Crop Suitability Modeling & Computer Vision Leaf Pathology
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_system_status_sidebar(gpu_active: bool, device_msg: str, models_ready: bool) -> None:
    """
    Renders a subtle, non-intrusive system status card in the sidebar.
    Avoids making GPU the hero and never implies the system is unavailable on CPU.
    """
    if st is None:
        return
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🖥️ System Status")
        
        device_label = "NVIDIA GPU" if gpu_active else "CPU"
        device_icon = "🟢" if gpu_active else "⚪"
        models_label = "Ready" if models_ready else "Error"
        models_icon = "🟢" if models_ready else "🔴"
        
        st.markdown(
            f"""
            <div style="background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px; padding: 12px; font-size: 0.88rem;">
                <div style="margin-bottom: 6px;"><strong>Inference Device:</strong> {device_icon} {device_label}</div>
                <div style="margin-bottom: 6px;"><strong>Neural Models:</strong> {models_icon} {models_label}</div>
                <div><strong>Quantile Profiles:</strong> 🟢 Loaded (9 Crops)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
def render_device_status(gpu_active: bool, device_msg: str, models_ready: bool = True) -> None:
    """
    Backward-compatible alias for render_system_status_sidebar.
    """
    render_system_status_sidebar(gpu_active, device_msg, models_ready)


def render_disclaimer() -> None:
    """
    Renders an advisory disclaimer notice at the bottom of application views.
    """
    if st is None:
        return
    st.markdown("---")
    st.markdown(
        """
        <div style="background-color: #fff9db; border-left: 4px solid #f59f00; padding: 12px 16px; border-radius: 4px; color: #664d03; font-size: 0.88rem; margin-top: 20px;">
            <strong>⚠️ Advisory Notice:</strong> This system provides AI-assisted agricultural recommendations for demonstration and decision-support purposes. Recommendations should be validated against local agro-ecological conditions and certified agricultural extension expertise before implementation.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workflow_diagram() -> None:
    """
    Renders a visual step-by-step decision workflow diagram.
    """
    if st is None:
        return
    st.markdown("### 🔄 End-to-End Decision Architecture")
    st.markdown(
        """
        <div style="display: flex; flex-wrap: wrap; align-items: center; justify-content: center; gap: 8px; padding: 16px; background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 10px; margin: 16px 0;">
            <div style="background: #ffffff; border: 1px solid #ced4da; padding: 8px 14px; border-radius: 6px; font-weight: 600; font-size: 0.9rem; text-align: center;">🌾 Farmer Input</div>
            <div style="color: #2d6a4f; font-weight: bold; font-size: 1.2rem;">→</div>
            <div style="background: #ffffff; border: 1px solid #ced4da; padding: 8px 14px; border-radius: 6px; font-weight: 600; font-size: 0.9rem; text-align: center;">🔍 Data Validation</div>
            <div style="color: #2d6a4f; font-weight: bold; font-size: 1.2rem;">→</div>
            <div style="background: #ffffff; border: 1px solid #ced4da; padding: 8px 14px; border-radius: 6px; font-weight: 600; font-size: 0.9rem; text-align: center;">🧠 ML Inference Engine</div>
            <div style="color: #2d6a4f; font-weight: bold; font-size: 1.2rem;">→</div>
            <div style="background: #ffffff; border: 1px solid #ced4da; padding: 8px 14px; border-radius: 6px; font-weight: 600; font-size: 0.9rem; text-align: center;">📊 Quantile Diagnostics</div>
            <div style="color: #2d6a4f; font-weight: bold; font-size: 1.2rem;">→</div>
            <div style="background: #2d6a4f; color: white; padding: 8px 14px; border-radius: 6px; font-weight: 600; font-size: 0.9rem; text-align: center;">📋 Actionable Prescription</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_status_badge(status: str) -> str:
    """
    Returns an HTML badge string with semantic styling for status displays.
    """
    badge_colors = {
        "GOOD": "#2d6a4f",
        "SUITABLE": "#2d6a4f",
        "ACCEPTABLE": "#1d3557",
        "NEEDS ATTENTION": "#e76f51",
        "NOT SUITABLE": "#d62828",
        "NORMAL": "#2d6a4f",
        "OPTIMAL": "#2d6a4f",
        "LOW": "#d62828",
        "HIGH": "#7209b7",
        "SLIGHTLY LOW": "#f77f00",
        "SLIGHTLY HIGH": "#4361ee",
    }
    color = badge_colors.get(status.upper(), "#6c757d")
    return (
        f'<span style="background-color: {color}; color: #ffffff; padding: 4px 10px; '
        f'border-radius: 6px; font-weight: 600; font-size: 0.85rem; letter-spacing: 0.5px;">'
        f'{status}</span>'
    )


def plot_soil_parameters_bar(
    diagnostics: List[Dict[str, Any]],
    save_path: Optional[pathlib.Path] = None,
) -> Optional[plt.Figure]:
    """
    Generates a horizontal bar chart highlighting normal vs deviated soil parameters.
    """
    if not diagnostics:
        return None

    parameters = [d["Parameter"] for d in diagnostics]
    values = [d["Value"] for d in diagnostics]
    statuses = [d["Status"] for d in diagnostics]

    color_map = {
        "NORMAL": "#2d6a4f",
        "SLIGHTLY LOW": "#f4a261",
        "LOW": "#e76f51",
        "SLIGHTLY HIGH": "#457b9d",
        "HIGH": "#9b5de5",
    }
    colors = [color_map.get(s, "#a8dadc") for s in statuses]

    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=100)
    bars = ax.barh(parameters, values, color=colors, edgecolor="#1d3557", alpha=0.9, height=0.6)

    ax.set_xlabel("Measured Reading", fontsize=10, fontweight="bold")
    ax.set_title("Soil & Environmental Readings Diagnostic Breakdown", fontsize=11, fontweight="bold", pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    max_val = max(values) if values else 1.0
    for bar, status in zip(bars, statuses):
        width = bar.get_width()
        ax.text(
            width + max_val * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{status}",
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
            color="#1d3557",
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def plot_crop_recommendations(
    recommendations: List[Dict[str, Any]],
    save_path: Optional[pathlib.Path] = None,
) -> Optional[plt.Figure]:
    """
    Renders a horizontal ranking chart of crop suitability scores.
    """
    if not recommendations:
        return None

    crops = [item["crop_display"] for item in reversed(recommendations)]
    scores = [item["score"] for item in reversed(recommendations)]

    colors = []
    for score in scores:
        if score >= 80.0:
            colors.append("#2d6a4f")
        elif score >= 60.0:
            colors.append("#1d3557")
        elif score >= 40.0:
            colors.append("#e76f51")
        else:
            colors.append("#d62828")

    fig, ax = plt.subplots(figsize=(7.5, max(3.5, len(crops) * 0.45)), dpi=100)
    bars = ax.barh(crops, scores, color=colors, edgecolor="#1b4332", alpha=0.9, height=0.55)

    ax.set_xlim(0, 100)
    ax.set_xlabel("Suitability Score (0 - 100)", fontsize=10, fontweight="bold")
    ax.set_title("Crop Suitability Ranking for Measured Soil Environment", fontsize=11, fontweight="bold", pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    for bar, score in zip(bars, scores):
        width = bar.get_width()
        ax.text(
            width + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.1f}%",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def render_disease_detection(leaf_result: Dict[str, Any]) -> None:
    """
    Renders disease diagnosis card, confidence meter, and pathology protocols.
    """
    if st is None:
        return
    st.markdown("### 🔬 Pathology Diagnosis")
    
    m1, m2 = st.columns(2)
    with m1:
        st.metric("Detected Crop Species", leaf_result["crop_display"])
    with m2:
        st.metric("Pathology Condition", leaf_result["disease"])

    conf_val = float(leaf_result.get("confidence", 0.0))
    st.markdown(f"**Diagnostic Confidence:** `{conf_val * 100:.1f}%`")
    st.progress(min(max(conf_val, 0.0), 1.0))
    st.caption(f"PlantVillage Neural Class: `{leaf_result['class_name']}`")

    if leaf_result.get("reliable", True):
        st.success("High confidence diagnosis. Suitable for targeted automated agronomic advisory.")
    else:
        st.warning("Confidence is below 60%. Agronomist review recommended before chemical treatment.")

    treatment = leaf_result.get("treatment", {})
    if treatment:
        with st.expander("📖 Pathology Symptoms & Management Protocols", expanded=True):
            st.markdown(f"**Observed Symptoms:**\n{treatment.get('symptoms', 'Foliar lesions consistent with pathogen.')}")
            st.markdown(f"**Cultural & Sanitation Practices:**\n{treatment.get('cultural', 'Prune affected tissue and maintain airflow.')}")
            st.markdown(f"**Chemical / Biological Interventions:**\n{treatment.get('chemical', 'Consult local agricultural extension.')}")


def render_advisory(advisory: Dict[str, Any]) -> None:
    """
    Renders structured agronomic recommendations, disease root causes,
    and soil target prescriptions.
    """
    if st is None:
        return
    
    # Section: Immediate Action
    recommendations = advisory.get("recommendations", [])
    if recommendations:
        st.markdown("#### 🚨 Immediate Recommended Actions")
        for rec in recommendations:
            st.markdown(f"- **{rec}**")

    # Section: Disease Root Causes
    disease_causes = advisory.get("disease_causes", [])
    if disease_causes:
        st.markdown("#### 🔬 Environmental & Soil Disease Drivers")
        st.info(
            f"The AI model correlated the following soil and weather conditions "
            f"directly with foliar stress and disease vulnerability for **{advisory.get('crop_display', 'this crop')}**:"
        )
        for cause in disease_causes:
            st.markdown(f"- ⚠️ **{cause}**")

    # Section: Nutrient & Soil Prescriptions
    prescriptions = advisory.get("prescriptions", [])
    if prescriptions:
        st.markdown(f"#### 🧪 Soil Nutrient Management & Corrective Prescriptions")
        st.caption(
            "Optimal concentration ranges and targeted fertilizer / cultural amendments required:"
        )
        presc_df = pd.DataFrame(prescriptions)
        st.dataframe(
            presc_df[["parameter", "current", "optimal", "status", "action"]].rename(
                columns={
                    "parameter": "Parameter",
                    "current": "Measured Value",
                    "optimal": "Target Range",
                    "status": "Status",
                    "action": "Corrective Amendment Required",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

"""
Visualization module for Farmer Crop Advisory System.
Implements the centralized CSS design system, high-contrast UI cards, status badges,
workflow architecture visualizations, and harmonized publication-grade Matplotlib plots.
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

# ============================================================
# DESIGN SYSTEM TOKENS & COLOR PALETTE
# ============================================================

THEME_COLORS = {
    "primary": "#2d6a4f",
    "primary_dark": "#1b4332",
    "primary_light": "#d8f3dc",
    "primary_soft": "#e8f5ed",
    "text_dark": "#1b2d24",
    "text_muted": "#495057",
    "card_bg": "#ffffff",
    "card_border": "#d2e3d8",
    "status_good": "#2d6a4f",
    "status_acceptable": "#1d3557",
    "status_attention": "#e76f51",
    "status_unsuitable": "#d62828",
    "status_neutral": "#6c757d",
}


def inject_custom_theme() -> None:
    """
    Injects centralized CSS tokens and responsive component styling into the Streamlit session.
    Guarantees strict high-contrast legibility across all surfaces, avoiding white-on-light bugs.
    """
    if st is None:
        return

    st.markdown(
        """
        <style>
        /* Base typography and background refinement */
        .stApp {
            color: #1b2d24;
        }

        /* Centralized Component Styles */
        .advisory-header {
            background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 100%);
            border-radius: 10px;
            padding: 22px 26px;
            margin-bottom: 22px;
            color: #ffffff;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.06);
        }
        .advisory-header h1 {
            color: #d8f3dc !important;
            margin: 0;
            font-size: 2.1rem;
            font-weight: 700;
        }
        .advisory-header p {
            color: #cce3d5 !important;
            margin: 6px 0 0 0;
            font-size: 1.02rem;
        }

        .advisory-card {
            background-color: #ffffff;
            border: 1px solid #d2e3d8;
            border-radius: 8px;
            padding: 18px 20px;
            color: #1b2d24;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.03);
            margin-bottom: 16px;
        }
        .advisory-card h3, .advisory-card h4 {
            color: #1b4332 !important;
            margin-top: 0;
            margin-bottom: 8px;
            font-weight: 600;
        }
        .advisory-card p {
            color: #495057;
            font-size: 0.92rem;
            line-height: 1.45;
            margin-bottom: 10px;
        }

        .ranking-card {
            background-color: #ffffff;
            border: 1px solid #d2e3d8;
            border-radius: 8px;
            padding: 16px;
            text-align: center;
            color: #1b2d24;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.03);
        }
        .ranking-card .rank-tag {
            font-size: 0.8rem;
            font-weight: 700;
            color: #6c757d;
            letter-spacing: 0.5px;
        }
        .ranking-card .crop-title {
            color: #1b4332;
            font-size: 1.25rem;
            font-weight: 700;
            margin: 4px 0;
        }
        .ranking-card .score-value {
            font-size: 1.65rem;
            font-weight: 700;
            color: #2d6a4f;
            margin: 4px 0;
        }

        .system-status-box {
            background-color: #f4f7f5;
            border: 1px solid #d2e3d8;
            border-radius: 8px;
            padding: 14px 16px;
            color: #1b2d24;
            font-size: 0.88rem;
            margin-top: 10px;
        }
        .system-status-box .status-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 6px;
        }
        .system-status-box .status-row:last-child {
            margin-bottom: 0;
        }

        .workflow-container {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: center;
            gap: 10px;
            padding: 16px;
            background-color: #f8faf9;
            border: 1px solid #d8e5dc;
            border-radius: 8px;
            margin: 16px 0;
        }
        .workflow-node {
            background: #ffffff;
            border: 1px solid #b7d5c2;
            color: #1b4332;
            padding: 8px 14px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.88rem;
            text-align: center;
        }
        .workflow-arrow {
            color: #2d6a4f;
            font-weight: bold;
            font-size: 1.1rem;
        }

        .disclaimer-box {
            background-color: #fef9e7;
            border-left: 4px solid #f39c12;
            padding: 12px 16px;
            border-radius: 4px;
            color: #7d5a00;
            font-size: 0.86rem;
            line-height: 1.4;
            margin-top: 24px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """
    Renders the agricultural application banner.
    """
    if st is None:
        return
    st.markdown(
        """
        <div class="advisory-header">
            <h1>🌱 Farmer Crop Advisory System</h1>
            <p>
                Decision Support: Crop Suitability Neural Modeling & Computer Vision Leaf Pathology Diagnosis
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_system_status_sidebar(
    gpu_active: bool,
    device_msg: str,
    models_ready: bool,
    detailed_telemetry: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Renders a subtle, non-intrusive system status card in the sidebar.
    Guarantees that CPU fallback is displayed neutrally rather than as an error.
    """
    if st is None:
        return

    device_label = "NVIDIA GPU (CUDA)" if gpu_active else "CPU"
    device_indicator = "🟢 Active" if gpu_active else "⚪ Standard"
    models_label = "Operational" if models_ready else "Error"
    models_indicator = "🟢 Loaded" if models_ready else "🔴 Missing"

    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🖥️ System Status")
        st.markdown(
            f"""
            <div class="system-status-box">
                <div class="status-row">
                    <span><strong>Inference Device:</strong></span>
                    <span>{device_label} ({device_indicator})</span>
                </div>
                <div class="status-row">
                    <span><strong>Neural Models:</strong></span>
                    <span>{models_label} ({models_indicator})</span>
                </div>
                <div class="status-row">
                    <span><strong>Quantile Profiles:</strong></span>
                    <span>9 Crop Profiles 🟢</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if detailed_telemetry:
            with st.expander("🔍 Detailed Model Telemetry", expanded=False):
                for k, v in detailed_telemetry.items():
                    status_icon = "✅" if v.get("exists") else "❌"
                    st.markdown(f"**{v.get('name')}**: {status_icon}")
                    st.caption(f"`{v.get('path')}`")


def render_device_status(gpu_active: bool, device_msg: str, models_ready: bool = True) -> None:
    """
    Backward-compatible alias for render_system_status_sidebar.
    """
    render_system_status_sidebar(gpu_active, device_msg, models_ready)


def render_disclaimer() -> None:
    """
    Renders the agricultural decision-support disclaimer.
    """
    if st is None:
        return
    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>⚠️ Decision-Support Advisory Notice:</strong>
            This system provides AI-assisted agricultural recommendations for research and decision-support purposes.
            Prescriptions should be cross-referenced with local agro-ecological conditions and certified extension services.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workflow_diagram() -> None:
    """
    Renders the step-by-step decision architecture workflow.
    """
    if st is None:
        return
    st.markdown("### 🔄 Decision Engine Architecture")
    st.markdown(
        """
        <div class="workflow-container">
            <div class="workflow-node">🌾 Farmer Input</div>
            <div class="workflow-arrow">→</div>
            <div class="workflow-node">🔍 Range Validation</div>
            <div class="workflow-arrow">→</div>
            <div class="workflow-node">🧠 Neural Inference</div>
            <div class="workflow-arrow">→</div>
            <div class="workflow-node">📊 Quantile Diagnostics</div>
            <div class="workflow-arrow">→</div>
            <div class="workflow-node" style="background: #2d6a4f; color: #ffffff;">📋 Agronomic Prescription</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_status_badge(status: str) -> str:
    """
    Returns a high-contrast HTML badge string with explicit white text on semantic backgrounds.
    """
    status_upper = str(status).strip().upper()
    badge_colors = {
        "GOOD": "#2d6a4f",
        "SUITABLE": "#2d6a4f",
        "OPTIMAL": "#2d6a4f",
        "NORMAL": "#2d6a4f",
        "ACCEPTABLE": "#1d3557",
        "NEEDS ATTENTION": "#d9534f",
        "ATTENTION": "#d9534f",
        "NOT SUITABLE": "#b52b27",
        "LOW": "#b52b27",
        "HIGH": "#6f42c1",
        "SLIGHTLY LOW": "#d35400",
        "SLIGHTLY HIGH": "#2980b9",
    }
    color = badge_colors.get(status_upper, "#495057")
    return (
        f'<span style="background-color: {color}; color: #ffffff !important; padding: 4px 10px; '
        f'border-radius: 4px; font-weight: 600; font-size: 0.82rem; letter-spacing: 0.5px; display: inline-block;">'
        f'{status}</span>'
    )


def plot_soil_parameters_bar(
    diagnostics: List[Dict[str, Any]],
    save_path: Optional[pathlib.Path] = None,
) -> Optional[plt.Figure]:
    """
    Generates a horizontal bar chart highlighting normal vs deviated soil parameters,
    harmonized with the application theme tokens.
    """
    if not diagnostics:
        return None

    parameters = [d["Parameter"] for d in diagnostics]
    values = [float(d["Value"]) for d in diagnostics]
    statuses = [d["Status"] for d in diagnostics]

    color_map = {
        "NORMAL": "#2d6a4f",
        "SLIGHTLY LOW": "#e67e22",
        "LOW": "#c0392b",
        "SLIGHTLY HIGH": "#2980b9",
        "HIGH": "#8e44ad",
    }
    colors = [color_map.get(s, "#7f8c8d") for s in statuses]

    fig, ax = plt.subplots(figsize=(7.8, 3.8), dpi=100)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafbfc")

    bars = ax.barh(parameters, values, color=colors, edgecolor="#1b2d24", alpha=0.9, height=0.55)

    ax.set_xlabel("Measured Soil / Climate Value", fontsize=9.5, fontweight="bold", color="#1b2d24")
    ax.set_title("Soil & Climate Reading Diagnostic Breakdown", fontsize=10.5, fontweight="bold", color="#1b4332", pad=12)
    ax.tick_params(colors="#1b2d24", labelsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#ced4da")
    ax.spines["bottom"].set_color("#ced4da")
    ax.grid(axis="x", linestyle="--", alpha=0.35, color="#ced4da")

    max_val = max(values) if values else 1.0
    for bar, status in zip(bars, statuses):
        width = bar.get_width()
        ax.text(
            width + max_val * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{status}",
            va="center",
            ha="left",
            fontsize=8,
            fontweight="bold",
            color="#1b4332",
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
    Renders a horizontal ranking chart of crop suitability scores,
    harmonized with the application theme tokens.
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
            colors.append("#e67e22")
        else:
            colors.append("#c0392b")

    fig, ax = plt.subplots(figsize=(7.8, max(3.5, len(crops) * 0.44)), dpi=100)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafbfc")

    bars = ax.barh(crops, scores, color=colors, edgecolor="#1b2d24", alpha=0.9, height=0.55)

    ax.set_xlim(0, 100)
    ax.set_xlabel("Suitability Score (0 - 100%)", fontsize=9.5, fontweight="bold", color="#1b2d24")
    ax.set_title("Alternative Crop Suitability Ranking (Neural Net Inference)", fontsize=10.5, fontweight="bold", color="#1b4332", pad=12)
    ax.tick_params(colors="#1b2d24", labelsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#ced4da")
    ax.spines["bottom"].set_color("#ced4da")
    ax.grid(axis="x", linestyle="--", alpha=0.35, color="#ced4da")

    for bar, score in zip(bars, scores):
        width = bar.get_width()
        ax.text(
            width + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.1f}%",
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
            color="#1b2d24",
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def render_disease_detection(leaf_result: Dict[str, Any]) -> None:
    """
    Renders pathology diagnosis card, confidence indicator, and management protocols.
    """
    if st is None:
        return

    st.markdown("### 🔬 Vision Diagnostic Report")

    m1, m2 = st.columns(2)
    with m1:
        st.metric("Detected Crop Species", leaf_result["crop_display"])
    with m2:
        st.metric("Pathology Condition", leaf_result["disease"])

    conf_val = float(leaf_result.get("confidence", 0.0))
    st.markdown(f"**Diagnostic Confidence:** `{conf_val * 100:.1f}%`")
    st.progress(min(max(conf_val, 0.0), 1.0))
    st.caption(f"PlantVillage Classification Label: `{leaf_result['class_name']}`")

    if leaf_result.get("reliable", True):
        st.success("High confidence diagnosis. Suitable for targeted agronomic management.")
    else:
        st.warning("Confidence is below 60%. Agronomist review recommended prior to treatment.")

    treatment = leaf_result.get("treatment", {})
    if treatment:
        with st.expander("📖 Pathology Symptoms & Management Protocols", expanded=True):
            st.markdown(f"**Observed Symptoms:**\n{treatment.get('symptoms', 'Foliar lesions consistent with pathogen.')}")
            st.markdown(f"**Cultural & Sanitation Practices:**\n{treatment.get('cultural', 'Prune affected tissue and maintain canopy airflow.')}")
            st.markdown(f"**Chemical / Biological Interventions:**\n{treatment.get('chemical', 'Consult local agricultural extension for registered preventative foliar sprays.')}")


def render_advisory(advisory: Dict[str, Any]) -> None:
    """
    Renders structured agronomic recommendations, disease root causes,
    and soil target prescriptions.
    """
    if st is None:
        return

    # Section 1: Immediate Action
    recommendations = advisory.get("recommendations", [])
    if recommendations:
        st.markdown("#### 🚨 Immediate Recommended Actions")
        for rec in recommendations:
            st.markdown(f"- **{rec}**")

    # Section 2: Disease Root Causes
    disease_causes = advisory.get("disease_causes", [])
    if disease_causes:
        st.markdown("#### 🔬 Environmental & Soil Disease Drivers")
        st.info(
            f"The advisory engine identified the following environmental and soil conditions "
            f"contributing to stress and foliar vulnerability for **{advisory.get('crop_display', 'this crop')}**:"
        )
        for cause in disease_causes:
            st.markdown(f"- ⚠️ **{cause}**")

    # Section 3: Nutrient Management & Prescriptions
    prescriptions = advisory.get("prescriptions", [])
    if prescriptions:
        st.markdown("#### 🧪 Soil Nutrient Management & Corrective Prescriptions")
        st.caption("Optimal concentration ranges and specific corrective amendments:")
        presc_df = pd.DataFrame(prescriptions)
        st.dataframe(
            presc_df[["parameter", "current", "optimal", "status", "action"]].rename(
                columns={
                    "parameter": "Parameter",
                    "current": "Measured Value",
                    "optimal": "Optimal Range",
                    "status": "Status",
                    "action": "Corrective Amendment Required",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

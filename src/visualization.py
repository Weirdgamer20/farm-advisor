"""
Visualization module for Farmer Crop Advisory System.
Implements the centralized CSS design system, high-contrast UI cards, status badges,
workflow architecture visualizations, and harmonized publication-grade Matplotlib plots.
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

try:
    import streamlit as st
except ImportError:
    st = None


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
        .stApp { color: #1b2d24; }
        .advisory-header {
            background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 100%);
            border-radius: 10px;
            padding: 22px 26px;
            margin-bottom: 22px;
            color: #ffffff;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.06);
        }
        .advisory-header h1 { color: #d8f3dc !important; margin: 0; font-size: 2.1rem; font-weight: 700; }
        .advisory-header p { color: #cce3d5 !important; margin: 6px 0 0 0; font-size: 1.02rem; }

        .advisory-card {
            background-color: #ffffff;
            border: 1px solid #d2e3d8;
            border-radius: 8px;
            padding: 18px 20px;
            color: #1b2d24;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.03);
            margin-bottom: 16px;
        }
        .advisory-card h3, .advisory-card h4 { color: #1b4332 !important; margin-top: 0; margin-bottom: 8px; font-weight: 600; }
        .advisory-card p { color: #495057; font-size: 0.92rem; line-height: 1.45; margin-bottom: 10px; }

        .ranking-card {
            background-color: #ffffff;
            border: 1px solid #d2e3d8;
            border-radius: 8px;
            padding: 16px;
            text-align: center;
            color: #1b2d24;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.03);
        }
        .ranking-card .rank-tag { font-size: 0.8rem; font-weight: 700; color: #6c757d; letter-spacing: 0.5px; }
        .ranking-card .crop-title { color: #1b4332; font-size: 1.25rem; font-weight: 700; margin: 4px 0; }
        .ranking-card .score-value { font-size: 1.65rem; font-weight: 700; color: #2d6a4f; margin: 4px 0; }

        .system-status-box {
            background-color: #f4f7f5;
            border: 1px solid #d2e3d8;
            border-radius: 8px;
            padding: 14px 16px;
            color: #1b2d24;
            font-size: 0.88rem;
            margin-top: 10px;
        }
        .system-status-box .status-row { display: flex; justify-content: space-between; margin-bottom: 6px; }
        .system-status-box .status-row:last-child { margin-bottom: 0; }

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
        .workflow-arrow { color: #2d6a4f; font-weight: bold; font-size: 1.1rem; }

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
    """Renders the agricultural application banner."""
    if st is None:
        return
    st.markdown(
        """
        <div class="advisory-header">
            <h1>🌱 Farmer Crop Advisory System</h1>
            <p>Decision Support: Crop Suitability Neural Modeling & Computer Vision Leaf Pathology Diagnosis</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_system_status_sidebar(
    gpu_active: bool,
    device_msg: str,
    models_ready: bool,
    detailed_telemetry: Optional[Dict[str, Any]] = None,
    runtime_snapshot: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Renders an accurate, transparent system status card in the sidebar.
    Decouples physical GPU hardware from TensorFlow CUDA execution.
    """
    if st is None:
        return

    hw = (runtime_snapshot or {}).get("hardware", {})
    physical_gpu = hw.get("physical_gpu")
    tf_cuda = hw.get("tf_cuda_active", gpu_active)

    if tf_cuda:
        tf_label = "CUDA GPU (Accelerated)"
        tf_indicator = "🟢 Active"
    else:
        tf_label = "CPU Inference"
        tf_indicator = "⚪ Standard"

    if physical_gpu:
        hw_label = f"{physical_gpu[:24]}..." if len(physical_gpu) > 24 else physical_gpu
        hw_sub = "🟢 Detected"
    else:
        hw_label = "Host CPU"
        hw_sub = "⚪ Detected"

    rt_status = (runtime_snapshot or {}).get("status", "READY" if models_ready else "NOT_STARTED")
    if rt_status == "READY":
        engine_label, engine_indicator = "Operational", "🟢 Ready"
    elif rt_status == "INITIALIZING":
        engine_label, engine_indicator = "Initializing", "⟳ Background"
    else:
        engine_label, engine_indicator = "Standby/Notice", "🟡 Pending"

    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🖥️ System & Hardware Status")
        st.markdown(
            f"""
            <div class="system-status-box">
                <div class="status-row">
                    <span><strong>Physical GPU:</strong></span>
                    <span>{hw_label} ({hw_sub})</span>
                </div>
                <div class="status-row">
                    <span><strong>TensorFlow Runtime:</strong></span>
                    <span>{tf_label} ({tf_indicator})</span>
                </div>
                <div class="status-row">
                    <span><strong>AI Engine:</strong></span>
                    <span>{engine_label} ({engine_indicator})</span>
                </div>
                <div class="status-row">
                    <span><strong>Empirical Profiles:</strong></span>
                    <span>9 Crop Profiles 🟢</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if detailed_telemetry:
            with st.expander("🔍 Disk Artifact Telemetry", expanded=False):
                for k, v in detailed_telemetry.items():
                    status_icon = "✅" if v.get("exists") else "❌"
                    st.markdown(f"**{v.get('name')}**: {status_icon}")
                    st.caption(f"`{v.get('path')}`")


def render_ai_engine_status_card(snapshot: Dict[str, Any]) -> None:
    """Renders real-time AI background runtime state banner with step breakdown."""
    if st is None or not snapshot:
        return

    status = snapshot.get("status", "NOT_STARTED")
    stages = snapshot.get("stages", {})
    msg = snapshot.get("progress_message", "")

    if status == "READY":
        st.markdown(
            """
            <div class="advisory-card" style="border-left: 5px solid #2d6a4f; padding: 14px 18px; margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 700; color: #1b4332; font-size: 1.05rem;">AI ENGINE: ● READY</span>
                    <span style="background: #e8f5ed; color: #2d6a4f; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 0.8rem;">Fully Operational</span>
                </div>
                <div style="margin-top: 8px; font-size: 0.86rem; color: #495057; display: grid; grid-template-columns: 1fr 1fr; gap: 4px;">
                    <div>Disease Vision Model: <strong style="color: #2d6a4f;">✓ Loaded</strong></div>
                    <div>Soil Neural Model: <strong style="color: #2d6a4f;">✓ Loaded</strong></div>
                    <div>Empirical Profiles: <strong style="color: #2d6a4f;">✓ Active (9 Crops)</strong></div>
                    <div>Inference Graph: <strong style="color: #2d6a4f;">✓ Warmed</strong></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif status == "INITIALIZING":
        s_tf = "✓" if stages.get("tensorflow") else "⟳"
        s_dm = "✓" if stages.get("disease_model") else "○"
        s_sm = "✓" if stages.get("soil_model") else "○"
        s_wm = "✓" if stages.get("warmup") else "○"
        st.markdown(
            f"""
            <div class="advisory-card" style="border-left: 5px solid #e76f51; padding: 14px 18px; margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 700; color: #e76f51; font-size: 1.05rem;">AI ENGINE: ⟳ INITIALIZING IN BACKGROUND</span>
                    <span style="background: #fef0ec; color: #e76f51; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 0.8rem;">Non-blocking Startup</span>
                </div>
                <div style="margin-top: 6px; font-size: 0.88rem; color: #1b2d24;">
                    <strong>Current Step:</strong> {msg}
                </div>
                <div style="margin-top: 8px; font-size: 0.84rem; color: #495057; display: flex; gap: 16px; flex-wrap: wrap;">
                    <span>TensorFlow: <strong>{s_tf}</strong></span>
                    <span>Pathology CNN: <strong>{s_dm}</strong></span>
                    <span>Suitability MLP: <strong>{s_sm}</strong></span>
                    <span>Graph Warm-up: <strong>{s_wm}</strong></span>
                </div>
                <div style="margin-top: 6px; font-size: 0.8rem; color: #6c757d;">
                    The interface is fully responsive. AI prediction buttons will activate once initialization concludes.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif status == "FAILED":
        st.markdown(
            f"""
            <div class="advisory-card" style="border-left: 5px solid #d62828; padding: 14px 18px; margin-bottom: 16px;">
                <div style="font-weight: 700; color: #d62828; font-size: 1.05rem;">AI ENGINE: ❌ INITIALIZATION ERROR</div>
                <div style="margin-top: 6px; font-size: 0.86rem; color: #495057;">{snapshot.get('error', 'Unknown runtime error')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )



def render_disclaimer() -> None:
    """Renders the agricultural decision-support disclaimer."""
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
    """Renders the step-by-step decision architecture workflow."""
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
    """Returns a high-contrast HTML badge string with explicit white text on semantic backgrounds."""
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
    color = badge_colors.get(str(status).strip().upper(), "#495057")
    return (
        f'<span style="background-color: {color}; color: #ffffff !important; padding: 4px 10px; '
        f'border-radius: 4px; font-weight: 600; font-size: 0.82rem; letter-spacing: 0.5px; display: inline-block;">'
        f'{status}</span>'
    )


def _setup_barh_plot(height: float, title: str, xlabel: str) -> Tuple[plt.Figure, plt.Axes]:
    """Helper creating a harmonized, theme-styled horizontal bar figure and axis."""
    fig, ax = plt.subplots(figsize=(7.8, height), dpi=100)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafbfc")
    ax.set_xlabel(xlabel, fontsize=9.5, fontweight="bold", color="#1b2d24")
    ax.set_title(title, fontsize=10.5, fontweight="bold", color="#1b4332", pad=12)
    ax.tick_params(colors="#1b2d24", labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#ced4da")
    ax.grid(axis="x", linestyle="--", alpha=0.35, color="#ced4da")
    return fig, ax


def _finalize_plot(fig: plt.Figure, save_path: Optional[pathlib.Path] = None) -> plt.Figure:
    """Helper completing layout tight pack and saving."""
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_soil_parameters_bar(
    diagnostics: List[Dict[str, Any]],
    save_path: Optional[pathlib.Path] = None,
) -> Optional[plt.Figure]:
    """Generates a horizontal bar chart highlighting normal vs deviated soil parameters."""
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

    fig, ax = _setup_barh_plot(
        height=3.8,
        title="Soil & Climate Reading Diagnostic Breakdown",
        xlabel="Measured Soil / Climate Value",
    )
    bars = ax.barh(parameters, values, color=colors, edgecolor="#1b2d24", alpha=0.9, height=0.55)

    max_val = max(values) if values else 1.0
    for bar, status in zip(bars, statuses):
        ax.text(
            bar.get_width() + max_val * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{status}",
            va="center",
            ha="left",
            fontsize=8,
            fontweight="bold",
            color="#1b4332",
        )

    return _finalize_plot(fig, save_path)


def plot_crop_recommendations(
    recommendations: List[Dict[str, Any]],
    save_path: Optional[pathlib.Path] = None,
) -> Optional[plt.Figure]:
    """Renders a horizontal ranking chart of crop suitability scores."""
    if not recommendations:
        return None

    crops = [item["crop_display"] for item in reversed(recommendations)]
    scores = [item["score"] for item in reversed(recommendations)]

    colors = [
        "#2d6a4f" if s >= 80.0 else ("#1d3557" if s >= 60.0 else ("#e67e22" if s >= 40.0 else "#c0392b"))
        for s in scores
    ]

    fig, ax = _setup_barh_plot(
        height=max(3.5, len(crops) * 0.44),
        title="Alternative Crop Suitability Ranking (Neural Net Inference)",
        xlabel="Suitability Score (0 - 100%)",
    )
    ax.set_xlim(0, 100)
    bars = ax.barh(crops, scores, color=colors, edgecolor="#1b2d24", alpha=0.9, height=0.55)

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_width() + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.1f}%",
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
            color="#1b2d24",
        )

    return _finalize_plot(fig, save_path)


def render_disease_detection(leaf_result: Dict[str, Any]) -> None:
    """Renders pathology diagnosis card, confidence indicator, and management protocols."""
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
    """Renders structured agronomic recommendations, disease root causes, and soil target prescriptions."""
    if st is None:
        return

    recommendations = advisory.get("recommendations", [])
    if recommendations:
        st.markdown("#### 🚨 Immediate Recommended Actions")
        for rec in recommendations:
            st.markdown(f"- **{rec}**")

    disease_causes = advisory.get("disease_causes", [])
    if disease_causes:
        st.markdown("#### 🔬 Environmental & Soil Disease Drivers")
        st.info(
            f"The advisory engine identified the following environmental and soil conditions "
            f"contributing to stress and foliar vulnerability for **{advisory.get('crop_display', 'this crop')}**:"
        )
        for cause in disease_causes:
            st.markdown(f"- ⚠️ **{cause}**")

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

"""
Visualization module for Farmer Crop Advisory System.
Responsible for Streamlit UI components, KPI metrics, status badges, and analytical charts.
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from config import FIGURES_DIR


def render_header() -> None:
    """
    Renders a modern, agricultural-themed application banner.
    """
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


def render_device_status(gpu_active: bool, device_msg: str) -> None:
    """
    Renders non-blocking hardware acceleration status in the sidebar.
    """
    with st.sidebar:
        st.markdown("### ⚙️ Hardware Status")
        if gpu_active:
            st.success(f"🚀 {device_msg}")
        else:
            st.info(f"💻 {device_msg}")
        st.caption("Inference runs seamlessly across CPU and GPU hardware.")


def format_status_badge(status: str) -> str:
    """
    Returns an HTML badge string with appropriate semantic styling for status displays.
    """
    badge_colors = {
        "GOOD": "#2d6a4f",
        "ACCEPTABLE": "#1d3557",
        "NEEDS ATTENTION": "#e76f51",
        "NOT SUITABLE": "#d62828",
        "NORMAL": "#2d6a4f",
        "LOW": "#d62828",
        "HIGH": "#7209b7",
        "SLIGHTLY LOW": "#f77f00",
        "SLIGHTLY HIGH": "#4361ee",
    }
    color = badge_colors.get(status, "#6c757d")
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
    Generates a visual horizontal bar chart highlighting normal vs deviated soil parameters.
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

    for bar, status in zip(bars, statuses):
        width = bar.get_width()
        ax.text(
            width + max(values) * 0.02,
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

    # Reverse order so highest score is at the top
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
    Renders disease diagnosis card and treatment recommendations.
    """
    st.subheader("Diagnostic Detection")
    m1, m2 = st.columns(2)
    with m1:
        st.metric("Detected Crop", leaf_result["crop_display"])
    with m2:
        st.metric("Condition / Disease", leaf_result["disease"])

    st.metric("Model Confidence", f"{leaf_result['confidence'] * 100:.2f}%")
    st.caption(f"PlantVillage Pathology Class: `{leaf_result['class_name']}`")

    if leaf_result["reliable"]:
        st.success("High confidence diagnosis. Suitable for targeted automated agronomic advisory.")
    else:
        st.warning("Confidence is below 60%. Agronomist review recommended before treatment application.")

    treatment = leaf_result.get("treatment", {})
    if treatment:
        with st.expander("📖 Pathology Symptoms & Management Protocols", expanded=True):
            st.markdown(f"**Observed Symptoms:** {treatment.get('symptoms', 'N/A')}")
            st.markdown(f"**Cultural & Sanitation Practices:** {treatment.get('cultural', 'N/A')}")
            st.markdown(f"**Chemical / Biological Interventions:** {treatment.get('chemical', 'N/A')}")


def render_advisory(advisory: Dict[str, Any]) -> None:
    """
    Renders structured agronomic recommendations.
    """
    deviations = advisory.get("deviations", [])
    if deviations:
        st.markdown("#### ⚠️ Identified Soil & Environmental Deviations")
        for dev in deviations:
            st.markdown(f"- {dev}")
    else:
        st.success("All soil and environmental metrics align with empirical optimal ranges.")

    recommendations = advisory.get("recommendations", [])
    if recommendations:
        st.markdown("#### 📋 Actionable Agronomic Recommendations")
        for rec in recommendations:
            st.markdown(f"- **{rec}**")

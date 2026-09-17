"""
Visualization module for Farmer Crop Advisory System.
Generates radar charts, parameter deviation plots, and styled presentation outputs.
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FIGURES_DIR


def plot_soil_parameters_bar(
    diagnostics: List[Dict[str, Any]],
    save_path: Optional[pathlib.Path] = None,
) -> Optional[plt.Figure]:
    """
    Generates a visual parameter deviation bar chart highlighting normal vs deviated inputs.
    """
    if not diagnostics:
        return None

    parameters = [d["Parameter"] for d in diagnostics]
    values = [d["Value"] for d in diagnostics]
    statuses = [d["Status"] for d in diagnostics]

    color_map = {
        "NORMAL": "#2ecc71",
        "SLIGHTLY LOW": "#f39c12",
        "LOW": "#e74c3c",
        "SLIGHTLY HIGH": "#3498db",
        "HIGH": "#9b59b6",
    }
    colors = [color_map.get(s, "#95a5a6") for s in statuses]

    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    bars = ax.barh(parameters, values, color=colors, edgecolor="#2c3e50", alpha=0.85)

    ax.set_xlabel("Value")
    ax.set_title("Soil & Environmental Readings Diagnostic Breakdown", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, status in zip(bars, statuses):
        width = bar.get_width()
        ax.text(
            width * 1.02,
            bar.get_y() + bar.get_height() / 2,
            f" {status}",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="semibold",
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def format_status_badge(status: str) -> str:
    """
    Returns an HTML badge string with appropriate semantic styling for status displays.
    """
    badge_colors = {
        "GOOD": "#28a745",
        "ACCEPTABLE": "#17a2b8",
        "NEEDS ATTENTION": "#ffc107",
        "NOT SUITABLE": "#dc3545",
        "NORMAL": "#28a745",
        "LOW": "#dc3545",
        "HIGH": "#6f42c1",
        "SLIGHTLY LOW": "#fd7e14",
        "SLIGHTLY HIGH": "#007bff",
    }
    color = badge_colors.get(status, "#6c757d")
    return f'<span style="background-color: {color}; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 0.85rem;">{status}</span>'

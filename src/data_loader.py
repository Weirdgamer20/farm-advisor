"""
Data loader module for Farmer Crop Advisory System.
Handles loading of dataset profiles, model artifacts, and class labels with robust caching.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from config import (
    CROP_DATA,
    CSV_CANDIDATES,
    DISEASE_CLASSES,
    DISEASE_MODEL,
    IMAGE_BACKUP_PATH,
    IMAGE_CPU_BEST_PATH,
    SOIL_FEATURES,
    SOIL_MODEL,
    SOIL_PROFILES,
)
from src.utils import canonical_crop

# Streamlit caching decorators when run inside Streamlit
try:
    import streamlit as st
    cache_resource = st.cache_resource
    cache_data = st.cache_data
except ImportError:
    # Standalone or non-Streamlit execution fallback
    def cache_resource(fn):
        return fn
    def cache_data(fn):
        return fn


@cache_resource
def load_models(
    image_model_path: pathlib.Path = DISEASE_MODEL,
    image_classes_path: pathlib.Path = DISEASE_CLASSES,
    soil_model_path: pathlib.Path = SOIL_MODEL,
) -> Tuple[Any, List[str], Any]:
    """
    Loads and validates the trained leaf disease model and soil suitability neural network.
    Gracefully falls back to backup model artifacts if primary weights are missing.
    """
    import tensorflow as tf

    # Verify or fallback for vision model
    target_image_model_path = image_model_path
    if not target_image_model_path.exists():
        if IMAGE_CPU_BEST_PATH.exists():
            target_image_model_path = IMAGE_CPU_BEST_PATH
        elif IMAGE_BACKUP_PATH.exists():
            target_image_model_path = IMAGE_BACKUP_PATH
        else:
            raise FileNotFoundError(
                f"Plant pathology vision model artifact not found at: {image_model_path}"
            )

    if not image_classes_path.exists():
        raise FileNotFoundError(
            f"Plant pathology class labels not found at: {image_classes_path}"
        )

    if not soil_model_path.exists():
        raise FileNotFoundError(
            f"Soil condition neural network artifact not found at: {soil_model_path}"
        )

    # Load artifacts
    image_model = tf.keras.models.load_model(target_image_model_path)
    soil_model = tf.keras.models.load_model(soil_model_path)

    with image_classes_path.open("r", encoding="utf-8") as f:
        image_classes: List[str] = json.load(f)

    # Validation checks
    if image_model.output_shape[-1] != len(image_classes):
        raise ValueError(
            f"Leaf model output count ({image_model.output_shape[-1]}) does not match class count ({len(image_classes)})."
        )

    if len(soil_model.inputs) != 2:
        raise ValueError("Soil model must take 2 inputs: 'numeric' and 'crop_index'.")

    return image_model, image_classes, soil_model


@cache_data
def load_crop_data(csv_path: Optional[pathlib.Path] = None) -> Optional[pd.DataFrame]:
    """
    Loads and validates the crop recommendation dataset used for empirical quantile diagnostics.
    """
    target_path = csv_path
    if target_path is None:
        target_path = next((p for p in CSV_CANDIDATES if p.exists()), None)

    if target_path is None or not target_path.exists():
        return None

    try:
        df = pd.read_csv(target_path)
        df.columns = [str(c).strip() for c in df.columns]

        required_columns = SOIL_FEATURES + ["crop"]
        if not all(c in df.columns for c in required_columns):
            return None

        for feature in SOIL_FEATURES:
            df[feature] = pd.to_numeric(df[feature], errors="coerce")

        df["crop"] = df["crop"].map(lambda v: canonical_crop(v, strict=False))
        return df.dropna(subset=required_columns)
    except Exception:
        return None


def load_profile_data(csv_path: Optional[pathlib.Path] = None) -> Optional[pd.DataFrame]:
    """
    Backward-compatible alias for load_crop_data.
    """
    return load_crop_data(csv_path)


@cache_data
def load_soil_profiles(profiles_path: pathlib.Path = SOIL_PROFILES) -> Dict[str, Any]:
    """
    Loads precalculated empirical distribution quantiles (p10, p25, median, p75, p90)
    for each supported crop.
    """
    if not profiles_path.exists():
        return {}

    try:
        with profiles_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

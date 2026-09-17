"""
Data loader module for Farmer Crop Advisory System.
Handles loading of dataset profiles, model artifacts, and class labels.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Tuple, Optional

import pandas as pd

from config import (
    IMAGE_MODEL_PATH,
    IMAGE_CLASSES_PATH,
    SOIL_MODEL_PATH,
    CSV_CANDIDATES,
    SOIL_FEATURES,
)
from src.utils import canonical_crop

# Optional Streamlit cache decorators if run within Streamlit
try:
    import streamlit as st
    cache_resource = st.cache_resource
    cache_data = st.cache_data
except ImportError:
    # No-op decorators for headless / standalone execution
    def cache_resource(fn):
        return fn
    def cache_data(fn):
        return fn


@cache_resource
def load_models(
    image_model_path: pathlib.Path = IMAGE_MODEL_PATH,
    image_classes_path: pathlib.Path = IMAGE_CLASSES_PATH,
    soil_model_path: pathlib.Path = SOIL_MODEL_PATH,
) -> Tuple[Any, list[str], Any]:
    """
    Loads and validates the trained leaf disease model and soil suitability neural network.
    """
    import tensorflow as tf

    required = [image_model_path, image_classes_path, soil_model_path]
    missing = [str(p) for p in required if not p.exists()]

    if missing:
        raise FileNotFoundError(
            "Required model artifacts are missing:\n"
            + "\n".join(missing)
            + "\n\nPlease ensure trained model files exist in models/ directory."
        )

    image_model = tf.keras.models.load_model(image_model_path)
    soil_model = tf.keras.models.load_model(soil_model_path)

    with image_classes_path.open("r", encoding="utf-8") as f:
        image_classes = json.load(f)

    if image_model.output_shape[-1] != len(image_classes):
        raise ValueError(
            f"Leaf model output count ({image_model.output_shape[-1]}) does not match class mapping ({len(image_classes)})."
        )

    if not isinstance(soil_model.inputs, list) or len(soil_model.inputs) != 2:
        raise ValueError(
            "Soil model must have two inputs: 'numeric' and 'crop_index'."
        )

    if soil_model.output_shape[-1] != 1:
        raise ValueError("Soil model must output a single suitability score.")

    return image_model, image_classes, soil_model


@cache_data
def load_profile_data(csv_path: Optional[pathlib.Path] = None) -> Optional[pd.DataFrame]:
    """
    Loads and validates the crop recommendation dataset used for empirical quantile diagnostics.
    """
    if csv_path is None:
        csv_path = next((p for p in CSV_CANDIDATES if p.exists()), None)

    if csv_path is None or not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    required_columns = SOIL_FEATURES + ["crop"]
    if not all(c in df.columns for c in required_columns):
        return None

    for feature in SOIL_FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")

    try:
        df["crop"] = df["crop"].map(canonical_crop)
    except (ValueError, KeyError):
        return None

    return df.dropna(subset=required_columns)

"""
Data loader module for Farmer Crop Advisory System.
Handles loading of dataset profiles, model artifacts, and class labels with robust caching,
transparent diagnostics, and optimized file scanning.
"""

from __future__ import annotations

import json
import logging
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
    TEST_DIR,
)
from src.utils import canonical_crop

logger = logging.getLogger(__name__)

# Streamlit caching decorators with standalone fallback
try:
    import streamlit as st
    cache_resource = st.cache_resource
    cache_data = st.cache_data
except ImportError:
    def cache_resource(fn):
        return fn
    def cache_data(fn):
        return fn


def check_artifact_availability() -> Dict[str, Dict[str, Any]]:
    """
    Non-blocking inspection of all required disk artifacts and datasets.
    Provides structured diagnostic telemetry for UI status and error reporting.
    """
    vision_model_path = DISEASE_MODEL
    if not vision_model_path.exists():
        if IMAGE_CPU_BEST_PATH.exists():
            vision_model_path = IMAGE_CPU_BEST_PATH
        elif IMAGE_BACKUP_PATH.exists():
            vision_model_path = IMAGE_BACKUP_PATH

    csv_path = next((p for p in CSV_CANDIDATES if p.exists()), None)

    return {
        "disease_model": {
            "name": "Plant Pathology Vision Model",
            "path": str(vision_model_path),
            "exists": vision_model_path.exists(),
        },
        "disease_classes": {
            "name": "Disease Class Taxonomy",
            "path": str(DISEASE_CLASSES),
            "exists": DISEASE_CLASSES.exists(),
        },
        "soil_model": {
            "name": "Soil Suitability Neural Model",
            "path": str(SOIL_MODEL),
            "exists": SOIL_MODEL.exists(),
        },
        "soil_profiles": {
            "name": "Empirical Quantile Profiles",
            "path": str(SOIL_PROFILES),
            "exists": SOIL_PROFILES.exists(),
        },
        "dataset_csv": {
            "name": "Crop Recommendation Dataset",
            "path": str(csv_path) if csv_path else "Not found in search candidates",
            "exists": csv_path is not None and csv_path.exists(),
        },
    }


@cache_resource
def load_models(
    image_model_path: pathlib.Path = DISEASE_MODEL,
    image_classes_path: pathlib.Path = DISEASE_CLASSES,
    soil_model_path: pathlib.Path = SOIL_MODEL,
    warmup: bool = False,
) -> Tuple[Any, List[str], Any]:
    """
    Loads and validates the trained leaf disease model and soil suitability neural network.
    Gracefully falls back to backup model artifacts if primary weights are missing.
    Warmup pass is optional to prevent blocking application startup.
    """
    import tensorflow as tf

    # Verify vision model path with fallback hierarchy
    target_image_model_path = image_model_path
    if not target_image_model_path.exists():
        if IMAGE_CPU_BEST_PATH.exists():
            target_image_model_path = IMAGE_CPU_BEST_PATH
            logger.info(f"Using CPU-best vision model fallback: {IMAGE_CPU_BEST_PATH}")
        elif IMAGE_BACKUP_PATH.exists():
            target_image_model_path = IMAGE_BACKUP_PATH
            logger.info(f"Using backup vision model fallback: {IMAGE_BACKUP_PATH}")
        else:
            raise FileNotFoundError(
                f"Plant pathology vision model artifact not found at primary ({image_model_path}) "
                f"or fallback locations ({IMAGE_CPU_BEST_PATH}, {IMAGE_BACKUP_PATH})."
            )

    if not image_classes_path.exists():
        raise FileNotFoundError(
            f"Plant pathology class labels JSON file not found at: {image_classes_path}"
        )

    if not soil_model_path.exists():
        raise FileNotFoundError(
            f"Soil condition neural network artifact not found at: {soil_model_path}"
        )

    # Load neural models and labels
    try:
        image_model = tf.keras.models.load_model(target_image_model_path)
    except Exception as exc:
        raise RuntimeError(f"Failed loading vision model from {target_image_model_path}: {exc}")

    try:
        soil_model = tf.keras.models.load_model(soil_model_path)
    except Exception as exc:
        raise RuntimeError(f"Failed loading soil model from {soil_model_path}: {exc}")

    try:
        with image_classes_path.open("r", encoding="utf-8") as f:
            image_classes: List[str] = json.load(f)
    except Exception as exc:
        raise RuntimeError(f"Failed reading disease classes JSON from {image_classes_path}: {exc}")

    # Architecture validation checks
    if image_model.output_shape[-1] != len(image_classes):
        raise ValueError(
            f"Vision model output count ({image_model.output_shape[-1]}) does not match "
            f"taxonomy class count ({len(image_classes)})."
        )

    if len(soil_model.inputs) != 2:
        raise ValueError(
            f"Soil model requires 2 inputs ('numeric', 'crop_index'), but got {len(soil_model.inputs)}."
        )

    if warmup:
        try:
            import numpy as np
            _dummy_img = np.zeros((1, 224, 224, 3), dtype=np.float32)
            _ = image_model(_dummy_img, training=False)
            _dummy_soil = {
                "numeric": np.zeros((1, len(SOIL_FEATURES)), dtype=np.float32),
                "crop_index": np.zeros((1, 1), dtype=np.int32),
            }
            _ = soil_model(_dummy_soil, training=False)
        except Exception:
            pass

    return image_model, image_classes, soil_model


@cache_data
def get_sample_images(sample_dir: Optional[pathlib.Path] = None) -> List[pathlib.Path]:
    """
    Discovers test sample foliar images with caching to avoid repetitive filesystem scans on reruns.
    """
    target_dir = sample_dir or TEST_DIR
    if not target_dir or not target_dir.exists():
        return []

    discovered: List[pathlib.Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.JPG", "*.png"):
        discovered.extend(list(target_dir.rglob(ext)))

    return sorted(discovered)


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
    except Exception as exc:
        logger.warning(f"Error reading crop dataset: {exc}")
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
    except Exception as exc:
        logger.warning(f"Error loading soil profiles from {profiles_path}: {exc}")
        return {}

"""
Utility functions for crop canonicalization, hardware detection, and helper methods.
"""

from __future__ import annotations

import re

from config import SOIL_CROPS

# Canonical alias mapping for crop names
CROP_ALIASES: dict[str, str] = {
    "apple": "apple",
    "bell pepper": "bell_pepper",
    "bell_pepper": "bell_pepper",
    "cherry": "cherry",
    "corn": "maize",
    "corn maize": "maize",
    "corn (maize)": "maize",
    "maize": "maize",
    "grape": "grapes",
    "grapes": "grapes",
    "peach": "peach",
    "potato": "potato",
    "strawberry": "strawberry",
    "tomato": "tomato",
}

# Pretty display names for the UI and reports
PRETTY_CROP_NAMES: dict[str, str] = {
    "apple": "Apple",
    "bell_pepper": "Bell Pepper",
    "cherry": "Cherry",
    "grapes": "Grape",
    "maize": "Corn (Maize)",
    "peach": "Peach",
    "potato": "Potato",
    "strawberry": "Strawberry",
    "tomato": "Tomato",
}


def canonical_crop(value: str) -> str:
    """
    Normalizes arbitrary crop strings, alias variants, or PlantVillage labels
    to the system standard canonical lowercase identifier.
    """
    key = str(value).strip().lower()

    if key in CROP_ALIASES:
        return CROP_ALIASES[key]

    compact = (
        key.replace("(", "")
        .replace(")", "")
        .replace("-", " ")
        .replace("_", " ")
    )

    if compact in CROP_ALIASES:
        return CROP_ALIASES[compact]

    # Explicit handling for the PlantVillage label format
    if "corn" in compact and "maize" in compact:
        return "maize"

    raise ValueError(f"Unsupported crop name: {value!r}")


def pretty_crop(crop: str) -> str:
    """
    Returns a human-readable, capitalized display name for a canonical crop key.
    """
    return PRETTY_CROP_NAMES.get(crop, crop.replace("_", " ").title())


def setup_device() -> tuple[bool, str]:
    """
    Configures TensorFlow runtime device, enabling memory growth for available GPUs.
    Returns a tuple (gpu_active, status_message).
    """
    try:
        import tensorflow as tf
    except ImportError:
        return False, "TensorFlow is not installed in the current environment."

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        return True, f"TensorFlow GPU active: {gpus[0].name}"
    return False, "TensorFlow GPU is not available. Inference is running on CPU."

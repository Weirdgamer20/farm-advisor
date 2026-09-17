"""
Preprocessing module for Farmer Crop Advisory System.
Handles image tensor transformations and soil numeric/categorical feature encoding.
"""

import io
import pathlib
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from PIL import Image, ImageOps

from config import IMAGE_SIZE, SOIL_CROPS, SOIL_FEATURES
from src.utils import canonical_crop


def preprocess_image(
    image: Union[Image.Image, str, pathlib.Path, bytes, bytearray, Any],
    target_size: tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    """
    Transforms a PIL Image, path, byte stream, or uploaded file into a batched float32
    NumPy tensor ready for EfficientNetB0 pathology inference.
    Automatically applies EXIF transposition to correct phone camera orientation.
    """
    if isinstance(image, Image.Image):
        pil_img = image
    elif hasattr(image, "getvalue"):
        pil_img = Image.open(io.BytesIO(image.getvalue()))
    elif isinstance(image, (bytes, bytearray)):
        pil_img = Image.open(io.BytesIO(image))
    elif isinstance(image, (str, pathlib.Path)):
        pil_img = Image.open(image)
    elif hasattr(image, "read"):
        pil_img = Image.open(image)
    else:
        pil_img = Image.open(image)

    # Correct EXIF rotation (critical for photos taken directly on smartphones)
    pil_img = ImageOps.exif_transpose(pil_img)

    # Ensure 3-channel standard RGB (strips alpha channel or converts 1-channel grayscale)
    rgb_image = pil_img.convert("RGB")
    resized_image = rgb_image.resize(target_size)
    array = np.asarray(resized_image, dtype=np.float32)
    batched = np.expand_dims(array, axis=0)
    return batched


def prepare_soil_inputs(
    values: Dict[str, float],
    crop: str,
) -> Dict[str, np.ndarray]:
    """
    Formats raw soil and environmental readings along with the crop identifier
    into model-compatible NumPy tensors matching the dual-input Keras architecture.
    """
    canonical = canonical_crop(crop)

    if canonical not in SOIL_CROPS:
        raise ValueError(f"Crop '{crop}' is not supported by the soil suitability model.")

    crop_index = SOIL_CROPS.index(canonical)

    numeric_tensor = np.asarray(
        [[float(values.get(feature, 0.0)) for feature in SOIL_FEATURES]],
        dtype=np.float32,
    )

    crop_id_tensor = np.asarray(
        [[crop_index]],
        dtype=np.int32,
    )

    return {
        "numeric": numeric_tensor,
        "crop_index": crop_id_tensor,
    }


def validate_soil_readings(values: Dict[str, float]) -> Tuple[bool, List[str]]:
    """
    Validates user-provided soil parameters against agronomically feasible ranges.
    Returns (is_valid, list_of_warning_messages).
    """
    warnings: List[str] = []

    limits = {
        "N": (0.0, 500.0, "Nitrogen"),
        "P": (0.0, 500.0, "Phosphorus"),
        "K": (0.0, 500.0, "Potassium"),
        "temperature": (-15.0, 65.0, "Temperature"),
        "humidity": (0.0, 100.0, "Relative Humidity"),
        "ph": (0.0, 14.0, "Soil pH"),
        "rainfall": (0.0, 4000.0, "Annual Rainfall"),
    }

    for feature, (low, high, label) in limits.items():
        if feature not in values:
            warnings.append(f"Missing parameter: {label} ({feature})")
            continue

        val = values[feature]
        if val < low or val > high:
            warnings.append(f"{label} ({val}) is outside normal range [{low}, {high}].")

    is_valid = len(warnings) == 0
    return is_valid, warnings

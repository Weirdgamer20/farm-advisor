"""
Preprocessing module for Farmer Crop Advisory System.
Handles image tensor transformations and soil numeric/categorical feature encoding.
"""

from __future__ import annotations

from typing import Any, Dict
import numpy as np

from config import IMAGE_SIZE, SOIL_FEATURES, SOIL_CROPS
from src.utils import canonical_crop


def preprocess_image(
    image: Any,
    target_size: tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    """
    Transforms a PIL Image into a normalized, batched float32 NumPy tensor
    ready for EfficientNetB0 inference.
    """
    from PIL import Image
    rgb_image = image.convert("RGB")
    resized_image = rgb_image.resize(target_size)
    array = np.asarray(resized_image, dtype=np.float32)
    batched = np.expand_dims(array, axis=0)
    return batched


def prepare_soil_inputs(
    values: Dict[str, float],
    crop: str,
) -> Dict[str, np.ndarray]:
    """
    Formats raw soil and environmental readings along with crop identifier
    into model-compatible NumPy tensors matching the dual-input architecture.
    """
    canonical = canonical_crop(crop)

    if canonical not in SOIL_CROPS:
        raise ValueError(f"Crop '{crop}' is not supported by the soil suitability model.")

    crop_index = SOIL_CROPS.index(canonical)

    numeric_tensor = np.asarray(
        [[values[feature] for feature in SOIL_FEATURES]],
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

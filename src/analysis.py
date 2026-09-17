"""
Analysis module for Farmer Crop Advisory System.
Performs ML inference, confidence evaluation, soil diagnostics, and advisory generation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from config import (
    IMAGE_CONFIDENCE_THRESHOLD,
    STATUS_THRESHOLDS,
    SOIL_FEATURES,
)
from src.preprocessing import preprocess_image, prepare_soil_inputs
from src.utils import canonical_crop, pretty_crop


def classify_leaf(
    image: Image.Image,
    image_model: tf.keras.Model,
    image_classes: List[str],
) -> Dict[str, Any]:
    """
    Executes leaf disease inference and decomposes the predicted class
    into crop identity, pathology condition, and statistical confidence.
    """
    batch_tensor = preprocess_image(image)

    probabilities = image_model.predict(batch_tensor, verbose=0)[0]
    index = int(np.argmax(probabilities))
    confidence = float(probabilities[index])

    class_name = str(image_classes[index])
    parts = class_name.split(" - ", 1)

    if len(parts) == 2:
        detected_crop_raw, disease = parts[0], parts[1]
    else:
        detected_crop_raw, disease = class_name, "Unknown"

    canonical = canonical_crop(detected_crop_raw)

    if disease.lower() == "healthy":
        disease = "Healthy"

    return {
        "class_name": class_name,
        "crop": canonical,
        "crop_display": pretty_crop(canonical),
        "disease": disease,
        "confidence": confidence,
        "probabilities": probabilities,
        "reliable": confidence >= IMAGE_CONFIDENCE_THRESHOLD,
    }


def predict_soil(
    values: Dict[str, float],
    crop: str,
    soil_model: tf.keras.Model,
) -> Dict[str, Any]:
    """
    Evaluates soil and environmental condition suitability using the dual-input neural network.
    """
    canonical = canonical_crop(crop)
    model_inputs = prepare_soil_inputs(values, canonical)

    prediction = soil_model.predict(model_inputs, verbose=0)
    raw_score = float(np.asarray(prediction).reshape(-1)[0])
    score = float(np.clip(raw_score, 0.0, 100.0))

    if score >= STATUS_THRESHOLDS["GOOD"]:
        status = "GOOD"
    elif score >= STATUS_THRESHOLDS["ACCEPTABLE"]:
        status = "ACCEPTABLE"
    elif score >= STATUS_THRESHOLDS["NEEDS ATTENTION"]:
        status = "NEEDS ATTENTION"
    else:
        status = "NOT SUITABLE"

    return {
        "crop": canonical,
        "score": score,
        "status": status,
    }


def diagnose_soil(
    values: Dict[str, float],
    crop: str,
    profile_df: Optional[pd.DataFrame],
) -> List[Dict[str, Any]]:
    """
    Computes quantile-based diagnostic deviations comparing current farmer inputs
    against empirical distributions for the specific crop.
    """
    if profile_df is None or profile_df.empty:
        return []

    canonical = canonical_crop(crop)
    crop_df = profile_df[profile_df["crop"] == canonical]

    if crop_df.empty:
        return []

    rows: List[Dict[str, Any]] = []

    for feature in SOIL_FEATURES:
        distribution = crop_df[feature].to_numpy(dtype=np.float64)

        p10 = float(np.percentile(distribution, 10))
        p25 = float(np.percentile(distribution, 25))
        p75 = float(np.percentile(distribution, 75))
        p90 = float(np.percentile(distribution, 90))

        value = float(values[feature])

        if value < p10:
            status = "LOW"
            message = f"{feature} is critically below 10th percentile for {pretty_crop(canonical)}."
        elif value < p25:
            status = "SLIGHTLY LOW"
            message = f"{feature} is below typical range for {pretty_crop(canonical)}."
        elif value > p90:
            status = "HIGH"
            message = f"{feature} is critically above 90th percentile for {pretty_crop(canonical)}."
        elif value > p75:
            status = "SLIGHTLY HIGH"
            message = f"{feature} is above typical range for {pretty_crop(canonical)}."
        else:
            status = "NORMAL"
            message = f"{feature} is optimal within dataset's typical range for {pretty_crop(canonical)}."

        rows.append({
            "Parameter": feature,
            "Value": value,
            "Status": status,
            "Analysis": message,
            "Optimal_Range": f"{p25:.1f} - {p75:.1f}",
        })

    return rows


def generate_advisory_summary(
    leaf_result: Dict[str, Any],
    soil_result: Dict[str, Any],
    diagnostics: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Synthesizes vision results, neural suitability scores, and soil diagnostics
    into a structured agronomic advisory summary.
    """
    crop_display = leaf_result["crop_display"]
    disease = leaf_result["disease"]
    confidence_pct = leaf_result["confidence"] * 100.0

    if disease.lower() == "healthy":
        leaf_summary = f"Leaf is healthy with {confidence_pct:.1f}% confidence ({crop_display})."
    else:
        leaf_summary = f"Detected {disease} on {crop_display} with {confidence_pct:.1f}% confidence."

    deviations = [row["Analysis"] for row in diagnostics if row["Status"] != "NORMAL"]

    recommendations = []
    if disease.lower() != "healthy":
        recommendations.append(f"Isolate affected foliage to control propagation of {disease}.")
    if soil_result["status"] in ["NEEDS ATTENTION", "NOT SUITABLE"]:
        recommendations.append("Adjust fertilization or irrigation according to identified parameter deviations.")
    if not deviations and soil_result["status"] == "GOOD":
        recommendations.append("Current environmental parameters are well-balanced for cultivation.")

    return {
        "crop_display": crop_display,
        "leaf_summary": leaf_summary,
        "suitability_score": soil_result["score"],
        "suitability_status": soil_result["status"],
        "deviations": deviations,
        "recommendations": recommendations,
        "reliable": leaf_result["reliable"],
    }

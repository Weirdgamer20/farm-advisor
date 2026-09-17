"""
Analysis module for Farmer Crop Advisory System.
Performs ML inference for crop recommendation, soil suitability prediction,
leaf disease pathology identification, and comprehensive agronomic advisory synthesis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from config import (
    IMAGE_CONFIDENCE_THRESHOLD,
    SOIL_CROPS,
    SOIL_FEATURES,
    STATUS_THRESHOLDS,
)
from src.preprocessing import preprocess_image, prepare_soil_inputs
from src.utils import canonical_crop, get_disease_treatment, pretty_crop


def classify_leaf(
    image: Any,
    image_model: Any,
    image_classes: List[str],
) -> Dict[str, Any]:
    """
    Executes leaf disease inference and decomposes the predicted class
    into crop identity, pathology condition, statistical confidence, and agronomic management.
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

    canonical = canonical_crop(detected_crop_raw, strict=False) or detected_crop_raw.lower()

    if disease.lower() == "healthy":
        disease = "Healthy"

    treatment = get_disease_treatment(disease)

    return {
        "class_name": class_name,
        "crop": canonical,
        "crop_display": pretty_crop(canonical),
        "disease": disease,
        "confidence": confidence,
        "probabilities": probabilities,
        "reliable": confidence >= IMAGE_CONFIDENCE_THRESHOLD,
        "treatment": treatment,
    }


def predict_soil(
    values: Dict[str, float],
    crop: str,
    soil_model: Any,
) -> Dict[str, Any]:
    """
    Evaluates soil and environmental condition suitability for a specific crop
    using the dual-input neural network.
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
        "crop_display": pretty_crop(canonical),
        "score": score,
        "status": status,
    }


def recommend_crops(
    values: Dict[str, float],
    soil_model: Any,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Evaluates all agricultural crop profiles against the farmer's soil and environmental
    parameters, returning a ranked list of best-suited crops.
    """
    ranked_crops: List[Dict[str, Any]] = []

    for crop_key in SOIL_CROPS:
        res = predict_soil(values, crop_key, soil_model)
        ranked_crops.append(res)

    ranked_crops.sort(key=lambda item: item["score"], reverse=True)
    return ranked_crops[:top_k]


def diagnose_soil(
    values: Dict[str, float],
    crop: str,
    profile_df: Optional[pd.DataFrame] = None,
    profiles_dict: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Computes quantile-based diagnostic deviations comparing current farmer inputs
    against empirical distributions for the target crop.
    Supports either precomputed profile quantiles or raw dataset DataFrame.
    """
    canonical = canonical_crop(crop)
    rows: List[Dict[str, Any]] = []

    # Approach A: Precomputed distribution profiles
    if profiles_dict and canonical in profiles_dict:
        crop_profile = profiles_dict[canonical]
        for feature in SOIL_FEATURES:
            if feature not in crop_profile:
                continue

            stats = crop_profile[feature]
            p10 = float(stats.get("p10", 0.0))
            p25 = float(stats.get("p25", 0.0))
            p75 = float(stats.get("p75", 100.0))
            p90 = float(stats.get("p90", 100.0))

            value = float(values.get(feature, 0.0))

            if value < p10:
                status = "LOW"
                message = f"{feature} is critically below optimal range for {pretty_crop(canonical)}."
            elif value < p25:
                status = "SLIGHTLY LOW"
                message = f"{feature} is marginally low for {pretty_crop(canonical)}."
            elif value > p90:
                status = "HIGH"
                message = f"{feature} is critically above optimal threshold for {pretty_crop(canonical)}."
            elif value > p75:
                status = "SLIGHTLY HIGH"
                message = f"{feature} is slightly elevated for {pretty_crop(canonical)}."
            else:
                status = "NORMAL"
                message = f"{feature} is well-balanced within the empirical optimal range for {pretty_crop(canonical)}."

            rows.append({
                "Parameter": feature,
                "Value": value,
                "Status": status,
                "Analysis": message,
                "Optimal_Range": f"{p25:.1f} - {p75:.1f}",
            })
        return rows

    # Approach B: Compute directly from DataFrame
    if profile_df is not None and not profile_df.empty:
        crop_df = profile_df[profile_df["crop"] == canonical]
        if not crop_df.empty:
            for feature in SOIL_FEATURES:
                distribution = crop_df[feature].to_numpy(dtype=np.float64)

                p10 = float(np.percentile(distribution, 10))
                p25 = float(np.percentile(distribution, 25))
                p75 = float(np.percentile(distribution, 75))
                p90 = float(np.percentile(distribution, 90))

                value = float(values.get(feature, 0.0))

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
                    message = f"{feature} is optimal within dataset typical range for {pretty_crop(canonical)}."

                rows.append({
                    "Parameter": feature,
                    "Value": value,
                    "Status": status,
                    "Analysis": message,
                    "Optimal_Range": f"{p25:.1f} - {p75:.1f}",
                })
            return rows

    return []


def generate_advisory_summary(
    leaf_result: Optional[Dict[str, Any]] = None,
    soil_result: Optional[Dict[str, Any]] = None,
    diagnostics: Optional[List[Dict[str, Any]]] = None,
    top_crops: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Synthesizes vision results, neural suitability scores, and soil diagnostics
    into a structured agronomic advisory summary.
    """
    recommendations: List[str] = []
    deviations: List[str] = []

    crop_name = "Crop"
    if leaf_result:
        crop_name = leaf_result.get("crop_display", "Crop")
        disease = leaf_result.get("disease", "Unknown")
        if disease.lower() != "healthy":
            recommendations.append(f"Isolate affected foliage immediately to control propagation of {disease}.")
            treatment = leaf_result.get("treatment", {})
            if treatment.get("cultural"):
                recommendations.append(f"Cultural Practice: {treatment['cultural']}")
            if treatment.get("chemical"):
                recommendations.append(f"Chemical Control: {treatment['chemical']}")
    elif soil_result:
        crop_name = soil_result.get("crop_display", "Crop")

    if diagnostics:
        for diag in diagnostics:
            if diag["Status"] != "NORMAL":
                deviations.append(f"{diag['Parameter']}: {diag['Analysis']} (Current: {diag['Value']:.1f}, Optimal: {diag['Optimal_Range']})")

    if soil_result:
        status = soil_result.get("status")
        score = soil_result.get("score", 0.0)
        if status in ["NEEDS ATTENTION", "NOT SUITABLE"]:
            recommendations.append(f"Soil suitability for {crop_name} is suboptimal ({score:.1f}/100). Adjust nutrient amendments as noted in deviations.")
        elif status == "GOOD" and not deviations:
            recommendations.append(f"Field soil conditions are excellent ({score:.1f}/100) for {crop_name} cultivation.")

    if top_crops and len(top_crops) > 0:
        best_crop = top_crops[0]
        if best_crop.get("crop_display") != crop_name:
            recommendations.append(
                f"Alternative crop suggestion: {best_crop['crop_display']} achieves higher suitability ({best_crop['score']:.1f}/100) under current field conditions."
            )

    return {
        "crop_display": crop_name,
        "suitability_score": soil_result["score"] if soil_result else None,
        "suitability_status": soil_result["status"] if soil_result else None,
        "deviations": deviations,
        "recommendations": recommendations,
    }

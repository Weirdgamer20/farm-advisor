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
    Uses direct tensor call to avoid model.predict() data adapter overhead.
    """
    batch_tensor = preprocess_image(image)

    try:
        raw_preds = image_model(batch_tensor, training=False)
        probabilities = np.asarray(raw_preds)[0]
    except Exception:
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
    using the dual-input neural network with ultra-low latency direct tensor execution.
    """
    canonical = canonical_crop(crop)
    model_inputs = prepare_soil_inputs(values, canonical)

    try:
        prediction = soil_model(model_inputs, training=False)
    except Exception:
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
    parameters in a SINGLE vectorized batch pass for near-instantaneous response time (<10ms).
    """
    num_crops = len(SOIL_CROPS)
    numeric_row = [float(values.get(feature, 0.0)) for feature in SOIL_FEATURES]
    numeric_batch = np.tile(numeric_row, (num_crops, 1)).astype(np.float32)
    crop_indices = np.arange(num_crops, dtype=np.int32).reshape(-1, 1)

    batch_inputs = {
        "numeric": numeric_batch,
        "crop_index": crop_indices,
    }

    try:
        raw_predictions = soil_model(batch_inputs, training=False)
    except Exception:
        raw_predictions = soil_model.predict(batch_inputs, verbose=0)

    scores = np.clip(np.asarray(raw_predictions).reshape(-1), 0.0, 100.0)

    ranked_crops: List[Dict[str, Any]] = []
    for i, crop_key in enumerate(SOIL_CROPS):
        score = float(scores[i])
        if score >= STATUS_THRESHOLDS["GOOD"]:
            status = "GOOD"
        elif score >= STATUS_THRESHOLDS["ACCEPTABLE"]:
            status = "ACCEPTABLE"
        elif score >= STATUS_THRESHOLDS["NEEDS ATTENTION"]:
            status = "NEEDS ATTENTION"
        else:
            status = "NOT SUITABLE"

        ranked_crops.append({
            "crop": crop_key,
            "crop_display": pretty_crop(crop_key),
            "score": score,
            "status": status,
        })

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
    values: Optional[Dict[str, float]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Synthesizes vision results, neural suitability scores, soil diagnostics,
    and environmental parameters into a structured, holistic agronomic advisory.
    Identifies why diseases occurred based on soil/climate conditions and provides
    exact target concentration adjustments for the crop.
    """
    recommendations: List[str] = []
    deviations: List[str] = []
    disease_causes: List[str] = []
    prescriptions: List[Dict[str, str]] = []

    crop_name = "Crop"
    disease = "Healthy"
    if leaf_result:
        crop_name = leaf_result.get("crop_display", "Crop")
        disease = leaf_result.get("disease", "Unknown")
        if disease.lower() != "healthy":
            recommendations.append(f"Isolate affected foliage immediately to control propagation of {disease}.")
            treatment = leaf_result.get("treatment", {})
            if treatment.get("cultural"):
                recommendations.append(f"Cultural Sanitation: {treatment['cultural']}")
            if treatment.get("chemical"):
                recommendations.append(f"Targeted Spray / Treatment: {treatment['chemical']}")
    elif soil_result:
        crop_name = soil_result.get("crop_display", "Crop")

    # Analyze root causes connecting disease to soil and environmental readings
    if values and disease.lower() != "healthy" and disease.lower() != "unknown":
        humidity = float(values.get("humidity", 0.0))
        rainfall = float(values.get("rainfall", 0.0))
        k_val = float(values.get("K", 0.0))
        n_val = float(values.get("N", 0.0))
        ph_val = float(values.get("ph", 6.5))

        # Fungal disease correlations
        if any(term in disease.lower() for term in ["blight", "rust", "spot", "mildew", "scab", "rot", "scorch", "measles"]):
            if humidity >= 70.0:
                disease_causes.append(
                    f"High relative humidity ({humidity:.1f}%) creates extended leaf wetness, directly fostering fungal spore germination for {disease}."
                )
            if rainfall >= 120.0:
                disease_causes.append(
                    f"Elevated rainfall ({rainfall:.1f} mm) causes soil splashing that transfers fungal spores from the soil onto lower foliage."
                )
            if k_val < 35.0:
                disease_causes.append(
                    f"Low soil potassium ({k_val:.1f} mg/kg) weakens plant epidermal cell walls, reducing natural mechanical defense against pathogen penetration."
                )
            if n_val > 100.0:
                disease_causes.append(
                    f"Excessive nitrogen ({n_val:.1f} mg/kg) triggers rapid, succulent vegetative growth with thin cuticle layers that fungi easily penetrate."
                )
        # Bacterial disease correlations
        elif "bacterial" in disease.lower():
            if humidity >= 70.0:
                disease_causes.append(
                    f"Warm and humid microclimate ({humidity:.1f}% RH) allows bacteria to enter leaf stomata and hydathodes rapidly."
                )
        # Viral disease correlations
        elif "virus" in disease.lower() or "curl" in disease.lower():
            disease_causes.append(
                f"{disease} is vector-borne (typically whiteflies or aphids); environmental stress and nutrient imbalance reduce crop tolerance to viral symptoms."
            )

        if ph_val < 5.8:
            disease_causes.append(
                f"Acidic soil (pH {ph_val:.1f}) restricts phosphorus and calcium bioavailability, limiting root vigor and systemic plant immunity."
            )
        elif ph_val > 7.5:
            disease_causes.append(
                f"Alkaline soil (pH {ph_val:.1f}) locks out essential micronutrients (iron, manganese, zinc), impairing cellular repair."
            )

    # Process diagnostic deviations and build exact soil prescriptions
    if diagnostics:
        for diag in diagnostics:
            param = diag["Parameter"]
            val = float(diag["Value"])
            opt = diag["Optimal_Range"]
            status = diag["Status"]

            if status != "NORMAL":
                deviations.append(f"{param}: {diag['Analysis']} (Current: {val:.1f}, Optimal: {opt})")

                # Generate targeted agronomic prescription for this parameter
                action = ""
                if param == "N":
                    action = f"Apply urea (46-0-0) or composted manure to raise Nitrogen into optimal {opt} mg/kg." if "LOW" in status else "Halt nitrogen fertilization to prevent excessive vegetative tenderness."
                elif param == "P":
                    action = f"Incorporate diammonium phosphate (DAP) or rock phosphate to reach optimal {opt} mg/kg for root development." if "LOW" in status else "Avoid phosphorus additives to prevent micronutrient lockout."
                elif param == "K":
                    action = f"Apply Muriate of Potash (MOP / 0-0-60) to attain optimal {opt} mg/kg; improves foliar cell wall resilience against pathogens." if "LOW" in status else "Maintain balanced irrigation; avoid potassium over-amendment."
                elif param == "ph":
                    action = f"Broadcast agricultural limestone or dolomite to raise soil pH toward optimal {opt}." if "LOW" in status else f"Incorporate agricultural gypsum or elemental sulfur to bring pH down toward optimal {opt}."
                elif param == "humidity":
                    action = "Improve plant spacing, prune dense lower foliage, and switch to drip irrigation to lower canopy humidity." if "HIGH" in status else "Consider light micro-sprinkling if low humidity induces transpiration stress."
                elif param == "rainfall":
                    action = "Ensure raised beds and adequate trench drainage to prevent waterlogging around roots." if "HIGH" in status else "Supplement with scheduled drip irrigation to avoid moisture deficit stress."
                elif param == "temperature":
                    action = "Utilize shade netting or mulch to moderate root zone temperature." if "HIGH" in status else "Apply straw mulching to conserve root zone heat."

                prescriptions.append({
                    "parameter": param,
                    "current": f"{val:.1f}",
                    "optimal": opt,
                    "status": status,
                    "action": action,
                })

    if soil_result:
        status = soil_result.get("status")
        score = soil_result.get("score", 0.0)
        if status in ["NEEDS ATTENTION", "NOT SUITABLE"]:
            recommendations.append(f"Soil suitability for {crop_name} is suboptimal ({score:.1f}/100). Implement the nutrient amendments in the prescription below.")
        elif status == "GOOD" and not deviations:
            recommendations.append(f"Field soil conditions are well-aligned ({score:.1f}/100) for {crop_name} productivity.")

    if top_crops and len(top_crops) > 0:
        best_crop = top_crops[0]
        if best_crop.get("crop_display") != crop_name:
            recommendations.append(
                f"Crop rotation insight: {best_crop['crop_display']} achieves higher natural suitability ({best_crop['score']:.1f}/100) under your current soil conditions."
            )

    return {
        "crop_display": crop_name,
        "disease": disease,
        "suitability_score": soil_result["score"] if soil_result else None,
        "suitability_status": soil_result["status"] if soil_result else None,
        "disease_causes": disease_causes,
        "prescriptions": prescriptions,
        "deviations": deviations,
        "recommendations": recommendations,
    }

"""
Analysis module for Farmer Crop Advisory System.
Implements plant leaf disease inference, soil suitability prediction, empirical quantile
diagnostics, and automated agronomic advisory synthesis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from config import IMAGE_CONFIDENCE_THRESHOLD, SOIL_CROPS, SOIL_FEATURES
from src.preprocessing import prepare_soil_inputs, preprocess_image
from src.utils import canonical_crop, get_disease_treatment, pretty_crop


def score_to_status(score: float) -> str:
    """Maps a numeric suitability score (0-100) to standard agronomic status."""
    if score >= 80.0:
        return "GOOD"
    if score >= 60.0:
        return "ACCEPTABLE"
    if score >= 40.0:
        return "NEEDS ATTENTION"
    return "NOT SUITABLE"


def classify_leaf(
    model: Any,
    image: Any,
    classes: List[str],
    confidence_threshold: float = IMAGE_CONFIDENCE_THRESHOLD,
) -> Dict[str, Any]:
    """
    Executes vision inference on leaf imagery, identifies plant pathology,
    and returns top candidate classes, confidence, and treatment guidance.
    """
    tensor = preprocess_image(image)
    preds = model.predict(tensor, verbose=0)[0]

    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx])
    raw_label = classes[top_idx]

    if "___" in raw_label:
        raw_crop, raw_disease = raw_label.split("___", 1)
    else:
        raw_crop, raw_disease = "unknown", raw_label

    disease_name = raw_disease.replace("_", " ").strip()
    detected_crop = canonical_crop(raw_crop, strict=False) or raw_crop.lower()

    top_indices = np.argsort(preds)[::-1][:3]
    top_candidates = [
        {
            "class_name": classes[i],
            "confidence": float(preds[i]),
            "percentage": f"{preds[i] * 100:.1f}%",
        }
        for i in top_indices
    ]

    return {
        "class_name": raw_label,
        "crop": detected_crop,
        "crop_display": pretty_crop(detected_crop),
        "disease": disease_name,
        "confidence": confidence,
        "reliable": confidence >= confidence_threshold,
        "top_candidates": top_candidates,
        "treatment": get_disease_treatment(disease_name),
    }


def predict_soil(
    soil_model: Any,
    readings: Dict[str, float],
    crop: str,
) -> Dict[str, Any]:
    """
    Runs neural inference for soil compatibility score of a single target crop.
    """
    canonical = canonical_crop(crop, strict=True)
    inputs = prepare_soil_inputs(readings, canonical)

    try:
        raw_score = soil_model(inputs, training=False)
    except Exception:
        raw_score = soil_model.predict(inputs, verbose=0)

    score = float(np.clip(np.asarray(raw_score).reshape(-1)[0], 0.0, 100.0))

    return {
        "crop": canonical,
        "crop_display": pretty_crop(canonical),
        "score": score,
        "status": score_to_status(score),
    }


def recommend_crops(
    soil_model: Any,
    readings: Dict[str, float],
) -> List[Dict[str, Any]]:
    """
    Simultaneously evaluates all 9 supported crops using a single vectorized batch pass
    for sub-10ms response latency, returning crops ranked by suitability.
    """
    num_crops = len(SOIL_CROPS)
    numeric_row = [float(readings.get(f, 0.0)) for f in SOIL_FEATURES]
    batch_inputs = {
        "numeric": np.tile(numeric_row, (num_crops, 1)).astype(np.float32),
        "crop_index": np.arange(num_crops, dtype=np.int32).reshape(-1, 1),
    }

    try:
        preds = soil_model(batch_inputs, training=False)
    except Exception:
        preds = soil_model.predict(batch_inputs, verbose=0)

    scores = np.clip(np.asarray(preds).reshape(-1), 0.0, 100.0)

    ranked = [
        {
            "crop": crop_key,
            "crop_display": pretty_crop(crop_key),
            "score": float(scores[i]),
            "status": score_to_status(float(scores[i])),
        }
        for i, crop_key in enumerate(SOIL_CROPS)
    ]
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


def diagnose_soil(
    readings: Dict[str, float],
    crop: str,
    profiles_dict: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Compares soil and climate readings against precomputed empirical distribution quantiles
    (p10, p25, p75, p90) for the target crop.
    """
    canonical = canonical_crop(crop, strict=False) or crop
    if not profiles_dict or canonical not in profiles_dict:
        return []

    crop_profile = profiles_dict[canonical]
    rows: List[Dict[str, Any]] = []

    for f in SOIL_FEATURES:
        stats = crop_profile.get(f, {})
        p10 = float(stats.get("p10", 0.0))
        p25 = float(stats.get("p25", 0.0))
        p75 = float(stats.get("p75", 100.0))
        p90 = float(stats.get("p90", 100.0))
        val = float(readings.get(f, 0.0))

        if val < p10:
            status, message = "LOW", f"{f} is critically low (below 10th percentile)."
        elif val < p25:
            status, message = "SLIGHTLY LOW", f"{f} is slightly low (below 25th percentile)."
        elif val > p90:
            status, message = "HIGH", f"{f} is critically high (above 90th percentile)."
        elif val > p75:
            status, message = "SLIGHTLY HIGH", f"{f} is slightly high (above 75th percentile)."
        else:
            status, message = "NORMAL", f"{f} is within optimal range."

        rows.append({
            "Parameter": f,
            "Value": val,
            "Status": status,
            "Analysis": message,
            "Optimal_Range": f"{p25:.1f} - {p75:.1f}",
        })

    return rows


def _get_prescriptive_action(param: str, status: str, opt: str) -> str:
    """Returns tailored agronomic remediation guidance for parameter deviations."""
    is_low = "LOW" in status
    actions = {
        "N": f"Apply urea (46-0-0) or compost to raise Nitrogen toward optimal {opt} mg/kg." if is_low else "Halt nitrogen fertilization to prevent excessive vegetative tenderness.",
        "P": f"Incorporate DAP or rock phosphate to reach optimal {opt} mg/kg for root development." if is_low else "Avoid phosphorus additives to prevent micronutrient lockout.",
        "K": f"Apply Muriate of Potash (MOP) to reach optimal {opt} mg/kg; improves cell wall resilience." if is_low else "Maintain balanced irrigation; avoid potassium over-amendment.",
        "ph": f"Broadcast agricultural lime or dolomite to raise soil pH toward optimal {opt}." if is_low else f"Incorporate agricultural gypsum or elemental sulfur to lower pH toward optimal {opt}.",
        "humidity": "Maintain scheduled irrigation." if is_low else "Improve plant spacing and switch to drip irrigation to reduce canopy humidity.",
        "rainfall": "Supplement with drip irrigation to avoid moisture deficit stress." if is_low else "Ensure raised beds and trench drainage to prevent waterlogging.",
        "temperature": "Apply straw mulching to conserve root zone heat." if is_low else "Utilize shade netting or mulch to moderate root zone temperature.",
    }
    return actions.get(param, "")


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
    and environmental readings into an agronomic advisory.
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
                recommendations.append(f"Targeted Spray: {treatment['chemical']}")
    elif soil_result:
        crop_name = soil_result.get("crop_display", "Crop")

    # Correlate disease etiology with environmental stressors
    if values and disease.lower() not in ["healthy", "unknown"]:
        humidity = float(values.get("humidity", 0.0))
        rainfall = float(values.get("rainfall", 0.0))
        k_val = float(values.get("K", 0.0))
        n_val = float(values.get("N", 0.0))
        ph_val = float(values.get("ph", 6.5))

        if any(term in disease.lower() for term in ["blight", "rust", "spot", "mildew", "scab", "rot", "scorch", "measles"]):
            if humidity >= 70.0:
                disease_causes.append(f"High relative humidity ({humidity:.1f}%) creates extended leaf wetness, fostering spore germination for {disease}.")
            if rainfall >= 120.0:
                disease_causes.append(f"Elevated rainfall ({rainfall:.1f} mm) causes soil splashing that transfers spores onto lower foliage.")
            if k_val < 35.0:
                disease_causes.append(f"Low soil potassium ({k_val:.1f} mg/kg) weakens epidermal cell walls, reducing natural defense against pathogen entry.")
            if n_val > 100.0:
                disease_causes.append(f"Excessive nitrogen ({n_val:.1f} mg/kg) triggers rapid, succulent vegetative growth with thin cuticle layers vulnerable to fungal attack.")
        elif "bacterial" in disease.lower():
            if humidity >= 70.0:
                disease_causes.append(f"Warm, humid microclimate ({humidity:.1f}% RH) allows bacteria to enter foliar stomata rapidly.")
        elif "virus" in disease.lower() or "curl" in disease.lower():
            disease_causes.append(f"{disease} is vector-borne; environmental stress and nutrient imbalance reduce crop tolerance to viral symptoms.")

        if ph_val < 5.8:
            disease_causes.append(f"Acidic soil (pH {ph_val:.1f}) restricts nutrient bioavailability, lowering overall plant immunity.")
        elif ph_val > 7.5:
            disease_causes.append(f"Alkaline soil (pH {ph_val:.1f}) locks out essential micronutrients (iron, manganese, zinc).")

    # Diagnostic deviations and prescriptions
    if diagnostics:
        for diag in diagnostics:
            param, val, opt, status = diag["Parameter"], float(diag["Value"]), diag["Optimal_Range"], diag["Status"]
            if status != "NORMAL":
                deviations.append(f"{param}: {diag['Analysis']} (Current: {val:.1f}, Optimal: {opt})")
                prescriptions.append({
                    "parameter": param,
                    "current": f"{val:.1f}",
                    "optimal": opt,
                    "status": status,
                    "action": _get_prescriptive_action(param, status, opt),
                })

    if soil_result:
        status, score = soil_result.get("status"), soil_result.get("score", 0.0)
        if status in ["NEEDS ATTENTION", "NOT SUITABLE"]:
            recommendations.append(f"Soil suitability for {crop_name} is suboptimal ({score:.1f}/100). Implement the nutrient amendments below.")
        elif status == "GOOD" and not deviations:
            recommendations.append(f"Field soil conditions are well-aligned ({score:.1f}/100) for {crop_name} productivity.")

    if top_crops and len(top_crops) > 0 and top_crops[0].get("crop_display") != crop_name:
        recommendations.append(
            f"Crop rotation insight: {top_crops[0]['crop_display']} achieves higher natural suitability ({top_crops[0]['score']:.1f}/100) under your current soil conditions."
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

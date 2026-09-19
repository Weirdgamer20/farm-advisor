"""
Farmer Crop Advisory Engine — Unified High-Performance Core
Consolidates neural model loading, hardware checks, vision pathology inference,
soil condition evaluation, agronomic advisory generation, and chart plotting.
"""

from __future__ import annotations

import io
import json
import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

try:
    import streamlit as st
    cache_resource = st.cache_resource
    cache_data = st.cache_data
except ImportError:
    def cache_resource(fn): return fn
    def cache_data(fn): return fn

# ============================================================
# CONSTANTS, PATHS & CROPS
# ============================================================

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
TEST_DIR = DATA_DIR / "raw" / "Plant Village Dataset" / "Test"

DISEASE_MODEL_PATH = MODEL_DIR / "plant_disease_model.keras"
SOIL_MODEL_PATH = MODEL_DIR / "soil_condition_model.keras"
DISEASE_CLASSES_PATH = MODEL_DIR / "plant_disease_classes.json"
SOIL_PROFILES_PATH = MODEL_DIR / "soil_condition_profiles.json"

SOIL_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
SOIL_CROPS = ["apple", "bell_pepper", "cherry", "grapes", "maize", "peach", "potato", "strawberry", "tomato"]

STATUS_THRESHOLDS = {
    "GOOD": 80.0,
    "ACCEPTABLE": 60.0,
    "NEEDS ATTENTION": 40.0,
}

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

DISEASE_TREATMENTS: dict[str, dict[str, str]] = {
    "Apple Scab": {
        "symptoms": "Olive-green to brown velvety spots on leaves, becoming dark and corky.",
        "cultural": "Prune trees to enhance air circulation; rake and destroy fallen leaves in autumn.",
        "chemical": "Apply preventative fungicides (Captan, Mancozeb, or copper-based sprays) at green-tip stage.",
    },
    "Black Rot": {
        "symptoms": "Circular brown leaf spots with dark borders (frog-eye spots); rotting fruit.",
        "cultural": "Remove mummified fruit, dead wood, and prunings; prune canopy for rapid drying.",
        "chemical": "Apply Captan or Myclobutanil fungicides starting at petal fall through cover sprays.",
    },
    "Cedar Apple Rust": {
        "symptoms": "Bright orange or yellow spots on the upper leaf surface.",
        "cultural": "Remove nearby eastern red cedar trees if practical; select rust-resistant cultivars.",
        "chemical": "Apply Myclobutanil or Mancozeb fungicide sprays at pink-bud through petal fall.",
    },
    "Bacterial Spot": {
        "symptoms": "Small, water-soaked, dark spots on leaves that turn brown/black with yellow halos.",
        "cultural": "Avoid overhead irrigation; practice 2-3 year crop rotation with non-solanaceous crops.",
        "chemical": "Apply preventative copper sprays mixed with Mancozeb; use bactericides where permitted.",
    },
    "Powdery Mildew": {
        "symptoms": "White to light gray powdery fungal patches on leaves and tender shoots.",
        "cultural": "Ensure adequate plant spacing; avoid excessive nitrogen fertilization.",
        "chemical": "Spray potassium bicarbonate, horticultural oil, or sulfur-based fungicides at first symptom.",
    },
    "Cercospora Leaf Spot": {
        "symptoms": "Small, circular spots with tan to gray centers and purple/brown margins.",
        "cultural": "Bury crop residues after harvest; rotate fields away from susceptible crops.",
        "chemical": "Apply strobilurin or triazole fungicides during early vegetative stages if pressure is high.",
    },
    "Common Rust": {
        "symptoms": "Small, powdery, cinnamon-brown to dark pustules on both upper and lower leaf surfaces.",
        "cultural": "Plant resistant hybrid seed varieties; practice early planting to escape peak spore loads.",
        "chemical": "Apply fungicides containing Azoxystrobin or Pyraclostrobin if pustules appear before tasseling.",
    },
    "Northern Leaf Blight": {
        "symptoms": "Long, elliptical grayish-green or tan lesions (cigar-shaped) on foliage.",
        "cultural": "Turn under crop residue promptly after harvest; implement minimum 1-2 year non-host crop rotation.",
        "chemical": "Apply strobilurin or DMI triazole fungicides at first sign of disease on lower leaves.",
    },
    "Esca": {
        "symptoms": "Tiger-stripe foliar chlorosis and necrosis; dark spots on berries; wood necrosis.",
        "cultural": "Protect pruning wounds with wound sealants; prune during dry weather; remove infected vine sections.",
        "chemical": "No curative chemical exists; treat pruning wounds with Trichoderma or authorized biocides.",
    },
    "Early Blight": {
        "symptoms": "Concentric dark brown rings with target-like appearance surrounded by yellow chlorotic halos.",
        "cultural": "Mulch soil around plants; avoid overhead watering; prune lower leaves touching the soil.",
        "chemical": "Apply copper hydroxide, Chlorothalonil, or Azoxystrobin on a 7-10 day schedule.",
    },
    "Late Blight": {
        "symptoms": "Water-soaked dark lesions with pale margins, often showing white fuzzy mold on leaf undersides.",
        "cultural": "Destroy volunteer potato/tomato plants; ensure excellent field drainage; avoid wet foliage.",
        "chemical": "Apply systemic fungicides (Mefenoxam, Cyazofamid, or Mancozeb) immediately upon first sighting.",
    },
    "Septoria Leaf Spot": {
        "symptoms": "Numerous circular brown spots with dark borders and small black dots in center.",
        "cultural": "Sterilize stakes and cages; avoid working among wet plants; clean up debris.",
        "chemical": "Apply Chlorothalonil or Copper fungicides preventatively after fruit set.",
    },
    "Yellow Leaf Curl Virus": {
        "symptoms": "Severe stunting, upward leaf curling, yellowing leaf margins, aborted flowers.",
        "cultural": "Control silverleaf whitefly vectors using insecticidal soap or yellow sticky traps; plant virus-resistant varieties.",
        "chemical": "Apply systemic insecticides targeting whitefly nymphs if economic thresholds are exceeded.",
    },
    "Leaf Scorch": {
        "symptoms": "Purplish-red spots that enlarge and dry into brown scorched patches on strawberry foliage.",
        "cultural": "Renovate beds after harvest; avoid overhead irrigation; improve soil aeration.",
        "chemical": "Apply Captan or Thiophanate-methyl during bloom and early runner production.",
    },
}

def canonical_crop(value: str) -> str:
    key = str(value).strip().lower().replace("(", "").replace(")", "").replace("-", " ").replace("_", " ")
    if "corn" in key and "maize" in key:
        return "maize"
    return CROP_ALIASES.get(key, key)

def pretty_crop(crop: str) -> str:
    return PRETTY_CROP_NAMES.get(crop, crop.replace("_", " ").title())

def get_disease_treatment(disease: str) -> Dict[str, str]:
    for key, treatment in DISEASE_TREATMENTS.items():
        if key.lower() in disease.lower() or disease.lower() in key.lower():
            return treatment
    return {
        "symptoms": "Pathogen symptoms observed on foliar tissue.",
        "cultural": "Ensure good crop hygiene, remove diseased tissue, and maintain balanced soil moisture.",
        "chemical": "Consult local agricultural extension service for registered preventative sprays.",
    }

# ============================================================
# HARDWARE & MODEL INITIALIZATION
# ============================================================

_CACHED_DEVICE_STATUS: Optional[Tuple[bool, str]] = None

@cache_resource
def setup_device() -> Tuple[bool, str]:
    global _CACHED_DEVICE_STATUS
    if _CACHED_DEVICE_STATUS is not None:
        return _CACHED_DEVICE_STATUS
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                try: tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError: pass
            _CACHED_DEVICE_STATUS = (True, f"TensorFlow GPU active: {gpus[0].name}")
        else:
            _CACHED_DEVICE_STATUS = (False, "Running on CPU (TensorFlow GPU not detected).")
    except Exception as exc:
        _CACHED_DEVICE_STATUS = (False, f"Device notice: {exc}")
    return _CACHED_DEVICE_STATUS

@cache_resource
def load_artifacts() -> Tuple[Any, List[str], Any, Dict[str, Any]]:
    import tensorflow as tf
    if not DISEASE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model missing: {DISEASE_MODEL_PATH}")
    if not SOIL_MODEL_PATH.exists():
        raise FileNotFoundError(f"Soil model missing: {SOIL_MODEL_PATH}")

    image_model = tf.keras.models.load_model(DISEASE_MODEL_PATH)
    soil_model = tf.keras.models.load_model(SOIL_MODEL_PATH)

    with DISEASE_CLASSES_PATH.open("r", encoding="utf-8") as f:
        image_classes = json.load(f)
    with SOIL_PROFILES_PATH.open("r", encoding="utf-8") as f:
        soil_profiles = json.load(f)

    # Warm-up pass to eliminate first-inference lag
    try:
        dummy_img = np.zeros((1, 224, 224, 3), dtype=np.float32)
        _ = image_model(dummy_img, training=False)
        dummy_soil = {
            "numeric": np.zeros((1, len(SOIL_FEATURES)), dtype=np.float32),
            "crop_index": np.zeros((1, 1), dtype=np.int32),
        }
        _ = soil_model(dummy_soil, training=False)
    except Exception:
        pass

    return image_model, image_classes, soil_model, soil_profiles

# ============================================================
# VISION INFERENCE & CROP AUTO-DETECTION
# ============================================================

def preprocess_image(image: Union[Image.Image, bytes, bytearray, Any]) -> np.ndarray:
    if isinstance(image, Image.Image):
        pil_img = image
    elif hasattr(image, "getvalue"):
        pil_img = Image.open(io.BytesIO(image.getvalue()))
    elif isinstance(image, (bytes, bytearray)):
        pil_img = Image.open(io.BytesIO(image))
    else:
        pil_img = Image.open(image)

    pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
    resized = pil_img.resize((224, 224))
    return np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)

def classify_leaf(image: Any, image_model: Any, image_classes: List[str]) -> Dict[str, Any]:
    tensor = preprocess_image(image)
    try:
        preds = image_model(tensor, training=False)
        probabilities = np.asarray(preds)[0]
    except Exception:
        probabilities = image_model.predict(tensor, verbose=0)[0]

    index = int(np.argmax(probabilities))
    confidence = float(probabilities[index])
    class_name = str(image_classes[index])

    parts = class_name.split(" - ", 1)
    detected_crop_raw = parts[0] if len(parts) == 2 else class_name
    disease = parts[1] if len(parts) == 2 else "Unknown"

    canonical = canonical_crop(detected_crop_raw)
    if disease.lower() == "healthy":
        disease = "Healthy"

    return {
        "class_name": class_name,
        "crop": canonical,
        "crop_display": pretty_crop(canonical),
        "disease": disease,
        "confidence": confidence,
        "treatment": get_disease_treatment(disease),
    }

# ============================================================
# SOIL EVALUATION & VECTORIZED CROP RECOMMENDATION
# ============================================================

def evaluate_soil_and_crops(
    readings: Dict[str, float],
    target_crop: str,
    soil_model: Any,
    profiles: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Evaluates target crop suitability and simultaneously ranks all 9 crops
    in a single vectorized batch pass for near-instant latency.
    """
    canonical_target = canonical_crop(target_crop)
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

    ranked_crops: List[Dict[str, Any]] = []
    target_score = 50.0
    for i, c_key in enumerate(SOIL_CROPS):
        s = float(scores[i])
        status = "GOOD" if s >= 80 else ("ACCEPTABLE" if s >= 60 else ("NEEDS ATTENTION" if s >= 40 else "NOT SUITABLE"))
        if c_key == canonical_target:
            target_score = s
        ranked_crops.append({
            "crop": c_key,
            "crop_display": pretty_crop(c_key),
            "score": s,
            "status": status,
        })
    ranked_crops.sort(key=lambda x: x["score"], reverse=True)

    target_status = "GOOD" if target_score >= 80 else ("ACCEPTABLE" if target_score >= 60 else ("NEEDS ATTENTION" if target_score >= 40 else "NOT SUITABLE"))
    target_result = {
        "crop": canonical_target,
        "crop_display": pretty_crop(canonical_target),
        "score": target_score,
        "status": target_status,
    }

    # Diagnostics using precomputed empirical quantiles
    diagnostics: List[Dict[str, Any]] = []
    crop_profile = profiles.get(canonical_target, {})
    for f in SOIL_FEATURES:
        stats = crop_profile.get(f, {})
        p10, p25 = float(stats.get("p10", 0.0)), float(stats.get("p25", 0.0))
        p75, p90 = float(stats.get("p75", 100.0)), float(stats.get("p90", 100.0))
        val = float(readings.get(f, 0.0))

        if val < p10: status, msg = "LOW", f"{f} is critically low."
        elif val < p25: status, msg = "SLIGHTLY LOW", f"{f} is slightly low."
        elif val > p90: status, msg = "HIGH", f"{f} is critically high."
        elif val > p75: status, msg = "SLIGHTLY HIGH", f"{f} is slightly high."
        else: status, msg = "NORMAL", f"{f} is within optimal range."

        diagnostics.append({
            "Parameter": f,
            "Value": val,
            "Status": status,
            "Analysis": msg,
            "Optimal_Range": f"{p25:.1f} - {p75:.1f}",
        })

    return target_result, ranked_crops, diagnostics

# ============================================================
# ADVISORY PRESCRIPTIONS SYNTHESIS
# ============================================================

def generate_advisory(
    leaf_res: Dict[str, Any],
    soil_res: Dict[str, Any],
    diagnostics: List[Dict[str, Any]],
    top_crops: List[Dict[str, Any]],
    readings: Dict[str, float],
) -> Dict[str, Any]:
    crop_name = leaf_res.get("crop_display", "Crop")
    disease = leaf_res.get("disease", "Unknown")
    is_healthy = disease.lower() == "healthy"

    disease_causes: List[str] = []
    prescriptions: List[Dict[str, str]] = []
    recommendations: List[str] = []

    if not is_healthy:
        treatment = leaf_res.get("treatment", {})
        if treatment.get("cultural"): recommendations.append(f"Cultural Sanitation: {treatment['cultural']}")
        if treatment.get("chemical"): recommendations.append(f"Targeted Spray: {treatment['chemical']}")

        humidity = float(readings.get("humidity", 0.0))
        rainfall = float(readings.get("rainfall", 0.0))
        k_val = float(readings.get("K", 0.0))
        n_val = float(readings.get("N", 0.0))
        ph_val = float(readings.get("ph", 6.5))

        if any(term in disease.lower() for term in ["blight", "rust", "spot", "mildew", "scab", "rot", "scorch"]):
            if humidity >= 70.0: disease_causes.append(f"High relative humidity ({humidity:.1f}%) directly fosters fungal spore germination.")
            if rainfall >= 120.0: disease_causes.append(f"Rainfall splash ({rainfall:.1f} mm) transfers fungal spores from soil onto lower foliage.")
            if k_val < 35.0: disease_causes.append(f"Low potassium ({k_val:.1f} mg/kg) weakens epidermal cell walls against pathogen penetration.")
            if n_val > 100.0: disease_causes.append(f"Excess nitrogen ({n_val:.1f} mg/kg) causes thin, tender cuticles vulnerable to infection.")
        elif "bacterial" in disease.lower():
            if humidity >= 70.0: disease_causes.append(f"Humid microclimate ({humidity:.1f}%) enables bacteria to enter leaf stomata.")
        elif "virus" in disease.lower() or "curl" in disease.lower():
            disease_causes.append(f"{disease} is vector-borne; soil and climate stress decrease crop tolerance.")

        if ph_val < 5.8: disease_causes.append(f"Acidic soil (pH {ph_val:.1f}) restricts nutrient bioavailability, lowering immunity.")
        elif ph_val > 7.5: disease_causes.append(f"Alkaline soil (pH {ph_val:.1f}) locks out essential micronutrients (iron, zinc).")

    for diag in diagnostics:
        param, val, opt, status = diag["Parameter"], diag["Value"], diag["Optimal_Range"], diag["Status"]
        if status != "NORMAL":
            action = ""
            if param == "N": action = f"Apply urea/compost to reach {opt} mg/kg." if "LOW" in status else "Halt nitrogen fertilization."
            elif param == "P": action = f"Incorporate DAP/rock phosphate to reach {opt} mg/kg." if "LOW" in status else "Avoid phosphorus additives."
            elif param == "K": action = f"Apply Muriate of Potash (MOP) to reach {opt} mg/kg." if "LOW" in status else "Maintain balanced irrigation."
            elif param == "ph": action = f"Broadcast lime/dolomite to raise pH to {opt}." if "LOW" in status else f"Apply sulfur/gypsum to lower pH to {opt}."
            elif param == "humidity": action = "Improve plant spacing & switch to drip irrigation." if "HIGH" in status else "Maintain adequate irrigation."
            elif param == "rainfall": action = "Ensure raised beds and trench drainage." if "HIGH" in status else "Supplement with drip irrigation."
            elif param == "temperature": action = "Use shade netting or mulch to moderate heat." if "HIGH" in status else "Apply straw mulching."

            prescriptions.append({
                "Parameter": param,
                "Your Reading": f"{val:.1f}",
                "Target Optimal Range": opt,
                "Status": status,
                "Corrective Action Required": action,
            })

    if soil_res.get("status") in ["NEEDS ATTENTION", "NOT SUITABLE"]:
        recommendations.append(f"Soil suitability for {crop_name} is suboptimal ({soil_res['score']:.1f}/100). Apply the corrective amendments below.")
    else:
        recommendations.append(f"Field soil environment is well-aligned ({soil_res['score']:.1f}/100) for {crop_name} productivity.")

    if top_crops and top_crops[0]["crop_display"] != crop_name:
        recommendations.append(f"Crop rotation insight: {top_crops[0]['crop_display']} achieves higher natural suitability ({top_crops[0]['score']:.1f}/100) on this soil.")

    return {
        "crop_display": crop_name,
        "disease": disease,
        "score": soil_res["score"],
        "status": soil_res["status"],
        "disease_causes": disease_causes,
        "prescriptions": prescriptions,
        "recommendations": recommendations,
    }

# ============================================================
# LIGHTWEIGHT HEADLESS CHART GENERATION
# ============================================================

def plot_soil_bars(diagnostics: List[Dict[str, Any]]) -> Optional[plt.Figure]:
    if not diagnostics: return None
    params = [d["Parameter"] for d in diagnostics]
    values = [d["Value"] for d in diagnostics]
    statuses = [d["Status"] for d in diagnostics]

    color_map = {"NORMAL": "#2d6a4f", "SLIGHTLY LOW": "#f4a261", "LOW": "#e76f51", "SLIGHTLY HIGH": "#457b9d", "HIGH": "#9b5de5"}
    colors = [color_map.get(s, "#a8dadc") for s in statuses]

    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=90)
    bars = ax.barh(params, values, color=colors, edgecolor="#1d3557", alpha=0.9, height=0.6)
    ax.set_xlabel("Measured Reading", fontsize=9, fontweight="bold")
    ax.set_title("Soil & Environmental Readings Diagnostic Breakdown", fontsize=10, fontweight="bold", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    for bar, status in zip(bars, statuses):
        ax.text(bar.get_width() + max(values)*0.02, bar.get_y() + bar.get_height()/2, status, va="center", ha="left", fontsize=8, fontweight="bold", color="#1d3557")
    plt.tight_layout()
    return fig

def plot_crop_bars(recommendations: List[Dict[str, Any]]) -> Optional[plt.Figure]:
    if not recommendations: return None
    crops = [item["crop_display"] for item in reversed(recommendations[:5])]
    scores = [item["score"] for item in reversed(recommendations[:5])]
    colors = ["#2d6a4f" if s >= 80 else ("#1d3557" if s >= 60 else ("#e76f51" if s >= 40 else "#d62828")) for s in scores]

    fig, ax = plt.subplots(figsize=(7.5, max(3.2, len(crops) * 0.42)), dpi=90)
    bars = ax.barh(crops, scores, color=colors, edgecolor="#1b4332", alpha=0.9, height=0.55)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Suitability Score (0 - 100)", fontsize=9, fontweight="bold")
    ax.set_title("Crop Suitability Ranking for Current Soil", fontsize=10, fontweight="bold", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height()/2, f"{score:.1f}%", va="center", ha="left", fontsize=8.5, fontweight="bold")
    plt.tight_layout()
    return fig

"""
Utility functions for crop canonicalization, hardware detection, and agronomic helpers.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

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

# Knowledge base of practical agronomic treatments for diagnosed plant pathologies
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
        "symptoms": "Cinnamon-brown pustules scattered over both leaf surfaces.",
        "cultural": "Plant rust-resistant hybrids; ensure balanced crop nutrition.",
        "chemical": "Fungicide application (Pyraclostrobin, Azoxystrobin) if pustules appear before tasseling.",
    },
    "Northern Leaf Blight": {
        "symptoms": "Long, elliptical grayish-green or tan lesions on lower leaves progressing upwards.",
        "cultural": "Manage crop residue via tilling; practice crop rotation.",
        "chemical": "Fungicide application at threshold if weather favors fungal spread (cool, moist conditions).",
    },
    "Esca (Black Measles)": {
        "symptoms": "Tiger-stripe leaf chlorosis and necrosis; dark spotting on berries.",
        "cultural": "Delay pruning until late winter; protect pruning wounds with wound sealants.",
        "chemical": "No curative fungicide; manage vine stress and prune infected cordons.",
    },
    "Early Blight": {
        "symptoms": "Concentric dark brown rings (target board pattern) on older lower foliage.",
        "cultural": "Mulch soil surface to prevent soil splash; practice drip irrigation; remove lower infected leaves.",
        "chemical": "Apply Chlorothalonil, Copper hydroxide, or Azoxystrobin on a 7-14 day interval.",
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


def canonical_crop(value: str, strict: bool = True) -> Optional[str]:
    """
    Normalizes arbitrary crop strings, alias variants, or PlantVillage labels
    to the system standard canonical lowercase identifier.

    Args:
        value: Raw crop name string.
        strict: If True (default), raises ValueError on unknown crops.
                If False, returns None for unrecognized crops (safe for bulk mapping).
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

    if strict:
        raise ValueError(f"Unsupported crop name: {value!r}")
    return None


def pretty_crop(crop: str) -> str:
    """
    Returns a human-readable, capitalized display name for a canonical crop key.
    """
    return PRETTY_CROP_NAMES.get(crop, crop.replace("_", " ").title())


def setup_device() -> Tuple[bool, str]:
    """
    Configures TensorFlow runtime device, enabling memory growth for available GPUs.
    Never blocks or crashes execution when GPU is unavailable.
    Returns a tuple (gpu_active, status_message).
    """
    try:
        import tensorflow as tf
    except ImportError:
        return False, "TensorFlow is not installed in the current environment."

    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError:
                    pass
            return True, f"TensorFlow GPU active: {gpus[0].name}"
        return False, "Running on CPU (TensorFlow GPU not detected)."
    except Exception as exc:
        return False, f"Device initialization notice: {exc}"


def get_disease_treatment(disease: str) -> Dict[str, str]:
    """
    Retrieves agronomic management and treatment advice for a diagnosed plant condition.
    """
    for key, treatment in DISEASE_TREATMENTS.items():
        if key.lower() in disease.lower() or disease.lower() in key.lower():
            return treatment

    return {
        "symptoms": "Specific pathogen symptoms observed on foliar tissue.",
        "cultural": "Ensure good crop hygiene, remove diseased tissue, and maintain balanced soil moisture.",
        "chemical": "Consult local agricultural extension service for registered preventative sprays.",
    }

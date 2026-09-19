"""
Utility module for Farmer Crop Advisory System.
Provides crop name canonicalization, human-readable formatting, disease treatment lookups,
and safe GPU/CPU hardware detection with CPU fallback.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ============================================================
# CROP MAPPINGS & ALIASES
# ============================================================

CROP_ALIASES: Dict[str, str] = {
    "apple": "apple",
    "bell pepper": "bell_pepper",
    "bell_pepper": "bell_pepper",
    "pepper bell": "bell_pepper",
    "pepper": "bell_pepper",
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

PRETTY_CROP_NAMES: Dict[str, str] = {
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

# ============================================================
# AGRONOMIC PATHOLOGY TREATMENTS
# ============================================================

DISEASE_TREATMENTS: Dict[str, Dict[str, str]] = {
    "Apple Scab": {
        "symptoms": "Olive-green to black velvety lesions on leaves and fruit, causing leaf distortion and premature defoliation.",
        "cultural": "Rake and destroy fallen leaves in autumn; prune canopies to maximize airflow and rapid drying; avoid overhead irrigation.",
        "chemical": "Apply preventative fungicides (Captan, Mancozeb, or copper-based sprays) at green-tip stage and repeat as guided by local infection warnings.",
    },
    "Black Rot": {
        "symptoms": "Circular brown leaf spots with dark borders (frog-eye spots); rotting fruit turning black and shriveled into hard mummies.",
        "cultural": "Remove mummified fruit, dead wood, and infected canes during dormancy; prune canopy to ensure rapid sunlight penetration.",
        "chemical": "Apply Captan or Myclobutanil fungicides starting at petal fall through pre-harvest cover sprays.",
    },
    "Cedar Apple Rust": {
        "symptoms": "Bright orange or yellow spots on the upper leaf surface that develop tubelike fungal structures underneath.",
        "cultural": "Remove nearby eastern red cedar or juniper trees if practical within 1 km; select rust-resistant cultivars.",
        "chemical": "Apply Myclobutanil, Mancozeb, or Propiconazole fungicide sprays at pink-bud through petal fall.",
    },
    "Bacterial Spot": {
        "symptoms": "Small, water-soaked, angular dark spots on foliage that become necrotic with chlorotic halos, causing defoliation.",
        "cultural": "Avoid working in fields when foliage is wet; use certified disease-free seeds; practice 2-3 year crop rotation with non-solanaceous crops.",
        "chemical": "Apply preventative copper hydroxide sprays mixed with Mancozeb; apply registered bactericides when conditions favor disease.",
    },
    "Powdery Mildew": {
        "symptoms": "White to light gray powdery fungal patches covering leaves, shoots, and flower buds.",
        "cultural": "Ensure adequate plant spacing; avoid excessive nitrogen fertilization that forces soft growth; overhead wash foliage if practical.",
        "chemical": "Spray potassium bicarbonate, neem oil, sulfur-based fungicides, or systemic triazoles at the first onset of mildew spots.",
    },
    "Cercospora Leaf Spot": {
        "symptoms": "Small, circular spots with tan to gray centers and prominent reddish-brown or purple margins on leaves.",
        "cultural": "Bury crop residues after harvest; avoid overhead sprinkler watering; rotate fields away from susceptible host crops.",
        "chemical": "Apply strobilurin or triazole fungicides during early vegetative stages when conditions are humid.",
    },
    "Common Rust": {
        "symptoms": "Small, powdery, cinnamon-brown to dark pustules on both upper and lower leaf surfaces.",
        "cultural": "Plant resistant hybrid seed varieties; practice early planting to escape peak spore loads; remove volunteer plants.",
        "chemical": "Apply fungicides containing Azoxystrobin, Pyraclostrobin, or Propiconazole if pustules appear prior to reproductive stages.",
    },
    "Northern Leaf Blight": {
        "symptoms": "Long, elliptical grayish-green or tan lesions (cigar-shaped) that coalesce and scorch entire leaves.",
        "cultural": "Turn under crop residue promptly after harvest; implement minimum 1-2 year non-host crop rotation.",
        "chemical": "Apply strobilurin or DMI triazole fungicides at first sign of disease on lower leaves, especially before tasseling.",
    },
    "Esca": {
        "symptoms": "Tiger-stripe foliar chlorosis and necrosis; dark spots on berries; wood necrosis inside vine trunk.",
        "cultural": "Protect pruning wounds with wound sealants; prune during dry weather; remove and burn severely infected vines.",
        "chemical": "No curative chemical exists; treat pruning wounds with Trichoderma or authorized biocides preventatively.",
    },
    "Early Blight": {
        "symptoms": "Concentric dark brown rings with target-like appearance surrounded by yellow chlorotic halos, starting on lower leaves.",
        "cultural": "Mulch soil around plants; avoid overhead watering; prune lower leaves touching the soil; rotate away from solanaceous plants.",
        "chemical": "Apply copper hydroxide, Chlorothalonil, or Azoxystrobin on a 7-10 day schedule during warm, rainy weather.",
    },
    "Late Blight": {
        "symptoms": "Large water-soaked dark lesions with pale margins, often showing white fuzzy mold on leaf undersides in cool, humid air.",
        "cultural": "Destroy volunteer potato/tomato plants; ensure excellent field drainage; avoid wet foliage overnight; immediately rogue infected plants.",
        "chemical": "Apply systemic fungicides (Mefenoxam, Cyazofamid, or Mancozeb) immediately upon first sighting or regional blight alerts.",
    },
    "Leaf Mold": {
        "symptoms": "Pale greenish-yellow spots on leaf upper surfaces with olive-green to brownish velvety mold beneath.",
        "cultural": "Reduce greenhouse/tunnel humidity below 85%; increase cross-ventilation; prune lower suckers to open canopy.",
        "chemical": "Apply Chlorothalonil, Copper, or Cyazofamid fungicides at early infection signs.",
    },
    "Septoria Leaf Spot": {
        "symptoms": "Numerous circular brown spots with dark borders and tiny black fruiting bodies (pycnidia) in centers.",
        "cultural": "Sterilize stakes and cages; avoid working among wet plants; clean up debris; apply organic mulch around plant base.",
        "chemical": "Apply Chlorothalonil or Copper fungicides preventatively after fruit set and continue through harvest.",
    },
    "Two-spotted Spider Mite": {
        "symptoms": "Fine stippling, yellow bronzing on upper leaf surfaces, and fine silky webbing on leaf undersides.",
        "cultural": "Maintain adequate irrigation to avoid drought stress; spray water to wash dust from leaves; introduce predatory mites (Phytoseiulus).",
        "chemical": "Apply insecticidal soap, neem oil, or selective miticides (Abamectin, Bifenazate) rotating modes of action.",
    },
    "Target Spot": {
        "symptoms": "Small brown spots that enlarge into concentric dark circles with necrotic centers on leaves and stems.",
        "cultural": "Provide wide plant spacing; stake plants off wet soil; avoid overhead irrigation.",
        "chemical": "Apply fungicides containing Azoxystrobin, Difenoconazole, or Boscalid.",
    },
    "Yellow Leaf Curl Virus": {
        "symptoms": "Severe stunting, upward leaf curling, yellowing leaf margins, puckering, and aborted flowers.",
        "cultural": "Control silverleaf whitefly vectors using yellow sticky cards and fine insect mesh netting; plant virus-resistant varieties.",
        "chemical": "Apply systemic insecticides or insecticidal soaps targeting whitefly nymphs if economic thresholds are exceeded.",
    },
    "Mosaic Virus": {
        "symptoms": "Mottled dark green and light green mosaic patterns on leaves, leaf distortion (shoestringing), and stunted growth.",
        "cultural": "Disinfect garden tools and wash hands before touching crops; do not use tobacco near plants; rogue infected plants immediately.",
        "chemical": "No chemical viricide exists; focus on controlling aphid vectors with horticultural oil or soaps.",
    },
    "Leaf Scorch": {
        "symptoms": "Purplish-red spots that enlarge and dry into dark brown scorched patches on strawberry foliage.",
        "cultural": "Renovate strawberry beds after harvest; avoid overhead irrigation; improve soil drainage and aeration.",
        "chemical": "Apply Captan or Thiophanate-methyl during bloom and early runner production if pressure is historically high.",
    },
}

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def canonical_crop(value: str, strict: bool = False) -> Optional[str]:
    """
    Standardizes crop strings across user inputs, dataset labels, and model categories.
    Handles aliases like 'corn (maize)', 'bell pepper', and case discrepancies.
    """
    if value is None:
        if strict:
            raise ValueError("Crop name cannot be None.")
        return None

    cleaned = (
        str(value)
        .strip()
        .lower()
        .replace("(", "")
        .replace(")", "")
        .replace("-", " ")
        .replace("_", " ")
    )

    # Specific compound alias handling
    if "corn" in cleaned and "maize" in cleaned:
        return "maize"

    canonical = CROP_ALIASES.get(cleaned, None)

    if canonical is None:
        if strict:
            raise ValueError(f"Crop '{value}' is not recognized in the canonical taxonomy.")
        return None

    return canonical


def pretty_crop(crop: str) -> str:
    """
    Converts a canonical crop identifier into a clean, human-readable display string.
    """
    if not crop:
        return ""
    canonical = canonical_crop(crop, strict=False) or crop.lower().strip()
    return PRETTY_CROP_NAMES.get(canonical, canonical.replace("_", " ").title())


def get_disease_treatment(disease: str) -> Dict[str, str]:
    """
    Looks up specific agronomic management protocols for a detected disease.
    Falls back gracefully to comprehensive hygienic guidelines for unknown or unlisted pathologies.
    """
    if not disease:
        disease = ""

    disease_clean = disease.strip().lower()

    for name, protocol in DISEASE_TREATMENTS.items():
        if name.lower() in disease_clean or disease_clean in name.lower():
            return protocol

    return {
        "symptoms": "Foliar discoloration or tissue necrosis consistent with plant pathogen stress.",
        "cultural": "Ensure good crop hygiene, remove diseased tissue, avoid overhead watering, and maintain balanced soil moisture.",
        "chemical": "Consult local agricultural extension agents for approved registered preventative foliar sprays in your region.",
    }


def setup_device() -> Tuple[bool, str]:
    """
    Configures runtime hardware acceleration safely with CPU fallback.
    Performs fast, non-blocking GPU detection to eliminate cold-start lag.
    """
    try:
        os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
        import shutil, subprocess, sys

        if "tensorflow" in sys.modules:
            gpus = sys.modules["tensorflow"].config.list_physical_devices("GPU")
            if gpus:
                return True, f"Hardware acceleration enabled: {len(gpus)} GPU(s) active"

        if shutil.which("nvidia-smi"):
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if res.returncode == 0 and res.stdout.strip():
                return True, f"Hardware acceleration enabled: GPU active ({res.stdout.strip().splitlines()[0]})"
    except Exception as exc:
        logger.debug(f"Device setup fallback: {exc}")

    return False, "Running on CPU (No CUDA-compatible GPU detected; standard latency)"


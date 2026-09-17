"""
Configuration module for Farmer Crop Advisory System.
Contains centralized paths, model artifacts, hyperparameters, and thresholds.
"""

from __future__ import annotations

import pathlib

# ============================================================
# DIRECTORY PATHS
# ============================================================

BASE_DIR = pathlib.Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

SRC_DIR = BASE_DIR / "src"
MODEL_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
DOCS_DIR = BASE_DIR / "docs"

# Ensure output directories exist
for directory in [OUTPUTS_DIR, FIGURES_DIR, REPORTS_DIR, PREDICTIONS_DIR, PROCESSED_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================
# DATASET PATHS
# ============================================================

PLANT_VILLAGE_DIR = RAW_DATA_DIR / "Plant Village Dataset"
TRAIN_DIR = PLANT_VILLAGE_DIR / "Train"
VAL_DIR = PLANT_VILLAGE_DIR / "Val"
TEST_DIR = PLANT_VILLAGE_DIR / "Test"

# Fallback paths for raw CSV
CSV_CANDIDATES = [
    RAW_DATA_DIR / "crop_recommendation_10000.csv",
    RAW_DATA_DIR / "crop_recommendation_10000(2).csv",
    BASE_DIR / "crop_recommendation_10000.csv",
]

# Preferred single path
DEFAULT_CSV_PATH = RAW_DATA_DIR / "crop_recommendation_10000.csv"

# ============================================================
# MODEL ARTIFACT PATHS
# ============================================================

IMAGE_MODEL_PATH = MODEL_DIR / "plant_disease_model.keras"
IMAGE_CLASSES_PATH = MODEL_DIR / "plant_disease_classes.json"
SOIL_MODEL_PATH = MODEL_DIR / "soil_condition_model.keras"
SOIL_METADATA_PATH = MODEL_DIR / "soil_condition_metadata.json"
SOIL_PROFILES_PATH = MODEL_DIR / "soil_condition_profiles.json"
IMAGE_BACKUP_PATH = MODEL_DIR / "plant_disease_model_backup.keras"
IMAGE_CPU_BEST_PATH = MODEL_DIR / "plant_disease_model_cpu_best.keras"

# ============================================================
# VISION PIPELINE HYPERPARAMETERS
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
IMAGE_CONFIDENCE_THRESHOLD = 0.60

# ============================================================
# SOIL PIPELINE FEATURES & PARAMETERS
# ============================================================

SOIL_FEATURES = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
]

SOIL_CROPS = [
    "apple",
    "bell_pepper",
    "cherry",
    "grapes",
    "maize",
    "peach",
    "potato",
    "strawberry",
    "tomato",
]

STATUS_THRESHOLDS = {
    "GOOD": 80.0,
    "ACCEPTABLE": 60.0,
    "NEEDS ATTENTION": 40.0,
}

SEED = 42

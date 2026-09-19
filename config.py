"""
Configuration module for Farmer Crop Advisory System.
Centralizes paths, model artifacts, hyperparameters, and feature lists.
"""

from __future__ import annotations

import pathlib

# Directory Paths
BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

for d in (PROCESSED_DATA_DIR, OUTPUTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Dataset & Model Artifact Paths
CROP_DATA = RAW_DATA_DIR / "crop_recommendation_10000.csv"
CSV_CANDIDATES = [
    CROP_DATA,
    RAW_DATA_DIR / "crop_recommendation_10000(2).csv",
    BASE_DIR / "crop_recommendation_10000.csv",
]
TEST_DIR = RAW_DATA_DIR / "Plant Village Dataset" / "Test"

DISEASE_MODEL = MODEL_DIR / "plant_disease_model.keras"
SOIL_MODEL = MODEL_DIR / "soil_condition_model.keras"
DISEASE_CLASSES = MODEL_DIR / "plant_disease_classes.json"
SOIL_PROFILES = MODEL_DIR / "soil_condition_profiles.json"
IMAGE_BACKUP_PATH = MODEL_DIR / "plant_disease_model_backup.keras"
IMAGE_CPU_BEST_PATH = MODEL_DIR / "plant_disease_model_cpu_best.keras"

# Vision Pipeline Settings
IMAGE_SIZE = (224, 224)
IMAGE_CONFIDENCE_THRESHOLD = 0.60

# Soil Pipeline Features & Supported Crops
SOIL_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
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

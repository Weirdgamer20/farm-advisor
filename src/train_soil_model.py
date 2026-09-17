"""
AI 3000 — Crop-Conditioned Soil/Environment Suitability Neural Network

Input:
    N, P, K, temperature, humidity, pH, rainfall + crop

Output:
    0-100 crop/environment suitability score

Important:
    The supplied CSV has no measured "soil_condition" target.
    Therefore the target is constructed from crop-specific empirical
    distributions in the CSV. This is a genuine neural network, but
    it learns the dataset-defined suitability function; it is not a
    laboratory soil-health classifier.
"""

from __future__ import annotations

import pathlib
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "soil_condition_model.keras"

CSV_CANDIDATES = [
    BASE_DIR / "data" / "raw" / "crop_recommendation_10000.csv",
    BASE_DIR / "data" / "raw" / "crop_recommendation_10000(2).csv",
    BASE_DIR / "crop_recommendation_10000.csv",
    BASE_DIR / "crop_recommendation_10000(2).csv",
]

FEATURES = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
]

EXPECTED_CROPS = [
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

SEED = 42
INITIAL_LR = 1e-3

STATUS_THRESHOLDS = {
    "good": 80.0,
    "acceptable": 60.0,
    "needs_attention": 40.0,
}


# ============================================================
# CROP NORMALIZATION
# ============================================================

def canonical_crop(value: str) -> str:
    key = str(value).strip().lower()

    aliases = {
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

    if key in aliases:
        return aliases[key]

    compact = (
        key.replace("(", "")
        .replace(")", "")
        .replace("-", " ")
        .replace("_", " ")
    )

    if compact in aliases:
        return aliases[compact]

    raise ValueError(f"Unsupported crop label: {value!r}")


# ============================================================
# REPRODUCIBILITY / DEVICE
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 72)
print("AI 3000 SOIL CONDITION / SUITABILITY MODEL")
print("=" * 72)
print("TensorFlow:", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print("GPU detected:", gpus)
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
else:
    print("WARNING: no TensorFlow GPU detected; training will use CPU.")


# ============================================================
# LOAD CSV
# ============================================================

csv_path = next((p for p in CSV_CANDIDATES if p.exists()), None)

if csv_path is None:
    raise FileNotFoundError(
        "No crop recommendation CSV found. Expected one of:\n"
        + "\n".join(str(p) for p in CSV_CANDIDATES)
    )

print("CSV:", csv_path)

df = pd.read_csv(csv_path)
df.columns = [str(c).strip() for c in df.columns]

required_columns = FEATURES + ["crop"]
missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(f"CSV is missing required columns: {missing}")

for feature in FEATURES:
    df[feature] = pd.to_numeric(df[feature], errors="coerce")

df["crop"] = df["crop"].astype(str).map(canonical_crop)

df = df.dropna(subset=FEATURES + ["crop"]).copy()

# Enforce the 9 crops used by the leaf model.
df = df[df["crop"].isin(EXPECTED_CROPS)].copy()

counts = df["crop"].value_counts().reindex(EXPECTED_CROPS).fillna(0)

missing_crops = counts[counts == 0].index.tolist()

if missing_crops:
    raise ValueError(
        "The soil CSV does not contain all 9 leaf-model crops. "
        f"Missing: {missing_crops}"
    )

print("Rows:", len(df))
print("Crop counts:")
for crop in EXPECTED_CROPS:
    print(f"  {crop:12s}: {int(counts[crop])}")


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

crop_to_index = {
    crop: i for i, crop in enumerate(EXPECTED_CROPS)
}

crop_indices = df["crop"].map(crop_to_index).to_numpy(dtype=np.int32)
X_numeric = df[FEATURES].to_numpy(dtype=np.float32)

indices = np.arange(len(df))

train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.30,
    random_state=SEED,
    stratify=crop_indices,
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    random_state=SEED,
    stratify=crop_indices[temp_idx],
)

print()
print("Split:")
print("  train:", len(train_idx))
print("  val:  ", len(val_idx))
print("  test: ", len(test_idx))


# ============================================================
# LEARN CROP PROFILES FROM TRAINING DATA ONLY
# ============================================================

train_df = df.iloc[train_idx].copy()
profiles: dict[str, dict[str, dict[str, float]]] = {}

for crop in EXPECTED_CROPS:
    crop_rows = train_df[train_df["crop"] == crop]

    if len(crop_rows) < 5:
        raise ValueError(f"Too few training rows for crop: {crop}")

    profiles[crop] = {}

    for feature in FEATURES:
        values = crop_rows[feature].to_numpy(dtype=np.float64)

        profiles[crop][feature] = {
            "p10": float(np.percentile(values, 10)),
            "p25": float(np.percentile(values, 25)),
            "median": float(np.percentile(values, 50)),
            "p75": float(np.percentile(values, 75)),
            "p90": float(np.percentile(values, 90)),
        }


# ============================================================
# DATASET-DERIVED TARGET
# ============================================================

def feature_suitability(value: float, profile: dict[str, float]) -> float:
    p10 = profile["p10"]
    p25 = profile["p25"]
    p75 = profile["p75"]
    p90 = profile["p90"]

    spread = max(p90 - p10, 1e-6)

    if p25 <= value <= p75:
        return 1.0

    if value < p25:
        distance = p25 - value
        score = 1.0 - 0.50 * (
            distance / max(p25 - p10, 1e-6)
        )

        if value < p10:
            extra = (p10 - value) / spread
            score = 0.50 * np.exp(-2.0 * extra)

        return float(np.clip(score, 0.0, 1.0))

    distance = value - p75
    score = 1.0 - 0.50 * (
        distance / max(p90 - p75, 1e-6)
    )

    if value > p90:
        extra = (value - p90) / spread
        score = 0.50 * np.exp(-2.0 * extra)

    return float(np.clip(score, 0.0, 1.0))


def suitability_score(values: np.ndarray, crop: str) -> float:
    profile = profiles[crop]

    scores = [
        feature_suitability(
            float(value),
            profile[feature],
        )
        for feature, value in zip(FEATURES, values)
    ]

    scores = np.asarray(scores, dtype=np.float64)

    geometric_mean = np.exp(
        np.mean(
            np.log(
                np.clip(scores, 1e-8, 1.0)
            )
        )
    )

    return float(
        np.clip(
            geometric_mean * 100.0,
            0.0,
            100.0,
        )
    )


y = np.asarray(
    [
        suitability_score(
            X_numeric[i],
            df.iloc[i]["crop"],
        )
        for i in range(len(df))
    ],
    dtype=np.float32,
)

print()
print("Generated target:")
print("  min :", float(y.min()))
print("  mean:", float(y.mean()))
print("  max :", float(y.max()))


# ============================================================
# NEURAL NETWORK WITH PREPROCESSING INSIDE THE MODEL
# ============================================================

numeric_train = X_numeric[train_idx]

normalizer = layers.Normalization(
    axis=-1,
    name="numeric_normalization",
)
normalizer.adapt(numeric_train)

numeric_input = keras.Input(
    shape=(len(FEATURES),),
    dtype=tf.float32,
    name="numeric",
)

crop_input = keras.Input(
    shape=(1,),
    dtype=tf.int32,
    name="crop_index",
)

numeric_scaled = normalizer(numeric_input)

crop_one_hot = layers.CategoryEncoding(
    num_tokens=len(EXPECTED_CROPS),
    output_mode="one_hot",
    name="crop_one_hot",
)(crop_input)

x = layers.Concatenate(name="combined_features")(
    [numeric_scaled, crop_one_hot]
)

x = layers.Dense(128, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.20)(x)

x = layers.Dense(64, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.15)(x)

x = layers.Dense(32, activation="relu")(x)
x = layers.Dense(16, activation="relu")(x)

x = layers.Dense(1, activation="sigmoid", name="score_01")(x)

output = layers.Rescaling(
    scale=100.0,
    name="soil_suitability_score",
)(x)

model = keras.Model(
    inputs={
        "numeric": numeric_input,
        "crop_index": crop_input,
    },
    outputs=output,
    name="AI 3000SoilConditionNet",
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=INITIAL_LR),
    loss=keras.losses.Huber(delta=5.0),
    metrics=[
        keras.metrics.MeanAbsoluteError(name="mae"),
        keras.metrics.RootMeanSquaredError(name="rmse"),
    ],
)

model.summary()


# ============================================================
# INPUT ARRAYS
# ============================================================

def inputs_for(index_array: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "numeric": X_numeric[index_array],
        "crop_index": crop_indices[index_array].reshape(-1, 1),
    }


# ============================================================
# TRAIN
# ============================================================

callbacks = [
    keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_mae",
        mode="min",
        save_best_only=True,
        verbose=1,
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_mae",
        mode="min",
        patience=12,
        restore_best_weights=True,
        verbose=1,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1,
    ),
]

print()
print("=" * 72)
print("TRAINING SOIL SUITABILITY NEURAL NETWORK")
print("=" * 72)

model.fit(
    inputs_for(train_idx),
    y[train_idx],
    validation_data=(
        inputs_for(val_idx),
        y[val_idx],
    ),
    epochs=100,
    batch_size=64,
    callbacks=callbacks,
    verbose=1,
)


# ============================================================
# FINAL TEST
# ============================================================

best_model = keras.models.load_model(MODEL_PATH)

test_results = best_model.evaluate(
    inputs_for(test_idx),
    y[test_idx],
    return_dict=True,
    verbose=1,
)

predictions = best_model.predict(
    inputs_for(test_idx),
    verbose=0,
).reshape(-1)

correlation = float(
    np.corrcoef(predictions, y[test_idx])[0, 1]
)

best_model.save(MODEL_PATH)

print()
print("=" * 72)
print("SOIL MODEL COMPLETE")
print("=" * 72)
print(f"Test MAE:        {test_results['mae']:.4f}")
print(f"Test RMSE:       {test_results['rmse']:.4f}")
print(f"Target corr.:    {correlation:.4f}")
print("Model:", MODEL_PATH)
print()
print("Supported crops:")
for i, crop in enumerate(EXPECTED_CROPS):
    print(f"  {i}: {crop}")
print()
print(
    "NOTE: This neural network predicts dataset-derived "
    "crop/environment suitability, not laboratory soil health."
)
print("=" * 72)

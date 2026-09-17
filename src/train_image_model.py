"""
AI 3000 — Leaf Disease / Crop Identification Neural Network

Input:
    Leaf image

Output:
    PlantVillage disease class (29 classes across 9 crops)

The script trains in two stages:
    1. Train the new classification head with EfficientNetB0 frozen.
    2. Fine-tune the last part of EfficientNetB0 at a low learning rate.

This fixes the previous training flow where a fresh model could jump
directly into low-LR fine-tuning before the randomly initialized head
had learned useful features.
"""

from __future__ import annotations

import json
import pathlib
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent

DATASET_CANDIDATES = [
    BASE_DIR / "data" / "raw" / "Plant Village Dataset",
    BASE_DIR / "Plant Village Dataset",
]
DATASET_DIR = next((p for p in DATASET_CANDIDATES if p.exists()), DATASET_CANDIDATES[0])
TRAIN_DIR = DATASET_DIR / "Train"
VAL_DIR = DATASET_DIR / "Val"
TEST_DIR = DATASET_DIR / "Test"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "plant_disease_model.keras"
CLASS_NAMES_PATH = MODEL_DIR / "plant_disease_classes.json"
BACKUP_PATH = MODEL_DIR / "plant_disease_model_backup.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

INITIAL_EPOCHS = 12
FINE_TUNE_EPOCHS = 20
FINE_TUNE_FRACTION = 0.30

INITIAL_LR = 1e-3
FINE_TUNE_LR = 1e-5

# Set True only if you want training to stop when TensorFlow
# cannot see the GPU. False makes the script usable on CPU too.
REQUIRE_GPU = False


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

print("=" * 72)
print("AI 3000 LEAF MODEL TRAINING")
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
    print("WARNING: TensorFlow GPU is not available; training will use CPU.")
    if REQUIRE_GPU:
        raise RuntimeError(
            "GPU required by configuration, but TensorFlow detected no GPU."
        )


# ============================================================
# DATASET VALIDATION
# ============================================================

for directory in (TRAIN_DIR, VAL_DIR, TEST_DIR):
    if not directory.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {directory}")

print("Dataset:", DATASET_DIR)

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

class_names = train_ds.class_names
num_classes = len(class_names)

if val_ds.class_names != class_names:
    raise RuntimeError("Train and validation class ordering differ.")

if test_ds.class_names != class_names:
    raise RuntimeError("Train and test class ordering differ.")

print("Classes:", num_classes)

for i, name in enumerate(class_names):
    print(f"  {i:02d}: {name}")

with CLASS_NAMES_PATH.open("w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=2, ensure_ascii=False)


# ============================================================
# PERFORMANCE PIPELINE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)


# ============================================================
# CLASS WEIGHTS
# ============================================================

labels = []

for _, batch_labels in train_ds.unbatch():
    labels.append(int(batch_labels.numpy()))

labels = np.asarray(labels, dtype=np.int32)

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=labels,
)

class_weights = {
    int(index): float(weight)
    for index, weight in enumerate(weights)
}


# ============================================================
# AUGMENTATION
# ============================================================

augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.15),
        layers.RandomTranslation(0.08, 0.08),
    ],
    name="leaf_augmentation",
)


# ============================================================
# MODEL
# ============================================================

inputs = keras.Input(
    shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    name="leaf_image",
)

x = augmentation(inputs)

# EfficientNetB0 in modern tf.keras includes the expected input
# rescaling/preprocessing in the application model.
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    name="efficientnetb0",
)

base_model.trainable = False

x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D(name="global_average_pool")(x)
x = layers.BatchNormalization(name="head_bn")(x)
x = layers.Dropout(0.35, name="head_dropout")(x)
x = layers.Dense(256, activation="relu", name="head_dense")(x)
x = layers.BatchNormalization(name="head_dense_bn")(x)
x = layers.Dropout(0.25, name="head_dense_dropout")(x)

outputs = layers.Dense(
    num_classes,
    activation="softmax",
    name="disease_class",
)(x)

model = keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="AI 3000PlantDiseaseNet",
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=INITIAL_LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

if model.output_shape[-1] != num_classes:
    raise RuntimeError("Model output class count does not match dataset.")


# ============================================================
# CALLBACK FACTORY
# ============================================================

callbacks = [
    keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=4,
        restore_best_weights=True,
        verbose=1,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    ),
]


# ============================================================
# STAGE 1 — TRAIN CLASSIFICATION HEAD
# ============================================================

print()
print("=" * 72)
print("STAGE 1 — TRAINING CLASSIFICATION HEAD")
print("=" * 72)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1,
)


# ============================================================
# STAGE 2 — FINE-TUNE LAST PART OF EFFICIENTNET
# ============================================================

print()
print("=" * 72)
print("STAGE 2 — FINE-TUNING EFFICIENTNETB0")
print("=" * 72)

base_model.trainable = True

freeze_until = int(len(base_model.layers) * (1.0 - FINE_TUNE_FRACTION))

for layer in base_model.layers[:freeze_until]:
    layer.trainable = False

# BatchNorm statistics should remain stable during small-dataset
# fine-tuning.
for layer in base_model.layers[freeze_until:]:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

trainable_count = sum(1 for layer in base_model.layers if layer.trainable)

print("EfficientNet layers:", len(base_model.layers))
print("Trainable EfficientNet layers:", trainable_count)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

fine_tune_callbacks = [
    keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    ),
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=class_weights,
    callbacks=fine_tune_callbacks,
    verbose=1,
)


# ============================================================
# FINAL TEST
# ============================================================

best_model = keras.models.load_model(MODEL_PATH)

test_loss, test_accuracy = best_model.evaluate(
    test_ds,
    verbose=1,
)

# Keep a backup after a successful training run.
if MODEL_PATH.exists():
    import shutil
    shutil.copy2(MODEL_PATH, BACKUP_PATH)

best_model.save(MODEL_PATH)

print()
print("=" * 72)
print("LEAF MODEL COMPLETE")
print("=" * 72)
print(f"Test loss:     {test_loss:.6f}")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")
print("Model:", MODEL_PATH)
print("Classes:", CLASS_NAMES_PATH)
print("Backup:", BACKUP_PATH)
print("=" * 72)

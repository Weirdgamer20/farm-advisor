# 15 - API & Module Documentation

## 1. `config` Module
- `BASE_DIR: Path`: Project root path.
- `RAW_DATA_DIR: Path`: `data/raw` path.
- `IMAGE_MODEL_PATH: Path`: Path to `plant_disease_model.keras`.
- `SOIL_MODEL_PATH: Path`: Path to `soil_condition_model.keras`.
- `IMAGE_CONFIDENCE_THRESHOLD: float`: Minimum confidence (0.60) for reliable diagnosis.
- `STATUS_THRESHOLDS: dict`: Threshold score dictionary (`GOOD`: 80.0, `ACCEPTABLE`: 60.0, `NEEDS ATTENTION`: 40.0).

## 2. `src.utils` Module
- `canonical_crop(value: str) -> str`: Normalizes diverse crop strings to system standard string.
- `pretty_crop(crop: str) -> str`: Returns capitalized, human-friendly crop label.
- `setup_device() -> tuple[bool, str]`: Initializes GPU memory growth and returns device report.

## 3. `src.data_loader` Module
- `load_models(...) -> tuple[tf.keras.Model, list[str], tf.keras.Model]`: Validates and loads vision and soil models.
- `load_profile_data(csv_path: Optional[Path]) -> Optional[pd.DataFrame]`: Loads historical crop parameters.

## 4. `src.preprocessing` Module
- `preprocess_image(image: Image.Image, target_size: tuple) -> np.ndarray`: Converts PIL image into batched tensor `(1, 224, 224, 3)`.
- `prepare_soil_inputs(values: dict, crop: str) -> dict[str, np.ndarray]`: Formats numeric and categorical inputs.

## 5. `src.analysis` Module
- `classify_leaf(image, image_model, image_classes) -> dict`: Returns pathology, crop, confidence, and reliability.
- `predict_soil(values, crop, soil_model) -> dict`: Returns score (0–100) and status string.
- `diagnose_soil(values, crop, profile_df) -> list[dict]`: Computes quantile bounds and parameter health.
- `generate_advisory_summary(...) -> dict`: Compiles high-level agricultural recommendations.

## 6. `src.visualization` Module
- `plot_soil_parameters_bar(diagnostics, save_path) -> plt.Figure`: Generates parameter deviation bar chart.
- `format_status_badge(status: str) -> str`: Produces stylized HTML status pill.

# Farmer Crop Advisory System

An end-to-end academic AI agriculture prototype providing:

- **Plant Leaf Disease Classification**: EfficientNet-based computer vision for foliar pathology diagnosis with top candidate rankings and confidence calibration.
- **Crop Suitability Neural Network**: Dual-input deep learning model conditioning numeric soil/meteorological readings with crop embeddings.
- **Empirical Quantile Diagnostics**: Statistical benchmarking against precalculated empirical quantiles (`p10`, `p25`, `median`, `p75`, `p90`) across 9 supported crop profiles.
- **Agronomic Advisory Synthesis**: Correlates environmental and soil drivers with plant disease vulnerability and computes actionable concentration adjustment prescriptions.
- **Streamlit Web Application**: Thin, responsive presentation layer with publication-ready Matplotlib visual analytics.

---

## 1. Project Structure

The codebase is strictly organized according to the academic ML evaluator architecture following the Single Responsibility Principle (SRP):

```text
farm-advisor/
├── .github/
│   └── workflows/
│       └── ci.yml                   # Automated CI workflow (compileall & pytest)
├── config.py                        # Centralized paths, model artifacts, and hyperparameters
├── main.py                          # Thin Streamlit application entry point
├── requirements.txt                 # Dependencies including pytest
├── README.md                        # Documentation & setup guide
├── runui.bat                        # Automated WSL launcher script
│
├── data/
│   ├── raw/                         # Raw datasets
│   │   ├── crop_recommendation_10000.csv
│   │   └── Plant Village Dataset/
│   │       ├── Train/
│   │       ├── Val/
│   │       └── Test/
│   └── processed/                   # Processed dataset storage
│
├── src/
│   ├── __init__.py                  # Package interface exports
│   ├── data_loader.py               # Cached model loading, validation & profile parsing
│   ├── preprocessing.py             # EXIF image correction, RGB tensor scaling & soil vectorization
│   ├── analysis.py                  # Core inference, quantile diagnostics & advisory synthesis
│   ├── visualization.py             # Charts, metrics & status badges
│   └── utils.py                     # Canonical crop mappings, disease treatments & device detection
│
├── models/
│   ├── plant_disease_model.keras    # Trained foliar pathology model
│   ├── plant_disease_classes.json   # 38 pathology class labels
│   ├── soil_condition_model.keras   # Trained dual-input suitability model
│   ├── soil_condition_profiles.json # Empirical quantile distributions
│   └── soil_condition_metadata.json # Training metadata
│
├── outputs/
│   ├── figures/                     # Generated diagnostic plots
│   ├── reports/                     # Model performance reports
│   └── predictions/                 # Inference logs
│
├── tests/
│   ├── __init__.py
│   ├── test_analysis.py             # Diagnostics, advisory & root cause synthesis tests
│   ├── test_data_loader.py          # Profile loading, missing artifact fallback tests
│   ├── test_preprocessing.py        # Image transforms, soil vectorization & bounds tests
│   └── test_utils.py                # Crop canonicalization, treatments & device check tests
│
└── docs/                            # Standardized project documentation modules
```

---

## 2. Component Responsibilities

| Module | Responsibility |
| :--- | :--- |
| **`config.py`** | Centralizes paths, model artifact references, image sizes, soil features, and agronomic thresholds. |
| **`src/data_loader.py`** | Handles cached loading and validation of Keras neural models, class labels, CSV datasets, and quantile profiles. |
| **`src/preprocessing.py`** | Implements EXIF rotation correction, standard RGB normalization, dual-input soil vectorization, and physical range validation. |
| **`src/analysis.py`** | Coordinates vision inference, single-crop suitability scoring, vectorized multi-crop batch ranking, quantile comparisons, and root-cause advisory synthesis. |
| **`src/visualization.py`** | Renders UI cards, banners, metric widgets, and headless Matplotlib charts for parameter deviations and crop rankings. |
| **`src/utils.py`** | Provides canonical crop name resolution, human-readable formatting, disease treatment lookups, and non-blocking GPU/CPU hardware configuration. |
| **`main.py`** | Thin Streamlit coordinator wiring data loading, user inputs, analysis, and visual presentations. |

---

## 3. Environment Setup

### System Prerequisites
- **Operating System**: Windows 10/11 with WSL2 (Ubuntu) or native Linux
- **Python**: 3.10 - 3.13
- **Acceleration**: NVIDIA GPU with CUDA support (CPU fallback automatically engaged when GPU is absent)

### Installation

1. Navigate to the project root directory:
   ```bash
   cd /mnt/d/PROJECT
   ```

2. Activate or create a virtual environment:
   ```bash
   source .venv/bin/activate
   ```

3. Install project dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 4. Running the Application

### Direct Terminal Command
```bash
python3 -m streamlit run main.py --server.address 0.0.0.0 --server.port 8501
```

Once started, navigate to:
```text
http://localhost:8501
```

### Windows One-Click Launcher
Double-click `runui.bat` from Windows File Explorer. It automatically detects WSL Ubuntu, starts Streamlit headlessly, and opens your default browser.

---

## 5. Automated Verification & Testing

### Running Syntax Validation (compileall)
```bash
python3 -m compileall config.py main.py src/ tests/
```

### Running Pytest Test Suite
Execute the full unit and integration test suite:
```bash
pytest tests/ -v
```

The test suite validates:
- Image tensor conversion (PIL RGB, RGBA, bytes, EXIF orientation).
- Soil input vectorization and physical range bounds checking.
- Empirical quantile calculation and deviation status mapping (`LOW`, `NORMAL`, `HIGH`).
- Root cause correlation for plant diseases with environmental drivers (humidity, rainfall, K, N, pH).
- Canonical crop normalization and alias resolution across naming variants.
- Non-blocking hardware acceleration detection and CPU fallback.
- Profile parsing and fallback handling on missing artifact paths.

---

## 6. Continuous Integration (GitHub Actions)

Continuous integration is configured in `.github/workflows/ci.yml`. On every push and pull request to `master` and `main`, CI performs:
1. Python 3.11 environment setup.
2. Automated dependency installation from `requirements.txt`.
3. Complete syntax compilation via `python -m compileall`.
4. Automated test suite execution via `pytest tests/ -v`.

---

## 7. Supported Crops & Soil Parameters

### Supported Crops (9 Species)
- Apple (`apple`)
- Bell Pepper (`bell_pepper`)
- Cherry (`cherry`)
- Grape (`grapes`)
- Corn / Maize (`maize`)
- Peach (`peach`)
- Potato (`potato`)
- Strawberry (`strawberry`)
- Tomato (`tomato`)

### Monitored Soil & Climate Parameters (7 Features)
- **Nitrogen (N)**: 0 – 400 mg/kg
- **Phosphorus (P)**: 0 – 400 mg/kg
- **Potassium (K)**: 0 – 400 mg/kg
- **Air Temperature**: -10 – 55 °C
- **Relative Humidity**: 10 – 100 %
- **Soil pH**: 3.5 – 10.0
- **Annual Rainfall**: 0 – 2500 mm

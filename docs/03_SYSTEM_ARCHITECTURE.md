# 03 - System Architecture

## 1. High-Level Architecture
The system employs a modular layered architecture separating Presentation, Business Logic / Model Inference, Data Access, and Model Persistence.

```text
[ Farmer / User Client ]
          │  (HTTP / Browser)
          ▼
┌─────────────────────────────────────────────────────────────┐
│                 PRESENTATION LAYER (main.py)                │
│    - Streamlit UI Dashboard                                 │
│    - File Uploader, Parameter Sliders & Number Inputs       │
│    - Status Cards, Diagnostic Tables, Matplotlib Charts     │
└──────────────┬───────────────────────────────┬──────────────┘
               │                               │
               ▼                               ▼
┌───────────────────────────────┐ ┌───────────────────────────┐
│     VISION INFERENCE (src)    │ │   SOIL SUITABILITY (src)  │
│ - src/preprocessing.py        │ │ - src/preprocessing.py    │
│   (RGB conversion, Resize)    │ │   (Vector assembly, Index)│
│ - src/analysis.py             │ │ - src/analysis.py         │
│   (EfficientNetB0 inference)  │ │   (Dual-input MLP infer)  │
└──────────────┬────────────────┘ └─────────────┬─────────────┘
               │                                │
               ▼                                ▼
┌─────────────────────────────────────────────────────────────┐
│                 DATA & MODEL ASSETS LAYER                   │
│ - models/plant_disease_model.keras                          │
│ - models/soil_condition_model.keras                         │
│ - data/raw/crop_recommendation_10000.csv (Empirical Quant.) │
└─────────────────────────────────────────────────────────────┘
```

## 2. Component Design & SRP Mapping
- **`config.py`**: Central source of truth for filesystem paths, constants, thresholds, and hyperparameters.
- **`src/utils.py`**: Name canonicalization, hardware detection, device memory growth configuration.
- **`src/data_loader.py`**: Cached retrieval of model weights and historical agronomic profiles.
- **`src/preprocessing.py`**: Input sanitization, tensor shaping, scaling, and categorical encoding.
- **`src/analysis.py`**: Model prediction execution, quantile deviations calculation, and recommendation synthesis.
- **`src/visualization.py`**: Bar charts, status badge generation, and visual reports.
- **`main.py`**: Orchestration and interactive Streamlit UI rendering.

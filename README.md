# Farmer Crop Advisory System

A presentation-ready AI agriculture prototype that provides:

- Plant leaf disease classification
- Crop recommendation from soil/environment data
- Soil/fertilizer advisory
- Irrigation guidance
- Streamlit web interface

The project is designed to run on **Windows through WSL Ubuntu**.


## Download dataset here ##
  url-- https://www.kaggle.com/datasets/emmarex/plantdisease

---

## 1. Project Structure

The project adheres to a modular submission architecture based on the Single Responsibility Principle (SRP):

```text
D:\PROJECT\
│
├── main.py                          # Streamlit UI & application entry point
├── config.py                        # Centralized paths, model artifacts, and hyperparameters
├── requirements.txt                 # Project dependencies
├── README.md                        # Documentation & setup guide
├── .gitignore                       # Clean submission exclusion file
├── run_web_ui.bat                   # Web interface launcher (WSL / fallback)
├── run_webui.bat                    # Backward compatibility launcher
├── run_training.bat                 # Dual-model training launcher
│
├── data/
│   ├── raw/
│   │   ├── crop_recommendation_10000.csv
│   │   └── Plant Village Dataset/
│   │       ├── Train/
│   │       ├── Val/
│   │       └── Test/
│   └── processed/
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # Cached model & dataset loading
│   ├── preprocessing.py             # Image transforms & feature vectorization
│   ├── analysis.py                  # Core inference, quantile diagnostics, advisory logic
│   ├── visualization.py             # Charts, metrics & status badges
│   ├── utils.py                     # Canonical crop mappings & hardware setup
│   ├── train_image_model.py         # Leaf disease training script
│   └── train_soil_model.py          # Soil suitability training script
│
├── models/
│   ├── plant_disease_model.keras
│   ├── plant_disease_classes.json
│   ├── soil_condition_model.keras
│   ├── plant_disease_model_backup.keras
│   ├── plant_disease_model_cpu_best.keras
│   ├── soil_condition_metadata.json
│   └── soil_condition_profiles.json
│
├── outputs/
│   ├── figures/                     # Generated diagnostic plots
│   ├── reports/                     # Model performance reports
│   └── predictions/                 # Inference logs
│
└── docs/                            # 18 Standardized documentation modules
    ├── 01_PROBLEM_STATEMENT.md
    ├── 02_SRS.md
    ├── 03_SYSTEM_ARCHITECTURE.md
    ├── 04_ER_DIAGRAM.md
    ├── 05_DATASET.md
    ├── 06_DATA_PREPROCESSING.md
    ├── 07_ML_MODEL.md
    ├── 08_MODEL_EVALUATION.md
    ├── 09_CHATBOT.md
    ├── 10_BACKEND_API.md
    ├── 11_DATABASE.md
    ├── 12_FRONTEND.md
    ├── 13_ADMIN_DASHBOARD.md
    ├── 14_TEST_CASES.md
    ├── 15_API_DOCUMENTATION.md
    ├── 16_DEPLOYMENT.md
    ├── 17_FINAL_REPORT.md
    └── 18_PROJECT_PRESENTATION.md
```

The `models/` directory contains pre-trained neural networks. If these weights already exist, **re-training is optional**.

---

# 2. Requirements

## Windows

Install:

- Windows 10/11
- NVIDIA driver if GPU acceleration is required
- WSL2
- Ubuntu WSL distribution
- Python 3.13 or the Python version supported by the installed project environment

## Python packages

The project uses packages listed in:

```text
requirements.txt
```

Main dependencies include:

```text
TensorFlow
NumPy
Pandas
Scikit-learn
Matplotlib
Seaborn
Pillow
Streamlit
```

---

# 3. Start WSL Ubuntu

Do NOT use the `docker-desktop` WSL distribution.

Open PowerShell:

```powershell
wsl -l -v
```

You should have an Ubuntu distribution similar to:

```text
NAME              STATE           VERSION
docker-desktop    Running         2
Ubuntu            Stopped         2
```

Start Ubuntu:

```powershell
wsl -d Ubuntu
```

You should see something similar to:

```text
(gpu) user@computer:/mnt/d/farmadvisdor$
```

---

# 4. Go to the Project

Inside Ubuntu:

```bash
cd /mnt/d/farmadvisdor
```

Verify:

```bash
pwd
ls
```

You should see:

```text
farm_advisor.py
train_image_model.py
train_soil_model.py
requirements.txt
crop_recommendation_10000.csv
Plant Village Dataset
models
```

---

# 5. Verify Python

Run:

```bash
python3 --version
```

Example:

```text
Python 3.13.15
```

Check the active environment:

```bash
echo $VIRTUAL_ENV
```

If an environment is already active, continue.

If you have a project virtual environment at:

```text
/mnt/d/farmadvisdor/.venv
```

activate it with:

```bash
source /mnt/d/farmadvisdor/.venv/bin/activate
```

If your existing environment has a different location, activate that environment instead.

Verify:

```bash
which python3
```

---

# 6. Install Dependencies

With the project environment active:

```bash
python3 -m pip install --upgrade pip
```

Then:

```bash
python3 -m pip install -r requirements.txt
```

Verify TensorFlow:

```bash
python3 -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
```

Verify Streamlit:

```bash
python3 -c "import streamlit; print('Streamlit:', streamlit.__version__)"
```

---

# 7. Verify GPU

Run:

```bash
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

If a GPU is available, TensorFlow should return something similar to:

```text
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

If it returns:

```text
[]
```

TensorFlow is running without a GPU.

The project can still run on CPU, but image-model training will be slower.

---

# 8. Check the Crop Dataset

The crop recommendation dataset is:

```text
crop_recommendation_10000.csv
```

Check that it exists:

```bash
ls -lh crop_recommendation_10000.csv
```

The dataset contains:

```text
N
P
K
temperature
humidity
ph
rainfall
crop
```

The soil/crop model uses these values to calculate crop/soil recommendations.

---

# 9. Check the Plant Disease Dataset

The Plant Village dataset must have:

```text
Plant Village Dataset/
├── Train/
├── Val/
└── Test/
```

Check:

```bash
ls "Plant Village Dataset"
```

Expected:

```text
Train
Val
Test
```

Check the training classes:

```bash
find "Plant Village Dataset/Train" -mindepth 1 -maxdepth 1 -type d
```

The class directories contain the leaf images.

---

# 10. Train the Disease Model

The disease model is trained using:

```text
train_image_model.py
```

Run:

```bash
python3 train_image_model.py
```

This is the most computationally expensive stage.

The process uses the Plant Village:

```text
Train → training
Val   → validation
Test  → final evaluation
```

After successful training, check:

```bash
ls -lh models/
```

The important files include:

```text
models/plant_disease_model.keras
models/plant_disease_classes.json
```

Depending on the training script, additional backup/best-model files may also be created.

---

# 11. Train the Soil/Crop Model

Run:

```bash
python3 train_soil_model.py
```

This uses:

```text
crop_recommendation_10000.csv
```

After successful training, verify:

```bash
ls -lh models/
```

The important output is:

```text
models/soil_condition_model.keras
```

Metadata/profile files may also be generated.

---

# 12. Train Both Models

If you want to run both training programs manually:

```bash
python3 train_image_model.py
python3 train_soil_model.py
```

Wait for both commands to finish successfully.

Then:

```bash
ls -lh models/
```

---

# 13. Important: Do Not Retrain Every Time

Once these files exist:

```text
models/plant_disease_model.keras
models/plant_disease_classes.json
models/soil_condition_model.keras
```

you normally **do not need to run the training scripts again**.

Training is only required when:

- The dataset changes
- The model code changes
- You want better accuracy
- You intentionally want to create a new model

For normal presentation use, load the existing models.

---

# 14. Run the Web UI

The main application is:

```text
farm_advisor.py
```

Run:

```bash
python3 -m streamlit run farm_advisor.py --server.address 0.0.0.0
```

Streamlit should display a URL such as:

```text
Local URL: http://localhost:8501
```

Open Windows Chrome and visit:

```text
http://localhost:8501
```

---

# 15. Using the Web UI

## Disease Detection

1. Open the web UI.
2. Find the leaf disease section.
3. Upload a leaf image.
4. The application loads the trained disease model.
5. The model predicts the disease class.
6. The application displays the prediction and confidence.

Pipeline:

```text
Leaf Image
    ↓
Image Preprocessing
    ↓
Trained Disease Model
    ↓
Disease Class
    ↓
Confidence
```

---

# 16. Crop/Soil Advisory

Enter:

```text
Nitrogen
Phosphorus
Potassium
Temperature
Humidity
pH
Rainfall
```

The application uses the trained soil/crop model.

Pipeline:

```text
N / P / K
Temperature
Humidity
pH
Rainfall
    ↓
Soil/Crop Model
    ↓
Crop / Soil Recommendation
```

---

# 17. One-Click Web UI

The project includes:

```text
run_web_ui.bat
```

From Windows, double-click:

```text
run_web_ui.bat
```

It launches the application through WSL Ubuntu and uses the existing trained models.

Alternatively, run it from PowerShell:

```powershell
D:\farmadvisdor\run_web_ui.bat
```

The browser can then be opened at:

```text
http://localhost:8501
```

---

# 18. Manual One-Click Equivalent

If the BAT file is unavailable, use:

```powershell
wsl -d Ubuntu
```

Then:

```bash
cd /mnt/d/farmadvisdor
```

Activate the project environment if necessary:

```bash
source /path/to/your/gpu/environment/bin/activate
```

Then:

```bash
python3 -m streamlit run farm_advisor.py --server.address 0.0.0.0
```

---

# 19. Complete First-Time Setup

For a completely fresh setup:

### PowerShell

```powershell
wsl -l -v
```

Start Ubuntu:

```powershell
wsl -d Ubuntu
```

### Ubuntu

```bash
cd /mnt/d/farmadvisdor
```

Check Python:

```bash
python3 --version
```

Activate the project environment:

```bash
source /path/to/your/gpu/environment/bin/activate
```

Install dependencies:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Check TensorFlow:

```bash
python3 -c "import tensorflow as tf; print(tf.__version__)"
```

Check GPU:

```bash
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Train disease model:

```bash
python3 train_image_model.py
```

Train soil/crop model:

```bash
python3 train_soil_model.py
```

Launch UI:

```bash
python3 -m streamlit run farm_advisor.py --server.address 0.0.0.0
```

Open:

```text
http://localhost:8501
```

---

# 20. Normal Presentation-Day Startup

If the models are already trained, you do NOT need the training commands.

Use only:

```powershell
wsl -d Ubuntu
```

Then:

```bash
cd /mnt/d/farmadvisdor
```

Activate the existing environment:

```bash
source /path/to/your/gpu/environment/bin/activate
```

Run:

```bash
python3 -m streamlit run farm_advisor.py --server.address 0.0.0.0
```

Or simply double-click:

```text
run_web_ui.bat
```

---

# 21. Troubleshooting

## `python3: not found`

You are probably inside the wrong WSL distribution.

Check:

```powershell
wsl -l -v
```

Do not use:

```text
docker-desktop
```

Use:

```powershell
wsl -d Ubuntu
```

---

## `apt: not found`

You are probably inside `docker-desktop`.

Exit:

```bash
exit
```

Then from PowerShell:

```powershell
wsl -d Ubuntu
```

---

## `Activate.ps1` not found

A Linux virtual environment uses:

```bash
source .venv/bin/activate
```

A Windows virtual environment uses:

```powershell
.\.venv\Scripts\Activate.ps1
```

Do not use the Windows activation command inside WSL.

---

## `streamlit: command not found`

Use:

```bash
python3 -m streamlit run farm_advisor.py --server.address 0.0.0.0
```

If Streamlit is not installed:

```bash
python3 -m pip install streamlit
```

---

## TensorFlow cannot find GPU

Check:

```bash
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

If it returns:

```text
[]
```

the application can still run, but training may be slower.

---

## Model file not found

Check:

```bash
ls -lh models/
```

The application requires the trained model files generated by the training scripts.

If they are missing, retrain:

```bash
python3 train_image_model.py
python3 train_soil_model.py
```

---

## Streamlit `ScriptRunContext` warnings

Do not run:

```bash
python3 farm_advisor.py
```

Run:

```bash
python3 -m streamlit run farm_advisor.py
```

---

# 22. System Workflow

```text
                    FARMER CROP ADVISORY
                             │
              ┌──────────────┴──────────────┐
              │                             │
         LEAF IMAGE                    FIELD DATA
              │                             │
              ▼                             ▼
      Disease Model                  Soil/Crop Model
              │                             │
              ▼                             ▼
       Disease Result               Crop/Soil Result
              │                             │
              └──────────────┬──────────────┘
                             ▼
                      FARM ADVISORY
```

---

# 23. Model Files

The application uses trained `.keras` models rather than retraining models during normal inference.

Disease:

```text
models/plant_disease_model.keras
```

Disease labels:

```text
models/plant_disease_classes.json
```

Soil/crop:

```text
models/soil_condition_model.keras
```

Therefore, the normal workflow is:

```text
Train Once
   ↓
Save Models
   ↓
Run Web UI
   ↓
Load Models
   ↓
Farmer Input
   ↓
Prediction
```

---

# 24. Presentation Demo

Recommended demonstration:

1. Start `run_web_ui.bat`.
2. Open `http://localhost:8501`.
3. Upload a Plant Village leaf image.
4. Show the detected disease and confidence.
5. Enter N/P/K, temperature, humidity, pH and rainfall.
6. Generate the crop/soil advisory.
7. Explain that the disease model was trained using Plant Village and the crop recommendation model uses the supplied agricultural CSV.
8. Show the trained model files in `models/`.

This demonstrates the complete working prototype without requiring the training process during the presentation.

---

## Quick Commands

### First-time training

```bash
wsl -d Ubuntu
cd /mnt/d/farmadvisdor
source /path/to/your/gpu/environment/bin/activate
python3 -m pip install -r requirements.txt
python3 train_image_model.py
python3 train_soil_model.py
python3 -m streamlit run farm_advisor.py --server.address 0.0.0.0
```

### Normal use

```bash
wsl -d Ubuntu
cd /mnt/d/farmadvisdor
source /path/to/your/gpu/environment/bin/activate
python3 -m streamlit run farm_advisor.py --server.address 0.0.0.0
```

Or from Windows:

```text
Double-click run_web_ui.bat
```

---

## Important

This is a **demonstration/prototype system**, not a production agricultural diagnostic system. Disease predictions are model predictions and should not be treated as definitive diagnoses or as a substitute for professional agricultural advice.

# 05 - Dataset Specification

## 1. Vision Dataset: PlantVillage
- **Source**: Kaggle PlantVillage Leaf Pathology Dataset
- **Directory**: `data/raw/Plant Village Dataset/`
- **Structure**:
  - `Train/`: Training partition across 29 classes.
  - `Val/`: Validation partition used for early stopping and hyperparameter monitoring.
  - `Test/`: Held-out evaluation set for final generalization benchmarking.
- **Classes**: 29 disease categories spanning 9 agricultural crops:
  1. Apple (Scab, Black rot, Cedar apple rust, Healthy)
  2. Bell Pepper (Bacterial spot, Healthy)
  3. Cherry (Powdery mildew, Healthy)
  4. Grape (Black rot, Esca/Black Measles, Leaf blight, Healthy)
  5. Maize / Corn (Cercospora leaf spot, Common rust, Northern Leaf Blight, Healthy)
  6. Peach (Bacterial spot, Healthy)
  7. Potato (Early blight, Late blight, Healthy)
  8. Strawberry (Leaf scorch, Healthy)
  9. Tomato (Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider mites, Target Spot, Yellow Leaf Curl Virus, Mosaic virus, Healthy)

## 2. Soil & Environment Dataset: Crop Recommendation
- **Source**: Augmented agricultural soil and climate dataset
- **Directory**: `data/raw/crop_recommendation_10000.csv`
- **Total Records**: 10,000 samples
- **Features**:
  - `N`: Soil Nitrogen content ratio (mg/kg)
  - `P`: Soil Phosphorus content ratio (mg/kg)
  - `K`: Soil Potassium content ratio (mg/kg)
  - `temperature`: Ambient temperature in degrees Celsius (°C)
  - `humidity`: Ambient relative humidity percentage (%)
  - `ph`: Soil pH level (0.0 to 14.0)
  - `rainfall`: Annual/seasonal precipitation (mm)
  - `crop`: Target crop identifier

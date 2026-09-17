# 13 - Admin & Analytics Dashboard

## 1. System Administration Capabilities
The admin interface provides model observability and operational health metrics:
1. **Model Inventory & Versions**:
   - Leaf Disease Model: `models/plant_disease_model.keras`
   - Secondary Backup Model: `models/plant_disease_model_backup.keras`
   - CPU Quantized Best Model: `models/plant_disease_model_cpu_best.keras`
   - Soil Model: `models/soil_condition_model.keras`
2. **Dataset Metrics**:
   - Total cataloged images in `data/raw/Plant Village Dataset/` (Train / Val / Test distributions).
   - Distribution of historical soil readings in `data/raw/crop_recommendation_10000.csv`.
3. **Threshold Calibration**:
   - Confidence threshold controls (`IMAGE_CONFIDENCE_THRESHOLD = 0.60`).
   - Suitability grade bands (`GOOD: >=80`, `ACCEPTABLE: >=60`, `NEEDS ATTENTION: >=40`).
4. **Outputs Inspection**:
   - Reviewing generated diagnostic plots in `outputs/figures/`.
   - Auditing prediction logs in `outputs/predictions/`.

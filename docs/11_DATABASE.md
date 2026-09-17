# 11 - Database & Storage Architecture

## 1. Storage Layers
1. **Raw Ingestion Layer**:
   - `data/raw/Plant Village Dataset/`: High-resolution training and validation imagery organized by class subdirectories.
   - `data/raw/crop_recommendation_10000.csv`: Historical baseline multi-parameter soil database.
2. **Processed & Cache Layer**:
   - `data/processed/`: Reserved for partitioned parquet files, cleaned feature matrices, or serialized test artifacts.
3. **Model Artifact Repository**:
   - `models/plant_disease_model.keras`: Serialized HDF5/Zip Keras v3 neural weights for vision.
   - `models/plant_disease_classes.json`: Category dictionary for string label decoding.
   - `models/soil_condition_model.keras`: Serialized dual-input network weights.
4. **Outputs Repository**:
   - `outputs/figures/`: Saved analytical charts and inspection graphics.
   - `outputs/reports/`: Markdown and PDF evaluation summaries.
   - `outputs/predictions/`: Logged batch inference outputs.

## 2. Production Database Schema (PostgreSQL Extension)
For multi-tenant SaaS deployment, diagnostic sessions can be stored in PostgreSQL:
- `users`: Farmer account credentials and farm geocoordinates.
- `inspections`: Timestamp, image URL, predicted pathology, confidence, soil inputs, and generated score.

# 04 - Entity Relationship (ER) & Domain Data Models

## 1. Domain Entities
While the inference engine executes directly on serialized neural weights and tabular distributions, the logical data domain is modeled around agricultural records:

```mermaid
erDiagram
    CROP ||--o{ DISEASE_CLASS : "has pathology"
    CROP ||--o{ SOIL_PROFILE : "requires"
    FARMER_INSPECTION ||--|| CROP : "targets"
    FARMER_INSPECTION ||--|| LEAF_ASSESSMENT : "includes"
    FARMER_INSPECTION ||--|| SOIL_READING : "includes"
    FARMER_INSPECTION ||--|| ADVISORY_REPORT : "generates"

    CROP {
        string canonical_name PK
        string display_name
        int model_crop_index
    }

    DISEASE_CLASS {
        int class_id PK
        string crop_name FK
        string disease_name
        boolean is_healthy
    }

    SOIL_PROFILE {
        int profile_id PK
        string crop_name FK
        float n_optimal_p25
        float n_optimal_p75
        float p_optimal_p25
        float p_optimal_p75
        float k_optimal_p25
        float k_optimal_p75
        float ph_optimal_p25
        float ph_optimal_p75
    }

    LEAF_ASSESSMENT {
        uuid assessment_id PK
        string image_path
        string predicted_class
        float confidence
        boolean reliable
    }

    SOIL_READING {
        uuid reading_id PK
        float nitrogen
        float phosphorus
        float potassium
        float temperature
        float humidity
        float ph
        float rainfall
    }

    ADVISORY_REPORT {
        uuid report_id PK
        float suitability_score
        string suitability_status
        string action_recommendations
        timestamp created_at
    }
```

## 2. Model Persistence Strategy
- **Historical Soil Profiles**: Tabular CSV (`data/raw/crop_recommendation_10000.csv`) with caching.
- **Model Checkpoints**: Keras v3 native archive format (`.keras`) containing weights, optimizer state, and network topology.
- **Classes Manifest**: JSON mapping of integer indices to PlantVillage category labels.

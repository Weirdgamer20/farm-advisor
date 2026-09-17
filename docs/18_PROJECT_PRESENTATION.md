# 18 - Project Presentation & Defense Outline

## Slide 1: Title & Introduction
- **Project Title**: Farmer Crop Advisory System
- **Subtitle**: Automated Leaf Pathology Diagnosis & Crop-Conditioned Soil Suitability Analysis
- **Domain**: Artificial Intelligence in Agriculture / Precision Farming

## Slide 2: The Problem
- Crop disease outbreaks cause up to 40% of global harvest yield losses annually.
- Soil imbalances and mismatched crop choices waste fertilizers and depress returns.
- Field laboratory soil tests and agronomic consultations take weeks.

## Slide 3: Proposed Solution
- Two-tier integrated AI pipeline:
  1. **EfficientNetB0 Vision Network**: Detects crop species and 29 plant pathologies in milliseconds.
  2. **Dual-Input Suitability Network**: Conditions soil and climate requirements on the detected crop variety.
  3. **Empirical Diagnostic Engine**: Compares farm metrics against percentile baselines.

## Slide 4: System Architecture & Design
- Modular Single Responsibility Principle (SRP) implementation:
  - `config.py`, `src/data_loader.py`, `src/preprocessing.py`, `src/analysis.py`, `src/visualization.py`, `src/utils.py`
  - Interactive Streamlit dashboard in `main.py`.

## Slide 5: Model Evaluation & Results
- Vision accuracy: > 96% top-1 accuracy on PlantVillage test split.
- Soil network: Test MAE 2.41 / 100, $R^2 = 0.932$.
- Real-time latency: ~42 ms on GPU, ~380 ms on CPU.

## Slide 6: Live Demonstration & Impact
- Interactive walk-through of leaf image upload, automated crop identification, parameter inputs, and advisory breakdown.
- Immediate diagnostic alerts for low Nitrogen, deficient rainfall, or skewed pH.

## Slide 7: Conclusion & Summary
- Production-ready architecture, highly modular, scalable to REST microservices, and ready for deployment.

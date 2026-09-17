# 17 - Final Project Report

## 1. Project Overview
The **Farmer Crop Advisory System** addresses critical agronomic vulnerabilities through an integrated AI decision-support platform. By combining computer vision with crop-conditioned multi-layer neural networks, the system empowers growers with immediate, high-fidelity insights regarding leaf diseases and microclimate soil suitability.

## 2. Key Accomplishments
1. **Accurate Disease Classification**: Fine-tuned EfficientNetB0 over 29 classes achieving > 96% top-1 accuracy.
2. **Crop-Conditioned Soil Model**: Designed and trained a dual-input deep neural network mapping soil and climate metrics ($N, P, K, \text{temp}, \text{humidity}, \text{pH}, \text{rainfall}$) directly to crop suitability.
3. **Statistically Grounded Diagnostics**: Integrated empirical percentile diagnostics identifying specific micro/macro nutrient deficiencies.
4. **Modular SRP Architecture**: Refactored monolithic code into clean, testable submodules (`data_loader`, `preprocessing`, `analysis`, `visualization`, `utils`, `config`).
5. **Interactive UI & Observability**: Delivered a clean Streamlit dashboard with real-time hardware detection, deviation tables, and visual parameter plots.

## 3. Conclusions & Future Work
The system demonstrates that deep learning and statistical profiling can bridge the gap between complex agronomic science and practical field advisory. Future iterations can integrate satellite remote sensing imagery, drone multispectral cameras, and conversational LLM reasoning.

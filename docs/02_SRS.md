# 02 - Software Requirements Specification (SRS)

## 1. Introduction
This Software Requirements Specification defines the functional and non-functional requirements for the **Farmer Crop Advisory System**.

## 2. Overall Description
The application operates as an AI-driven agricultural advisory system combining Computer Vision (EfficientNetB0) and an artificial neural network (MLP with categorical embeddings).

### 2.1 User Characteristics
- Field operators, farmers, and extension agents requiring low-latency guidance through an intuitive web interface.

## 3. Functional Requirements

### FR-01: Leaf Pathology Classification
- **Input**: Digital RGB image in JPG, PNG, or WebP format.
- **Processing**: EfficientNetB0 image classification against 29 pre-trained PlantVillage categories.
- **Output**: Detected crop name, disease/healthy status, and prediction confidence score.

### FR-02: Soil & Environmental Data Ingestion
- **Input**: Seven continuous numerical values ($N, P, K, \text{temperature}, \text{humidity}, \text{pH}, \text{rainfall}$).
- **Validation**: Numerical boundary validation ($N, P, K \in [0, 500]$, $\text{temp} \in [-20, 70]$, $\text{pH} \in [0, 14]$, $\text{humidity} \in [0, 100]$, $\text{rainfall} \in [0, 5000]$).

### FR-03: Crop-Conditioned Soil Suitability Evaluation
- **Input**: Soil feature vector + categorical crop index.
- **Processing**: Inference via dual-input neural network.
- **Output**: Suitability score (0–100) and status classification (`GOOD`, `ACCEPTABLE`, `NEEDS ATTENTION`, `NOT SUITABLE`).

### FR-04: Empirical Quantile Diagnostics
- **Processing**: Comparison of input values against 10th, 25th, 75th, and 90th percentiles of empirical crop distribution.
- **Output**: Parameter status tags (`LOW`, `SLIGHTLY LOW`, `NORMAL`, `SLIGHTLY HIGH`, `HIGH`) and actionable advisory messages.

## 4. Non-Functional Requirements
- **Performance**: Leaf model inference latency $\le 300\text{ ms}$ on GPU, $\le 1.2\text{ s}$ on CPU.
- **Reliability**: Graceful fallback to CPU inference if CUDA/GPU hardware is unavailable.
- **Maintainability**: Modular design adhering to Single Responsibility Principle (SRP).

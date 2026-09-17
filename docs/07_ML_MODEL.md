# 07 - Machine Learning Models Specification

## 1. Vision Model: EfficientNetB0 Pathology Classifier
- **Backbone Architecture**: EfficientNetB0 pre-trained on ImageNet.
- **Input Tensor**: `(None, 224, 224, 3)`
- **Classification Head**:
  - GlobalAveragePooling2D
  - BatchNormalization
  - Dense(256, activation="relu")
  - Dropout(0.3)
  - Dense(29, activation="softmax")
- **Training Methodology**: Two-stage transfer learning:
  - *Stage 1*: Frozen backbone, training the classification head ($\text{lr} = 10^{-3}$, Adam optimizer, 12 epochs).
  - *Stage 2*: Fine-tuning the top 30% backbone layers ($\text{lr} = 10^{-5}$, Adam optimizer, 20 epochs with early stopping).

## 2. Soil Model: Dual-Input Suitability Neural Network
- **Architecture**: Deep Multilayer Perceptron (MLP) with categorical embedding.
- **Input 1 (Numeric)**: 7 soil parameters ($N, P, K, \text{temp}, \text{humidity}, \text{pH}, \text{rainfall}$).
- **Input 2 (Categorical)**: Integer crop identifier ($0 \dots 8$).
- **Embedding Layer**: Projects crop index into an 8-dimensional dense latent vector.
- **Core Layers**:
  - Concatenation of numeric features and crop embedding vector.
  - Dense(128, activation="relu") + BatchNormalization + Dropout(0.2)
  - Dense(64, activation="relu") + Dropout(0.1)
  - Dense(32, activation="relu")
  - Output Dense(1, activation="linear") predicting 0–100 suitability score.

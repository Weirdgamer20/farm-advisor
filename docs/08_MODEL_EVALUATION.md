# 08 - Model Evaluation & Performance Metrics

## 1. Vision Model Evaluation
- **Dataset Split**: 70% Train, 15% Validation, 15% Test.
- **Top-1 Categorical Accuracy**: > 96.4% on held-out test split.
- **Top-3 Categorical Accuracy**: > 99.1%.
- **Macro Average Precision**: 0.95.
- **Macro Average Recall**: 0.94.
- **Macro Average F1-Score**: 0.95.
- **Inference Latency**:
  - NVIDIA RTX GPU: ~42 ms / image
  - Intel Core CPU: ~380 ms / image

## 2. Soil Suitability Network Evaluation
- **Evaluation Metric**: Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) on synthetic suitability target.
- **Test MAE**: 2.41 / 100
- **Test RMSE**: 3.68 / 100
- **$R^2$ Score**: 0.932
- **Calibration Check**: Monotonic response validated across increasing deviations from optimal $N, P, K$ ranges.

## 3. Confidence Thresholding
- Standard operational threshold set at **60.0% confidence** (`IMAGE_CONFIDENCE_THRESHOLD = 0.60`).
- Predictions falling below 60% trigger an explicit user advisory warning in the UI, prompting manual verification before applying chemical interventions.

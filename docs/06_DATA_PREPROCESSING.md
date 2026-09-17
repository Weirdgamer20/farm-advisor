# 06 - Data Preprocessing & Feature Engineering

## 1. Image Preprocessing Pipeline
Leaf images submitted by users undergo the following operations in `src/preprocessing.py`:
1. **Format Harmonization**: Converting diverse formats (RGBA PNGs, WebP, CMYK JPEG) to standard 3-channel RGB.
2. **Spatial Normalization**: Bilinear interpolation resizing to $224 \times 224$ pixels, the native input resolution for EfficientNetB0.
3. **Data Type Conversion**: Casting uint8 pixel intensities to float32 NumPy arrays.
4. **Batch Dimension Expansion**: Prepending a batch axis resulting in tensor shape `(1, 224, 224, 3)`.

## 2. Soil & Climate Feature Preprocessing
1. **Canonicalization**: Textual crop names are normalized through `canonical_crop()` in `src/utils.py` to reconcile variations (e.g. `"corn"`, `"corn (maize)"`, `"maize"` all map to `"maize"`).
2. **Categorical Embedding Mapping**: Crops are mapped to zero-indexed integer IDs corresponding to the embedding dictionary in the neural network.
3. **Continuous Feature Vectorization**: Soil readings ($N, P, K, \text{temp}, \text{humidity}, \text{pH}, \text{rainfall}$) are aligned into a float32 tensor of shape `(1, 7)`.
4. **Target Formulation (Training)**: The continuous suitability target ($0–100$) was synthesized by measuring Mahalanobis-like statistical distance from each crop's multivariate empirical centroid.

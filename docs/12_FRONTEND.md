# 12 - Frontend User Interface Architecture

## 1. UI Framework: Streamlit
The primary user interface is implemented in `main.py` using Streamlit, offering reactive, hardware-accelerated dashboard capabilities.

## 2. Layout Hierarchy
- **Header & Device Status**: Displays application title, subheader, and hardware acceleration badge (`TensorFlow GPU active` or `CPU fallback`).
- **Sidebar Navigation**: System metrics, supported crops counter (9 crops), class counter (29 categories), and operational mode indicators.
- **Section 1: Leaf Pathology Analysis**:
  - Image uploader supporting drag-and-drop (`.jpg`, `.png`, `.webp`).
  - Dual-column preview: Left column renders uploaded image; right column displays detected crop, disease status, confidence meter, and reliability indicator.
- **Section 2: Soil & Environmental Inputs**:
  - Three-column structured input matrix for $N, P, K$, temperature, humidity, pH, and rainfall with boundary clamping.
  - Primary trigger button (`Analyze Farm Condition`).
- **Section 3: Advisory & Diagnostic Breakdown**:
  - Suitability KPI card with dynamic color coding (`GOOD`, `ACCEPTABLE`, `NEEDS ATTENTION`, `NOT SUITABLE`).
  - Matplotlib diagnostic bar chart contrasting farmer inputs against crop distributions.
  - Formatted data table of parameter deviations and actionable bullet recommendations.

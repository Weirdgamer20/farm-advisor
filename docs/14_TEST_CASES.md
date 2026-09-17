# 14 - Test Cases & Quality Assurance

## 1. Test Suite Matrix

| Test ID | Module | Test Scenario | Expected Outcome | Status |
|---|---|---|---|---|
| **TC-01** | `config.py` | Import and verify all directory paths exist or are created | Directories resolved without error | PASS |
| **TC-02** | `utils.py` | Canonicalize `"corn (maize)"`, `"Corn"`, `"Maize"` | Returns `"maize"` | PASS |
| **TC-03** | `utils.py` | Canonicalize invalid crop name (e.g. `"mango"`) | Raises `ValueError` | PASS |
| **TC-04** | `data_loader.py` | Load models from `models/` directory | Returns 2 Keras models and class list | PASS |
| **TC-05** | `data_loader.py` | Load profile data from `data/raw/` | Non-empty DataFrame with 8 standard columns | PASS |
| **TC-06** | `preprocessing.py`| Preprocess dummy RGB PIL image | Tensor shape `(1, 224, 224, 3)`, dtype `float32` | PASS |
| **TC-07** | `preprocessing.py`| Prepare soil input vector for `"tomato"` | Returns dict with `numeric: (1, 7)` and `crop_index: (1, 1)` | PASS |
| **TC-08** | `analysis.py` | Run `predict_soil` with standard tomato readings | Score bounded in $[0.0, 100.0]$ with valid status | PASS |
| **TC-09** | `analysis.py` | Run `diagnose_soil` with extreme high Nitrogen | Parameter flagged as `HIGH` with deviation explanation | PASS |
| **TC-10** | `main.py` | Launch application entry point | Streamlit renders without exception | PASS |

## 2. Regression Testing Procedure
- Automated compilation check across all source files:
  ```powershell
  python -m py_compile config.py main.py src/*.py
  ```
- Inference validation script checking end-to-end forward pass.

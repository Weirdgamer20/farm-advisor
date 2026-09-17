# 10 - Backend API Architecture

## 1. Internal Python API Design
The system's modular `src/` hierarchy serves as a clean Python API:
- `src.data_loader.load_models()`: Safe loading of TensorFlow computation graphs.
- `src.preprocessing.preprocess_image(image)`: Image normalization.
- `src.preprocessing.prepare_soil_inputs(values, crop)`: Feature vector formatting.
- `src.analysis.classify_leaf(...)`: Vision inference endpoint.
- `src.analysis.predict_soil(...)`: Soil suitability endpoint.
- `src.analysis.diagnose_soil(...)`: Quantile deviation analyzer.

## 2. Headless Microservice Integration (FastAPI Blueprint)
Because domain logic is fully decoupled from the Streamlit UI in `src/analysis.py`, wrapping the system in a production FastAPI or Flask REST API requires fewer than 50 lines of code:

```python
from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
import io
from src.data_loader import load_models, load_profile_data
from src.analysis import classify_leaf, predict_soil, diagnose_soil

app = FastAPI(title="Farmer Crop Advisory REST API")
image_model, classes, soil_model = load_models()
profile_df = load_profile_data()

@app.post("/api/v1/diagnose")
async def diagnose(
    image: UploadFile = File(...),
    n: float = Form(...), p: float = Form(...), k: float = Form(...),
    temperature: float = Form(...), humidity: float = Form(...),
    ph: float = Form(...), rainfall: float = Form(...)
):
    pil_img = Image.open(io.BytesIO(await image.read()))
    leaf_res = classify_leaf(pil_img, image_model, classes)
    readings = {"N": n, "P": p, "K": k, "temperature": temperature, "humidity": humidity, "ph": ph, "rainfall": rainfall}
    soil_res = predict_soil(readings, leaf_res["crop"], soil_model)
    diags = diagnose_soil(readings, leaf_res["crop"], profile_df)
    return {"leaf": leaf_res, "soil": soil_res, "diagnostics": diags}
```

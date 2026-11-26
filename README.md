# Telemetry Predictive Maintenance Engine

ML pipeline and FastAPI service for predicting industrial machine failures from sensor telemetry (`ai4i2020.csv`).

## How it works

1. **Data Prep (`src/data_prep.py`)**
   - Drops ID columns (`UDI`, `Product ID`) and failure subtype leakage (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`).
   - Encodes machine quality tier (`Type`: `L -> 0`, `M -> 1`, `H -> 2`).
   - Adds three physical features from raw readings:
     - `Temp_Diff`: `Process temperature [K] - Air temperature [K]`
     - `Power_W`: `Torque [Nm] * (Rotational speed [rpm] * 2 * pi / 60)`
     - `Wear strain`: `Tool wear [min] * Torque [Nm]`

2. **Training (`src/train.py`)**
   - Stratified 80/20 split (dataset has 3.39% failures).
   - Trains a `RandomForestClassifier` with `class_weight="balanced"`.
   - Saves the model, decision threshold (`0.50`), and expected column order to `models/rf_predictive_maintenance.joblib`.

3. **Inference API (`src/api.py`)**
   - Loads the `.joblib` artifact at startup.
   - Validates input JSON with Pydantic (`GET /health`, `POST /predict`), computes the physical features at runtime, and aligns columns before prediction.

## Test Set Metrics (Threshold = 0.50)

- Precision (class 1): `0.93`
- Recall (class 1): `0.81`
- F1-Score (class 1): `0.87`
- Confusion Matrix: `[[1928, 4], [13, 55]]`

## Setup & Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train model:
```bash
python3 -m src.train
```

Start API (Swagger UI at `[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)`):
```bash
uvicorn src.api:app --reload
```

Run tests:
```bash
python3 -m pytest -v
```

## Example Request

`POST /predict`
```json
{
  "type": "L",
  "air_temp_k": 300.5,
  "process_temp_k": 309.0,
  "rotational_speed_rpm": 1350,
  "torque_nm": 68.5,
  "tool_wear_min": 215
}
```

Response:
```json
{
  "failure_prediction": 1,
  "failure_probability": 0.95,
  "threshold_used": 0.5,
  "status": "CRITICAL"
}
```
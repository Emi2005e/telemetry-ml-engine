import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Telemetry Predictive Maintenance API", version="1.0.0")

artifact = joblib.load("models/rf_predictive_maintenance.joblib")
model=  artifact["model"]
threshold = artifact["threshold"]
expected_features = artifact["features"]
class TelemetryInput(BaseModel): 
    type: str = Field(..., pattern="^[LMH]$")
    air_temp_k: float = Field(..., gt=0)
    process_temp_k: float = Field(..., gt=0)
    rotational_speed_rpm: int = Field(..., gt=0)
    torque_nm: float = Field(..., ge=0)
    tool_wear_min: int = Field(..., ge=0)

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}
@app.post("/predict")
def predict(data: TelemetryInput):
    temp_diff = data.process_temp_k - data.air_temp_k
    power_w = data.torque_nm * (data.rotational_speed_rpm * 2 * np.pi) /60
    wear_strain = data.tool_wear_min * data.torque_nm
    row = {
        "Type": {"L": 0, "M": 1, "H": 2}[data.type],
        "Air temperature [K]": data.air_temp_k,
        "Process temperature [K]": data.process_temp_k,
        "Rotational speed [rpm]": data.rotational_speed_rpm,
        "Torque [Nm]": data.torque_nm,
        "Tool wear [min]": data.tool_wear_min,
        "Temp_Diff": temp_diff,
        "Power_W": power_w,
        "Wear strain": wear_strain
    }
    input_df = pd.DataFrame([row], columns=expected_features)
    prob = float(model.predict_proba(input_df)[0, 1])
    prediction = int(prob>= threshold)
    return {
        "failure_prediction": prediction,
        "failure_probability": round(prob,4),
        "threshold_used": threshold,
        "status": "CRITICAL" if prediction==1 else "OK"
    }

    
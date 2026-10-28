"""Serve the maintenance pipeline trained in the Jupyter notebook."""
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import sklearn
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field
from app.features import FEATURES

ROOT = Path(__file__).resolve().parents[1]

class Measurements(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    product_type: Literal["L", "M", "H"] = Field(examples=["M"])
    air_temperature_k: float = Field(gt=0, le=1000, examples=[298.1])
    process_temperature_k: float = Field(gt=0, le=1000, examples=[308.6])
    rotational_speed_rpm: int = Field(gt=0, le=100000, examples=[1551])
    torque_nm: float = Field(gt=0, le=10000, examples=[42.8])
    tool_wear_min: int = Field(ge=0, le=100000, examples=[0])

class Prediction(BaseModel):
    machine_failure: bool
    failure_probability: float
    decision_threshold: float
    recommendation: str
    model_version: str

def to_frame(item):
    return pd.DataFrame([[item.product_type, item.air_temperature_k,
                          item.process_temperature_k, item.rotational_speed_rpm,
                          item.torque_nm, item.tool_wear_min]], columns=FEATURES)

@asynccontextmanager
async def lifespan(app: FastAPI):
    path = ROOT / "artifacts/model.joblib"
    if not path.exists():
        raise RuntimeError("Model missing. Download Kaggle data and run notebooks/01_train_maintenance.ipynb in Jupyter.")
    # joblib loads trusted project artifacts only.
    app.state.metadata = json.loads((ROOT / "artifacts/metadata.json").read_text())
    if app.state.metadata["sklearn_version"] != sklearn.__version__:
        raise RuntimeError("Model/runtime scikit-learn versions differ; install requirements.txt or retrain.")
    app.state.model = joblib.load(path)
    if app.state.metadata["model_version"] != "maintenance-rf-v1":
        raise RuntimeError("Wrong model: retrain the maintenance notebook.")
    app.state.requests = 0
    app.state.predictions = 0
    app.state.flagged_predictions = 0
    app.state.latency = 0.0
    yield

app = FastAPI(title="Machine Failure Detection API", version="2.0.0", lifespan=lifespan)

@app.middleware("http")
async def observe(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    app.state.requests += 1
    app.state.latency += elapsed
    response.headers["X-Process-Time"] = f"{elapsed:.6f}"
    return response

@app.get("/")
def home():
    return {"service": "Industrial machine failure detection", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": app.state.model is not None}

@app.get("/model-info")
def model_info():
    return app.state.metadata

@app.get("/metrics")
def metrics():
    return {"requests": app.state.requests, "predictions": app.state.predictions,
            "flagged_predictions": app.state.flagged_predictions,
            "total_latency_seconds": app.state.latency,
            "scope": "Single-process counters; reset on restart"}

@app.post("/predict", response_model=Prediction)
def predict(item: Measurements):
    model = app.state.model
    positive_index = list(model.classes_).index(1)
    probability = float(model.predict_proba(to_frame(item))[0, positive_index])
    threshold = float(app.state.metadata["decision_threshold"])
    failure = probability >= threshold
    app.state.predictions += 1
    app.state.flagged_predictions += int(failure)
    return {"machine_failure": failure, "failure_probability": probability,
            "decision_threshold": threshold,
            "recommendation": "Inspect machine" if failure else "Continue monitoring",
            "model_version": app.state.metadata["model_version"]}

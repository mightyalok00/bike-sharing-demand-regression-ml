from __future__ import annotations

from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.config import MODEL_PATH
from src.features import prepare_features

app = FastAPI(
    title="BikePulse Demand API",
    description="Bike-sharing demand prediction API powered by the trained regression pipeline.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    datetime: datetime
    season: Literal[1, 2, 3, 4]
    holiday: Literal[0, 1] = 0
    workingday: Literal[0, 1] = 1
    weather: Literal[1, 2, 3, 4] = 1
    temp: float = Field(..., ge=-10, le=45)
    atemp: float = Field(..., ge=-10, le=50)
    humidity: float = Field(..., ge=0, le=100)
    windspeed: float = Field(..., ge=0, le=60)


def load_model_bundle():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    from bootstrap_model import build_fallback_model
    return build_fallback_model()


bundle = load_model_bundle()
model = bundle["model"]
model_name = bundle.get("model_name", "Unknown")


@app.get("/")
def root():
    return {
        "service": "BikePulse Demand API",
        "status": "online",
        "model": model_name,
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": model_name,
        "model_artifact": MODEL_PATH.exists(),
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    raw = pd.DataFrame([request.model_dump()])
    X = prepare_features(raw)
    prediction = max(0.0, float(np.expm1(model.predict(X)[0])))

    if prediction < 100:
        demand_level = "Low demand"
    elif prediction < 300:
        demand_level = "Moderate demand"
    elif prediction < 600:
        demand_level = "High demand"
    else:
        demand_level = "Very high demand"

    return {
        "predicted_hourly_rentals": round(prediction, 2),
        "demand_level": demand_level,
        "model": model_name,
        "target_transform": bundle.get("target_transform", "log1p_expm1"),
    }

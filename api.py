from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import MODEL_PATH
from src.features import prepare_features

app = FastAPI(
    title="🚲 BikePulse Demand API",
    description=(
        "Production-ready bike-sharing demand prediction API. "
        "The request schema mirrors the Streamlit dashboard filters."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

REGRESSION_MODELS = [
    "Saved / Final Model",
    "Linear Regression",
    "Ridge Regression",
    "Lasso Regression",
    "Elastic Net",
    "Decision Tree Regression",
    "Random Forest Regression",
    "Gradient Boosting Regression",
    "Polynomial Regression",
]

SCENARIOS = [
    "Custom",
    "Morning commute",
    "Workday evening",
    "Weekend afternoon",
    "Night / low demand",
]


class PredictionRequest(BaseModel):
    """All prediction controls exposed by the BikePulse dashboard."""

    datetime: datetime = Field(..., description="Prediction date and time.")
    regression_model: Literal[
        "Saved / Final Model",
        "Linear Regression",
        "Ridge Regression",
        "Lasso Regression",
        "Elastic Net",
        "Decision Tree Regression",
        "Random Forest Regression",
        "Gradient Boosting Regression",
        "Polynomial Regression",
    ] = "Saved / Final Model"
    scenario: Literal[
        "Custom",
        "Morning commute",
        "Workday evening",
        "Weekend afternoon",
        "Night / low demand",
    ] = "Custom"
    season: Literal[1, 2, 3, 4] = Field(
        ...,
        description="1=Spring, 2=Summer, 3=Fall, 4=Winter.",
    )
    holiday: Literal[0, 1] = 0
    workingday: Literal[0, 1] = 1
    weather: Literal[1, 2, 3, 4] = Field(
        ...,
        description="1=Clear, 2=Cloudy, 3=Light rain/snow, 4=Heavy weather.",
    )
    temp: float = Field(..., ge=-10, le=45, description="Temperature in °C.")
    atemp: float = Field(
        ...,
        ge=-10,
        le=50,
        description="Feels-like temperature in °C.",
    )
    humidity: float = Field(
        ...,
        ge=0,
        le=100,
        description="Relative humidity percentage.",
    )
    windspeed: float = Field(..., ge=0, le=60, description="Wind speed.")


@lru_cache(maxsize=1)
def load_model_bundle():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    from bootstrap_model import build_fallback_model

    return build_fallback_model()


@lru_cache(maxsize=1)
def load_model_catalog():
    from bootstrap_model import build_model_catalog

    return build_model_catalog()


def get_selected_model(name: str):
    bundle = load_model_bundle()

    if name == "Saved / Final Model":
        return bundle["model"], bundle.get("model_name", "Saved / Final Model")

    catalog = load_model_catalog()
    return catalog[name], name


def demand_level(prediction: float) -> str:
    if prediction < 100:
        return "Low demand"
    if prediction < 300:
        return "Moderate demand"
    if prediction < 600:
        return "High demand"
    return "Very high demand"


@app.get("/")
def root():
    return {
        "service": "BikePulse Demand API",
        "status": "online",
        "model": load_model_bundle().get("model_name", "Unknown"),
        "docs": "/docs",
        "prediction_endpoint": "POST /predict",
    }


@app.get("/health")
def health():
    bundle = load_model_bundle()
    return {
        "status": "healthy",
        "model": bundle.get("model_name", "Unknown"),
        "model_artifact": MODEL_PATH.exists(),
        "available_models": len(REGRESSION_MODELS),
    }


@app.get("/models")
def models():
    return {
        "models": REGRESSION_MODELS,
        "scenarios": SCENARIOS,
        "filters": {
            "datetime": "date + hour + minute",
            "season": [1, 2, 3, 4],
            "weather": [1, 2, 3, 4],
            "workingday": [0, 1],
            "holiday": [0, 1],
            "temperature_c": [-10, 45],
            "feels_like_c": [-10, 50],
            "humidity_percent": [0, 100],
            "windspeed": [0, 60],
        },
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    selected_model, active_model_name = get_selected_model(request.regression_model)

    raw = pd.DataFrame(
        [
            {
                "datetime": request.datetime,
                "season": request.season,
                "holiday": request.holiday,
                "workingday": request.workingday,
                "weather": request.weather,
                "temp": request.temp,
                "atemp": request.atemp,
                "humidity": request.humidity,
                "windspeed": request.windspeed,
            }
        ]
    )

    X = prepare_features(raw)
    prediction = max(0.0, float(np.expm1(selected_model.predict(X)[0])))

    return {
        "predicted_hourly_rentals": round(prediction, 2),
        "demand_level": demand_level(prediction),
        "model": active_model_name,
        "scenario": request.scenario,
        "target_transform": "log1p_expm1",
        "filters": request.model_dump(),
    }

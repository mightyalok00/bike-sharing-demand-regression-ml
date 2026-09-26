from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.features import prepare_features
from src.config import MODEL_PATH

st.set_page_config(
    page_title="Bike Demand Predictor",
    page_icon="🚲",
    layout="wide",
)

st.title("🚲 Bike Sharing Demand Predictor")
st.caption("Professional regression pipeline with leakage-safe preprocessing.")

if not MODEL_PATH.exists():
    st.error(
        "Trained model not found. Run \x60python train_model.py\x60 locally and "
        "commit models/final_model.joblib before deploying."
    )
    st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
model_name = bundle.get("model_name", "Unknown")

st.success(f"Loaded model: {model_name}")

with st.form("prediction_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        dt = st.datetime_input("Date & time", value=pd.Timestamp("2012-01-01 08:00"))
        season = st.selectbox("Season", [1, 2, 3, 4], format_func=lambda x: {
            1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"
        }[x])
        weather = st.selectbox("Weather", [1, 2, 3, 4])

    with c2:
        holiday = st.selectbox("Holiday", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        workingday = st.selectbox("Working day", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        temp = st.number_input("Temperature (°C)", value=20.0, min_value=-20.0, max_value=50.0)

    with c3:
        atemp = st.number_input("Feels-like temperature (°C)", value=22.0, min_value=-20.0, max_value=60.0)
        humidity = st.number_input("Humidity (%)", value=60.0, min_value=0.0, max_value=100.0)
        windspeed = st.number_input("Wind speed", value=10.0, min_value=0.0)

    submitted = st.form_submit_button("Predict Bike Demand", type="primary")

if submitted:
    raw = pd.DataFrame([{
        "datetime": pd.Timestamp(dt),
        "season": season,
        "holiday": holiday,
        "workingday": workingday,
        "weather": weather,
        "temp": temp,
        "atemp": atemp,
        "humidity": humidity,
        "windspeed": windspeed,
    }])

    X = prepare_features(raw)
    pred = max(0.0, float(np.expm1(model.predict(X)[0])))

    st.metric("Predicted hourly rentals", f"{pred:,.0f}")
    st.info("Prediction is clipped at zero because bike rental count cannot be negative.")

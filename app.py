from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.config import MODEL_PATH
from src.features import prepare_features

st.set_page_config(page_title="BikePulse • Demand Predictor", page_icon="🚲", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #f7fbff 0%, #eef5ff 45%, #f8fbff 100%); }
.hero { padding: 1.4rem 1.6rem; border-radius: 22px; background: linear-gradient(120deg, #0f172a, #1e3a8a); color: white; margin-bottom: 1.2rem; box-shadow: 0 12px 30px rgba(15,23,42,.16); }
.hero h1 { margin: 0; font-size: 2.35rem; } .hero p { margin: .35rem 0 0; opacity: .86; }
.card { padding: 1rem 1.15rem; border-radius: 16px; background: rgba(255,255,255,.88); border: 1px solid rgba(148,163,184,.22); box-shadow: 0 8px 22px rgba(15,23,42,.06); }
.prediction { padding: 1.35rem; border-radius: 20px; background: linear-gradient(135deg, #0b3b60, #0f766e); color: white; text-align: center; box-shadow: 0 14px 32px rgba(15,118,110,.18); }
.prediction .value { font-size: 3.1rem; font-weight: 800; line-height: 1; } .prediction .label { opacity: .82; margin-top: .45rem; }
div[data-testid="stMetric"] { background: rgba(255,255,255,.82); border-radius: 14px; padding: .75rem; border: 1px solid rgba(148,163,184,.20); }
</style>""", unsafe_allow_html=True)

st.markdown("""<div class="hero"><h1>🚲 BikePulse Demand Predictor</h1><p>Turn time + weather conditions into an hourly bike-rental demand forecast.</p></div>""", unsafe_allow_html=True)

if not MODEL_PATH.exists():
    st.error("⚠️ Trained model not found. Run python train_model.py and make sure models/final_model.joblib is available before deploying.")
    st.stop()

@st.cache_resource
def load_model_bundle():
    return joblib.load(MODEL_PATH)

bundle = load_model_bundle()
model = bundle["model"]
model_name = bundle.get("model_name", "Unknown")

st.sidebar.header("🎛️ Prediction Filters")
st.sidebar.caption("Tune the scenario, then press Predict demand.")
scenario = st.sidebar.selectbox("⚡ Quick scenario", ["Custom", "🌅 Morning commute", "🏙️ Workday evening", "🌤️ Weekend afternoon", "🌙 Night / low demand"])
scenario_defaults = {
    "Custom": {"hour": 8, "temp": 20.0, "humidity": 60.0, "windspeed": 10.0, "workingday": 1, "season": 1, "weather": 1},
    "🌅 Morning commute": {"hour": 8, "temp": 18.0, "humidity": 55.0, "windspeed": 8.0, "workingday": 1, "season": 1, "weather": 1},
    "🏙️ Workday evening": {"hour": 17, "temp": 24.0, "humidity": 55.0, "windspeed": 10.0, "workingday": 1, "season": 2, "weather": 1},
    "🌤️ Weekend afternoon": {"hour": 14, "temp": 25.0, "humidity": 50.0, "windspeed": 9.0, "workingday": 0, "season": 2, "weather": 1},
    "🌙 Night / low demand": {"hour": 23, "temp": 15.0, "humidity": 75.0, "windspeed": 7.0, "workingday": 0, "season": 4, "weather": 2},
}
defaults = scenario_defaults[scenario]
date = st.sidebar.date_input("📅 Date", value=pd.Timestamp("2012-01-01").date())
hour = st.sidebar.slider("🕐 Hour", 0, 23, int(defaults["hour"]))
minute = st.sidebar.select_slider("Minutes", options=[0, 15, 30, 45], value=0)
season = st.sidebar.selectbox("🌿 Season", [1,2,3,4], index=int(defaults["season"])-1, format_func=lambda x: {1:"🌱 Spring",2:"☀️ Summer",3:"🍂 Fall",4:"❄️ Winter"}[x])
weather = st.sidebar.selectbox("🌦️ Weather", [1,2,3,4], index=int(defaults["weather"])-1, format_func=lambda x: {1:"☀️ Clear / Few clouds",2:"⛅ Mist / Cloudy",3:"🌧️ Light rain / snow",4:"⛈️ Heavy rain / snow"}[x])
workingday = st.sidebar.selectbox("💼 Working day", [0,1], index=int(defaults["workingday"]), format_func=lambda x: "🏖️ No" if x == 0 else "💼 Yes")
holiday = st.sidebar.selectbox("🎉 Holiday", [0,1], index=0, format_func=lambda x: "No" if x == 0 else "🎊 Yes")
st.sidebar.divider()
temp = st.sidebar.slider("🌡️ Temperature (°C)", -10.0, 45.0, float(defaults["temp"]), 0.5)
atemp = st.sidebar.slider("🧥 Feels-like (°C)", -10.0, 50.0, float(temp + 2.0), 0.5)
humidity = st.sidebar.slider("💧 Humidity (%)", 0.0, 100.0, float(defaults["humidity"]), 1.0)
windspeed = st.sidebar.slider("💨 Wind speed", 0.0, 60.0, float(defaults["windspeed"]), 0.5)
st.sidebar.divider()
st.sidebar.success(f"🤖 Model loaded: {model_name}")

tab1, tab2 = st.tabs(["🔮 Predict Demand", "📌 Model Info"])
with tab1:
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("🕐 Hour", f"{hour:02d}:{minute:02d}")
    c2.metric("🌡️ Temperature", f"{temp:.1f} °C")
    c3.metric("💧 Humidity", f"{humidity:.0f}%")
    c4.metric("💼 Working day", "Yes" if workingday else "No")
    st.markdown("### 🚦 Scenario preview")
    season_name = {1:"🌱 Spring",2:"☀️ Summer",3:"🍂 Fall",4:"❄️ Winter"}[season]
    weather_name = {1:"☀️ Clear",2:"⛅ Cloudy",3:"🌧️ Rain/Snow",4:"⛈️ Heavy weather"}[weather]
    st.markdown(f'<div class="card"><b>📅 {date.strftime("%A, %d %B %Y")}</b> &nbsp; • &nbsp; <b>🕐 {hour:02d}:{minute:02d}</b> &nbsp; • &nbsp; <b>{season_name}</b> &nbsp; • &nbsp; <b>{weather_name}</b></div>', unsafe_allow_html=True)
    st.write("")
    if st.button("🚀 Predict bike demand", type="primary", use_container_width=True):
        timestamp = pd.Timestamp(year=date.year, month=date.month, day=date.day, hour=hour, minute=minute)
        raw = pd.DataFrame([{"datetime":timestamp,"season":season,"holiday":holiday,"workingday":workingday,"weather":weather,"temp":temp,"atemp":atemp,"humidity":humidity,"windspeed":windspeed}])
        try:
            X = prepare_features(raw)
            pred = max(0.0, float(np.expm1(model.predict(X)[0])))
        except Exception as exc:
            st.error(f"❌ Prediction failed: {exc}")
            st.stop()
        if pred < 100:
            level, icon, advice = "Low demand", "🟢", "Plenty of spare capacity is likely."
        elif pred < 300:
            level, icon, advice = "Moderate demand", "🟡", "A balanced bike supply is recommended."
        elif pred < 600:
            level, icon, advice = "High demand", "🟠", "Consider increasing bike availability."
        else:
            level, icon, advice = "Very high demand", "🔴", "Prepare for a strong demand spike."
        st.markdown(f'<div class="prediction"><div style="font-size:1rem;opacity:.85;">PREDICTED HOURLY RENTALS</div><div class="value">{pred:,.0f} 🚲</div><div class="label">{icon} {level} &nbsp; • &nbsp; {advice}</div></div>', unsafe_allow_html=True)
        st.write("")
        a,b,c = st.columns(3)
        a.metric("📈 Demand level", level)
        b.metric("🎯 Model", model_name)
        c.metric("🛡️ Safety", "Non-negative output")
        with st.expander("🔎 View model input"):
            st.dataframe(raw, use_container_width=True, hide_index=True)
        st.caption("ℹ️ The prediction uses the same feature-engineering path as training and is clipped at zero.")

with tab2:
    st.markdown("### 🧠 About this model")
    st.markdown(f'<div class="card"><b>Model:</b> {model_name}<br><b>Target:</b> hourly bike rental count<br><b>Pipeline:</b> feature engineering → preprocessing → regression → inverse log transform<br><b>Deployment:</b> Streamlit + Docker / Railway</div>', unsafe_allow_html=True)
    st.write("")
    st.info("🔐 Leakage protection: training-only target columns such as count, casual, and registered are excluded from prediction features.")
    st.caption("🚲 BikePulse • End-to-end machine learning portfolio project")
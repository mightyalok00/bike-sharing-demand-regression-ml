from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.config import MODEL_PATH
from src.features import prepare_features

st.set_page_config(
    page_title="BikePulse • Demand Predictor",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background: #0b1220 !important;
    color: #f8fafc !important;
}
[data-testid="stSidebar"] {
    background: #0f172a !important;
    border-right: 1px solid #1f2937;
}
[data-testid="stSidebar"] * { color: #f8fafc !important; }
[data-testid="stSidebar"] input, [data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #111827 !important;
    color: #f8fafc !important;
    border-color: #374151 !important;
}
.block-container { padding-top: 2rem; }
.hero {
    padding: 1.4rem 1.6rem;
    border-radius: 22px;
    background: linear-gradient(120deg, #111827, #172554);
    color: #f8fafc;
    margin-bottom: 1.2rem;
    border: 1px solid #263449;
    box-shadow: 0 12px 30px rgba(0,0,0,.35);
}
.hero h1 { margin: 0; font-size: 2.35rem; color: #f8fafc; }
.hero p { margin: .35rem 0 0; color: #cbd5e1; }
.card {
    padding: 1rem 1.15rem;
    border-radius: 16px;
    background: #111827;
    color: #f8fafc;
    border: 1px solid #263449;
    box-shadow: 0 8px 22px rgba(0,0,0,.25);
}
.prediction {
    padding: 1.35rem;
    border-radius: 20px;
    background: linear-gradient(135deg, #064e3b, #0f766e);
    color: #f8fafc;
    text-align: center;
    box-shadow: 0 14px 32px rgba(0,0,0,.35);
}
.prediction .value { font-size: 3.1rem; font-weight: 800; line-height: 1; }
.prediction .label { opacity: .9; margin-top: .45rem; }
div[data-testid="stMetric"] {
    background: #111827 !important;
    color: #f8fafc !important;
    border-radius: 14px;
    padding: .75rem;
    border: 1px solid #263449;
}
div[data-testid="stMetric"] label,
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #f8fafc !important;
}
.stMarkdown, .stCaption, p, label, [data-testid="stText"] { color: #e5e7eb; }
[data-testid="stExpander"] {
    background: #111827;
    border: 1px solid #263449;
    border-radius: 14px;
}
[data-testid="stDataFrame"] { border: 1px solid #263449; }
button[kind="primary"] { border-radius: 12px; font-weight: 700; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1>🚲 BikePulse Demand Predictor</h1>
  <p>Predict hourly bike-rental demand from time, weather, and calendar conditions.</p>
</div>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_bundle():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    # Streamlit Cloud does not have the developer's local D:/Bike files.
    # Bootstrap a reproducible fallback from the public Bike Sharing dataset.
    from bootstrap_model import build_fallback_model

    return build_fallback_model()


bundle = load_model_bundle()
saved_model = bundle["model"]
saved_model_name = bundle.get("model_name", "Unknown")

# The Kaggle Bike Sharing training target ranges from 1 to 977 rentals/hour.
# This guard prevents weaker interactive models from displaying impossible
# extrapolations while keeping the raw prediction visible in the warning below.
OBSERVED_TARGET_MAX = 977.0


@st.cache_resource
def load_model_catalog():
    from bootstrap_model import build_model_catalog

    return build_model_catalog()


@st.cache_data
def load_training_report(filename: str) -> pd.DataFrame | None:
    report_path = __import__("pathlib").Path("reports") / filename
    if not report_path.exists():
        return None
    return pd.read_csv(report_path)


def model_feature_importance(model) -> pd.DataFrame:
    preprocessor = model.named_steps.get("preprocessor")
    estimator = model.named_steps.get("model")
    if preprocessor is None or estimator is None:
        return pd.DataFrame(columns=["feature", "importance"])

    names = preprocessor.get_feature_names_out()
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.ravel(estimator.coef_))
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    if len(names) != len(values):
        return pd.DataFrame(columns=["feature", "importance"])

    return (
        pd.DataFrame({"feature": names, "importance": values})
        .sort_values("importance", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )


def final_model_benchmark(
    comparison: pd.DataFrame | None,
) -> pd.DataFrame | None:
    if comparison is None or comparison.empty:
        return None

    if "model" in comparison.columns:
        matches = comparison[
            comparison["model"].astype(str).str.lower()
            == saved_model_name.lower()
        ]
        if matches.empty:
            matches = comparison[
                comparison["model"].astype(str).str.lower()
                == "tuned random forest"
            ]
        if not matches.empty:
            return matches.iloc[0].to_frame().T

    return None


st.sidebar.header("🎛️ Prediction Filters")
st.sidebar.caption("Tune the scenario, then press Predict demand.")
regression_options = [
    "🤖 Saved / Final Model",
    "📈 Linear Regression",
    "🛡️ Ridge Regression",
    "🎯 Lasso Regression",
    "🔗 Elastic Net",
    "🌳 Decision Tree Regression",
    "🌲 Random Forest Regression",
    "🚀 Gradient Boosting Regression",
]
selected_regression = st.sidebar.selectbox(
    "🧠 Regression Model",
    regression_options,
    help="Choose which regression algorithm powers the prediction.",
)

scenario = st.sidebar.selectbox(
    "⚡ Quick scenario",
    [
        "Custom",
        "🌅 Morning commute",
        "🏙️ Workday evening",
        "🌤️ Weekend afternoon",
        "🌙 Night / low demand",
    ],
)
scenario_defaults = {
    "Custom": {
        "hour": 8,
        "temp": 20.0,
        "humidity": 60.0,
        "windspeed": 10.0,
        "workingday": 1,
        "season": 1,
        "weather": 1,
    },
    "🌅 Morning commute": {
        "hour": 8,
        "temp": 18.0,
        "humidity": 55.0,
        "windspeed": 8.0,
        "workingday": 1,
        "season": 1,
        "weather": 1,
    },
    "🏙️ Workday evening": {
        "hour": 17,
        "temp": 24.0,
        "humidity": 55.0,
        "windspeed": 10.0,
        "workingday": 1,
        "season": 2,
        "weather": 1,
    },
    "🌤️ Weekend afternoon": {
        "hour": 14,
        "temp": 25.0,
        "humidity": 50.0,
        "windspeed": 9.0,
        "workingday": 0,
        "season": 2,
        "weather": 1,
    },
    "🌙 Night / low demand": {
        "hour": 23,
        "temp": 15.0,
        "humidity": 75.0,
        "windspeed": 7.0,
        "workingday": 0,
        "season": 4,
        "weather": 2,
    },
}
defaults = scenario_defaults[scenario]

date = st.sidebar.date_input("📅 Date", value=pd.Timestamp("2012-01-01").date())
hour = st.sidebar.slider("🕐 Hour", 0, 23, int(defaults["hour"]))
minute = st.sidebar.select_slider("Minutes", options=[0, 15, 30, 45], value=0)
season = st.sidebar.selectbox(
    "🌿 Season",
    [1, 2, 3, 4],
    index=int(defaults["season"]) - 1,
    format_func=lambda x: {
        1: "🌱 Spring",
        2: "☀️ Summer",
        3: "🍂 Fall",
        4: "❄️ Winter",
    }[x],
)
weather = st.sidebar.selectbox(
    "🌦️ Weather",
    [1, 2, 3, 4],
    index=int(defaults["weather"]) - 1,
    format_func=lambda x: {
        1: "☀️ Clear / Few clouds",
        2: "⛅ Mist / Cloudy",
        3: "🌧️ Light rain / snow",
        4: "⛈️ Heavy rain / snow",
    }[x],
)
workingday = st.sidebar.selectbox(
    "💼 Working day",
    [0, 1],
    index=int(defaults["workingday"]),
    format_func=lambda x: "🏖️ No" if x == 0 else "💼 Yes",
)
holiday = st.sidebar.selectbox(
    "🎉 Holiday",
    [0, 1],
    index=0,
    format_func=lambda x: "No" if x == 0 else "🎊 Yes",
)
st.sidebar.divider()
temp = st.sidebar.slider(
    "🌡️ Temperature (°C)", -10.0, 45.0, float(defaults["temp"]), 0.5
)
atemp = st.sidebar.slider(
    "🧥 Feels-like (°C)", -10.0, 50.0, float(temp + 2.0), 0.5
)
humidity = st.sidebar.slider(
    "💧 Humidity (%)", 0.0, 100.0, float(defaults["humidity"]), 1.0
)
windspeed = st.sidebar.slider(
    "💨 Wind speed", 0.0, 60.0, float(defaults["windspeed"]), 0.5
)
st.sidebar.divider()

selected_model = saved_model
model_name = saved_model_name

if selected_regression != "🤖 Saved / Final Model":
    try:
        model_catalog = load_model_catalog()
        model_name = (
            selected_regression.replace("📈 ", "")
            .replace("🛡️ ", "")
            .replace("🎯 ", "")
            .replace("🔗 ", "")
            .replace("🌳 ", "")
            .replace("🌲 ", "")
            .replace("🚀 ", "")
        )
        selected_model = model_catalog[model_name]
    except Exception as exc:
        st.sidebar.error(f"Model filter unavailable: {exc}")
        selected_model = saved_model
        model_name = saved_model_name

st.sidebar.success(f"🤖 Active model: {model_name}")

tab1, tab2, tab3 = st.tabs(["🔮 Predict Demand", "📊 Analytics", "📌 Model Info"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🕐 Hour", f"{hour:02d}:{minute:02d}")
    c2.metric("🌡️ Temperature", f"{temp:.1f} °C")
    c3.metric("💧 Humidity", f"{humidity:.0f}%")
    c4.metric("💼 Working day", "Yes" if workingday else "No")

    st.markdown("### 🚦 Scenario preview")
    season_name = {
        1: "🌱 Spring",
        2: "☀️ Summer",
        3: "🍂 Fall",
        4: "❄️ Winter",
    }[season]
    weather_name = {
        1: "☀️ Clear",
        2: "⛅ Cloudy",
        3: "🌧️ Rain/Snow",
        4: "⛈️ Heavy weather",
    }[weather]
    st.markdown(
        f'<div class="card"><b>📅 {date.strftime("%A, %d %B %Y")}</b> '
        f'&nbsp; • &nbsp; <b>🕐 {hour:02d}:{minute:02d}</b> '
        f'&nbsp; • &nbsp; <b>{season_name}</b> '
        f'&nbsp; • &nbsp; <b>{weather_name}</b></div>',
        unsafe_allow_html=True,
    )
    st.write("")

    if st.button("🚀 Predict bike demand", type="primary", width="stretch"):
        timestamp = pd.Timestamp(
            year=date.year,
            month=date.month,
            day=date.day,
            hour=hour,
            minute=minute,
        )
        raw = pd.DataFrame(
            [
                {
                    "datetime": timestamp,
                    "season": season,
                    "holiday": holiday,
                    "workingday": workingday,
                    "weather": weather,
                    "temp": temp,
                    "atemp": atemp,
                    "humidity": humidity,
                    "windspeed": windspeed,
                }
            ]
        )
        try:
            X = prepare_features(raw)
            raw_log_prediction = float(selected_model.predict(X)[0])
            raw_pred = max(0.0, float(np.expm1(raw_log_prediction)))
            pred = min(raw_pred, OBSERVED_TARGET_MAX)
        except Exception as exc:
            st.error(f"❌ Prediction failed: {exc}")
            st.stop()

        was_capped = raw_pred > OBSERVED_TARGET_MAX

        if pred < 100:
            level, icon, advice = (
                "Low demand",
                "🟢",
                "Plenty of spare capacity is likely.",
            )
        elif pred < 300:
            level, icon, advice = (
                "Moderate demand",
                "🟡",
                "A balanced bike supply is recommended.",
            )
        elif pred < 600:
            level, icon, advice = (
                "High demand",
                "🟠",
                "Consider increasing bike availability.",
            )
        else:
            level, icon, advice = (
                "Very high demand",
                "🔴",
                "Prepare for a strong demand spike.",
            )

        st.markdown(
            f'<div class="prediction"><div style="font-size:1rem;opacity:.85;">'
            f"PREDICTED HOURLY RENTALS</div><div class='value'>"
            f"{pred:,.0f} 🚲</div><div class='label'>"
            f"{icon} {level} &nbsp; • &nbsp; {advice}</div></div>",
            unsafe_allow_html=True,
        )
        st.write("")

        if was_capped:
            st.warning(
                f"⚠️ {model_name} produced an extrapolated raw estimate of "
                f"{raw_pred:,.0f} rentals/hour. The app caps the displayed value "
                f"at the observed training maximum of {OBSERVED_TARGET_MAX:,.0f}."
            )

        st.markdown("#### 📊 Demand intensity")
        progress = int(round((pred / OBSERVED_TARGET_MAX) * 100))
        st.progress(
            progress,
            text=f"{pred:,.0f} rentals/hour • {progress}% of observed maximum",
        )
        threshold_a, threshold_b, threshold_c = st.columns(3)
        threshold_a.metric("Low", "<100")
        threshold_b.metric("High", "300–599")
        threshold_c.metric("Observed max", "977")
        st.caption(
            "The progress scale uses the project's observed target range "
            "(0–977 rentals/hour)."
        )

        a, b, c = st.columns(3)
        a.metric("📈 Demand level", level)
        b.metric("🎯 Model", model_name)
        c.metric("🛡️ Output guard", "0–977 rentals/hour")

        with st.expander("🔎 View model input"):
            st.dataframe(raw, width="stretch", hide_index=True)

        st.caption(
            "ℹ️ The prediction uses the same feature-engineering path as training. "
            "Negative predictions are clipped at zero and extrapolated values above "
            "the observed training maximum are capped."
        )

with tab2:
    st.markdown("### 📊 Model Analytics")
    st.caption(
        "Inspect the active model and reproducible training benchmarks "
        "generated from the project training pipeline."
    )

    info_a, info_b, info_c = st.columns(3)
    info_a.metric("🤖 Active model", model_name)
    feature_count = len(
        selected_model.named_steps["preprocessor"].get_feature_names_out()
    )
    info_b.metric("🔢 Input features", feature_count)
    info_c.metric("🎯 Target transform", "log1p → expm1")

    comparison = load_training_report("model_comparison.csv")
    benchmark = final_model_benchmark(comparison)

    if benchmark is not None:
        st.markdown("#### 🏆 Final model benchmark")
        perf_a, perf_b, perf_c = st.columns(3)
        perf_a.metric("RMSE", f"{float(benchmark.iloc[0]['RMSE']):.2f}")
        perf_b.metric("MAE", f"{float(benchmark.iloc[0]['MAE']):.2f}")
        perf_c.metric("R²", f"{float(benchmark.iloc[0]['R2']) * 100:.2f}%")
        st.caption(
            f"Verified holdout metrics for {benchmark.iloc[0]['model']}. "
            "The full model comparison is shown below."
        )

    importance = model_feature_importance(selected_model)
    if not importance.empty:
        st.markdown("#### 🔎 Top model features")
        st.bar_chart(importance.set_index("feature")["importance"])
        st.dataframe(importance, width="stretch", hide_index=True)
    else:
        st.info(
            "Feature importance is not exposed by this model type. "
            "Try a tree-based model such as Random Forest or Gradient Boosting."
        )

    if comparison is not None and not comparison.empty:
        st.markdown("#### 🏁 Training benchmark comparison")
        display_columns = [
            c for c in ["model", "RMSE", "MAE", "R2", "cv_rmse_log_mean"]
            if c in comparison.columns
        ]
        st.dataframe(
            comparison[display_columns].sort_values("RMSE"),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info(
            "No training report is available in this deployment. "
            "Run train_model.py on the Kaggle dataset to generate "
            "reports/model_comparison.csv."
        )

with tab3:
    st.markdown("### 🧠 About this model")
    st.markdown(
        f'<div class="card"><b>Model:</b> {model_name}<br>'
        "<b>Target:</b> hourly bike rental count<br>"
        "<b>Pipeline:</b> feature engineering → preprocessing → regression → "
        "inverse log transform<br>"
        "<b>Deployment:</b> Streamlit Community Cloud</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    st.info(
        "🔐 Leakage protection: training-only target columns such as count, "
        "casual, and registered are excluded from prediction features."
    )
    st.caption("🚲 BikePulse • End-to-end machine learning portfolio project")

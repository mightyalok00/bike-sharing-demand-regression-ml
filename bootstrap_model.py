from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_DIR, MODEL_PATH
from src.features import prepare_features, split_feature_types
from src.modeling import make_models

LOGGER = logging.getLogger(__name__)

# Public mirror used first when the committed model artifact is unavailable.
DATA_URL = "https://raw.githubusercontent.com/TeamLab/machine_learning_from_scratch_with_python/master/code/ch8/data/train.csv"


def _synthetic_training_data(n_rows: int = 5000) -> pd.DataFrame:
    """Deterministic emergency dataset so the demo still works without network access."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2011-01-01", periods=n_rows, freq="h")
    hour = dates.hour.to_numpy()
    weekday = dates.weekday.to_numpy()
    month = dates.month.to_numpy()

    temp = 8 + 18 * np.sin(2 * np.pi * (month - 1) / 12) + rng.normal(0, 2.5, n_rows)
    temp = np.clip(temp, -5, 38)
    atemp = temp + rng.normal(1.5, 1.0, n_rows)
    humidity = np.clip(72 - 18 * np.sin(2 * np.pi * (month - 1) / 12) + rng.normal(0, 8, n_rows), 15, 100)
    windspeed = np.clip(rng.normal(12, 5, n_rows), 0, 45)
    workingday = (weekday < 5).astype(int)
    holiday = ((weekday == 6) & (rng.random(n_rows) < 0.12)).astype(int)
    season = np.select(
        [month.isin([1, 2, 12]), month.isin([3, 4, 5]), month.isin([6, 7, 8])],
        [4, 1, 2],
        default=3,
    )
    weather = np.select(
        [humidity > 90, humidity > 78],
        [3, 2],
        default=1,
    )

    commute = (
        180 * np.exp(-((hour - 8) / 2.2) ** 2)
        + 240 * np.exp(-((hour - 17) / 2.7) ** 2)
        + 70 * np.exp(-((hour - 13) / 4.5) ** 2)
    )
    weekend_effect = np.where(workingday == 1, 1.0, 0.78)
    weather_effect = np.where(weather == 1, 1.0, np.where(weather == 2, 0.82, 0.58))
    seasonal_effect = 1.0 + 0.18 * np.sin(2 * np.pi * (month - 3) / 12)
    trend = np.linspace(0.75, 1.25, n_rows)
    noise = rng.normal(0, 22, n_rows)

    count = np.maximum(
        0,
        25 + commute * weekend_effect * weather_effect
        + 5.5 * temp * seasonal_effect
        + 0.7 * (100 - humidity)
        + 35 * seasonal_effect
        + noise,
    )

    return pd.DataFrame({
        "datetime": dates,
        "season": season.astype(int),
        "holiday": holiday.astype(int),
        "workingday": workingday,
        "weather": weather.astype(int),
        "temp": temp,
        "atemp": atemp,
        "humidity": humidity,
        "windspeed": windspeed,
        "count": count,
    })


def _train(train: pd.DataFrame, model_name: str):
    required = {
        "datetime", "season", "holiday", "workingday", "weather",
        "temp", "atemp", "humidity", "windspeed", "count"
    }
    missing = required.difference(train.columns)
    if missing:
        raise ValueError(f"Training dataset is missing columns: {sorted(missing)}")

    X = prepare_features(train)
    y = np.log1p(train["count"].astype(float))
    numeric, categorical = split_feature_types(X)
    model = make_models(numeric, categorical)["Gradient Boosting Regression"]
    model.fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "target_transform": "log1p_expm1",
        "model_name": model_name,
    }
    joblib.dump(bundle, MODEL_PATH)
    return bundle


def build_fallback_model():
    try:
        train = pd.read_csv(DATA_URL)
        return _train(train, "Gradient Boosting Regression • Cloud fallback")
    except Exception as exc:
        LOGGER.warning("Public training-data bootstrap failed: %s", exc)
        return _train(
            _synthetic_training_data(),
            "Gradient Boosting Regression • Offline fallback",
        )

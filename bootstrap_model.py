from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_DIR, MODEL_PATH
from src.features import prepare_features, split_feature_types
from src.modeling import make_models

LOGGER = logging.getLogger(__name__)

# Public mirror of the original Kaggle Bike Sharing Demand training data.
# The app uses this only when the pre-trained artifact is not present.
DATA_URL = "https://raw.githubusercontent.com/TeamLab/machine_learning_from_scratch_with_python/master/code/ch8/data/train.csv"


def build_fallback_model():
    train = pd.read_csv(DATA_URL)
    required = {"datetime", "season", "holiday", "workingday", "weather",
                "temp", "atemp", "humidity", "windspeed", "count"}
    missing = required.difference(train.columns)
    if missing:
        raise ValueError(f"Training dataset is missing columns: {sorted(missing)}")

    X = prepare_features(train)
    y = np.log1p(train["count"].astype(float))

    numeric, categorical = split_feature_types(X)
    models = make_models(numeric, categorical)

    # Fast deployment fallback. A committed final_model.joblib is always preferred.
    model = models["Gradient Boosting Regression"]
    model.fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "target_transform": "log1p_expm1",
        "model_name": "Gradient Boosting Regression (deployment fallback)",
    }
    joblib.dump(bundle, MODEL_PATH)
    LOGGER.info("Created fallback model at %s", MODEL_PATH)
    return bundle

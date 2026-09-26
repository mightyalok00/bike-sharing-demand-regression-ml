import numpy as np
import pandas as pd

from src.features import prepare_features, split_feature_types
from src.modeling import make_models, make_polynomial_pipeline


def training_frame():
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2011-01-01", periods=12, freq="h"),
            "season": [1, 1, 1, 1] * 3,
            "holiday": [0] * 12,
            "workingday": [1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
            "weather": [1, 2, 1, 1] * 3,
            "temp": np.linspace(8, 20, 12),
            "atemp": np.linspace(9, 21, 12),
            "humidity": np.linspace(45, 85, 12),
            "windspeed": np.linspace(3, 15, 12),
        }
    )


def test_all_models_fit_and_predict():
    frame = training_frame()
    features = prepare_features(frame)
    numeric, categorical = split_feature_types(features)
    target = np.log1p(np.arange(12, dtype=float) + 20)

    for name, model in make_models(numeric, categorical).items():
        model.fit(features, target)
        predictions = model.predict(features)

        assert len(predictions) == len(features)
        assert np.isfinite(predictions).all(), name


def test_polynomial_pipeline_fits():
    frame = training_frame()
    features = prepare_features(frame)
    numeric, categorical = split_feature_types(features)
    target = np.log1p(np.arange(12, dtype=float) + 20)

    model = make_polynomial_pipeline(numeric, categorical, degree=2)
    model.fit(features, target)

    assert model.predict(features).shape == (12,)

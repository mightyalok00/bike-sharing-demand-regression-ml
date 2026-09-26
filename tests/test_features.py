import numpy as np
import pandas as pd
import pytest

from src.features import prepare_features, add_datetime_features


def sample_frame():
    return pd.DataFrame(
        {
            "datetime": ["2011-01-01 08:00:00", "2011-01-02 17:00:00"],
            "season": [1, 1],
            "holiday": [0, 0],
            "workingday": [1, 0],
            "weather": [1, 2],
            "temp": [9.8, 12.4],
            "atemp": [12.0, 14.2],
            "humidity": [70, 80],
            "windspeed": [10, 15],
            "count": [100, 120],
            "casual": [20, 25],
            "registered": [80, 95],
        }
    )


def test_datetime_features_are_deterministic():
    frame = sample_frame()
    first = add_datetime_features(frame)
    second = add_datetime_features(frame)

    pd.testing.assert_frame_equal(first, second)
    assert first.loc[0, "hour"] == 8
    assert first.loc[0, "is_weekend"] == 1


def test_prepare_features_removes_leakage_and_datetime():
    result = prepare_features(sample_frame())

    assert "datetime" not in result.columns
    assert not {"count", "casual", "registered"}.intersection(result.columns)


def test_cyclical_features_are_bounded():
    result = prepare_features(sample_frame())

    for column in ["hour_sin", "hour_cos", "month_sin", "month_cos"]:
        assert np.all(np.abs(result[column]) <= 1.0 + 1e-12)


def test_invalid_datetime_raises():
    frame = sample_frame()
    frame.loc[0, "datetime"] = "not-a-date"

    with pytest.raises(ValueError, match="Invalid datetime"):
        add_datetime_features(frame)

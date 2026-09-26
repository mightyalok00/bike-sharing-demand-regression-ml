import pandas as pd
import pytest

from src.data import basic_clean, validate_test, validate_train


def valid_train():
    return pd.DataFrame(
        {
            "datetime": ["2011-01-01 00:00:00"],
            "season": [1],
            "holiday": [0],
            "workingday": [0],
            "weather": [1],
            "temp": [9.8],
            "atemp": [14.4],
            "humidity": [81],
            "windspeed": [0],
            "count": [16],
        }
    )


def test_validate_train_accepts_required_schema():
    validate_train(valid_train())


def test_validate_test_rejects_missing_feature():
    frame = valid_train().drop(columns=["humidity", "count"])

    with pytest.raises(ValueError, match="missing required columns"):
        validate_test(frame)


def test_basic_clean_removes_duplicates():
    frame = pd.concat([valid_train(), valid_train()], ignore_index=True)

    cleaned = basic_clean(frame)

    assert len(cleaned) == 1
    assert pd.api.types.is_datetime64_any_dtype(cleaned["datetime"])

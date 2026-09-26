import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)


def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {path}\n"
            "Check the path or pass --train/--test/--sample explicitly."
        )
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CSV is empty: {path}")
    LOGGER.info("Loaded %s: %s rows x %s columns", path, *df.shape)
    return df


def validate_train(df: pd.DataFrame) -> None:
    required = {
        "datetime", "season", "holiday", "workingday", "weather",
        "temp", "atemp", "humidity", "windspeed", "count"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Training data is missing required columns: {sorted(missing)}")


def validate_test(df: pd.DataFrame) -> None:
    required = {
        "datetime", "season", "holiday", "workingday", "weather",
        "temp", "atemp", "humidity", "windspeed"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Test data is missing required columns: {sorted(missing)}")


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    out = out.drop_duplicates().reset_index(drop=True)
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    if out["datetime"].isna().any():
        raise ValueError("Some datetime values could not be parsed.")
    return out

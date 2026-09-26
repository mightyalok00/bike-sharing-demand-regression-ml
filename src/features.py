import numpy as np
import pandas as pd

TARGET = "count"
LEAKAGE_COLUMNS = {"count", "casual", "registered"}


def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out["datetime"], errors="coerce")

    if dt.isna().any():
        raise ValueError("Invalid datetime values found during feature engineering.")

    out["year"] = dt.dt.year
    out["month"] = dt.dt.month
    out["day"] = dt.dt.day
    out["hour"] = dt.dt.hour
    out["weekday"] = dt.dt.weekday
    out["weekofyear"] = dt.dt.isocalendar().week.astype(int)
    out["is_weekend"] = (dt.dt.weekday >= 5).astype(int)
    out["rush_hour"] = dt.dt.hour.isin([7, 8, 9, 17, 18, 19]).astype(int)
    out["peak_hour"] = dt.dt.hour.isin([8, 17, 18]).astype(int)

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["weekday_sin"] = np.sin(2 * np.pi * out["weekday"] / 7)
    out["weekday_cos"] = np.cos(2 * np.pi * out["weekday"] / 7)

    return out


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    out = add_datetime_features(df)
    out = out.drop(columns=["datetime"], errors="ignore")
    out = out.drop(columns=list(LEAKAGE_COLUMNS), errors="ignore")
    return out


def split_feature_types(X: pd.DataFrame):
    categorical = [
        c for c in ["season", "holiday", "workingday", "weather"]
        if c in X.columns
    ]
    numeric = [c for c in X.columns if c not in categorical]
    return numeric, categorical

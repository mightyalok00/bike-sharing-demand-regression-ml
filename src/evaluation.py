import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def summarize_cv(scores):
    scores = np.asarray(scores)
    return {
        "CV_RMSE_mean": float(-scores.mean()),
        "CV_RMSE_std": float(scores.std()),
    }


def feature_importance_table(model, X):
    final_model = model.named_steps["model"]
    preprocessor = model.named_steps["preprocessor"]

    if not hasattr(final_model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])

    names = preprocessor.get_feature_names_out()
    values = final_model.feature_importances_
    return (
        pd.DataFrame({"feature": names, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

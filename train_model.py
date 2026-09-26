from __future__ import annotations

import argparse
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)

from src.config import (
    COMPARISON_PATH,
    HOLDOUT_PATH,
    IMPORTANCE_PATH,
    MODEL_DIR,
    MODEL_PATH,
    RANDOM_STATE,
    REPORT_DIR,
    TEST_SIZE,
    TRAIN_PATH,
)
from src.data import basic_clean, load_csv, validate_train
from src.evaluation import feature_importance_table, regression_metrics
from src.features import prepare_features, split_feature_types
from src.modeling import make_models, make_polynomial_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=TRAIN_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    train = basic_clean(load_csv(args.train))
    validate_train(train)

    # Q6/Q7: cleaning + feature engineering.
    X_all = prepare_features(train)
    y = train["count"].astype(float)

    # Q11-15: holdout and cross-validation.
    X_train, X_holdout, y_train, _ = train_test_split(
        X_all, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    numeric, categorical = split_feature_types(X_train)
    models = make_models(numeric, categorical)

    # Q12: polynomial regression is evaluated separately.
    models["Polynomial Regression"] = make_polynomial_pipeline(
        numeric, categorical, degree=2
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    # Log target transformation.
    y_train_log = np.log1p(y_train)

    for name, model in models.items():
        LOGGER.info("Evaluating %s", name)
        scores = cross_val_score(
            model,
            X_train,
            y_train_log,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=-1,
        )
        model.fit(X_train, y_train_log)

        pred_log = model.predict(X_holdout)
        pred = np.maximum(0, np.expm1(pred_log))
        metrics = regression_metrics(y_holdout, pred)

        rows.append(
            {
                "model": name,
                "cv_rmse_log_mean": -scores.mean(),
                "cv_rmse_log_std": scores.std(),
                **metrics,
            }
        )

    comparison = pd.DataFrame(rows).sort_values("RMSE")
    comparison.to_csv(COMPARISON_PATH, index=False)

    # Q16: tune the strongest tree ensemble candidates.
    rf = models["Random Forest Regression"]
    rf_grid = {
        "model__n_estimators": [150, 250],
        "model__max_depth": [None, 15, 25],
        "model__min_samples_leaf": [1, 2, 4],
    }
    rf_search = RandomizedSearchCV(
        rf,
        rf_grid,
        n_iter=6,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_search.fit(X_train, y_train_log)

    gb = models["Gradient Boosting Regression"]
    gb_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_depth": [2, 3, 4],
    }
    gb_search = GridSearchCV(
        gb,
        gb_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
    )
    gb_search.fit(X_train, y_train_log)

    tuned = [
        ("Tuned Random Forest", rf_search.best_estimator_),
        ("Tuned Gradient Boosting", gb_search.best_estimator_),
    ]

    for name, model in tuned:
        pred = np.maximum(0, np.expm1(model.predict(X_holdout)))
        metrics = regression_metrics(y_holdout, pred)
        rows.append(
            {
                "model": name,
                "cv_rmse_log_mean": np.nan,
                "cv_rmse_log_std": np.nan,
                **metrics,
            }
        )

    final_table = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    final_table.to_csv(COMPARISON_PATH, index=False)

    # Final model: best RMSE on holdout, then refit on all training data.
    best_name = final_table.iloc[0]["model"]
    LOGGER.info("Selected model by holdout RMSE: %s", best_name)

    candidates = dict(models)
    candidates.update(dict(tuned))
    final_model = candidates[best_name]
    final_model.fit(X_all, np.log1p(y))

    joblib.dump(
        {
            "model": final_model,
            "target_transform": "log1p_expm1",
            "model_name": best_name,
        },
        MODEL_PATH,
    )

    holdout_pred = np.maximum(0, np.expm1(final_model.predict(X_holdout)))
    pd.DataFrame([regression_metrics(y_holdout, holdout_pred)]).to_csv(
        HOLDOUT_PATH, index=False
    )

    importance = feature_importance_table(final_model, X_all)
    importance.to_csv(IMPORTANCE_PATH, index=False)

    LOGGER.info("Saved final model to %s", MODEL_PATH)
    LOGGER.info("Saved comparison to %s", COMPARISON_PATH)


if __name__ == "__main__":
    main()

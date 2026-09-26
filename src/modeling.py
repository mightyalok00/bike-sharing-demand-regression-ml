from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

LOGGER = logging.getLogger(__name__)


def make_preprocessor(numeric, categorical, scaler="standard"):
    scaler_obj = StandardScaler() if scaler == "standard" else StandardScaler()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler_obj),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ], remainder="drop", verbose_feature_names_out=False)


def make_models(numeric, categorical) -> Dict[str, Pipeline]:
    prep = make_preprocessor(numeric, categorical)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.001, max_iter=20000),
        "Elastic Net": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=20000),
        "Decision Tree Regression": DecisionTreeRegressor(
            random_state=42, max_depth=20, min_samples_leaf=2
        ),
        "Random Forest Regression": RandomForestRegressor(
            n_estimators=250, random_state=42, n_jobs=-1,
            max_depth=None, min_samples_leaf=1
        ),
        "Gradient Boosting Regression": GradientBoostingRegressor(
            random_state=42, n_estimators=200, learning_rate=0.05,
            max_depth=3, loss="huber"
        ),
    }

    return {
        name: Pipeline([
            ("preprocessor", prep),
            ("model", estimator),
        ])
        for name, estimator in models.items()
    }


def make_polynomial_pipeline(numeric, categorical, degree=2):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    prep = ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ], remainder="drop", verbose_feature_names_out=False)

    return Pipeline([
        ("preprocessor", prep),
        ("model", Ridge(alpha=1.0)),
    ])

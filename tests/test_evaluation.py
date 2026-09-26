import numpy as np
import pytest

from src.evaluation import regression_metrics


def test_regression_metrics_are_correct():
    metrics = regression_metrics([1, 2, 3], [1, 2, 4])

    assert metrics["MAE"] == pytest.approx(1 / 3)
    assert metrics["RMSE"] == pytest.approx(np.sqrt(1 / 3))
    assert metrics["R2"] == pytest.approx(0.5)


def test_regression_metrics_rejects_nan_predictions():
    metrics = regression_metrics([1, 2], [1, np.nan])

    assert not np.isfinite(metrics["RMSE"])

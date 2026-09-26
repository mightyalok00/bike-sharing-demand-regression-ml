import pytest

from fastapi.testclient import TestClient

from api import app


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    payload = response.json()
    assert payload["service"] == "BikePulse Demand API"
    assert payload["prediction_endpoint"] == "POST /predict"


def test_models_endpoint():
    response = client.get("/models")

    assert response.status_code == 200
    payload = response.json()
    assert "Random Forest Regression" in payload["models"]
    assert "datetime" in payload["filters"]

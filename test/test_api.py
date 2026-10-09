import logging

import pytest
from fastapi.testclient import TestClient

import predict
from main import app

# A typical Barolo: high proline, high flavanoids
SAMPLE = {
    "alcohol": 13.2, "malic_acid": 1.78, "ash": 2.14, "alcalinity_of_ash": 11.2,
    "magnesium": 100.0, "total_phenols": 2.65, "flavanoids": 2.76,
    "nonflavanoid_phenols": 0.26, "proanthocyanins": 1.28, "color_intensity": 4.38,
    "hue": 1.05, "od280_od315_of_diluted_wines": 3.4, "proline": 1050.0,
}


@pytest.fixture
def client(trained_model):
    with TestClient(app) as test_client:
        yield test_client


# ----------------- Endpoints ----------------- #
def test_health_lists_the_endpoints(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["endpoints"] == ["/predict", "/predict/detailed"]


def test_predict_returns_class_and_name(client):
    response = client.post("/predict", json=SAMPLE)
    assert response.status_code == 200
    assert response.json() == {"predicted_class": 0, "class_name": "Barolo"}


def test_detailed_probabilities_are_consistent(client):
    body = client.post("/predict/detailed", json=SAMPLE).json()
    probabilities = body["probabilities"]
    assert set(probabilities) == {"Barolo", "Grignolino", "Barbera"}
    assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-3)
    assert body["confidence"] == max(probabilities.values())
    assert body["class_name"] == max(probabilities, key=probabilities.get)


# ----------------- Error handling ----------------- #
def test_missing_field_returns_422_and_is_logged(client, caplog):
    incomplete = {k: v for k, v in SAMPLE.items() if k != "proline"}
    with caplog.at_level(logging.WARNING):
        response = client.post("/predict", json=incomplete)
    assert response.status_code == 422
    assert any("Validation error on /predict" in r.message for r in caplog.records)


def test_missing_model_returns_503(missing_model):
    with TestClient(app) as test_client:
        response = test_client.post("/predict", json=SAMPLE)
    assert response.status_code == 503


# ----------------- Logging and caching ----------------- #
def test_middleware_logs_status_and_latency(client, caplog):
    with caplog.at_level(logging.INFO):
        client.post("/predict", json=SAMPLE)
    finished = [r.message for r in caplog.records if r.message.startswith("Request finished")]
    assert finished and "status 200" in finished[-1] and finished[-1].endswith("ms")


def test_model_is_loaded_once_for_many_requests(client):
    for _ in range(5):
        client.post("/predict", json=SAMPLE)
    assert predict.load_model.cache_info().misses == 1

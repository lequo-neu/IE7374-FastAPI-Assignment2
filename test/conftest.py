import joblib
import pytest

import predict
from data import load_data, split_data
from train import fit_model


@pytest.fixture
def trained_model(tmp_path, monkeypatch):
    """Train the real model into a temporary file and point the API at it."""
    X, y, _, _ = load_data()
    X_train, _, y_train, _ = split_data(X, y)
    path = tmp_path / "wine_model.pkl"
    joblib.dump(fit_model(X_train, y_train), path)

    monkeypatch.setattr(predict, "MODEL_PATH", str(path))
    predict.load_model.cache_clear()
    yield str(path)
    predict.load_model.cache_clear()


@pytest.fixture
def missing_model(tmp_path, monkeypatch):
    """Point the API at a model file that does not exist."""
    monkeypatch.setattr(predict, "MODEL_PATH", str(tmp_path / "missing.pkl"))
    predict.load_model.cache_clear()
    yield
    predict.load_model.cache_clear()

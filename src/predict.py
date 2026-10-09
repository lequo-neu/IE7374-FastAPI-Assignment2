# src/predict.py
# Extended from original: the model is loaded once and kept in memory, and
# predict_proba() support was added so /predict/detailed can return
# confidence scores alongside the predicted class label.

import os
from functools import lru_cache

import joblib
import numpy as np
from logger import get_logger

logger = get_logger(__name__)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "model", "wine_model.pkl"
)


@lru_cache(maxsize=1)
def load_model():
    """
    Load the trained model from disk once and keep it in memory.

    The original lab read the file again on every request. With the cache, only
    the first call touches the disk. A missing file raises an error that is not
    cached, so the model is picked up as soon as train.py has been run. After
    retraining an existing model, restart the server to load the new file.

    Returns:
        The fitted classifier.
    Raises:
        FileNotFoundError: If the model file does not exist yet.
    """
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found at {MODEL_PATH}. Run train.py first.")
        raise FileNotFoundError("Model not found. Run train.py before starting the API.")
    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded from {MODEL_PATH}")
    return model


def predict_data(X):
    """
    Return the predicted class label(s) for the given feature matrix.

    Args:
        X (list of list of float): Feature rows.

    Returns:
        numpy.ndarray: Predicted class indices.
    """
    logger.debug(f"predict_data called. Input shape: {np.array(X).shape}")
    y_pred = load_model().predict(X)
    logger.info(f"Prediction result: {y_pred.tolist()}")
    return y_pred


def predict_proba_data(X):
    """
    Return the class probability distribution for the given feature matrix.

    Args:
        X (list of list of float): Feature rows.

    Returns:
        numpy.ndarray: Probability array of shape (n_samples, n_classes).
    """
    logger.debug(f"predict_proba_data called. Input shape: {np.array(X).shape}")
    probas = load_model().predict_proba(X)
    logger.info(f"Probability scores: {probas.tolist()}")
    return probas

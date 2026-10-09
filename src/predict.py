# src/predict.py
# Extended from original: added predict_proba() support so the
# /predict/detailed endpoint can return confidence scores alongside
# the predicted class label.

import os
import joblib
import numpy as np
from logger import get_logger

logger = get_logger(__name__)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "model", "wine_model.pkl"
)


def load_model():
    if not os.path.exists(MODEL_PATH):
        logger.error(
            f"Model file not found at {MODEL_PATH}. "
            "Run train.py first."
        )
        raise FileNotFoundError(
            "Model not found. Run train.py before starting the API."
        )
    model = joblib.load(MODEL_PATH)
    logger.debug("Model loaded from disk.")
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
    model = load_model()
    y_pred = model.predict(X)
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
    model = load_model()
    probas = model.predict_proba(X)
    logger.info(f"Probability scores: {probas.tolist()}")
    return probas

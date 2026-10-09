# src/train.py
# Modification: replaced DecisionTreeClassifier with GradientBoostingClassifier.
# GradientBoosting is an ensemble method that iteratively corrects its own
# errors, producing a stronger learner than a single decision tree.
# n_estimators=100, learning_rate=0.1, random_state=12 (kept consistent
# with the original lab's random_state convention).

import os
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from data import load_data, split_data
from logger import get_logger

logger = get_logger(__name__)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "model", "wine_model.pkl"
)


def fit_model(X_train, y_train):
    """
    Train a GradientBoostingClassifier and save it to disk.

    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training labels.

    Returns:
        model: Trained GradientBoostingClassifier.
    """
    logger.info(
        "Training GradientBoostingClassifier "
        "(n_estimators=100, learning_rate=0.1, random_state=12)."
    )
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=12,
    )
    model.fit(X_train, y_train)
    logger.info("Model training complete.")
    return model


if __name__ == "__main__":
    logger.info("=== Training pipeline started ===")

    X, y, feature_names, target_names = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = fit_model(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    logger.info(f"Test accuracy: {acc:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")
    logger.info("=== Training pipeline complete ===")

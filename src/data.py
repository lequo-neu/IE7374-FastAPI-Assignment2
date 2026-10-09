# src/data.py
# Modification: replaced Iris dataset with Wine dataset.
# Wine has 178 samples, 13 numeric features, and 3 target classes
# representing wine cultivars grown in the same region in Italy.
# This is a more complex classification problem than Iris and is a
# natural fit for the drug-supply-chain domain context of PharmTrack.

import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from logger import get_logger

logger = get_logger(__name__)


def load_data():
    """
    Load the Wine dataset and return features and targets.

    Returns:
        X (numpy.ndarray): 178 x 13 feature matrix.
        y (numpy.ndarray): Target class labels (0, 1, 2).
        feature_names (list): Names of the 13 input features.
        target_names (list): Names of the 3 wine cultivar classes.
    """
    logger.info("Loading Wine dataset (178 samples, 13 features, 3 classes).")
    wine = load_wine()
    logger.debug(f"Features: {list(wine.feature_names)}")
    logger.debug(f"Classes:  {list(wine.target_names)}")
    return wine.data, wine.target, list(wine.feature_names), list(wine.target_names)


def split_data(X, y):
    """
    Split data into training and testing sets (70/30 split).

    Args:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Target labels.

    Returns:
        X_train, X_test, y_train, y_test (tuple)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=12
    )
    logger.info(
        f"Train/test split complete. "
        f"Train: {len(X_train)} samples | Test: {len(X_test)} samples."
    )
    return X_train, X_test, y_train, y_test

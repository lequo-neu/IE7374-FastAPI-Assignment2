# src/main.py
# Lab 1 (FastAPI): serve a GradientBoostingClassifier trained on the Wine
#   dataset through /predict (class only) and /predict/detailed (class plus
#   confidence scores). The model is loaded once when the server starts.
# Lab 2 (Logging): HTTP middleware logs every request with method, path,
#   status code and latency. Errors are logged with full tracebacks and
#   validation failures are logged as warnings.

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from predict import load_model, predict_data, predict_proba_data
from logger import get_logger

logger = get_logger(__name__)

CLASS_NAMES = {
    0: "Barolo",
    1: "Grignolino",
    2: "Barbera",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model before the first request so no caller pays for it."""
    try:
        load_model()
        logger.info("Model loaded and cached at startup.")
    except FileNotFoundError:
        logger.warning(
            "No trained model found at startup. Prediction endpoints return 503 "
            "until train.py has been run."
        )
    yield


app = FastAPI(
    title="Wine Cultivar Classifier API",
    description=(
        "IE7374 Lab Assignment 2, covering FastAPI and Logging. "
        "Serves a GradientBoostingClassifier trained on the Wine dataset."
    ),
    version="1.1.0",
    lifespan=lifespan,
)


# Lab 2: request and response logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    logger.info(f"Request started: {request.method} {request.url.path}")
    response = await call_next(request)
    ms = (time.time() - start) * 1000
    logger.info(
        f"Request finished: {request.method} {request.url.path}, "
        f"status {response.status_code}, {ms:.1f} ms"
    )
    return response


# Lab 2: validation error logging
@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Validation error on {request.url.path}: {exc.errors()}")
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


# Input and output models
class WineFeatures(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float


class WineResponse(BaseModel):
    predicted_class: int
    class_name: str


class WineDetailedResponse(BaseModel):
    predicted_class: int
    class_name: str
    confidence: float
    probabilities: dict


def _build_feature_list(wine: WineFeatures) -> list:
    return [[
        wine.alcohol, wine.malic_acid, wine.ash,
        wine.alcalinity_of_ash, wine.magnesium,
        wine.total_phenols, wine.flavanoids,
        wine.nonflavanoid_phenols, wine.proanthocyanins,
        wine.color_intensity, wine.hue,
        wine.od280_od315_of_diluted_wines, wine.proline,
    ]]


# Endpoints
@app.get("/", status_code=status.HTTP_200_OK)
async def health_check():
    """Confirm the API is running and list what it serves."""
    logger.debug("Health check endpoint called.")
    return {
        "status": "healthy",
        "model": "GradientBoostingClassifier",
        "dataset": "Wine (sklearn)",
        "endpoints": ["/predict", "/predict/detailed"],
    }


@app.post("/predict", response_model=WineResponse)
async def predict(wine: WineFeatures):
    """
    Predict the wine cultivar class from 13 chemical features.
    Returns the predicted class index and its name.
    """
    logger.info("POST /predict called.")
    try:
        result = predict_data(_build_feature_list(wine))
        cls = int(result[0])
        logger.info(f"Prediction: {cls} ({CLASS_NAMES[cls]})")
        return WineResponse(predicted_class=cls, class_name=CLASS_NAMES[cls])
    except FileNotFoundError as e:
        logger.error(str(e))
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in /predict: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error.")


@app.post("/predict/detailed", response_model=WineDetailedResponse)
async def predict_detailed(wine: WineFeatures):
    """
    Predict the wine cultivar and return the full probability distribution
    across all three classes, plus a confidence score (the probability of the
    predicted class).

    Callers can see not just what the model predicted but how certain it is,
    so low confidence predictions can be sent to a human for review.
    """
    logger.info("POST /predict/detailed called.")
    try:
        probas = predict_proba_data(_build_feature_list(wine))[0]
        cls = int(probas.argmax())
        confidence = round(float(probas[cls]), 4)
        prob_dict = {CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(probas)}
        logger.info(
            f"Detailed prediction: {cls} ({CLASS_NAMES[cls]}) confidence={confidence:.4f}"
        )
        return WineDetailedResponse(
            predicted_class=cls,
            class_name=CLASS_NAMES[cls],
            confidence=confidence,
            probabilities=prob_dict,
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in /predict/detailed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error.")

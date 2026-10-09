# IE7374 Lab Assignment 2 — FastAPI + Logging

## Overview

This project exposes a trained machine learning model as a REST API using
FastAPI and uvicorn. It is based on the API_Labs/FastAPI_Labs template from
the course repository and covers both lab activities for this week: Lab 1
(FastAPI) and Lab 2 (Logging). The model is a GradientBoostingClassifier
trained on the Wine dataset, served through three endpoints. Every request,
prediction, and error is logged to both the console and a rotating log file
using Python's built-in logging module.

## Changes Made to the Lab

**Data (src/data.py)**

The original lab used the Iris dataset, which has 150 samples, 4 features,
and 3 classes. This was replaced with the Wine dataset, which has 178 samples,
13 numeric chemical features, and 3 target classes representing wine cultivars
grown in the same region in Italy. The train/test split ratio and random state
were kept the same as the original (70/30, random_state=12) so the structural
change is clearly visible by comparison.

**Model (src/train.py)**

DecisionTreeClassifier(max_depth=3) was replaced with
GradientBoostingClassifier(n_estimators=100, learning_rate=0.1). Gradient
boosting is an ensemble method that trains trees sequentially, each one
correcting the residual errors of the previous, producing a stronger learner
than a single tree. The trained model is saved to model/wine_model.pkl.

**Prediction (src/predict.py)**

The original predict.py only supported hard class predictions. A second
function, predict_proba_data(), was added to return the full probability
distribution across all three classes. This is needed to power the advanced
endpoint described below.

**API (src/main.py)**

The original API had two endpoints: GET / and POST /predict. This version
retains both and adds a third: POST /predict/detailed, which returns the
predicted class, its name, the model's confidence in that prediction, and the
complete probability breakdown across all three classes. The IrisData and
IrisResponse models were replaced with WineFeatures (13 fields) and
WineResponse. The health check response was enriched to include model name,
dataset, and available endpoints.

**Logging (src/logger.py) — Lab 2**

The original lab had no logging. A central logger.py module was added with a
shared logging configuration that applies to every file in the project.
Logs are written simultaneously to the console at INFO level and to a rotating
file handler at DEBUG level, capped at 5 MB per file with 3 backup files.
Every module imports get_logger(__name__) from this file. An HTTP middleware
in main.py logs the method, path, status code, and latency of every request.
Validation errors and exceptions are caught and logged with appropriate levels
before returning the HTTP error response.

## Advanced Extension — Confidence Scores via /predict/detailed

**Why this was chosen**

A model that returns only a class label gives callers no way to know how
certain the prediction is. In production ML systems, a prediction made with
32% confidence and one made with 97% confidence are fundamentally different
signals, but both look identical when the API returns just an integer. This
matters especially in high-stakes domains like drug supply-chain management,
where an uncertain prediction should trigger a human review rather than
an automated action.

Adding a dedicated endpoint for detailed predictions makes the uncertainty
visible and actionable. Callers can branch on the confidence field: if it
falls below a threshold they define on their side, they escalate to a human;
if it is high, they proceed automatically. This pattern is widely used in
production ML APIs and is a natural next step from a basic classifier endpoint.

**How it works**

The /predict/detailed endpoint calls predict_proba() on the trained
GradientBoostingClassifier, which returns a probability distribution across
all three wine cultivar classes. The class with the highest probability is
selected as the predicted class, and its probability is returned as the
confidence score. The full probability dictionary is included in the response
keyed by class name rather than by index, so the caller does not need to know
the internal class mapping. Every step is logged at INFO level.

**What it produces**

A caller sending the same 13 chemical features to /predict gets back one
integer. A caller sending the same features to /predict/detailed gets back
the class index, the class name, a confidence score between 0 and 1, and
a breakdown showing the model's probability estimate for each cultivar.
This output is directly usable for building confidence thresholds,
dashboards, or audit trails without any post-processing on the caller's side.

## Prerequisites

The following must be available on your machine before running locally.

Python 3.10 or later (developed and tested with Python 3.14). Download from
https://www.python.org/downloads and verify with: python3 --version

Git. Download from https://git-scm.com and verify with: git --version

Key Python packages and versions (all listed in requirements.txt):
fastapi[all] 0.111.1, scikit-learn 1.5.1, joblib 1.4.2. These will be
installed automatically by pip in the Environment Setup step.

## Environment Setup

Clone the repository.

```
git clone https://github.com/lequo-neu/IE7374-FastAPI-Assignment2.git
cd IE7374-FastAPI-Assignment2
```

Create and activate a virtual environment.

```
python3 -m venv venv
source venv/bin/activate
```

Install all dependencies.

```
pip install -r requirements.txt
```

Verify the key packages installed correctly.

```
python -c "import fastapi, sklearn, joblib; print('All dependencies OK')"
```

## Running the Code

Step 1: Move into the src/ folder. All scripts must be run from there so
that relative paths to the model/ and logs/ directories resolve correctly.

```
cd src
```

Step 2: Train the model. This generates model/wine_model.pkl and writes
training logs to both the console and logs/app.log.

```
python train.py
```

Step 3: Start the API server.

```
uvicorn main:app --reload
```

Step 4: Open the interactive API documentation in a browser.

```
http://127.0.0.1:8000/docs
```

From the docs page, expand any endpoint, click Try it out, fill in the
request body, and click Execute to see the live response.

## What a Successful Run Looks Like

After running train.py, the console output follows this pattern.

```
2026-10-09 [__main__] INFO: === Training pipeline started ===
2026-10-09 [data]     INFO: Loading Wine dataset (178 samples, 13 features, 3 classes).
2026-10-09 [data]     INFO: Train/test split complete. Train: 124 | Test: 54.
2026-10-09 [__main__] INFO: Training GradientBoostingClassifier...
2026-10-09 [__main__] INFO: Test accuracy: 0.9815
2026-10-09 [__main__] INFO: Model saved to ../model/wine_model.pkl
2026-10-09 [__main__] INFO: === Training pipeline complete ===
```

After starting the server and calling POST /predict/detailed with valid
Wine feature values, the response looks like this.

```json
{
  "predicted_class": 0,
  "class_name": "Barolo",
  "confidence": 0.9823,
  "probabilities": {
    "Barolo": 0.9823,
    "Grignolino": 0.0141,
    "Barbera": 0.0036
  }
}
```

Each API call simultaneously produces a console log entry and writes to
logs/app.log.

```
2026-10-09 [main] INFO: → POST /predict/detailed
2026-10-09 [predict] INFO: Detailed prediction: 0 (Barolo) confidence=0.9823
2026-10-09 [main] INFO: ← POST /predict/detailed | 200 | 4.2ms
```

## Project Structure

```
IE7374-FastAPI-Assignment2/
    logs/
    model/
        wine_model.pkl          (generated after running train.py)
    src/
        __init__.py
        logger.py               (Lab 2: central logging configuration)
        data.py                 (modified: Wine dataset instead of Iris)
        train.py                (modified: GradientBoostingClassifier instead of DecisionTree)
        predict.py              (extended: added predict_proba_data function)
        main.py                 (Lab 1 + Lab 2: FastAPI app with logging middleware)
    .gitignore
    README.md
    requirements.txt
```

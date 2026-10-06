import os
import joblib
import pandas as pd

_dir = os.path.dirname(__file__)
_model = joblib.load(os.path.join(_dir, "../model/pts_model.joblib"))
_model_low = joblib.load(os.path.join(_dir, "../model/pts_model_low.joblib"))
_model_high = joblib.load(os.path.join(_dir, "../model/pts_model_high.joblib"))

FEATURES = [
    "last5_avg_pts",
    "last5_avg_ast",
    "last5_avg_reb",
    "last5_avg_min",
    "days_rest",
    "is_home",
    "avg_pts_vs_opponent",
    "opponent_encoded",
]


def predict_pts(features: dict) -> float:
    X = pd.DataFrame([features])[FEATURES]
    return float(_model.predict(X)[0])


def predict_with_interval(features: dict) -> dict:
    X = pd.DataFrame([features])[FEATURES]
    return {
        "predicted_pts": round(max(0.0, float(_model.predict(X)[0])), 1),
        "pts_low": round(max(0.0, float(_model_low.predict(X)[0])), 1),
        "pts_high": round(max(0.0, float(_model_high.predict(X)[0])), 1),
    }

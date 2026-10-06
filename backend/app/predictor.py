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


MIN_REMAINING = 2.0


def predict_live(features: dict) -> dict:
    base = predict_with_interval(features)
    so_far = features["pts_so_far"]
    played = features["min_so_far"]
    expected = float(features["last5_avg_min"])

    remaining = max(expected - played, MIN_REMAINING)
    frac = min(remaining / expected, 1.0) if expected > 0 else 0.0

    return {
        "predicted_pts": round(so_far + base["predicted_pts"] * frac, 1),
        "pts_low": round(so_far + base["pts_low"] * frac, 1),
        "pts_high": round(so_far + base["pts_high"] * frac, 1),
        "pts_so_far": so_far,
        "min_so_far": played,
    }

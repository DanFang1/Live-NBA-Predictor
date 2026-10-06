import numpy as np
import pandas as pd
import app.predictor as predictor
from app.predictor import predict_with_interval, FEATURES
from app.live import FEATURES_PATH


def sample_rows(n=200):
    df = pd.read_csv(FEATURES_PATH).dropna(subset=FEATURES)
    return df.sample(n, random_state=0)[FEATURES].to_dict("records")


def test_returns_point_and_interval():
    result = predict_with_interval(sample_rows(1)[0])
    assert set(result) == {"predicted_pts", "pts_low", "pts_high"}
    assert all(isinstance(v, float) for v in result.values())


def test_interval_is_ordered_on_real_rows():
    results = [predict_with_interval(r) for r in sample_rows()]
    ordered = [r["pts_low"] <= r["predicted_pts"] <= r["pts_high"] for r in results]
    assert sum(ordered) / len(ordered) >= 0.99


def test_point_estimates_are_plausible_points_totals():
    for r in sample_rows():
        assert 0 <= predict_with_interval(r)["predicted_pts"] <= 70


class StubModel:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.array([self.value])


def test_negative_model_output_is_clipped_to_zero(monkeypatch):
    monkeypatch.setattr(predictor, "_model", StubModel(1.0))
    monkeypatch.setattr(predictor, "_model_low", StubModel(-3.2))
    monkeypatch.setattr(predictor, "_model_high", StubModel(9.0))
    result = predict_with_interval(sample_rows(1)[0])
    assert result == {"predicted_pts": 1.0, "pts_low": 0.0, "pts_high": 9.0}

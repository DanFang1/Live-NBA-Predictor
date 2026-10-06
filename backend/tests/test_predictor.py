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


def live_features(so_far, played):
    return {**sample_rows(1)[0], "last5_avg_min": 34.0,
            "pts_so_far": so_far, "min_so_far": played}


def stub_pregame(monkeypatch, low=15.0, mid=25.0, high=35.0):
    monkeypatch.setattr(predictor, "_model", StubModel(mid))
    monkeypatch.setattr(predictor, "_model_low", StubModel(low))
    monkeypatch.setattr(predictor, "_model_high", StubModel(high))


def test_live_equals_pregame_at_tipoff(monkeypatch):
    stub_pregame(monkeypatch)
    r = predictor.predict_live(live_features(0, 0))
    assert (r["pts_low"], r["predicted_pts"], r["pts_high"]) == (15.0, 25.0, 35.0)


def test_live_halftime_moves_toward_current_pace(monkeypatch):
    stub_pregame(monkeypatch)
    r = predictor.predict_live(live_features(10, 17))
    assert r["predicted_pts"] == 22.5
    assert (r["pts_low"], r["pts_high"]) == (17.5, 27.5)


def test_live_interval_narrows_as_game_goes_on(monkeypatch):
    stub_pregame(monkeypatch)
    widths = []
    for played in (0, 17, 30):
        r = predictor.predict_live(live_features(0, played))
        widths.append(r["pts_high"] - r["pts_low"])
    assert widths[0] > widths[1] > widths[2]


def test_live_never_drops_below_points_already_scored(monkeypatch):
    stub_pregame(monkeypatch)
    r = predictor.predict_live(live_features(30, 40))   # played past average
    assert r["pts_low"] >= 30

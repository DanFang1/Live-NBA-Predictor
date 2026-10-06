import numpy as np
import pandas as pd
from model.train import evaluate_interval, baseline_mae, FEATURES, TARGET


class Const:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


def frame(pts):
    df = pd.DataFrame({f: 0.0 for f in FEATURES}, index=range(len(pts)))
    df[TARGET] = pts
    df["last5_avg_pts"] = 10.0
    return df


def test_interval_coverage_and_tail_rates():
    stats = evaluate_interval(Const(8.0), Const(20.0), frame([5, 10, 15, 25]))
    assert stats["coverage"] == 0.5      # 10 and 15 fall inside [8, 20]
    assert stats["below_low"] == 0.25    # 5
    assert stats["above_high"] == 0.25   # 25
    assert stats["mean_width"] == 12.0


def test_baseline_mae_is_error_of_last5_average():
    assert baseline_mae(frame([10, 14])) == 2.0

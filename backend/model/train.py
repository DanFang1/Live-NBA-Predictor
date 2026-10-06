import pandas as pd
import os
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error


DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/raw/features.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pts_model.joblib")
MODEL_LOW_PATH = os.path.join(os.path.dirname(__file__), "pts_model_low.joblib")
MODEL_HIGH_PATH = os.path.join(os.path.dirname(__file__), "pts_model_high.joblib")

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

TARGET = "PTS"


def load_and_prepare(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    df["opponent_encoded"] = df["opponent"].astype("category").cat.codes
    df = df.dropna(subset=FEATURES + [TARGET])
    return df


def time_split(df: pd.DataFrame, test_fraction: float = 0.2):
    cutoff = int(len(df) * (1 - test_fraction))
    return df.iloc[:cutoff], df.iloc[cutoff:]


def train_quantile(df_train: pd.DataFrame, alpha: float) -> XGBRegressor:
    X = df_train[FEATURES]
    y = df_train[TARGET]

    model = XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=alpha,
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X, y)
    return model

 
def train(df_train: pd.DataFrame) -> XGBRegressor:
    X = df_train[FEATURES]
    y = df_train[TARGET]

    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X, y)
    return model


def evaluate(model: XGBRegressor, df_test: pd.DataFrame) -> float:
    X = df_test[FEATURES]
    y = df_test[TARGET]
    preds = model.predict(X)
    return mean_absolute_error(y, preds)


def evaluate_interval(model_low, model_high, df_test: pd.DataFrame) -> dict:
    X = df_test[FEATURES]
    y = df_test[TARGET].to_numpy()
    lo, hi = model_low.predict(X), model_high.predict(X)
    return {
        "coverage": float(((y >= lo) & (y <= hi)).mean()),
        "below_low": float((y < lo).mean()),
        "above_high": float((y > hi).mean()),
        "mean_width": float((hi - lo).mean()),
    }


def baseline_mae(df_test: pd.DataFrame) -> float:
    return mean_absolute_error(df_test[TARGET], df_test["last5_avg_pts"])


if __name__ == "__main__":
    print("Loading data...")
    df = load_and_prepare(DATA_PATH)
    print(f"  {len(df)} rows after dropping nulls")

    df_train, df_test = time_split(df)
    print(f"  Train: {len(df_train)} rows | Test: {len(df_test)} rows")

    print("Training XGBoost...")
    model = train(df_train)
    mae = evaluate(model, df_test)
    print(f"  Test MAE: {mae:.2f} pts")
    joblib.dump(model, MODEL_PATH)
    print(f"  Model saved → {MODEL_PATH}")

    print("Training quantile models...")
    model_low = train_quantile(df_train, alpha=0.1)
    joblib.dump(model_low, MODEL_LOW_PATH)
    print(f"  Low bound (p10) saved → {MODEL_LOW_PATH}")

    model_high = train_quantile(df_train, alpha=0.9)
    joblib.dump(model_high, MODEL_HIGH_PATH)
    print(f"  High bound (p90) saved → {MODEL_HIGH_PATH}")

    stats = evaluate_interval(model_low, model_high, df_test)
    print(f"  Interval coverage (p10-p90): {stats['coverage']:.1%} (nominal 80%)")
    print(f"  Below p10: {stats['below_low']:.1%} | Above p90: {stats['above_high']:.1%}")
    print(f"  Mean interval width: {stats['mean_width']:.1f} pts")
    print(f"  Baseline MAE (last-5 average): {baseline_mae(df_test):.2f} pts")

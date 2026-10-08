"""
Day 6 — Baseline Forecasting Model.

Trains an XGBoost model to predict PM2.5 (and other key pollutants) 1 hour ahead,
using the lag/rolling/weather features from Day 5. Compares against a naive
baseline (persistence: "next hour = same as now") so you know whether the model
is actually adding value, not just producing numbers that look plausible.

Usage: python pipelines/forecasting/train_model.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import json

DATA_DIR = Path(__file__).parent.parent.parent / "data"
FEATURES_CSV = DATA_DIR / "processed" / "features.csv"
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

TARGET_POLLUTANT = "PM2.5"  # start with the one that matters most for health advisories

FEATURE_COLS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
    "PM2.5_lag1h", "PM2.5_lag2h", "PM2.5_lag3h", "PM2.5_lag6h",
    "PM2.5_roll3h_mean", "PM2.5_roll6h_mean",
    "temp_c", "humidity_pct", "wind_speed_ms", "wind_deg",
    "PM10_lag1h", "NO2_lag1h", "OZONE_lag1h", "CO_lag1h",
    "temp_c", "humidity_pct", "wind_speed_ms", "wind_deg",
]


def load_data():
    df = pd.read_csv(FEATURES_CSV, parse_dates=["timestamp"])
    df = df.dropna(subset=[TARGET_POLLUTANT])  # target must be known to train/evaluate on it
    return df


def naive_baseline_mae(df):
    """Persistence baseline: predict next value = PM2.5_lag1h (i.e. 'no change from last reading')."""
    valid = df.dropna(subset=["PM2.5_lag1h", TARGET_POLLUTANT])
    if valid.empty:
        return None
    return mean_absolute_error(valid[TARGET_POLLUTANT], valid["PM2.5_lag1h"])


def run():
    print("=== Day 6: Baseline Forecasting Model ===\n")
    df = load_data()

    # Only keep rows where we actually have a real 1-hour-ago reading — this is
    # the genuine "predict 1 hour ahead" task. Rows without it aren't a fair
    # 1-hour forecast problem, they're a different, harder problem (predicting
    # across a gap), and mixing them in breaks the baseline comparison.
    df = df.dropna(subset=["PM2.5_lag1h"])
    print(f"Rows with known {TARGET_POLLUTANT} AND a real 1-hour-ago reading: {len(df)}")

    available_features = [c for c in FEATURE_COLS if c in df.columns]
    missing_features = set(FEATURE_COLS) - set(available_features)
    if missing_features:
        print(f"Note: these expected feature columns weren't found, skipping them: {missing_features}")

    X = df[available_features]
    y = df[TARGET_POLLUTANT]

    if len(df) < 50:
        print(f"\nWARNING: only {len(df)} usable rows. Results won't be reliable yet.")

    df_sorted = df.sort_values("timestamp")
    split_idx = int(len(df_sorted) * 0.8)
    train_idx = df_sorted.index[:split_idx]
    test_idx = df_sorted.index[split_idx:]

    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    print(f"\nTrain rows: {len(X_train)}, Test rows: {len(X_test)}")

    if len(X_test) == 0:
        print("Not enough data for a test split yet.")
        return

    model = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    model_mae = mean_absolute_error(y_test, preds)
    model_rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"\n=== Model Performance (test set) ===")
    print(f"MAE:  {model_mae:.2f}")
    print(f"RMSE: {model_rmse:.2f}")

    # Baseline computed on the SAME test rows now — fair comparison
    test_df = df_sorted.loc[test_idx]
    baseline_mae = mean_absolute_error(test_df[TARGET_POLLUTANT], test_df["PM2.5_lag1h"])
    print(f"\n=== Naive Persistence Baseline (same test rows) ===")
    print(f"Baseline MAE: {baseline_mae:.2f}")

    if model_mae < baseline_mae:
        improvement = 100 * (baseline_mae - model_mae) / baseline_mae
        print(f"\nModel beats naive baseline by {improvement:.1f}% — model is adding real value.")
    else:
        print(f"\nModel is NOT beating the naive baseline. Given how strongly "
              f"autocorrelated PM2.5 is hour-to-hour, persistence is a genuinely "
              f"tough baseline — this is common in early air-quality forecasting "
              f"work, not necessarily a bug.")

    importance = dict(zip(available_features, model.feature_importances_.tolist()))
    importance = dict(sorted(importance.items(), key=lambda x: -x[1]))
    print(f"\n=== Top 5 Most Important Features ===")
    for feat, imp in list(importance.items())[:5]:
        print(f"  {feat}: {imp:.3f}")

    model.save_model(str(MODELS_DIR / "pm25_model.json"))
    with open(MODELS_DIR / "pm25_feature_importance.json", "w") as f:
        json.dump(importance, f, indent=2)

    print(f"\n=== Model saved to {MODELS_DIR / 'pm25_model.json'} ===")

if __name__ == "__main__":
    run()
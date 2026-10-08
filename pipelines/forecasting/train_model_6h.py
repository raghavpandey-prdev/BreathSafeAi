"""
6-Hour-Ahead Forecasting Model.

Same pipeline as the 24h version, but at a horizon where persistence shouldn't
have the "secretly aligned with the daily cycle" advantage — a real test of
whether the model can add value once that structural advantage is removed.

Usage: python pipelines/forecasting/train_model_6h.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import json
import sys
sys.path.append(str(Path(__file__).parent.parent / "features"))
from inference_helpers import load_baselines, get_station_baseline, build_city_dummies, CITY_DUMMY_COLS

DATA_DIR = Path(__file__).parent.parent.parent / "data"
FEATURES_CSV = DATA_DIR / "processed" / "features.csv"
HISTORICAL_CSV = DATA_DIR / "processed" / "historical_features.csv"
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

TARGET_POLLUTANT = "PM2.5"
FORECAST_HORIZON_HOURS = 6

FEATURE_COLS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend",
    "PM2.5_lag1h", "PM2.5_lag2h", "PM2.5_lag3h", "PM2.5_lag6h", "PM2.5_lag24h",
    "PM2.5_trend_24h",
    "PM2.5_roll3h_mean", "PM2.5_roll6h_mean",
    "PM10_lag1h", "NO2_lag1h", "OZONE_lag1h", "CO_lag1h",
    "temp_c", "humidity_pct", "wind_speed_ms", "wind_deg",
] + CITY_DUMMY_COLS


def load_data():
    live = pd.read_csv(FEATURES_CSV, parse_dates=["timestamp"])
    live["data_source"] = "live"

    if HISTORICAL_CSV.exists():
        hist = pd.read_csv(HISTORICAL_CSV, parse_dates=["timestamp"])
        hist["data_source"] = "historical"

        # UNION of columns, not intersection — historical data has no weather
        # columns, but that shouldn't silently strip weather from LIVE rows too.
        # Missing columns get filled with NaN; XGBoost handles NaN natively.
        all_cols = list(set(live.columns) | set(hist.columns))
        for col in all_cols:
            if col not in live.columns:
                live[col] = np.nan
            if col not in hist.columns:
                hist[col] = np.nan

        df = pd.concat([live[all_cols], hist[all_cols]], ignore_index=True)
        print(f"Combined: {len(live)} live rows + {len(hist)} historical rows = {len(df)} total")
    else:
        df = live
        print(f"No historical_features.csv found — using live data only ({len(df)} rows)")

    return df


def add_future_target(df):
    future = df[["station", "timestamp", "PM2.5"]].copy()
    future["timestamp"] = future["timestamp"] - pd.Timedelta(hours=FORECAST_HORIZON_HOURS)
    future = future.rename(columns={"PM2.5": f"PM2.5_target_{FORECAST_HORIZON_HOURS}h"})
    df = df.merge(future, on=["station", "timestamp"], how="left")
    return df


def add_station_baseline(df, train_idx=None):
    """Uses the PERSISTED baseline lookup (same file inference uses) instead
    of an on-the-fly train-only calculation — this is the fix: train and
    inference now use the identical source of truth."""
    baselines = load_baselines()
    df["station_baseline_pm25"] = df.apply(
        lambda r: get_station_baseline(r["station"], r["city"], baselines), axis=1
    )
    city_dummies = df["city"].apply(build_city_dummies).apply(pd.Series)
    df = pd.concat([df, city_dummies], axis=1)
    return df


def run():
    target_col = f"PM2.5_target_{FORECAST_HORIZON_HOURS}h"
    print(f"=== Forecasting Model: {FORECAST_HORIZON_HOURS}h ahead ===\n")

    df = load_data()
    df = add_future_target(df)
    df = df.dropna(subset=["PM2.5_lag1h", target_col, "PM2.5"])
    print(f"Rows with a real current reading AND a real {FORECAST_HORIZON_HOURS}h-future reading: {len(df)}")

    available_features = [c for c in FEATURE_COLS if c in df.columns]
    missing_features = set(FEATURE_COLS) - set(available_features)
    if missing_features:
        print(f"Note: expected feature columns not found, skipping: {missing_features}")

    y = df[target_col]

    df_sorted = df.sort_values("timestamp")
    split_idx = int(len(df_sorted) * 0.8)
    train_idx = df_sorted.index[:split_idx]
    test_idx = df_sorted.index[split_idx:]

    df_sorted = add_station_baseline(df_sorted)

    feature_cols_with_baseline = available_features + ["station_baseline_pm25"] + CITY_DUMMY_COLS
    feature_cols_with_baseline = [c for c in feature_cols_with_baseline if c in df_sorted.columns]
    X_train = df_sorted.loc[train_idx, feature_cols_with_baseline]
    X_test = df_sorted.loc[test_idx, feature_cols_with_baseline]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    print(f"\nTrain rows: {len(X_train)}, Test rows: {len(X_test)}")
    print(f"Test set sources: {df_sorted.loc[test_idx, 'data_source'].value_counts().to_dict()}")

    if len(X_test) == 0:
        print("Not enough data for a test split.")
        return

    model = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    print(f"\nFeatures the model actually used: {list(X_train.columns)}")

    preds = model.predict(X_test)
    model_mae = mean_absolute_error(y_test, preds)
    model_rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"\n=== Model Performance (test set, {FORECAST_HORIZON_HOURS}h ahead) ===")
    print(f"MAE:  {model_mae:.2f}")
    print(f"RMSE: {model_rmse:.2f}")

    test_df = df_sorted.loc[test_idx]
    baseline_mae = mean_absolute_error(test_df[target_col], test_df["PM2.5"])
    print(f"\n=== Naive Persistence Baseline ('in {FORECAST_HORIZON_HOURS}h = now') ===")
    print(f"Baseline MAE: {baseline_mae:.2f}")

    if model_mae < baseline_mae:
        improvement = 100 * (baseline_mae - model_mae) / baseline_mae
        print(f"\nModel beats naive baseline by {improvement:.1f}% — genuine forecasting value.")
    else:
        print(f"\nModel still not beating baseline at {FORECAST_HORIZON_HOURS}h. This would suggest "
              f"the issue isn't horizon-specific, and points more toward needing spatial/weather-"
              f"forecast features, or a fundamentally different model architecture.")

    importance = dict(zip(feature_cols_with_baseline, model.feature_importances_.tolist()))
    importance = dict(sorted(importance.items(), key=lambda x: -x[1]))
    print(f"\n=== Top 5 Most Important Features ===")
    for feat, imp in list(importance.items())[:5]:
        print(f"  {feat}: {imp:.3f}")

    model.save_model(str(MODELS_DIR / f"pm25_model_{FORECAST_HORIZON_HOURS}h.json"))
    with open(MODELS_DIR / f"pm25_feature_importance_{FORECAST_HORIZON_HOURS}h.json", "w") as f:
        json.dump(importance, f, indent=2)

    print(f"\n=== Model saved to {MODELS_DIR / f'pm25_model_{FORECAST_HORIZON_HOURS}h.json'} ===")


if __name__ == "__main__":
    run()
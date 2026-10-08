"""
Day 6/7 — 24-Hour-Ahead Forecasting Model (with historical backfill).

Combines 5.5 years of Kaggle historical data with live collection to train a
much more data-rich model. Historical and live data are concatenated but their
different station-naming schemes are preserved as-is (not force-merged), since
station identity isn't perfectly reconciled between the two sources.

Usage: python pipelines/forecasting/train_model_24h.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import json

DATA_DIR = Path(__file__).parent.parent.parent / "data"
FEATURES_CSV = DATA_DIR / "processed" / "features.csv"
HISTORICAL_CSV = DATA_DIR / "processed" / "historical_features.csv"
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

TARGET_POLLUTANT = "PM2.5"
FORECAST_HORIZON_HOURS = 24

FEATURE_COLS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend",
    "PM2.5_lag1h", "PM2.5_lag2h", "PM2.5_lag3h", "PM2.5_lag6h",
    "temp_c", "humidity_pct", "wind_speed_ms", "wind_deg",
    "PM2.5_roll3h_mean", "PM2.5_roll6h_mean", "PM2.5_lag24h", "PM2.5_trend_24h",
    "PM10_lag1h", "NO2_lag1h", "OZONE_lag1h", "CO_lag1h",
] # NOTE: weather cols (temp_c etc.) dropped from this list — historical data has no weather


def load_data():
    live = pd.read_csv(FEATURES_CSV, parse_dates=["timestamp"])
    live["data_source"] = "live"

    if HISTORICAL_CSV.exists():
        hist = pd.read_csv(HISTORICAL_CSV, parse_dates=["timestamp"])
        hist["data_source"] = "historical"
        common_cols = list(set(live.columns) & set(hist.columns))
        df = pd.concat([live[common_cols], hist[common_cols]], ignore_index=True)
        print(f"Combined: {len(live)} live rows + {len(hist)} historical rows = {len(df)} total")
    else:
        df = live
        print(f"No historical_features.csv found — using live data only ({len(df)} rows)")

    return df


def add_future_target(df):
    future = df[["station", "timestamp", "PM2.5"]].copy()
    future["timestamp"] = future["timestamp"] - pd.Timedelta(hours=FORECAST_HORIZON_HOURS)
    future = future.rename(columns={"PM2.5": "PM2.5_target_24h"})
    df = df.merge(future, on=["station", "timestamp"], how="left")
    return df


def add_station_baseline(df, train_idx):
    station_avg = df.loc[train_idx].groupby("station")["PM2.5"].mean()
    df["station_baseline_pm25"] = df["station"].map(station_avg)
    df["station_baseline_pm25"] = df["station_baseline_pm25"].fillna(df.loc[train_idx, "PM2.5"].mean())
    return df


def run():
    print(f"=== Forecasting Model: {FORECAST_HORIZON_HOURS}h ahead (with historical backfill) ===\n")

    df = load_data()
    df = add_future_target(df)
    df = df.dropna(subset=["PM2.5_lag1h", "PM2.5_target_24h","PM2.5"])
    print(f"Rows with a real current reading AND a real {FORECAST_HORIZON_HOURS}h-future reading: {len(df)}")

    available_features = [c for c in FEATURE_COLS if c in df.columns]
    missing_features = set(FEATURE_COLS) - set(available_features)
    if missing_features:
        print(f"Note: expected feature columns not found, skipping: {missing_features}")

    y = df["PM2.5_target_24h"]

    df_sorted = df.sort_values("timestamp")
    split_idx = int(len(df_sorted) * 0.8)
    train_idx = df_sorted.index[:split_idx]
    test_idx = df_sorted.index[split_idx:]

    df_sorted = add_station_baseline(df_sorted, train_idx)

    feature_cols_with_baseline = available_features + ["station_baseline_pm25"]
    X_train = df_sorted.loc[train_idx, feature_cols_with_baseline]
    X_test = df_sorted.loc[test_idx, feature_cols_with_baseline]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    print(f"\nTrain rows: {len(X_train)}, Test rows: {len(X_test)}")
    print(f"Test set sources: {df_sorted.loc[test_idx, 'data_source'].value_counts().to_dict()}")

    if len(X_test) == 0:
        print("Not enough data for a test split.")
        return

    # More data now supports more model complexity again
    model = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    model_mae = mean_absolute_error(y_test, preds)
    model_rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"\n=== Model Performance (test set, {FORECAST_HORIZON_HOURS}h ahead) ===")
    print(f"MAE:  {model_mae:.2f}")
    print(f"RMSE: {model_rmse:.2f}")

    test_df = df_sorted.loc[test_idx]
    baseline_mae = mean_absolute_error(test_df["PM2.5_target_24h"], test_df["PM2.5"])
    print(f"\n=== Naive Persistence Baseline ('tomorrow = today') ===")
    print(f"Baseline MAE: {baseline_mae:.2f}")

    if model_mae < baseline_mae:
        improvement = 100 * (baseline_mae - model_mae) / baseline_mae
        print(f"\nModel beats naive baseline by {improvement:.1f}% — genuine forecasting value.")
    else:
        print(f"\nModel still not beating baseline. With ~2M rows now, this would be a "
              f"more concerning signal — investigate feature quality if this persists.")

    importance = dict(zip(feature_cols_with_baseline, model.feature_importances_.tolist()))
    importance = dict(sorted(importance.items(), key=lambda x: -x[1]))
    print(f"\n=== Top 5 Most Important Features ===")
    for feat, imp in list(importance.items())[:5]:
        print(f"  {feat}: {imp:.3f}")

    model.save_model(str(MODELS_DIR / "pm25_model_24h.json"))
    with open(MODELS_DIR / "pm25_feature_importance_24h.json", "w") as f:
        json.dump(importance, f, indent=2)

    print(f"\n=== Model saved to {MODELS_DIR / 'pm25_model_24h.json'} ===")


if __name__ == "__main__":
    run()
"""
Model Evaluation — XGBoost (conformal) vs. Naive Baselines.

Answers the question every interviewer asks first: "How do you know your
model is actually better than doing nothing clever?" Evaluates the deployed
6h-ahead model against two standard forecasting baselines on the SAME test
split used at training time:

  1. Persistence  — "PM2.5 in 6h = PM2.5 right now" (no model at all)
  2. Climatology   — "PM2.5 in 6h = this station's average for that hour of
                       day", computed ONLY from the training portion (no leakage)

Reports MAE, RMSE, MAPE, and R^2 for all three, plus the model's empirical
conformal interval coverage, and saves a comparison bar chart.

Usage: python pipelines/evaluation/evaluate_model.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).parent.parent / "features"))
from inference_helpers import load_baselines, get_station_baseline, build_city_dummies, CITY_DUMMY_COLS

DATA_DIR = Path(__file__).parent.parent.parent / "data"
FEATURES_CSV = DATA_DIR / "processed" / "features.csv"
HISTORICAL_CSV = DATA_DIR / "processed" / "historical_features.csv"
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
REPORTS_DIR = Path(__file__).parent.parent.parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

FORECAST_HORIZON_HOURS = 6
TARGET_CONFIDENCE = 0.90

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
        all_cols = list(set(live.columns) | set(hist.columns))
        for col in all_cols:
            if col not in live.columns:
                live[col] = np.nan
            if col not in hist.columns:
                hist[col] = np.nan
        df = pd.concat([live[all_cols], hist[all_cols]], ignore_index=True)
    else:
        df = live

    return df


def add_future_target(df):
    future = df[["station", "timestamp", "PM2.5"]].copy()
    future["timestamp"] = future["timestamp"] - pd.Timedelta(hours=FORECAST_HORIZON_HOURS)
    future = future.rename(columns={"PM2.5": f"PM2.5_target_{FORECAST_HORIZON_HOURS}h"})
    df = df.merge(future, on=["station", "timestamp"], how="left")
    return df


def add_station_baseline(df):
    baselines = load_baselines()
    df["station_baseline_pm25"] = df.apply(
        lambda r: get_station_baseline(r["station"], r["city"], baselines), axis=1
    )
    city_dummies = df["city"].apply(build_city_dummies).apply(pd.Series)
    df = pd.concat([df, city_dummies], axis=1)
    return df


def mape(y_true, y_pred, eps=1.0):
    """Standard MAPE, floored to avoid blowing up near-zero PM2.5 readings."""
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def metrics_row(name, y_true, y_pred):
    return {
        "model": name,
        "MAE": round(mean_absolute_error(y_true, y_pred), 2),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
        "MAPE_%": round(mape(y_true, y_pred), 1),
        "R2": round(r2_score(y_true, y_pred), 3),
    }


def run():
    target_col = f"PM2.5_target_{FORECAST_HORIZON_HOURS}h"
    print("=== Evaluation: XGBoost (conformal) vs. Naive Baselines ===\n")

    df = load_data()
    df = add_future_target(df)
    df = df.dropna(subset=["PM2.5_lag1h", target_col, "PM2.5"])
    df = df.sort_values("timestamp")

    n = len(df)
    train_end = int(n * 0.70)
    train_idx = df.index[:train_end]
    remaining_idx = df.index[train_end:]
    remaining_df = df.loc[remaining_idx]
    cal_idx, test_idx = train_test_split(
        remaining_idx, test_size=0.5, stratify=remaining_df["month"], random_state=42
    )

    df = add_station_baseline(df)
    feature_cols = [c for c in FEATURE_COLS if c in df.columns] + ["station_baseline_pm25"]
    feature_cols = [c for c in dict.fromkeys(feature_cols) if c in df.columns]  # dedupe, keep order

    X_test = df.loc[test_idx, feature_cols]
    y_test = df.loc[test_idx, target_col].values

    # --- Load the already-trained, already-deployed model ---
    model = xgb.XGBRegressor()
    model.load_model(str(MODELS_DIR / "pm25_model_6h_conformal.json"))
    with open(MODELS_DIR / "pm25_conformal_params.json") as f:
        conformal_params = json.load(f)

    model_preds = model.predict(X_test)

    # --- Baseline 1: Persistence (current PM2.5 = forecasted PM2.5) ---
    persistence_preds = df.loc[test_idx, "PM2.5"].values

    # --- Baseline 2: Climatology (station's hour-of-day average, TRAIN-ONLY) ---
    train_df = df.loc[train_idx]
    hourly_station_avg = train_df.groupby(["station", "hour"])["PM2.5"].mean()
    global_hourly_avg = train_df.groupby("hour")["PM2.5"].mean()
    global_avg = train_df["PM2.5"].mean()

    def climatology_lookup(row):
        key = (row["station"], row["hour"])
        if key in hourly_station_avg.index:
            return hourly_station_avg.loc[key]
        if row["hour"] in global_hourly_avg.index:
            return global_hourly_avg.loc[row["hour"]]
        return global_avg

    climatology_preds = df.loc[test_idx].apply(climatology_lookup, axis=1).values

    # --- Compare ---
    results = [
        metrics_row("Persistence (naive)", y_test, persistence_preds),
        metrics_row("Climatology (hour-of-day avg)", y_test, climatology_preds),
        metrics_row("XGBoost (conformal, deployed)", y_test, model_preds),
    ]
    results_df = pd.DataFrame(results)

    print(results_df.to_string(index=False))

    best_mae = results_df.loc[results_df["MAE"].idxmin(), "model"]
    xgb_mae = results_df.loc[results_df["model"] == "XGBoost (conformal, deployed)", "MAE"].values[0]
    persistence_mae = results_df.loc[results_df["model"] == "Persistence (naive)", "MAE"].values[0]
    improvement = (persistence_mae - xgb_mae) / persistence_mae * 100

    print(f"\nBest model by MAE: {best_mae}")
    print(f"XGBoost improves on persistence by {improvement:.1f}% (MAE {persistence_mae} -> {xgb_mae})")

    # --- Conformal interval coverage, re-verified on this exact test slice ---
    def q_hat_for_row(row):
        month = row["month"]
        if 1 <= month <= 2 or month == 12:
            season = "winter"
        elif 3 <= month <= 5:
            season = "summer"
        else:
            season = "monsoon"
        return conformal_params["season_q_hats"].get(season, conformal_params["global_q_hat"])

    q_hats = df.loc[test_idx].apply(q_hat_for_row, axis=1).values
    lower = model_preds - q_hats
    upper = model_preds + q_hats
    coverage = float(np.mean((y_test >= lower) & (y_test <= upper)))
    print(f"\n90% prediction interval empirical coverage on this test slice: {coverage*100:.1f}%")

    # --- Save results for the README / resume writeup ---
    results_df.to_csv(REPORTS_DIR / "baseline_comparison.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(results_df["model"], results_df["MAE"], color=["#999999", "#6c9bd1", "#2e7d32"])
    ax.set_ylabel("MAE (µg/m³, lower is better)")
    ax.set_title(f"{FORECAST_HORIZON_HOURS}h-Ahead PM2.5 Forecast: Model vs. Baselines")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "baseline_comparison.png", dpi=150)
    print(f"\nSaved: {REPORTS_DIR / 'baseline_comparison.csv'}")
    print(f"Saved: {REPORTS_DIR / 'baseline_comparison.png'}")


if __name__ == "__main__":
    run()
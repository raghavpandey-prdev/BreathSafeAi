"""
Day 8 — Uncertainty Quantification via Split Conformal Prediction.

Wraps the 6h PM2.5 forecasting model with honest prediction intervals, instead
of a single (falsely precise) point number. Uses split-conformal prediction:
a held-out calibration set measures how wrong the model actually tends to be,
and that's used to build intervals with a guaranteed (approximate) coverage
rate — e.g. "the true value falls in this range ~90% of the time," which is
exactly the kind of claim you can actually stand behind in a health advisory.

Usage: python pipelines/forecasting/train_model_6h_conformal.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error
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

FORECAST_HORIZON_HOURS = 6
TARGET_CONFIDENCE = 0.90  # 90% prediction intervals

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
    print(f"=== Day 8: Conformal Prediction for {FORECAST_HORIZON_HOURS}h Forecast (season-stratified) ===\n")

    df = load_data()
    df = add_future_target(df)
    df = df.dropna(subset=["PM2.5_lag1h", target_col, "PM2.5"])
    print(f"Usable rows: {len(df)}")

    available_features = [c for c in FEATURE_COLS if c in df.columns]
    y = df[target_col]

    # Three-way TIME-based split: train (70%) -> calibration (15%) -> test (15%)
    from sklearn.model_selection import train_test_split

    df_sorted = df.sort_values("timestamp")
    n = len(df_sorted)
    train_end = int(n * 0.70)

    train_idx = df_sorted.index[:train_end]
    remaining_idx = df_sorted.index[train_end:]

    # Training stays strictly earliest-in-time (no leakage into the model).
    # But calibration/test are drawn as a STRATIFIED random split by month from
    # everything after training — this guarantees every season is represented
    # in calibration, instead of risking another narrow chronological window
    # that happens to miss a whole season (as just happened with spring).
    remaining_df = df_sorted.loc[remaining_idx]
    cal_idx, test_idx = train_test_split(
        remaining_idx, test_size=0.5, stratify=remaining_df["month"], random_state=42
    )

    df_sorted = add_station_baseline(df_sorted)
    feature_cols_with_baseline = available_features + ["station_baseline_pm25"] + CITY_DUMMY_COLS
    feature_cols_with_baseline = [c for c in feature_cols_with_baseline if c in df_sorted.columns]

    X_train = df_sorted.loc[train_idx, feature_cols_with_baseline]
    X_cal = df_sorted.loc[cal_idx, feature_cols_with_baseline]
    X_test = df_sorted.loc[test_idx, feature_cols_with_baseline]
    y_train = y.loc[train_idx]
    y_cal = y.loc[cal_idx]
    y_test = y.loc[test_idx]

    print(f"Train: {len(X_train)}, Calibration: {len(X_cal)}, Test: {len(X_test)}")

    # Train the point-prediction model
    model = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)

    # --- Conformal calibration step, STRATIFIED BY SEASON ---
    # A single global interval width gets sized to cover worst-case variance
    # (winter spikes), making it needlessly wide the rest of the year. Instead,
    # compute a separate q_hat per season so calm periods get tighter, more
    # useful intervals, and volatile periods get appropriately wider ones.
    df_sorted["season"] = pd.cut(
        df_sorted["month"], bins=[0, 2, 5, 9, 12],
        labels=["winter", "summer", "monsoon", "winter2"], ordered=False
    )
    df_sorted["season"] = df_sorted["season"].replace("winter2", "winter")

    cal_preds = model.predict(X_cal)
    cal_df = df_sorted.loc[cal_idx].copy()
    cal_df["residual"] = np.abs(y_cal.values - cal_preds)

    season_q_hats = {}
    for season in cal_df["season"].unique():
        season_residuals = cal_df.loc[cal_df["season"] == season, "residual"]
        n_season = len(season_residuals)
        if n_season < 30:
            continue  # not enough calibration data for this season, skip
        q_level = min(1.0, np.ceil((n_season + 1) * TARGET_CONFIDENCE) / n_season)
        season_q_hats[season] = float(np.quantile(season_residuals, q_level))

    print(f"\nPer-season q_hat (interval half-width):")
    for season, q in season_q_hats.items():
        print(f"  {season}: {q:.2f}")

    # --- Apply to test set, using each row's own season's q_hat ---
    test_df = df_sorted.loc[test_idx].copy()
    test_preds = model.predict(X_test)
    test_df["pred"] = test_preds
    test_df["q_hat"] = test_df["season"].map(season_q_hats)

    global_q_level = min(1.0, np.ceil((len(cal_df) + 1) * TARGET_CONFIDENCE) / len(cal_df))
    global_q_hat = float(np.quantile(cal_df["residual"], global_q_level))
    test_df["q_hat"] = test_df["q_hat"].fillna(global_q_hat)

    lower = test_df["pred"] - test_df["q_hat"]
    upper = test_df["pred"] + test_df["q_hat"]

    mae = mean_absolute_error(y_test, test_preds)
    print(f"\nPoint-forecast MAE on test set: {mae:.2f}")

    within_interval = (y_test.values >= lower.values) & (y_test.values <= upper.values)
    empirical_coverage = within_interval.mean()
    avg_interval_width = (upper - lower).mean()

    print(f"\n=== Conformal Prediction Results (season-stratified) ===")
    print(f"Target coverage: {TARGET_CONFIDENCE*100:.0f}%")
    print(f"Empirical coverage on test set: {empirical_coverage*100:.1f}%")
    print(f"Average interval width: {avg_interval_width:.2f}")

    if abs(empirical_coverage - TARGET_CONFIDENCE) < 0.03:
        print("\nCalibration looks good — empirical coverage closely matches target.")
    elif empirical_coverage < TARGET_CONFIDENCE:
        print("\nEmpirical coverage is LOWER than target — intervals too narrow.")
    else:
        print("\nEmpirical coverage is HIGHER than target — intervals still conservative, "
              "but should be meaningfully tighter than the non-stratified version.")

    model.save_model(str(MODELS_DIR / "pm25_model_6h_conformal.json"))
    with open(MODELS_DIR / "pm25_conformal_params.json", "w") as f:
        json.dump({
            "season_q_hats": season_q_hats,
            "global_q_hat": global_q_hat,
            "target_confidence": TARGET_CONFIDENCE,
            "empirical_coverage": float(empirical_coverage),
            "forecast_horizon_hours": FORECAST_HORIZON_HOURS,
        }, f, indent=2)

    print(f"\n=== Saved model + season-stratified conformal params to {MODELS_DIR} ===")


if __name__ == "__main__":
    run()
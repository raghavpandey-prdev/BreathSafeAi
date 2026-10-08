"""
Phase 1b — Train multi-horizon PM2.5 forecasting models (1, 3, 6, 12, 24h).

Fixes vs. the old training script:
  * TIME-BLOCKED split, not random-within-tail: train on 2015-2018, calibrate
    on 2019, test on 2020. The old split stratified cal/test randomly by month
    from the same tail window, so adjacent hours could land in both sets —
    that leaks information and inflates the coverage number.
  * Current PM2.5 reading is now a model input (it was missing before).
  * 12h/24h models get "forecast weather" (temp/humidity/wind/precip at
    t+12h, t+24h) as an input, not just weather at t=0 — this is what makes
    longer horizons workable at all.
  * Every model is benchmarked against persistence and climatology, per city,
    not just in aggregate — so a model that's only good in Delhi can't hide
    behind a good-looking overall number.

Usage: python pipelines/forecasting/train_multi_horizon.py
Output: models/pm25_model_{h}h.json, models/pm25_conformal_{h}h.json,
        models/model_manifest.json, reports/multi_horizon_evaluation.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_DIR = Path(__file__).parent.parent.parent / "data"
TRAINING_PARQUET = DATA_DIR / "processed" / "training_data.parquet"
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
REPORTS_DIR = Path(__file__).parent.parent.parent / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

HORIZONS = [1, 3, 6, 12, 24]
TARGET_CONFIDENCE = 0.90
TRAIN_YEARS_MAX = 2018   # train: <= this year
CAL_YEAR = 2019          # calibration: exactly this year
TEST_YEAR = 2020         # test: exactly this year

BASE_FEATURES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend",
    "PM2.5",
    "PM2.5_lag1h", "PM2.5_lag2h", "PM2.5_lag3h", "PM2.5_lag6h", "PM2.5_lag12h", "PM2.5_lag24h",
    "PM2.5_change_1h", "PM2.5_change_24h",
    "PM2.5_roll3h_mean", "PM2.5_roll6h_mean", "PM2.5_roll24h_mean", "PM2.5_roll24h_std",
    "PM10_lag1h", "NO2_lag1h", "OZONE_lag1h", "CO_lag1h",
    "temp_c", "humidity_pct", "wind_speed_ms", "wind_deg", "pressure_hpa", "precip_mm",
]

# Nearest cached future-weather horizon (6/12/24h) to use for each target horizon.
FUTURE_WEATHER_FOR_HORIZON = {1: 6, 3: 6, 6: 6, 12: 12, 24: 24}
FUTURE_WEATHER_VARS = ["temp_c", "humidity_pct", "wind_speed_ms", "precip_mm"]


def city_dummy_cols(cities):
    return [f"city_{c.replace(' ', '_')}" for c in sorted(cities)]


def add_city_dummies(df, cities):
    for c in sorted(cities):
        df[f"city_{c.replace(' ', '_')}"] = (df["city"] == c).astype(np.int8)
    return df


def add_station_baselines(df, train_mask):
    """Per-station mean PM2.5, computed ONLY from training rows (no leakage),
    with a city-level fallback for stations too sparse to trust on their own."""
    train = df.loc[train_mask]
    station_mean = train.groupby("station")["PM2.5"].mean()
    city_mean = train.groupby("city")["PM2.5"].mean()
    global_mean = train["PM2.5"].mean()

    df["station_baseline_pm25"] = df["station"].map(station_mean)
    df["station_baseline_pm25"] = df["station_baseline_pm25"].fillna(df["city"].map(city_mean))
    df["station_baseline_pm25"] = df["station_baseline_pm25"].fillna(global_mean)

    baselines = {
        "stations": {k: round(float(v), 1) for k, v in station_mean.items()},
        "cities": {k: round(float(v), 1) for k, v in city_mean.items()},
        "global": round(float(global_mean), 1),
    }
    return df, baselines


def climatology_predict(df, mask, train_mask):
    """Station's hour-of-day average PM2.5, computed from training rows only."""
    train = df.loc[train_mask]
    hourly_station = train.groupby(["station", "hour"])["PM2.5"].mean()
    hourly_city = train.groupby(["city", "hour"])["PM2.5"].mean()
    global_mean = train["PM2.5"].mean()

    sub = df.loc[mask]

    def lookup(row):
        key = (row["station"], row["hour"])
        if key in hourly_station.index:
            return hourly_station.loc[key]
        key2 = (row["city"], row["hour"])
        if key2 in hourly_city.index:
            return hourly_city.loc[key2]
        return global_mean

    return sub.apply(lookup, axis=1).values


def metrics_row(name, y_true, y_pred):
    return {
        "model": name,
        "MAE": round(mean_absolute_error(y_true, y_pred), 2),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
    }


def train_one_horizon(df, horizon, cities, dummy_cols):
    target_col = f"PM2.5_target_{horizon}h"
    fw_h = FUTURE_WEATHER_FOR_HORIZON[horizon]
    future_weather_cols = [f"{v}_fut{fw_h}h" for v in FUTURE_WEATHER_VARS] if horizon > 3 else []

    feature_cols = [c for c in BASE_FEATURES if c in df.columns] + future_weather_cols + dummy_cols
    feature_cols += ["station_baseline_pm25"]

    d = df.dropna(subset=["PM2.5_lag1h", target_col]).copy()

    train_mask = d["year"] <= TRAIN_YEARS_MAX
    cal_mask = d["year"] == CAL_YEAR
    test_mask = d["year"] == TEST_YEAR

    d, baselines = add_station_baselines(d, train_mask)
    feature_cols = [c for c in dict.fromkeys(feature_cols) if c in d.columns]

    X_train, y_train = d.loc[train_mask, feature_cols], d.loc[train_mask, target_col]
    X_cal, y_cal = d.loc[cal_mask, feature_cols], d.loc[cal_mask, target_col]
    X_test, y_test = d.loc[test_mask, feature_cols], d.loc[test_mask, target_col]

    print(f"\n--- {horizon}h horizon ---")
    print(f"Train: {len(X_train):,} ({d.loc[train_mask,'year'].min()}-{TRAIN_YEARS_MAX}) | "
          f"Cal: {len(X_cal):,} ({CAL_YEAR}) | Test: {len(X_test):,} ({TEST_YEAR})")

    if len(X_train) < 1000 or len(X_cal) < 200 or len(X_test) < 200:
        print(f"  Skipping {horizon}h — not enough rows in one of the splits for this city set.")
        return None

    model = xgb.XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    cal_preds = model.predict(X_cal)
    cal_df = d.loc[cal_mask].copy()
    cal_persistence = cal_df["PM2.5"].values

    # --- per-city blend weight, chosen on the CALIBRATION set only ---
    # weight=1.0 means "trust the model fully"; lower values shrink toward
    # persistence for cities where the pooled model hasn't earned that trust.
    blend_weights = {}
    for city in cities:
        m = (cal_df["city"] == city).values
        if m.sum() < 200:
            continue
        best_w, best_mae = 1.0, mean_absolute_error(y_cal.values[m], cal_preds[m])
        for w in np.arange(0.0, 1.01, 0.1):
            blended = w * cal_preds[m] + (1 - w) * cal_persistence[m]
            mae = mean_absolute_error(y_cal.values[m], blended)
            if mae < best_mae:
                best_w, best_mae = float(w), mae
        blend_weights[city] = round(best_w, 1)
    default_weight = round(float(np.median(list(blend_weights.values()))), 1) if blend_weights else 1.0

    print(f"  Blend weights (1.0=pure model, 0.0=pure persistence): {blend_weights}")

    cal_weight_arr = cal_df["city"].map(blend_weights).fillna(default_weight).values
    cal_blended = cal_weight_arr * cal_preds + (1 - cal_weight_arr) * cal_persistence

    # --- season-stratified conformal calibration, on the BLENDED predictions
    # (intervals have to match what's actually served, not the raw model) ---
    cal_df["residual"] = np.abs(y_cal.values - cal_blended)
    cal_df["season"] = pd.cut(
        cal_df["month"], bins=[0, 2, 5, 9, 12],
        labels=["winter", "summer", "monsoon", "winter2"], ordered=False
    ).replace("winter2", "winter")

    season_q_hats = {}
    for season in cal_df["season"].unique():
        residuals = cal_df.loc[cal_df["season"] == season, "residual"]
        n = len(residuals)
        if n < 30:
            continue
        q_level = min(1.0, np.ceil((n + 1) * TARGET_CONFIDENCE) / n)
        season_q_hats[str(season)] = float(np.quantile(residuals, q_level))

    global_q_level = min(1.0, np.ceil((len(cal_df) + 1) * TARGET_CONFIDENCE) / len(cal_df))
    global_q_hat = float(np.quantile(cal_df["residual"], global_q_level))

    # --- test set: raw model vs. blended model vs. persistence vs. climatology ---
    test_df = d.loc[test_mask].copy()
    test_preds = model.predict(X_test)
    persistence_preds = test_df["PM2.5"].values
    test_weight_arr = test_df["city"].map(blend_weights).fillna(default_weight).values
    blended_preds = test_weight_arr * test_preds + (1 - test_weight_arr) * persistence_preds

    test_df["season"] = pd.cut(
        test_df["month"], bins=[0, 2, 5, 9, 12],
        labels=["winter", "summer", "monsoon", "winter2"], ordered=False
    ).replace("winter2", "winter")
    q_hats = test_df["season"].map(season_q_hats).fillna(global_q_hat).values
    lower, upper = blended_preds - q_hats, blended_preds + q_hats
    coverage = float(np.mean((y_test.values >= lower) & (y_test.values <= upper)))

    climatology_preds = climatology_predict(d, test_mask, train_mask)

    overall = [
        metrics_row("Persistence", y_test, persistence_preds),
        metrics_row("Climatology", y_test, climatology_preds),
        metrics_row("XGBoost (raw)", y_test, test_preds),
        metrics_row("XGBoost (blended, deployed)", y_test, blended_preds),
    ]
    for row in overall:
        row["horizon_h"] = horizon
        row["city"] = "ALL"
    print(pd.DataFrame(overall).to_string(index=False))

    per_city_rows = []
    for city in cities:
        m = (test_df["city"] == city).values
        if m.sum() < 100:
            continue
        per_city_rows.append({**metrics_row("XGBoost (blended)", y_test[m], blended_preds[m]),
                              "horizon_h": horizon, "city": city})
        per_city_rows.append({**metrics_row("Persistence", y_test[m], persistence_preds[m]),
                              "horizon_h": horizon, "city": city})

    model.save_model(str(MODELS_DIR / f"pm25_model_{horizon}h.json"))
    with open(MODELS_DIR / f"pm25_conformal_{horizon}h.json", "w") as f:
        json.dump({
            "horizon_hours": horizon,
            "feature_cols": feature_cols,
            "blend_weights": blend_weights,
            "default_blend_weight": default_weight,
            "season_q_hats": season_q_hats,
            "global_q_hat": global_q_hat,
            "target_confidence": TARGET_CONFIDENCE,
            "empirical_coverage": coverage,
            "test_mae_blended": round(mean_absolute_error(y_test, blended_preds), 2),
            "test_mae_raw_model": round(mean_absolute_error(y_test, test_preds), 2),
            "persistence_mae": round(mean_absolute_error(y_test, persistence_preds), 2),
            "climatology_mae": round(mean_absolute_error(y_test, climatology_preds), 2),
        }, f, indent=2)

    return overall + per_city_rows, baselines


def run():
    print("=== Training multi-horizon models (time-blocked: train<=2018, cal=2019, test=2020) ===")
    df = pd.read_parquet(TRAINING_PARQUET)
    cities = sorted(df["city"].unique())
    dummy_cols = city_dummy_cols(cities)
    df = add_city_dummies(df, cities)
    print(f"Loaded {len(df):,} rows across {len(cities)} cities: {cities}")

    all_rows = []
    station_baselines = None
    for h in HORIZONS:
        result = train_one_horizon(df, h, cities, dummy_cols)
        if result is None:
            continue
        rows, baselines = result
        all_rows.extend(rows)
        if station_baselines is None:
            station_baselines = baselines  # same across horizons, save once

    report = pd.DataFrame(all_rows)
    report.to_csv(REPORTS_DIR / "multi_horizon_evaluation.csv", index=False)
    print(f"\nSaved evaluation report: {REPORTS_DIR / 'multi_horizon_evaluation.csv'}")

    with open(MODELS_DIR / "station_baselines_v2.json", "w") as f:
        json.dump(station_baselines, f, indent=2)

    with open(MODELS_DIR / "model_manifest.json", "w") as f:
        json.dump({
            "horizons": [h for h in HORIZONS if (MODELS_DIR / f"pm25_model_{h}h.json").exists()],
            "cities": cities,
            "city_dummy_cols": dummy_cols,
            "train_years": f"<= {TRAIN_YEARS_MAX}", "cal_year": CAL_YEAR, "test_year": TEST_YEAR,
        }, f, indent=2)
    print(f"Saved model manifest: {MODELS_DIR / 'model_manifest.json'}")


if __name__ == "__main__":
    run()
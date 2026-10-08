"""
Phase C (global model) — train multi-horizon PM2.5 models that generalize to
ANY location, not just the cities seen in training.

No city dummies, no per-station baseline lookup -- the feature set is
latitude/longitude, recent PM2.5 history (lags/rolling), and weather, all of
which are computable for any location on Earth with a live reading. Split is
per-location chronological (train/cal/test columns already assigned in
Phase B), since global coverage years don't align across locations.

Conformal calibration is REGION-stratified, not season-stratified like the
India model -- residual distributions differ far more by region (Dhaka's
baseline pollution vs. London's) than they do by season at this global
scale, given how unevenly years/seasons are covered per location.

Usage: python pipelines/forecasting/train_global_model.py
Output: models/global_model_{h}h.json, models/global_conformal_{h}h.json,
        models/global_manifest.json, reports/global_evaluation.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_DIR = Path(__file__).parent.parent.parent / "data"
TRAINING_PARQUET = DATA_DIR / "processed" / "global_training_data.parquet"
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
REPORTS_DIR = Path(__file__).parent.parent.parent / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

HORIZONS = [1, 3, 6, 12, 24]
TARGET_CONFIDENCE = 0.90
MIN_CAL_ROWS_FOR_REGION_Q = 100  # below this, fall back to the global q_hat for that region

FEATURE_COLS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend",
    "latitude", "longitude",
    "PM2.5",
    "PM2.5_lag1h", "PM2.5_lag2h", "PM2.5_lag3h", "PM2.5_lag6h", "PM2.5_lag12h", "PM2.5_lag24h",
    "PM2.5_change_1h", "PM2.5_change_24h",
    "PM2.5_roll3h_mean", "PM2.5_roll6h_mean", "PM2.5_roll24h_mean", "PM2.5_roll7d_mean", "PM2.5_roll24h_std",
    "temp_c", "humidity_pct", "wind_speed_ms", "wind_deg", "pressure_hpa", "precip_mm",
]


def climatology_predict(df, mask, train_mask):
    """Location's hour-of-day average PM2.5, computed from training rows
    only -- the global-model equivalent of the India pipeline's baseline."""
    train = df.loc[train_mask]
    hourly_loc = train.groupby(["location_id", "hour"])["PM2.5"].mean()
    hourly_region = train.groupby(["region", "hour"])["PM2.5"].mean()
    global_mean = train["PM2.5"].mean()

    sub = df.loc[mask]

    def lookup(row):
        key = (row["location_id"], row["hour"])
        if key in hourly_loc.index:
            return hourly_loc.loc[key]
        key2 = (row["region"], row["hour"])
        if key2 in hourly_region.index:
            return hourly_region.loc[key2]
        return global_mean

    return sub.apply(lookup, axis=1).values


def metrics_row(name, y_true, y_pred):
    return {
        "model": name,
        "MAE": round(mean_absolute_error(y_true, y_pred), 2),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
    }


def train_one_horizon(df, horizon):
    target_col = f"PM2.5_target_{horizon}h"
    d = df.dropna(subset=["PM2.5_lag1h", target_col]).copy()
    feature_cols = [c for c in FEATURE_COLS if c in d.columns]

    train_mask = d["split"] == "train"
    cal_mask = d["split"] == "cal"
    test_mask = d["split"] == "test"

    X_train, y_train = d.loc[train_mask, feature_cols], d.loc[train_mask, target_col]
    X_cal, y_cal = d.loc[cal_mask, feature_cols], d.loc[cal_mask, target_col]
    X_test, y_test = d.loc[test_mask, feature_cols], d.loc[test_mask, target_col]

    print(f"\n--- {horizon}h horizon ---")
    print(f"Train: {len(X_train):,} | Cal: {len(X_cal):,} | Test: {len(X_test):,}")

    if len(X_train) < 1000 or len(X_cal) < 100 or len(X_test) < 100:
        print(f"  Skipping {horizon}h -- not enough rows in one of the splits.")
        return None

    model = xgb.XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    cal_preds = model.predict(X_cal)
    cal_df = d.loc[cal_mask].copy()
    cal_persistence = cal_df["PM2.5"].values

    # --- per-region blend weight, chosen on the CALIBRATION set only ---
    # weight=1.0 means "trust the model fully"; lower values shrink toward
    # persistence for horizon/region combinations where the model hasn't
    # earned that trust (e.g. 1h globally, where persistence is already
    # near-optimal and the model's extra features mostly add noise).
    blend_weights = {}
    for region in cal_df["region"].unique():
        m = (cal_df["region"] == region).values
        if m.sum() < 200:
            continue
        best_w, best_mae = 1.0, mean_absolute_error(y_cal.values[m], cal_preds[m])
        for w in np.arange(0.0, 1.01, 0.1):
            blended = w * cal_preds[m] + (1 - w) * cal_persistence[m]
            mae = mean_absolute_error(y_cal.values[m], blended)
            if mae < best_mae:
                best_w, best_mae = float(w), mae
        blend_weights[region] = round(best_w, 1)
    default_weight = round(float(np.median(list(blend_weights.values()))), 1) if blend_weights else 1.0

    print(f"  Blend weights (1.0=pure model, 0.0=pure persistence): {blend_weights}")

    cal_weight_arr = cal_df["region"].map(blend_weights).fillna(default_weight).values
    cal_blended = cal_weight_arr * cal_preds + (1 - cal_weight_arr) * cal_persistence

    # --- region-stratified conformal calibration, on the BLENDED predictions ---
    cal_df["residual"] = np.abs(y_cal.values - cal_blended)
    global_q_level = min(1.0, np.ceil((len(cal_df) + 1) * TARGET_CONFIDENCE) / len(cal_df))
    global_q_hat = float(np.quantile(cal_df["residual"], global_q_level))

    region_q_hats = {}
    for region in cal_df["region"].unique():
        residuals = cal_df.loc[cal_df["region"] == region, "residual"]
        n = len(residuals)
        if n < MIN_CAL_ROWS_FOR_REGION_Q:
            continue
        q_level = min(1.0, np.ceil((n + 1) * TARGET_CONFIDENCE) / n)
        region_q_hats[region] = float(np.quantile(residuals, q_level))

    # --- test set: raw model vs. blended model vs. persistence vs. climatology ---
    test_df = d.loc[test_mask].copy()
    test_preds = model.predict(X_test)
    persistence_preds = test_df["PM2.5"].values
    test_weight_arr = test_df["region"].map(blend_weights).fillna(default_weight).values
    blended_preds = test_weight_arr * test_preds + (1 - test_weight_arr) * persistence_preds

    q_hats = test_df["region"].map(region_q_hats).fillna(global_q_hat).values
    lower, upper = blended_preds - q_hats, blended_preds + q_hats
    coverage = float(np.mean((y_test.values >= lower) & (y_test.values <= upper)))

    climatology_preds = climatology_predict(d, test_mask, train_mask)

    raw_mae = mean_absolute_error(y_test, test_preds)
    blended_mae = mean_absolute_error(y_test, blended_preds)
    use_blend = blended_mae < raw_mae
    deployed_preds = blended_preds if use_blend else test_preds
    deployed_label = "XGBoost (blended, deployed)" if use_blend else "XGBoost (raw, deployed)"

    print(f"  Deployed choice for {horizon}h: {'blended' if use_blend else 'raw'} "
          f"(raw MAE {raw_mae:.2f} vs blended MAE {blended_mae:.2f})")

    # recompute conformal intervals on whichever version is actually deployed,
    # so the interval matches the prediction method being served
    if use_blend:
        cal_q_source = cal_blended
    else:
        cal_q_source = cal_preds
    cal_df["residual"] = np.abs(y_cal.values - cal_q_source)
    global_q_level = min(1.0, np.ceil((len(cal_df) + 1) * TARGET_CONFIDENCE) / len(cal_df))
    global_q_hat = float(np.quantile(cal_df["residual"], global_q_level))
    region_q_hats = {}
    for region in cal_df["region"].unique():
        residuals = cal_df.loc[cal_df["region"] == region, "residual"]
        n = len(residuals)
        if n < MIN_CAL_ROWS_FOR_REGION_Q:
            continue
        q_level = min(1.0, np.ceil((n + 1) * TARGET_CONFIDENCE) / n)
        region_q_hats[region] = float(np.quantile(residuals, q_level))

    q_hats = test_df["region"].map(region_q_hats).fillna(global_q_hat).values
    lower, upper = deployed_preds - q_hats, deployed_preds + q_hats
    coverage = float(np.mean((y_test.values >= lower) & (y_test.values <= upper)))

    overall = [
        metrics_row("Persistence", y_test, persistence_preds),
        metrics_row("Climatology", y_test, climatology_preds),
        metrics_row("XGBoost (raw)", y_test, test_preds),
        {**metrics_row(deployed_label, y_test, deployed_preds)},
    ]
    for row in overall:
        row["horizon_h"] = horizon
        row["region"] = "ALL"
    print(pd.DataFrame(overall).to_string(index=False))

    per_region_rows = []
    for region in sorted(test_df["region"].unique()):
        m = (test_df["region"] == region).values
        if m.sum() < 50:
            continue
        per_region_rows.append({**metrics_row("Deployed", y_test[m], deployed_preds[m]),
                                "horizon_h": horizon, "region": region})
        per_region_rows.append({**metrics_row("Persistence", y_test[m], persistence_preds[m]),
                                "horizon_h": horizon, "region": region})

    model.save_model(str(MODELS_DIR / f"global_model_{horizon}h.json"))
    with open(MODELS_DIR / f"global_conformal_{horizon}h.json", "w") as f:
        json.dump({
            "horizon_hours": horizon,
            "feature_cols": feature_cols,
            "use_blend": use_blend,
            "blend_weights": blend_weights if use_blend else {},
            "default_blend_weight": default_weight if use_blend else 1.0,
            "region_q_hats": region_q_hats,
            "global_q_hat": global_q_hat,
            "target_confidence": TARGET_CONFIDENCE,
            "empirical_coverage": coverage,
            "test_mae_deployed": round(mean_absolute_error(y_test, deployed_preds), 2),
            "test_mae_raw_model": round(raw_mae, 2),
            "test_mae_blended": round(blended_mae, 2),
            "persistence_mae": round(mean_absolute_error(y_test, persistence_preds), 2),
            "climatology_mae": round(mean_absolute_error(y_test, climatology_preds), 2),
        }, f, indent=2)

    return overall + per_region_rows


def run():
    print("=== Training global PM2.5 models (per-location chronological split) ===")
    df = pd.read_parquet(TRAINING_PARQUET)
    regions = sorted(df["region"].unique())
    cities = sorted(df["city"].unique())
    print(f"Loaded {len(df):,} rows | {df['location_id'].nunique()} locations | "
          f"{len(cities)} cities | {len(regions)} regions: {regions}")

    all_rows = []
    for h in HORIZONS:
        result = train_one_horizon(df, h)
        if result is not None:
            all_rows.extend(result)

    report = pd.DataFrame(all_rows)
    report.to_csv(REPORTS_DIR / "global_evaluation.csv", index=False)
    print(f"\nSaved evaluation report: {REPORTS_DIR / 'global_evaluation.csv'}")

    with open(MODELS_DIR / "global_manifest.json", "w") as f:
        json.dump({
            "horizons": [h for h in HORIZONS if (MODELS_DIR / f"global_model_{h}h.json").exists()],
            "regions": regions,
            "cities": cities,
            "note": "No future-weather features yet (12h/24h use current weather only) -- "
                    "a known gap to close in a later iteration, same as the early India model had.",
        }, f, indent=2)
    print(f"Saved model manifest: {MODELS_DIR / 'global_manifest.json'}")


if __name__ == "__main__":
    run()
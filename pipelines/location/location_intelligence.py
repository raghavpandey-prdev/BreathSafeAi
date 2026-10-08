"""
Day 11 — Location/Route Intelligence.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
import json
import sys

sys.path.append(str(Path(__file__).parent.parent / "personalization"))
sys.path.append(str(Path(__file__).parent.parent / "features"))

from risk_engine import Forecast, UserProfile, generate_advisory
from inference_helpers import load_baselines, get_station_baseline, build_city_dummies, CITY_DUMMY_COLS, city_has_sufficient_data

DATA_DIR = Path(__file__).parent.parent.parent / "data"
FEATURES_CSV = DATA_DIR / "processed" / "features.csv"
HISTORICAL_CSV = DATA_DIR / "processed" / "historical_features.csv"
MODELS_DIR = Path(__file__).parent.parent.parent / "models"

FEATURE_COLS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend",
    "PM2.5_lag1h", "PM2.5_lag2h", "PM2.5_lag3h", "PM2.5_lag6h", "PM2.5_lag24h",
    "PM2.5_trend_24h",
    "PM2.5_roll3h_mean", "PM2.5_roll6h_mean",
    "PM10_lag1h", "NO2_lag1h", "OZONE_lag1h", "CO_lag1h",
    "temp_c", "humidity_pct", "wind_speed_ms", "wind_deg",
]


def forecast_for_row(row, model, params):
    baselines = load_baselines()
    row = row.copy()
    row["station_baseline_pm25"] = get_station_baseline(row["station"], row["city"], baselines)
    dummies = build_city_dummies(row["city"])
    for k, v in dummies.items():
        row[k] = v

    full_feature_cols = FEATURE_COLS + ["station_baseline_pm25"] + CITY_DUMMY_COLS
    available_features = [c for c in full_feature_cols if c in row.index]
    X = row[available_features].to_frame().T.astype(float)
    for col in full_feature_cols:
        if col not in X.columns:
            X[col] = np.nan
    X = X[full_feature_cols]

    point = float(model.predict(X)[0])
    q_hat = get_season_q_hat(int(row["month"]), params)
    return Forecast(point_estimate=point, lower_bound=max(0, point - q_hat), upper_bound=point + q_hat)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_model_and_params():
    model = xgb.XGBRegressor()
    model.load_model(str(MODELS_DIR / "pm25_model_6h_conformal.json"))
    with open(MODELS_DIR / "pm25_conformal_params.json") as f:
        params = json.load(f)
    return model, params


def get_season_q_hat(month: int, params: dict) -> float:
    if 1 <= month <= 2 or month == 12:
        season = "winter"
    elif 3 <= month <= 5:
        season = "summer"
    else:
        season = "monsoon"
    return params["season_q_hats"].get(season, params["global_q_hat"])


def find_nearest_station(lat, lon, df: pd.DataFrame) -> str:
    """Find the station in the live feature data closest to a given lat/lon.
    NOTE: features.csv doesn't currently carry lat/lon — this is a real gap
    flagged below rather than silently faked."""
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return None
    stations = df[["station", "latitude", "longitude"]].drop_duplicates()
    stations["distance_km"] = haversine_km(lat, lon, stations["latitude"], stations["longitude"])
    nearest = stations.sort_values("distance_km").iloc[0]
    return nearest["station"], nearest["distance_km"]


def get_latest_row_for_station(station: str, df: pd.DataFrame):
    """Latest row that actually has a PM2.5 reading. The very last row for a
    station is often an empty poll (NaN), which breaks the forecast and JSON."""
    station_rows = df[df["station"] == station].sort_values("timestamp")
    if station_rows.empty:
        return None
    valid = station_rows.dropna(subset=["PM2.5"])
    if valid.empty:
        return None
    return valid.iloc[-1]


def compare_locations(locations: dict, profile: UserProfile):
    """locations: dict of name -> station name (direct match against your live data)."""
    if not FEATURES_CSV.exists():
        print("No features.csv found — run build_features.py first.")
        return

    df = pd.read_csv(FEATURES_CSV, parse_dates=["timestamp"])
    model, params = load_model_and_params()

    results = []
    for label, station_name in locations.items():
        row = get_latest_row_for_station(station_name, df)
        if row is None:
            print(f"WARNING: no live data found for station '{station_name}' (location: {label}). Skipping.")
            continue

        forecast = forecast_for_row(row, model, params)
        advisory = generate_advisory(forecast, profile)
        results.append({
            "location": label,
            "station": station_name,
            "current_pm25": row["PM2.5"],
            "forecast_6h": forecast.point_estimate,
            "forecast_range": f"{forecast.lower_bound:.0f}-{forecast.upper_bound:.0f}",
            "risk_level": advisory["personal_risk_level"],
            "advice": advisory["advice"],
        })

    if not results:
        print("No valid locations to compare.")
        return

    results_df = pd.DataFrame(results)
    print("\n=== Location Comparison ===")
    print(results_df.to_string(index=False))

    risk_order = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    results_df["risk_rank"] = results_df["risk_level"].apply(lambda x: risk_order.index(x))

    min_rank = results_df["risk_rank"].min()
    max_rank = results_df["risk_rank"].max()

    if min_rank == max_rank:
        print(f"\nAll compared locations are currently at the same risk level "
              f"({results_df.iloc[0]['risk_level']}) — no meaningful difference "
              f"between them right now.")
    else:
        best = results_df.loc[results_df["risk_rank"].idxmin()]
        worst = results_df.loc[results_df["risk_rank"].idxmax()]
        print(f"\nBest location right now: {best['location']} ({best['risk_level']})")
        print(f"Worst location right now: {worst['location']} ({worst['risk_level']})")
        print(f"\nConsider: if your plans are flexible, {best['location']} is currently "
              f"the safer choice compared to {worst['location']}.")


def best_hour_today():
    """Uses historical diurnal averages to suggest which hour tends to be calmest."""
    if not HISTORICAL_CSV.exists():
        print("\nNo historical_features.csv found — skipping best-hour suggestion.")
        return

    hist = pd.read_csv(HISTORICAL_CSV, usecols=["hour", "PM2.5"])
    hourly_avg = hist.groupby("hour")["PM2.5"].mean().sort_values()

    print("\n=== Historically Calmest Hours (Delhi, all-time average) ===")
    print("Best 3 hours:")
    for hour, avg in hourly_avg.head(3).items():
        print(f"  {hour:02d}:00 — avg {avg:.1f} µg/m³")
    print("Worst 3 hours:")
    for hour, avg in hourly_avg.tail(3).items():
        print(f"  {hour:02d}:00 — avg {avg:.1f} µg/m³")


if __name__ == "__main__":
    print("=== Day 11: Location Intelligence Demo ===\n")

    print("IMPORTANT: replace these with actual station names from YOUR live "
          "aqi_history.csv (check the 'station' column for exact spelling) — "
          "these are just placeholder examples and will likely need adjusting.\n")

    example_locations = {
        "Home": "Anand Vihar, Delhi - DPCC",
        "Work": "Wazirpur, Delhi - DPCC",
    }

    healthy_adult = UserProfile(planned_activity="general_outdoor")
    compare_locations(example_locations, healthy_adult)

    best_hour_today()
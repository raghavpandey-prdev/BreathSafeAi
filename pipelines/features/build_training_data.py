"""
Phase 1 — Build the multi-city training dataset.

Fixes the problems found in the data audit:
  * Uses 10 metros from the Kaggle CPCB history, not only Delhi.
  * Puts every station on a regular hourly grid, so lags/rolling windows are
    exact even when readings are missing.
  * Includes the CURRENT PM2.5 reading as an input (the old model never saw it).
  * Adds real weather for the whole history (Open-Meteo ERA5), in IST so it
    lines up with CPCB timestamps (no UTC offset bug).
  * Adds targets for several horizons (1, 3, 6, 12, 24 h) and future-weather
    columns, so multi-horizon forecasting becomes possible.

Usage:
    python pipelines/features/build_training_data.py
    python pipelines/features/build_training_data.py --cities Delhi   # quick test

Output: data/processed/training_data.parquet
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).parent.parent.parent / "data"
KAGGLE_DIR = DATA_DIR / "raw" / "kaggle_historical"
WEATHER_CACHE = DATA_DIR / "raw" / "weather_era5"
OUT_PATH = DATA_DIR / "processed" / "training_data.parquet"

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# City-centre coordinates. ERA5 is a ~25 km grid, so one point per city is enough.
CITY_COORDS = {
    "Delhi": (28.6139, 77.2090),
    "Bengaluru": (12.9716, 77.5946),
    "Hyderabad": (17.3850, 78.4867),
    "Lucknow": (26.8467, 80.9462),
    "Chennai": (13.0827, 80.2707),
    "Mumbai": (19.0760, 72.8777),
    "Kolkata": (22.5726, 88.3639),
    "Jaipur": (26.9124, 75.7873),
    "Gurugram": (28.4595, 77.0266),
    "Patna": (25.5941, 85.1376),
}

POLLUTANTS = ["PM2.5", "PM10", "NO2", "OZONE", "CO", "SO2", "NH3"]

# Values outside these ranges are sensor faults. PM2.5/PM10 readings of exactly
# 1000 are the sensor ceiling (saturation), so the upper bound is exclusive.
PLAUSIBLE = {
    "PM2.5": (0, 999), "PM10": (0, 999), "NO2": (0, 500),
    "OZONE": (0, 500), "CO": (0, 200), "SO2": (0, 500), "NH3": (0, 500),
}

WEATHER_VARS = {
    "temperature_2m": "temp_c",
    "relative_humidity_2m": "humidity_pct",
    "wind_speed_10m": "wind_speed_ms",
    "wind_direction_10m": "wind_deg",
    "surface_pressure": "pressure_hpa",
    "precipitation": "precip_mm",
}

HORIZONS = [1, 3, 6, 12, 24]                 # forecast targets (hours ahead)
FUTURE_WEATHER_HOURS = [6, 12, 24]           # weather "as forecast" at t+h
FUTURE_WEATHER_VARS = ["temp_c", "humidity_pct", "wind_speed_ms", "precip_mm"]
PM25_LAGS = [1, 2, 3, 6, 12, 24]
ROLL_WINDOWS = [3, 6, 24]


# ---------------------------------------------------------------- weather ---

def fetch_city_weather(city, start, end):
    """Hourly ERA5 weather for one city, cached per calendar year on disk.
    Timestamps are IST (Asia/Kolkata) so they match CPCB/Kaggle timestamps."""
    WEATHER_CACHE.mkdir(parents=True, exist_ok=True)
    lat, lon = CITY_COORDS[city]
    frames = []

    for year in range(start.year, end.year + 1):
        cache_file = WEATHER_CACHE / f"{city}_{year}.csv"
        if cache_file.exists():
            frames.append(pd.read_csv(cache_file, parse_dates=["timestamp"]))
            continue

        y_start = max(pd.Timestamp(f"{year}-01-01"), start.normalize())
        y_end = min(pd.Timestamp(f"{year}-12-31"), end.normalize())
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": y_start.strftime("%Y-%m-%d"),
            "end_date": y_end.strftime("%Y-%m-%d"),
            "hourly": ",".join(WEATHER_VARS),
            "wind_speed_unit": "ms",
            "timezone": "Asia/Kolkata",
        }

        data = None
        for attempt in range(6):
            try:
                resp = requests.get(ARCHIVE_URL, params=params, timeout=60)
                if resp.status_code == 429 or resp.status_code >= 500:
                    wait = min(90, 10 * 2 ** attempt)
                    print(f"    {city} {year}: HTTP {resp.status_code}, retrying in {wait}s")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            except requests.RequestException as e:
                wait = min(90, 10 * 2 ** attempt)
                print(f"    {city} {year}: {e} — retrying in {wait}s")
                time.sleep(wait)
        if data is None or "hourly" not in data:
            raise RuntimeError(f"Could not fetch weather for {city} {year}")

        w = pd.DataFrame(data["hourly"])
        w["timestamp"] = pd.to_datetime(w.pop("time"))
        w = w.rename(columns=WEATHER_VARS)
        w.to_csv(cache_file, index=False)
        frames.append(w)
        time.sleep(1.0)  # be polite to the free API

    w = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp")
    return w.sort_values("timestamp").reset_index(drop=True)


def prepare_city_weather(city, start, end):
    """Weather on a complete hourly index + future-weather (t+h) columns."""
    w = fetch_city_weather(city, start, end).set_index("timestamp")
    w = w.reindex(pd.date_range(w.index.min(), w.index.max(), freq="h"))
    w.index.name = "timestamp"
    for h in FUTURE_WEATHER_HOURS:
        for v in FUTURE_WEATHER_VARS:
            w[f"{v}_fut{h}h"] = w[v].shift(-h)
    w["city"] = city
    return w.reset_index()


# ------------------------------------------------------------- CPCB data ----

def load_kaggle(cities):
    stations = pd.read_csv(KAGGLE_DIR / "stations.csv", encoding="utf-8-sig")
    stations = stations[stations["City"].isin(cities)]
    print(f"Stations in target cities: {len(stations)}")

    usecols = ["StationId", "Datetime", "PM2.5", "PM10", "NO2", "NH3", "CO", "SO2", "O3"]
    df = pd.read_csv(KAGGLE_DIR / "station_hour.csv", usecols=usecols)
    df = df[df["StationId"].isin(stations["StationId"])]
    df = df.rename(columns={"O3": "OZONE"})
    df["timestamp"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).drop(columns="Datetime")

    df = df.merge(stations[["StationId", "StationName", "City"]], on="StationId")
    df["station"] = df["StationName"].fillna(df["StationId"])
    df["city"] = df["City"]

    for col, (lo, hi) in PLAUSIBLE.items():
        df[col] = pd.to_numeric(df[col], errors="coerce")
        bad = (df[col] < lo) | (df[col] >= hi)
        if bad.any():
            print(f"  {col}: removed {int(bad.sum())} implausible values")
        df.loc[bad, col] = np.nan

    return df[["station", "city", "timestamp"] + POLLUTANTS]


# ---------------------------------------------------------- feature build ---

def build_station_frame(g, station, city):
    """Regular hourly grid for one station, then lags/rolling/targets by shift."""
    g = g.drop_duplicates("timestamp", keep="last").set_index("timestamp").sort_index()
    grid = pd.date_range(g.index.min(), g.index.max(), freq="h")
    g = g.reindex(grid)
    g.index.name = "timestamp"
    pm = g["PM2.5"]

    out = pd.DataFrame(index=g.index)
    out["station"] = station
    out["city"] = city
    for col in POLLUTANTS:
        out[col] = g[col]

    for lag in PM25_LAGS:
        out[f"PM2.5_lag{lag}h"] = pm.shift(lag)
    for col in ["PM10", "NO2", "OZONE", "CO"]:
        out[f"{col}_lag1h"] = g[col].shift(1)

    out["PM2.5_change_1h"] = pm - out["PM2.5_lag1h"]
    out["PM2.5_change_24h"] = pm - out["PM2.5_lag24h"]
    for w in ROLL_WINDOWS:
        out[f"PM2.5_roll{w}h_mean"] = pm.rolling(w, min_periods=max(1, w // 2)).mean()
    out["PM2.5_roll24h_std"] = pm.rolling(24, min_periods=12).std()

    for h in HORIZONS:
        out[f"PM2.5_target_{h}h"] = pm.shift(-h)

    return out.reset_index()


def add_time_features(df):
    ts = df["timestamp"]
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["year"] = ts.dt.year
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(np.int8)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def quality_report(df):
    print("\n=== Data quality report ===")
    print(f"Rows: {len(df):,} | stations: {df['station'].nunique()} | "
          f"{df['timestamp'].min()} -> {df['timestamp'].max()}")
    rep = df.groupby("city").agg(
        stations=("station", "nunique"),
        rows=("PM2.5", "size"),
        pm25_mean=("PM2.5", "mean"),
        weather_pct=("temp_c", lambda s: 100 * s.notna().mean()),
        has_lag24_pct=("PM2.5_lag24h", lambda s: 100 * s.notna().mean()),
        has_target6_pct=("PM2.5_target_6h", lambda s: 100 * s.notna().mean()),
    ).round(1).sort_values("rows", ascending=False)
    print(rep.to_string())
    print("\nRows per year:")
    print(df.groupby("year").size().to_string())


def run(cities):
    print("=== Building multi-city training dataset ===\n")
    raw = load_kaggle(cities)
    print(f"Raw rows in target cities: {len(raw):,}")

    print("\nBuilding hourly-grid features per station...")
    frames = []
    for (station, city), g in raw.groupby(["station", "city"]):
        if g["PM2.5"].notna().sum() < 500:  # too little data to learn from
            continue
        frames.append(build_station_frame(g, station, city))
    df = pd.concat(frames, ignore_index=True)

    # keep only hours with a real current PM2.5 reading (the forecast input)
    df = df.dropna(subset=["PM2.5"])
    df = add_time_features(df)
    print(f"Feature rows (with a current PM2.5 reading): {len(df):,}")

    print("\nAdding ERA5 weather (IST)...")
    weather_frames = []
    for city in sorted(df["city"].unique()):
        c = df[df["city"] == city]["timestamp"]
        print(f"  {city}: {c.min().date()} -> {c.max().date()}")
        weather_frames.append(prepare_city_weather(city, c.min(), c.max()))
    weather = pd.concat(weather_frames, ignore_index=True)
    df = df.merge(weather, on=["city", "timestamp"], how="left")

    # compact memory/disk footprint
    float_cols = df.select_dtypes("float64").columns
    df[float_cols] = df[float_cols].astype("float32")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    quality_report(df)
    print(f"\nSaved: {OUT_PATH}  ({OUT_PATH.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", default=",".join(CITY_COORDS),
                        help="comma-separated subset, e.g. Delhi,Mumbai")
    args = parser.parse_args()
    chosen = [c.strip() for c in args.cities.split(",") if c.strip() in CITY_COORDS]
    if not chosen:
        raise SystemExit(f"No valid cities. Choose from: {', '.join(CITY_COORDS)}")
    run(chosen)
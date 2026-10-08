"""
Phase B (global model) — turn the raw OpenAQ pulls into a trainable dataset.

Key differences from the India-specific pipeline (build_training_data.py),
all deliberate, for a model meant to generalize to places never seen in
training:
  * NO per-station baseline lookup table, NO city one-hot dummies — neither
    would mean anything for a city the model has never seen. Latitude and
    longitude are kept as raw numeric features instead (real geography a
    model can actually use), plus a rolling 7-day mean computed on the fly
    (generalizes to any location with recent history, not a hardcoded table).
  * PER-LOCATION chronological 70/15/15 split, not a shared global year
    cutoff — global coverage years don't line up (confirmed by the real
    pull: some locations only have 2021 data, others only 2018-2019), so a
    shared cutoff would silently drop locations from one split or another.
  * Locations are filtered by reporting DENSITY (readings/day), not raw
    calendar span — a location with dense multi-year coverage should never
    be rejected just because its span is long; only genuinely sparse
    locations (few readings spread across a huge gap) are dropped.

Usage: python pipelines/features/build_global_training_data.py
Output: data/processed/global_training_data.parquet
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw" / "openaq_global"
WEATHER_CACHE = DATA_DIR / "raw" / "weather_era5_global"
OUT_PATH = DATA_DIR / "processed" / "global_training_data.parquet"

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

MIN_ROWS_PER_LOCATION = 1500   # below this, a location's history is too thin to learn from
MIN_READINGS_PER_DAY = 1.0     # reject by DENSITY (sparse readings over a huge span), not raw span --
                                # a location with 2+ dense years is exactly what we want to keep
MAX_GRID_DAYS = 24 * 366 * 5   # hard safety ceiling only: reject truly pathological spans (hours)

HORIZONS = [1, 3, 6, 12, 24]
PM25_LAGS = [1, 2, 3, 6, 12, 24]
ROLL_WINDOWS = [3, 6, 24, 24 * 7]  # 24*7 = rolling 7-day mean, the generalizable stand-in
                                    # for the India model's per-station baseline lookup

WEATHER_VARS = {
    "temperature_2m": "temp_c",
    "relative_humidity_2m": "humidity_pct",
    "wind_speed_10m": "wind_speed_ms",
    "wind_direction_10m": "wind_deg",
    "surface_pressure": "pressure_hpa",
    "precipitation": "precip_mm",
}


def load_locations():
    """One DataFrame per raw CSV, keyed by (city, location_id)."""
    files = sorted(RAW_DIR.glob("*.csv"))
    files = [f for f in files if f.name != "pull_summary.csv"]
    locations = []
    for f in files:
        df = pd.read_csv(f, parse_dates=["timestamp"])
        if df.empty:
            continue
        locations.append(df)
    return locations


def fetch_location_weather(loc_id, lat, lon, start, end):
    """Hourly ERA5 weather for one location's exact coordinates, cached per
    location+year so re-runs don't re-fetch."""
    WEATHER_CACHE.mkdir(parents=True, exist_ok=True)
    frames = []
    for year in range(start.year, end.year + 1):
        cache_file = WEATHER_CACHE / f"loc{loc_id}_{year}.csv"
        if cache_file.exists():
            frames.append(pd.read_csv(cache_file, parse_dates=["timestamp"]))
            continue

        y_start = max(pd.Timestamp(f"{year}-01-01"), start.tz_localize(None).normalize())
        y_end = min(pd.Timestamp(f"{year}-12-31"), end.tz_localize(None).normalize())
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": y_start.strftime("%Y-%m-%d"), "end_date": y_end.strftime("%Y-%m-%d"),
            "hourly": ",".join(WEATHER_VARS), "wind_speed_unit": "ms", "timezone": "UTC",
        }
        data = None
        for attempt in range(5):
            try:
                resp = requests.get(ARCHIVE_URL, params=params, timeout=60)
                if resp.status_code == 429 or resp.status_code >= 500:
                    time.sleep(min(60, 10 * 2 ** attempt))
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            except requests.RequestException:
                time.sleep(min(60, 10 * 2 ** attempt))
        if data is None or "hourly" not in data:
            print(f"    weather fetch failed for location {loc_id}, year {year} -- skipping that year")
            continue

        w = pd.DataFrame(data["hourly"])
        w["timestamp"] = pd.to_datetime(w.pop("time"), utc=True)
        w = w.rename(columns=WEATHER_VARS)
        w.to_csv(cache_file, index=False)
        frames.append(w)
        time.sleep(1.0)

    if not frames:
        return pd.DataFrame(columns=["timestamp"] + list(WEATHER_VARS.values()))
    return pd.concat(frames, ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp")


def build_location_features(df):
    """Regular hourly grid, lags/rolling/targets for one location -- same
    shape of logic as the India pipeline, minus anything city-identity-based."""
    city, loc_id, region = df["city"].iloc[0], df["location_id"].iloc[0], df["region"].iloc[0]
    lat, lon = df["latitude"].iloc[0], df["longitude"].iloc[0]

    g = df[["timestamp", "pm25"]].drop_duplicates("timestamp").set_index("timestamp").sort_index()
    span_hours = (g.index.max() - g.index.min()).total_seconds() / 3600
    span_days = max(span_hours / 24, 1)

    if span_hours > MAX_GRID_DAYS:
        print(f"    {city} ({loc_id}): span implausibly large ({span_days:.0f} days), skipping")
        return None
    readings_per_day = len(g) / span_days
    if readings_per_day < MIN_READINGS_PER_DAY:
        print(f"    {city} ({loc_id}): too sparse ({readings_per_day:.2f} readings/day "
              f"over {span_days:.0f} days), skipping")
        return None

    grid = pd.date_range(g.index.min(), g.index.max(), freq="h")
    g = g.reindex(grid)
    g.index.name = "timestamp"
    pm = g["pm25"]

    out = pd.DataFrame(index=g.index)
    out["city"], out["location_id"], out["region"] = city, loc_id, region
    out["latitude"], out["longitude"] = lat, lon
    out["PM2.5"] = pm

    for lag in PM25_LAGS:
        out[f"PM2.5_lag{lag}h"] = pm.shift(lag)
    out["PM2.5_change_1h"] = pm - out["PM2.5_lag1h"]
    out["PM2.5_change_24h"] = pm - out["PM2.5_lag24h"]
    for w in ROLL_WINDOWS:
        label = f"PM2.5_roll{w}h_mean" if w != 24 * 7 else "PM2.5_roll7d_mean"
        out[label] = pm.rolling(w, min_periods=max(1, w // 4)).mean()
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


def assign_chronological_split(df):
    """70/15/15 by each location's OWN time order -- necessary because global
    coverage years don't align across locations (confirmed by the real
    pull), so a shared absolute-year cutoff would drop some locations from
    one split entirely."""
    df = df.sort_values(["location_id", "timestamp"]).reset_index(drop=True)
    splits = []
    for loc_id, g in df.groupby("location_id"):
        n = len(g)
        train_end = int(n * 0.70)
        cal_end = int(n * 0.85)
        split = np.where(np.arange(n) < train_end, "train",
                 np.where(np.arange(n) < cal_end, "cal", "test"))
        splits.append(pd.Series(split, index=g.index))
    df["split"] = pd.concat(splits).sort_index()
    return df


def quality_report(df):
    print("\n=== Global dataset quality report ===")
    print(f"Rows: {len(df):,} | locations: {df['location_id'].nunique()} | "
          f"cities: {df['city'].nunique()} | regions: {df['region'].nunique()}")
    rep = df.groupby(["region", "city"]).agg(
        rows=("PM2.5", "size"),
        pm25_mean=("PM2.5", "mean"),
        weather_pct=("temp_c", lambda s: 100 * s.notna().mean()),
        train=("split", lambda s: (s == "train").sum()),
        cal=("split", lambda s: (s == "cal").sum()),
        test=("split", lambda s: (s == "test").sum()),
    ).round(1)
    print(rep.to_string())


def run():
    print("=== Building global training dataset ===\n")
    raw_locations = load_locations()
    print(f"Loaded {len(raw_locations)} raw location files")

    frames = []
    for df in raw_locations:
        city, loc_id = df["city"].iloc[0], df["location_id"].iloc[0]
        if len(df) < MIN_ROWS_PER_LOCATION:
            print(f"  Dropping {city} ({loc_id}): only {len(df)} rows (< {MIN_ROWS_PER_LOCATION} minimum)")
            continue
        feat = build_location_features(df)
        if feat is not None:
            frames.append(feat)

    if not frames:
        raise SystemExit("No locations met the minimum row threshold -- nothing to build.")

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["PM2.5"])
    df = add_time_features(df)
    print(f"\nFeature rows after filtering: {len(df):,}")

    print("\nFetching per-location ERA5 weather...")
    weather_frames = []
    for loc_id, g in df.groupby("location_id"):
        lat, lon = g["latitude"].iloc[0], g["longitude"].iloc[0]
        start, end = g["timestamp"].min(), g["timestamp"].max()
        city = g["city"].iloc[0]
        print(f"  {city} ({loc_id}): {start.date()} -> {end.date()}")
        w = fetch_location_weather(loc_id, lat, lon, start, end)
        w["location_id"] = loc_id
        weather_frames.append(w)
    weather = pd.concat(weather_frames, ignore_index=True)

    df = df.merge(weather, on=["location_id", "timestamp"], how="left")
    df = assign_chronological_split(df)

    float_cols = df.select_dtypes("float64").columns
    df[float_cols] = df[float_cols].astype("float32")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    quality_report(df)
    print(f"\nSaved: {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    run()
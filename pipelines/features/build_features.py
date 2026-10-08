"""
Day 5 — Feature Engineering.

Takes the raw accumulated CPCB + weather data and builds a clean, model-ready
feature table: lag features, rolling averages, cyclical time encodings, and
merged weather — while applying the lessons from Day 4's EDA (drop unreliable
stations, filter stale/bad timestamps).

Usage: python pipelines/features/build_features.py
Output: data/processed/features.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AQI_CSV = RAW_DIR / "aqi_history.csv"
WEATHER_CSV = RAW_DIR / "weather_history.csv"

# From Day 4 EDA — stations with >50% missing data are excluded rather than imputed
MISSING_DATA_THRESHOLD = 0.5

# Core pollutants we care about most (PM2.5/PM10 for health, others as context)
KEY_POLLUTANTS = ["PM2.5", "PM10", "NO2", "OZONE", "CO", "SO2", "NH3"]

LAG_HOURS = [1, 2, 3, 6, 12, 24]
ROLLING_WINDOWS = [3, 6, 24]


def load_and_clean_aqi():
    df = pd.read_csv(AQI_CSV)
    df = df[df["source"] == "cpcb_data_gov_in"].copy()  # CPCB is primary; aqicn stays supplementary for now

    df["timestamp"] = pd.to_datetime(df["last_update"], format="%d-%m-%Y %H:%M:%S", errors="coerce")
    df["avg_value"] = pd.to_numeric(df["avg_value"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["timestamp"])
    print(f"Dropped {before - len(df)} rows with unparseable timestamps")

    # Filter out stale readings (catches the aqicn-style cached-old-data issue seen in EDA)
    cutoff = datetime.now() - timedelta(days=14)
    before = len(df)
    df = df[df["timestamp"] >= cutoff]
    print(f"Dropped {before - len(df)} rows older than 14 days (stale/bad timestamps)")

    # Drop stations with too much missing data, per Day 4 EDA
    missing_by_station = df.groupby("station")["avg_value"].apply(lambda x: x.isna().mean())
    bad_stations = missing_by_station[missing_by_station > MISSING_DATA_THRESHOLD].index.tolist()
    if bad_stations:
        print(f"Excluding {len(bad_stations)} stations with >{int(MISSING_DATA_THRESHOLD*100)}% missing data:")
        for s in bad_stations:
            print(f"  - {s}")
    df = df[~df["station"].isin(bad_stations)]
    if df.empty:
        raise SystemExit(
            "No CPCB rows in the last 14 days. The data.gov.in poll is failing — "
            "fix poll_aqi.py and run it before building features."
        )

    return df


def pivot_to_wide(df):
    """One row per station+timestamp, one column per pollutant."""
    wide = df.pivot_table(
        index=["station", "city", "latitude", "longitude", "timestamp"],
        columns="pollutant_id",
        values="avg_value",
        aggfunc="mean",
    ).reset_index()
    wide.columns.name = None
    return wide


def add_time_features(df):
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month"] = df["timestamp"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_lag_and_rolling_features(df):
    df = df.sort_values(["station", "timestamp"]).reset_index(drop=True)
    pollutant_cols = [c for c in KEY_POLLUTANTS if c in df.columns]

    # Time-aware lags: only fill a "lag Nh" value if a reading actually exists
    # at exactly that past hour for that station — NOT just N rows back, since
    # gaps mean row-position and real elapsed time don't match.
    for lag in LAG_HOURS:
        shifted = df[["station", "timestamp"] + pollutant_cols].copy()
        shifted["timestamp"] = shifted["timestamp"] + pd.Timedelta(hours=lag)
        shifted = shifted.rename(columns={c: f"{c}_lag{lag}h" for c in pollutant_cols})
        df = df.merge(shifted, on=["station", "timestamp"], how="left")
        # Trend feature: is pollution rising or falling compared to this time yesterday?
    if "PM2.5_lag1h" in df.columns and "PM2.5_lag24h" in df.columns:
        df["PM2.5_trend_24h"] = df["PM2.5_lag1h"] - df["PM2.5_lag24h"]
    if "PM2.5" in df.columns and "PM2.5_lag1h" in df.columns:
        df["PM2.5_change_1h"] = df["PM2.5"] - df["PM2.5_lag1h"]
    if "PM2.5" in df.columns and "PM2.5_lag24h" in df.columns:
        df["PM2.5_change_24h"] = df["PM2.5"] - df["PM2.5_lag24h"]
    # Time-aware rolling means: actual N-hour window by clock time, not by row count
    df = df.set_index("timestamp")
    roll_frames = []
    for window in ROLLING_WINDOWS:
        rolled = (
            df.groupby("station")[pollutant_cols]
            .rolling(f"{window}h", closed="left")
            .mean()
            .rename(columns={c: f"{c}_roll{window}h_mean" for c in pollutant_cols})
        )
        roll_frames.append(rolled)
    if "PM2.5" in pollutant_cols:
        std24 = (
            df.groupby("station")[["PM2.5"]]
            .rolling("24h", closed="left")
            .std()
            .rename(columns={"PM2.5": "PM2.5_roll24h_std"})
        )
        roll_frames.append(std24)
    df = df.reset_index()

    for rolled in roll_frames:
        rolled = rolled.reset_index()
        df = df.merge(rolled, on=["station", "timestamp"], how="left")

    return df


def merge_weather(df):
    if not WEATHER_CSV.exists():
        print("No weather_history.csv found — skipping weather merge.")
        return df

    weather = pd.read_csv(WEATHER_CSV)
    weather["timestamp"] = pd.to_datetime(weather["collected_at_utc"], errors="coerce")
    if weather["timestamp"].dt.tz is not None:
        weather["timestamp"] = weather["timestamp"].dt.tz_localize(None)
    weather = weather.dropna(subset=["timestamp"])

    df = df.sort_values("timestamp")
    weather = weather.sort_values("timestamp")

    merged_parts = []
    for city in df["city"].unique():
        city_aqi = df[df["city"] == city]
        city_weather = weather[weather["city"] == city]
        if city_weather.empty:
            merged_parts.append(city_aqi)
            continue
        merged = pd.merge_asof(
            city_aqi, city_weather[["timestamp", "temp_c", "humidity_pct", "wind_speed_ms", "wind_deg"]],
            on="timestamp", direction="nearest", tolerance=pd.Timedelta("2h")
        )
        merged_parts.append(merged)

    result = pd.concat(merged_parts, ignore_index=True)
    matched = result["temp_c"].notna().sum()
    print(f"Weather merged: {matched}/{len(result)} rows got a weather match (within 2h tolerance)")
    return result


def run():
    print("=== Day 5: Feature Engineering ===\n")

    print("Loading and cleaning AQI data...")
    aqi = load_and_clean_aqi()
    print(f"Clean AQI rows: {len(aqi)}\n")

    print("Pivoting to wide format (one row per station+timestamp)...")
    wide = pivot_to_wide(aqi)
    print(f"Wide table shape: {wide.shape}\n")

    print("Adding time features...")
    wide = add_time_features(wide)

    print("Adding lag and rolling features...")
    wide = add_lag_and_rolling_features(wide)

    print("Merging weather data...")
    wide = merge_weather(wide)

    out_path = OUT_DIR / "features.csv"
    wide.to_csv(out_path, index=False)
    print(f"\n=== Done. Feature table saved to {out_path} ===")
    print(f"Shape: {wide.shape}")
    print(f"Columns: {list(wide.columns)}")
    print("\nNote: early rows per station will have NaN lag/rolling features "
          "(not enough history yet) — expected, handled at model-training time (Day 6).")


if __name__ == "__main__":
    run()
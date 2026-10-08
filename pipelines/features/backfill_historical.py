"""
Historical Backfill from Kaggle CPCB Dataset (per-station version).

Uses station_hour.csv + stations.csv from the Kaggle "Air Quality Data in India"
dataset — real per-station historical readings, joined to get station/city names.
Kept as a SEPARATE feature table from live features.csv (station naming won't
exactly match your live CPCB pull's naming, so don't assume 1:1 station identity
between historical and live data — this is real historical density for the model
to learn from, not a perfect merge with your live stations).

Usage: python pipelines/features/backfill_historical.py
Output: data/processed/historical_features.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data"
STATION_HOUR_CSV = DATA_DIR / "raw" / "kaggle_historical" / "station_hour.csv"
STATIONS_CSV = DATA_DIR / "raw" / "kaggle_historical" / "stations.csv"
OUT_DIR = DATA_DIR / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CITIES = {
    "Delhi"
}

COLUMN_RENAME = {
    "O3": "OZONE",
}

KEY_POLLUTANTS = ["PM2.5", "PM10", "NO2", "OZONE", "CO", "SO2", "NH3"]
LAG_HOURS = [1, 2, 3, 6,24]
ROLLING_WINDOWS = [3, 6]


def load_and_clean():
    if not STATION_HOUR_CSV.exists():
        print(f"ERROR: {STATION_HOUR_CSV} not found. Download station_hour.csv "
              f"from Kaggle and place it there first.")
        return None
    if not STATIONS_CSV.exists():
        print(f"ERROR: {STATIONS_CSV} not found. Download stations.csv from "
              f"Kaggle and place it there first.")
        return None

    stations = pd.read_csv(STATIONS_CSV)
    df = pd.read_csv(STATION_HOUR_CSV)
    df = df.rename(columns=COLUMN_RENAME)

    df = df.merge(stations[["StationId", "StationName", "City"]], on="StationId", how="left")

    df["timestamp"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    df = df[df["City"].isin(TARGET_CITIES)]
    print(f"Rows for target cities: {len(df)}")

    df["station"] = df["StationName"].fillna(df["StationId"])
    df["city"] = df["City"]

    keep_cols = ["station", "city", "timestamp"] + [c for c in KEY_POLLUTANTS if c in df.columns]
    df = df[keep_cols]

    for col in KEY_POLLUTANTS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # The Kaggle CPCB dataset is known to contain sensor errors and impossible
    # values (negative readings, extreme spikes). Clip to plausible physical
    # ranges rather than trusting raw values — otherwise these silently corrupt
    # the model instead of just being missing data.
    PLAUSIBLE_RANGES = {
        "PM2.5": (0, 1000), "PM10": (0, 1000), "NO2": (0, 500),
        "OZONE": (0, 500), "CO": (0, 200), "SO2": (0, 500), "NH3": (0, 500),
    }
    print("\nCleaning implausible values:")
    for col, (lo, hi) in PLAUSIBLE_RANGES.items():
        if col in df.columns:
            before = df[col].notna().sum()
            df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan
            after = df[col].notna().sum()
            if before != after:
                print(f"  {col}: removed {before - after} implausible values ({before} -> {after} valid)")
            else:
                print(f"  {col}: no implausible values found")

    return df


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

    for lag in LAG_HOURS:
        shifted = df[["station", "timestamp"] + pollutant_cols].copy()
        shifted["timestamp"] = shifted["timestamp"] + pd.Timedelta(hours=lag)
        shifted = shifted.rename(columns={c: f"{c}_lag{lag}h" for c in pollutant_cols})
        df = df.merge(shifted, on=["station", "timestamp"], how="left")
        # Trend feature: is pollution rising or falling compared to this time yesterday?
    if "PM2.5_lag1h" in df.columns and "PM2.5_lag24h" in df.columns:
        df["PM2.5_trend_24h"] = df["PM2.5_lag1h"] - df["PM2.5_lag24h"]
    df = df.set_index("timestamp")
    for window in ROLLING_WINDOWS:
        rolled = (
            df.groupby("station")[pollutant_cols]
            .rolling(f"{window}h", closed="left")
            .mean()
            .rename(columns={c: f"{c}_roll{window}h_mean" for c in pollutant_cols})
        )
        rolled = rolled.reset_index()
        df = df.reset_index().merge(rolled, on=["station", "timestamp"], how="left")
        df = df.set_index("timestamp")

    df = df.reset_index()
    return df


def run():
    print("=== Historical Backfill from Kaggle CPCB Dataset (per-station) ===\n")
    df = load_and_clean()
    if df is None:
        return

    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Cities found: {sorted(df['city'].unique().tolist())}")
    print(f"Unique historical stations: {df['station'].nunique()}")

    df = add_time_features(df)
    df = add_lag_and_rolling_features(df)

    out_path = OUT_DIR / "historical_features.csv"
    df.to_csv(out_path, index=False)
    print(f"\n=== Done. Saved to {out_path} ===")
    print(f"Shape: {df.shape}")
    print("\nKept separate from live features.csv — station names won't exactly "
          "match your live CPCB pull, but this gives the model real multi-year "
          "seasonal/weekly pattern exposure per station, which your live data "
          "can't yet provide.")


if __name__ == "__main__":
    run()
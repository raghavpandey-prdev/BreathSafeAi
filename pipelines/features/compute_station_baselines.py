"""
Computes station-level and city-level PM2.5 baselines from ALL available data
(live + historical combined) and saves them to disk. This fixes a real bug:
station_baseline_pm25 was previously only computed during training and never
saved, so it was silently NaN at every actual prediction (dashboard, location
comparison, manual entry) — meaning the model never had access to the one
feature specifically designed to stop it from misapplying Delhi's pattern to
other stations/cities.

Usage: python pipelines/features/compute_station_baselines.py
Output: models/station_baselines.json
"""

import pandas as pd
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data"
FEATURES_CSV = DATA_DIR / "processed" / "features.csv"
HISTORICAL_CSV = DATA_DIR / "processed" / "historical_features.csv"
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def run():
    print("=== Computing Station & City Baselines ===\n")

    live = pd.read_csv(FEATURES_CSV, usecols=["station", "city", "PM2.5"])
    frames = [live]

    if HISTORICAL_CSV.exists():
        hist = pd.read_csv(HISTORICAL_CSV, usecols=["station", "city", "PM2.5"])
        frames.append(hist)

    df = pd.concat(frames, ignore_index=True).dropna(subset=["PM2.5"])

    station_baselines = df.groupby("station")["PM2.5"].mean().to_dict()
    city_baselines = df.groupby("city")["PM2.5"].mean().to_dict()
    global_baseline = float(df["PM2.5"].mean())

    city_row_counts = df.groupby("city").size().to_dict()

    output = {
        "station_baselines": {k: float(v) for k, v in station_baselines.items()},
        "city_baselines": {k: float(v) for k, v in city_baselines.items()},
        "city_row_counts": {k: int(v) for k, v in city_row_counts.items()},
        "global_baseline": global_baseline,
    }

    out_path = MODELS_DIR / "station_baselines.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Stations covered: {len(station_baselines)}")
    print(f"Cities covered: {len(city_baselines)}")
    print("\nCity baselines and data volume:")
    for city, baseline in sorted(city_baselines.items(), key=lambda x: -city_row_counts[x[0]]):
        print(f"  {city}: {baseline:.1f} µg/m³ avg ({city_row_counts[city]} rows)")

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    run()
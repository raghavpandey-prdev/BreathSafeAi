"""
Match live CPCB station names (from the hourly poll) to the Kaggle station
names the models were trained on. Kaggle's stations.csv has no lat/lon, so
geographic matching isn't possible — this matches by name instead, with a
prefix fallback for stations whose monitoring agency changed over time
(e.g. "Aya Nagar, Delhi - IITM" now vs. "Aya Nagar, Delhi - IMD" in Kaggle).

Anything that still doesn't match falls back to its city's average at
serving time — that fallback already exists in the training pipeline.

Usage: python pipelines/location/station_match.py
Output: models/station_name_map.json
"""

import json
import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent.parent / "models"


def norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())


def prefix(s):
    return norm(s.rsplit(" - ", 1)[0])


def run():
    live = pd.read_csv(DATA_DIR / "raw" / "aqi_history.csv")
    live = live[live["source"] == "cpcb_data_gov_in"]
    live_stations = sorted(live["station"].dropna().unique())

    kg = pd.read_csv(DATA_DIR / "raw" / "kaggle_historical" / "stations.csv", encoding="utf-8-sig")
    kg["StationName"] = kg["StationName"].str.strip()

    kg_exact = {norm(s): s for s in kg["StationName"]}
    kg_prefix = {}
    for s in kg["StationName"]:
        kg_prefix.setdefault(prefix(s), s)

    mapping = {}
    counts = {"exact": 0, "prefix": 0, "none": 0}
    for s in live_stations:
        if norm(s) in kg_exact:
            mapping[s] = {"matched_station": kg_exact[norm(s)], "method": "exact"}
            counts["exact"] += 1
        elif prefix(s) in kg_prefix:
            mapping[s] = {"matched_station": kg_prefix[prefix(s)], "method": "prefix"}
            counts["prefix"] += 1
        else:
            mapping[s] = {"matched_station": None, "method": "none"}
            counts["none"] += 1

    with open(MODELS_DIR / "station_name_map.json", "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"Matched {counts['exact']} exact + {counts['prefix']} by prefix "
          f"= {counts['exact'] + counts['prefix']} / {len(live_stations)} live stations.")
    print(f"{counts['none']} stations will use their city average instead.")
    print(f"Saved: {MODELS_DIR / 'station_name_map.json'}")


if __name__ == "__main__":
    run()
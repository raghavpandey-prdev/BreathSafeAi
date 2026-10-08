"""
Day 2 — Core ingestion script. Run this repeatedly (manually at first, scheduled later)
to build up historical AQI data for Delhi/NCR stations.

Each run:
1. Pulls all current CPCB records from data.gov.in (via curl.exe, since we confirmed
   that's the reliable path on this machine)
2. Filters down to Delhi/NCR cities only
3. Also pulls a few key Delhi/NCR station readings from aqicn.org as a cross-check source
4. Appends everything to data/raw/aqi_history.csv, skipping exact duplicate readings
   (same station + pollutant + timestamp) so re-running often doesn't create junk

Usage: python pipelines/ingestion/poll_aqi.py
"""

import os
import subprocess
import json
import csv
from pathlib import Path
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

load_dotenv()

DATA_GOV_IN_KEY = os.environ.get("DATA_GOV_IN_API_KEY", "")
AQICN_TOKEN = os.environ.get("AQICN_TOKEN", "demo")
DATA_GOV_IN_RESOURCE_ID = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"

# NCR = Delhi + these adjacent cities across UP/Haryana. Adjust this list if your
# station data uses slightly different city name spellings — check the raw data
# on your first run and expand this list if you notice relevant stations missing.
NCR_CITIES = {
    "Delhi", "Noida", "Greater Noida", "Gurugram", "Gurgaon",
    "Faridabad", "Ghaziabad", "Bahadurgarh", "Sonipat"
}

# Major metros beyond NCR — added for broader national coverage.
# Kept separate so NCR can still be treated as the primary, most-validated region.
OTHER_MAJOR_CITIES = {
    "Mumbai", "Bangalore", "Bengaluru", "Kolkata", "Chennai", "Hyderabad", "Pune",
    "Lucknow", "Jaipur", "Patna",
}

ALL_TARGET_CITIES = NCR_CITIES | OTHER_MAJOR_CITIES

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = DATA_DIR / "aqi_history.csv"

CSV_FIELDS = [
    "collected_at_utc", "source", "country", "state", "city", "station",
    "last_update", "latitude", "longitude", "pollutant_id",
    "min_value", "max_value", "avg_value"
]

# A few well-known Delhi/NCR station identifiers for aqicn.org's feed endpoint.
# These are supplementary — data.gov.in/CPCB is the primary source.
AQICN_STATIONS = [
    "delhi", "noida", "gurugram", "ghaziabad", "faridabad",
    "bengaluru", "hyderabad", "lucknow", "chennai", "mumbai", "kolkata", "jaipur", "patna",
]

def fetch_data_gov_in():
    """Fetch all current CPCB records via curl.exe (proven reliable path)."""
    url = (
        f"https://api.data.gov.in/resource/{DATA_GOV_IN_RESOURCE_ID}"
        f"?api-key={DATA_GOV_IN_KEY}&format=json&limit=5000"
    )
    for attempt in range(3):
        try:
            result = subprocess.run(
                ["curl.exe", "-sS", "-m", "90", url],
                capture_output=True, text=True, timeout=100
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            print(f"  data.gov.in fetch failed (attempt {attempt + 1}/3): {e}")
            continue
        if result.returncode == 0:
            break
        print(f"  data.gov.in curl error (attempt {attempt + 1}/3), "
              f"exit code {result.returncode}: {result.stderr[:200]}")
    else:
        return []

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"  data.gov.in returned invalid JSON. Raw response (first 500 chars):")
        print(f"  {result.stdout[:500]!r}")
        return []

    records = data.get("records", [])
    if not records:
        print(f"  data.gov.in response had no records. Raw response: {str(data)[:400]}")
    print(f"  Fetched {len(records)} total national records from data.gov.in")

    ncr_records = [r for r in records if r.get("city") in ALL_TARGET_CITIES]
    print(f"  Filtered to {len(ncr_records)} NCR records")
    return ncr_records


def fetch_aqicn():
    """Fetch a few key Delhi/NCR stations from aqicn.org as a supplementary source."""
    rows = []
    for station in AQICN_STATIONS:
        url = f"https://api.waqi.info/feed/{station}/?token={AQICN_TOKEN}"
        try:
            resp = requests.get(url, timeout=10)
            data = resp.json()
        except (requests.RequestException, ValueError):
            continue

        if data.get("status") != "ok":
            continue

        d = data.get("data", {})
        iaqi = d.get("iaqi", {})
        city_name = d.get("city", {}).get("name", station)
        time_s = d.get("time", {}).get("s", "")

        for pollutant, val in iaqi.items():
            rows.append({
                "collected_at_utc": datetime.now(timezone.utc).isoformat(),
                "source": "aqicn",
                "country": "India",
                "state": "",
                "city": city_name,
                "station": city_name,
                "last_update": time_s,
                "latitude": d.get("city", {}).get("geo", [None, None])[0],
                "longitude": d.get("city", {}).get("geo", [None, None])[1],
                "pollutant_id": pollutant.upper(),
                "min_value": "",
                "max_value": "",
                "avg_value": val.get("v", ""),
            })

    print(f"  Fetched {len(rows)} readings from aqicn.org")
    return rows


def load_existing_keys():
    """Load (station, pollutant_id, last_update) keys already in the CSV, to skip dupes."""
    if not CSV_PATH.exists():
        return set()
    keys = set()
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys.add((row["source"], row["station"], row["pollutant_id"], row["last_update"]))
    return keys


def append_rows(rows, existing_keys):
    file_exists = CSV_PATH.exists()
    new_count = 0
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            key = (row["source"], row["station"], row["pollutant_id"], row["last_update"])
            if key in existing_keys:
                continue
            writer.writerow(row)
            existing_keys.add(key)
            new_count += 1
    return new_count


def run():
    print(f"=== Polling AQI data — {datetime.now().isoformat()} ===")
    existing_keys = load_existing_keys()

    print("\nFetching data.gov.in (CPCB, primary)...")
    cpcb_records = fetch_data_gov_in()
    cpcb_rows = [
        {
            "collected_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": "cpcb_data_gov_in",
            "country": r.get("country", ""),
            "state": r.get("state", ""),
            "city": r.get("city", ""),
            "station": r.get("station", ""),
            "last_update": r.get("last_update", ""),
            "latitude": r.get("latitude", ""),
            "longitude": r.get("longitude", ""),
            "pollutant_id": r.get("pollutant_id", ""),
            "min_value": r.get("min_value", ""),
            "max_value": r.get("max_value", ""),
            "avg_value": r.get("avg_value", ""),
        }
        for r in cpcb_records
    ]

    print("\nFetching aqicn.org (supplementary)...")
    aqicn_rows = fetch_aqicn()

    all_rows = cpcb_rows + aqicn_rows
    new_count = append_rows(all_rows, existing_keys)

    print(f"\n=== Done. {new_count} new rows appended (duplicates skipped). ===")
    print(f"Total dataset: {CSV_PATH}")


if __name__ == "__main__":
    run()

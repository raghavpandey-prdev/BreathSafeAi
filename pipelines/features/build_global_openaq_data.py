"""
Phase A (global model) — pull real PM2.5 history from OpenAQ v3 for a set of
cities spanning every continent, as the raw material for a model that
generalizes beyond India.

This is deliberately separate from feature engineering: OpenAQ is a live,
paginated, rate-limited API (unlike the static Kaggle CSV the India model
used), so this script's only job is to pull and validate the raw data is
actually there and usable before anything gets built on top of it.

Usage:
    python pipelines/features/build_global_openaq_data.py --cities "Delhi,London"   # smoke test
    python pipelines/features/build_global_openaq_data.py                          # full pull

Requires OPENAQ_API_KEY in .env (free: https://explore.openaq.org/register).
Output: data/raw/openaq_global/<city>_<location_id>.csv (cached per location)
        data/raw/openaq_global/pull_summary.csv
"""

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

OPENAQ_BASE = "https://api.openaq.org/v3"
OPENAQ_API_KEY = os.environ.get("OPENAQ_API_KEY")
HEADERS = {"X-API-Key": OPENAQ_API_KEY} if OPENAQ_API_KEY else {}

RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "openaq_global"

MAX_LOCATIONS_PER_CITY = 3      # keep each city's pull bounded
SEARCH_RADIUS_M = 25000          # 25km, matches the India pipeline's station-matching radius
MAX_PAGES_PER_SENSOR = 150     # safety cap: ~100 rows/page -> up to ~4000 hourly rows/sensor
REQUEST_DELAY_S = 1.1            # stays safely under OpenAQ's 60 req/min free-tier limit
GLOBAL_START = "2018-01-01T00:00:00Z"
GLOBAL_END = "2021-12-31T23:59:59Z"

# One representative city per major world region. Deliberately NOT all in
# India/South Asia -- the whole point of this pull is geographic diversity.
TARGET_CITIES = {
    "Delhi":        (28.6139, 77.2090, "South Asia"),
    "Dhaka":        (23.8103, 90.4125, "South Asia"),
    "Beijing":      (39.9042, 116.4074, "East Asia"),
    "Seoul":        (37.5665, 126.9780, "East Asia"),
    "Jakarta":      (-6.2088, 106.8456, "Southeast Asia"),
    "Bangkok":      (13.7563, 100.5018, "Southeast Asia"),
    "London":       (51.5072, -0.1276, "Europe"),
    "Warsaw":       (52.2297, 21.0122, "Europe"),
    "Los Angeles":  (34.0522, -118.2437, "North America"),
    "Mexico City":  (19.4326, -99.1332, "North America"),
    "Toronto":      (43.6532, -79.3832, "North America"),
    "Sao Paulo":    (-23.5505, -46.6333, "South America"),
    "Santiago":     (-33.4489, -70.6693, "South America"),
    "Cairo":        (30.0444, 31.2357, "Africa / Middle East"),
    "Lagos":        (6.5244, 3.3792, "Africa / Middle East"),
    "Johannesburg": (-26.2041, 28.0473, "Africa / Middle East"),
    "Riyadh":       (24.7136, 46.6753, "Africa / Middle East"),
    "Sydney":       (-33.8688, 151.2093, "Oceania"),
}

_session = requests.Session()
_session.headers.update(HEADERS)
_pm25_parameter_id = None


def _get(path, params=None, retries=3):
    """GET with rate-limit pacing and backoff on 429/5xx. Returns parsed JSON
    or None on failure -- callers must handle None gracefully, never assume
    success."""
    url = f"{OPENAQ_BASE}{path}"
    for attempt in range(retries):
        time.sleep(REQUEST_DELAY_S)
        try:
            resp = _session.get(url, params=params, timeout=20)
        except requests.RequestException as e:
            print(f"    request error: {e}")
            continue
        if resp.status_code == 200:
            try:
                return resp.json()
            except ValueError:
                print(f"    non-JSON response from {url}")
                return None
        if resp.status_code in (429, 500, 502, 503):
            wait = 5 * (attempt + 1)
            print(f"    HTTP {resp.status_code} from {url}, retrying in {wait}s")
            time.sleep(wait)
            continue
        print(f"    HTTP {resp.status_code} from {url}: {resp.text[:200]}")
        return None
    return None


def get_pm25_parameter_id():
    global _pm25_parameter_id
    if _pm25_parameter_id is not None:
        return _pm25_parameter_id
    data = _get("/parameters", params={"limit": 200})
    if not data or "results" not in data:
        raise RuntimeError("Could not fetch /v3/parameters -- check OPENAQ_API_KEY in .env")
    for p in data["results"]:
        if p.get("name", "").lower() == "pm25":
            _pm25_parameter_id = p["id"]
            print(f"  Resolved pm25 parameter id = {_pm25_parameter_id}")
            return _pm25_parameter_id
    raise RuntimeError("No 'pm25' entry found in /v3/parameters -- API shape may have changed")


def find_locations_near(lat, lon, pm25_id, limit=10):
    data = _get("/locations", params={
        "coordinates": f"{lat},{lon}",
        "radius": SEARCH_RADIUS_M,
        "parameters_id": pm25_id,
        "limit": limit,
    })
    if not data or "results" not in data:
        return []
    return data["results"]


def get_pm25_sensor(location_id):
    data = _get(f"/locations/{location_id}/sensors")
    if not data or "results" not in data:
        return None
    for s in data["results"]:
        param = s.get("parameter", {})
        if param.get("name", "").lower() == "pm25":
            return s
    return None


def fetch_hourly_measurements(sensor_id):
    rows = []
    page = 1
    while page <= MAX_PAGES_PER_SENSOR:
        data = _get(f"/sensors/{sensor_id}/measurements/hourly", params={
            "limit": 100, "page": page,
            "datetime_from": GLOBAL_START, "datetime_to": GLOBAL_END,
        })
        if not data or "results" not in data or not data["results"]:
            break
        for r in data["results"]:
            try:
                value = r.get("value")
                period = r.get("period", {})
                ts = period.get("datetimeFrom", {}).get("utc")
                if value is None or ts is None:
                    continue
                rows.append({"timestamp": ts, "pm25": value})
            except (AttributeError, TypeError) as e:
                print(f"    skipping malformed row: {e}")
                continue
        if len(data["results"]) < 100:
            break  # last page
        page += 1
    return pd.DataFrame(rows)


def pull_city(city, lat, lon, region, pm25_id):
    print(f"\n--- {city} ({region}) ---")
    locations = find_locations_near(lat, lon, pm25_id)
    if not locations:
        print("  No OpenAQ locations found nearby.")
        return []

    summaries = []
    for loc in locations[:MAX_LOCATIONS_PER_CITY]:
        loc_id = loc.get("id")
        loc_name = loc.get("name", f"location_{loc_id}")
        coords = loc.get("coordinates", {})
        print(f"  Location: {loc_name} (id={loc_id})")

        sensor = get_pm25_sensor(loc_id)
        if sensor is None:
            print("    No pm25 sensor at this location, skipping.")
            continue

        df = fetch_hourly_measurements(sensor["id"])
        if df.empty:
            print("    No measurements returned, skipping.")
            continue

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp"])
        df["city"] = city
        df["region"] = region
        df["location_id"] = loc_id
        df["location_name"] = loc_name
        df["latitude"] = coords.get("latitude", lat)
        df["longitude"] = coords.get("longitude", lon)

        RAW_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RAW_DIR / f"{city.replace(' ', '_')}_{loc_id}.csv"
        df.to_csv(out_path, index=False)

        print(f"    Saved {len(df)} rows, {df['timestamp'].min()} -> {df['timestamp'].max()}")
        summaries.append({
            "city": city, "region": region, "location_id": loc_id, "location_name": loc_name,
            "rows": len(df), "start": df["timestamp"].min(), "end": df["timestamp"].max(),
        })

    return summaries


def run(cities):
    if not OPENAQ_API_KEY:
        raise SystemExit("OPENAQ_API_KEY not found in .env -- register free at "
                          "https://explore.openaq.org/register and add it before running this.")

    print("=== Pulling global PM2.5 history from OpenAQ ===")
    pm25_id = get_pm25_parameter_id()

    all_summaries = []
    for city, (lat, lon, region) in cities.items():
        all_summaries.extend(pull_city(city, lat, lon, region, pm25_id))

    if not all_summaries:
        print("\nNo data was pulled for any city. Check the warnings above.")
        return

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(RAW_DIR / "pull_summary.csv", index=False)

    print("\n=== Pull summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nCities with data: {summary_df['city'].nunique()} / {len(cities)}")
    print(f"Total rows: {summary_df['rows'].sum():,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", default=None, help="comma-separated subset, e.g. Delhi,London")
    args = parser.parse_args()

    if args.cities:
        chosen = {c.strip(): TARGET_CITIES[c.strip()] for c in args.cities.split(",") if c.strip() in TARGET_CITIES}
        if not chosen:
            raise SystemExit(f"No valid cities. Choose from: {', '.join(TARGET_CITIES)}")
    else:
        chosen = TARGET_CITIES

    run(chosen)
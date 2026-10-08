"""
Day 3 — Weather data collection for NCR, using OpenWeatherMap's Current Weather Data
endpoint (free tier, no card required).

Rather than fetching weather for every single station (there are dozens in NCR, and
weather doesn't vary that much station-to-station within a city), we fetch for a
handful of representative NCR locations and later merge by nearest-city-match when
building features. This keeps API usage well within free-tier limits.

Usage: python pipelines/ingestion/poll_weather.py
"""

import os
import csv
from pathlib import Path
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

load_dotenv()

OPENWEATHER_KEY = os.environ.get("OPENWEATHER_API_KEY", "")

# Representative coordinates for NCR cities — matches the NCR_CITIES list in poll_aqi.py
LOCATIONS = {
    "Delhi": (28.6139, 77.2090),
    "Noida": (28.5355, 77.3910),
    "Greater Noida": (28.4744, 77.5040),
    "Gurugram": (28.4595, 77.0266),
    "Faridabad": (28.4089, 77.3178),
    "Ghaziabad": (28.6692, 77.4538),
    "Bahadurgarh": (28.6931, 76.9209),
    "Sonipat": (28.9931, 77.0151),
    "Mumbai": (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Kolkata": (22.5726, 88.3639),
    "Chennai": (13.0827, 80.2707),
    "Hyderabad": (17.3850, 78.4867),
    "Pune": (18.5204, 73.8567),
    "Lucknow": (26.8467, 80.9462),
    "Jaipur": (26.9124, 75.7873),
    "Patna": (25.5941, 85.1376),
}

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = DATA_DIR / "weather_history.csv"

CSV_FIELDS = [
    "collected_at_utc", "city", "latitude", "longitude",
    "temp_c", "humidity_pct", "wind_speed_ms", "wind_deg",
    "pressure_hpa", "weather_main", "weather_desc"
]


def fetch_weather(city, lat, lon):
    if not OPENWEATHER_KEY:
        return None
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&appid={OPENWEATHER_KEY}&units=metric"
    )
    try:
        resp = requests.get(url, timeout=10)
    except requests.RequestException as e:
        print(f"  {city}: request failed — {e}")
        return None

    if resp.status_code == 401:
        print(f"  {city}: 401 Unauthorized — your key may still be activating "
              f"(can take up to ~2 hours after signup). Try again later.")
        return None
    if resp.status_code != 200:
        print(f"  {city}: unexpected status {resp.status_code} — {resp.text[:150]}")
        return None

    try:
        data = resp.json()
    except ValueError:
        print(f"  {city}: invalid JSON response")
        return None

    main = data.get("main", {})
    wind = data.get("wind", {})
    weather_list = data.get("weather", [{}])

    return {
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "city": city,
        "latitude": lat,
        "longitude": lon,
        "temp_c": main.get("temp", ""),
        "humidity_pct": main.get("humidity", ""),
        "wind_speed_ms": wind.get("speed", ""),
        "wind_deg": wind.get("deg", ""),
        "pressure_hpa": main.get("pressure", ""),
        "weather_main": weather_list[0].get("main", "") if weather_list else "",
        "weather_desc": weather_list[0].get("description", "") if weather_list else "",
    }


def run():
    start_time = datetime.now()
    print(f"=== Polling weather data — {start_time.isoformat()} ===")

    if not OPENWEATHER_KEY:
        print("OPENWEATHER_API_KEY not set in .env — nothing to do. Set it up first "
              "(see docs/api_setup.md).")
        return

    rows = []
    for city, (lat, lon) in LOCATIONS.items():
        row = fetch_weather(city, lat, lon)
        if row:
            rows.append(row)
            print(f"  {city}: {row['temp_c']}°C, {row['humidity_pct']}% humidity, "
                  f"wind {row['wind_speed_ms']} m/s, {row['weather_desc']}")

    if not rows:
        print("\nNo weather data collected this run.")
        return

    file_exists = CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n=== Done. {len(rows)} rows appended to {CSV_PATH} ===")
    print(f"Time taken: {elapsed:.1f} seconds")


if __name__ == "__main__":
    run()

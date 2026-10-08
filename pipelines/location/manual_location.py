"""
Manual Location Entry + Global Coverage Tiering.

Lets a user type ANY location in the world, geocodes it, then gives the best
honest answer available, in tiers:
  1. Near a station our trained model has real coverage for -> full ML forecast
  2. A live station nearby (any distance) -> live reading, with a confidence
     label that's honest about how far away it actually is
  3. No live station at all (aqicn has nothing anywhere reasonably close,
     e.g. open ocean, very remote regions) -> general country-level reference
     context, clearly labeled as NOT a live reading

The guiding principle throughout: never show a number without saying how
trustworthy it is, and never return a dead end when a lower-confidence
answer is honestly available.
"""

import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent / "features"))
from inference_helpers import load_baselines, city_has_sufficient_data

load_dotenv()
AQICN_TOKEN = os.environ.get("AQICN_TOKEN", "demo")

NOMINATIM_HEADERS = {
    "User-Agent": "BreatheSafeAI-research-project (personal, non-commercial)"
}

NEAREST_STATION_MAX_KM = 50  # only trust ML if a real station is genuinely close

# Rough, well-known general reference points (NOT live data) for when no
# live station exists anywhere reasonably close. Approximate annual PM2.5
# averages by broad public air-quality reporting (WHO/IQAir-style figures),
# used only as last-resort context so the app never returns a flat dead end.
COUNTRY_REFERENCE_PM25 = {
    "India": ("often high, especially Oct-Feb (crop burning + winter inversion); "
              "many regions regularly exceed WHO guidelines"),
    "China": ("historically high in northern industrial cities, improving in recent years"),
    "Pakistan": ("frequently among the highest in the world, especially in Punjab in winter"),
    "Bangladesh": ("consistently high, particularly in Dhaka"),
    "United States": ("generally moderate; wildfire season can cause sharp regional spikes"),
    "United Kingdom": ("generally good to moderate"),
    "Germany": ("generally good to moderate"),
    "Australia": ("generally good; bushfire season can cause severe regional spikes"),
    "Canada": ("generally good; wildfire season can cause severe regional spikes"),
    "Japan": ("generally good to moderate"),
    "Nepal": ("often poor, especially in the Kathmandu valley"),
    "Indonesia": ("often poor, particularly during dry-season haze/burning periods"),
}


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def geocode_location(query: str, retries: int = 2):
    """Turn a typed address/place name into (lat, lon, display_name, country).
    Retries on timeout, since the free Nominatim service can be slow/flaky."""
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": query, "format": "json", "limit": 1, "addressdetails": 1}

    last_error = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, headers=NOMINATIM_HEADERS, timeout=15)
            resp.raise_for_status()
            results = resp.json()
            break
        except (requests.RequestException, ValueError) as e:
            last_error = e
            if attempt < retries:
                import time
                time.sleep(1.5)  # respect Nominatim's rate limit, then retry
                continue
            return None, None, f"Geocoding failed after {retries + 1} attempts: {last_error}", None

    if not results:
        return None, None, "Location not found. Try a more specific address or city name.", None

    lat = float(results[0]["lat"])
    lon = float(results[0]["lon"])
    display_name = results[0]["display_name"]
    country = results[0].get("address", {}).get("country")
    return lat, lon, display_name, country


def find_nearest_live_station(lat: float, lon: float, df: pd.DataFrame):
    """Only meaningful within ML coverage — finds nearest station with real
    trained-model support."""
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return None, None
    stations = df[["station", "latitude", "longitude"]].drop_duplicates().dropna()
    if stations.empty:
        return None, None
    stations["distance_km"] = haversine_km(lat, lon, stations["latitude"], stations["longitude"])
    nearest = stations.sort_values("distance_km").iloc[0]
    return nearest["station"], nearest["distance_km"]


def confidence_tier(distance_km):
    if distance_km <= 50:
        return "local", "This reading is from a nearby station and should be fairly accurate."
    if distance_km <= 150:
        return "regional", f"Nearest live station is {distance_km:.0f} km away — treat as a regional estimate, not hyperlocal."
    if distance_km <= 500:
        return "country", f"Nearest live station is {distance_km:.0f} km away — this is only a rough, country/region-level estimate."
    return "very_approximate", f"Nearest live station is {distance_km:.0f} km away — this is a very rough estimate and may not reflect local conditions at all."


def get_global_live_conditions(lat: float, lon: float) -> dict:
    """Real current AQI/pollutant data for ANY location worldwide, via aqicn's
    geo-based feed. Always returns the nearest reading found (however far),
    tagged with an honest confidence tier instead of a hard cutoff — a
    distant, clearly-labeled estimate is more useful than a dead end."""
    url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={AQICN_TOKEN}"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
    except requests.RequestException as e:
        return {"error": f"Could not fetch live data: {e}"}

    if data.get("status") != "ok":
        return {"error": f"No live station data available near this location "
                          f"(aqicn status: {data.get('status')})."}

    d = data.get("data", {})
    station_geo = d.get("city", {}).get("geo", [None, None])
    distance = haversine_km(lat, lon, station_geo[0], station_geo[1]) if station_geo[0] is not None else None
    tier, tier_note = confidence_tier(distance) if distance is not None else (None, None)

    iaqi = d.get("iaqi", {})
    return {
        "station_name": d.get("city", {}).get("name", "Unknown station"),
        "aqi": d.get("aqi"),
        "pm25": iaqi.get("pm25", {}).get("v"),
        "pm10": iaqi.get("pm10", {}).get("v"),
        "measured_at": d.get("time", {}).get("s"),
        "distance_km": distance,
        "confidence_tier": tier,
        "confidence_note": tier_note,
    }


def resolve_location(query: str, live_df: pd.DataFrame) -> dict:
    lat, lon, display_name, country = geocode_location(query)
    if lat is None:
        return {"mode": "error", "message": display_name}

    baselines = load_baselines()
    station, distance = find_nearest_live_station(lat, lon, live_df)

    if station is not None and distance <= NEAREST_STATION_MAX_KM:
        city = live_df[live_df["station"] == station]["city"].iloc[0]
        if city_has_sufficient_data(city, baselines):
            return {
                "mode": "forecast",
                "display_name": display_name,
                "lat": lat, "lon": lon,
                "station": station,
                "distance_km": distance,
                "confidence_note": ("Full confidence" if city == "Delhi"
                                     else f"Limited history for {city} — forecast may be less reliable than Delhi/NCR."),
            }

    live_conditions = get_global_live_conditions(lat, lon)

    if "error" not in live_conditions and live_conditions.get("pm25") is not None:
        return {
            "mode": "live_only",
            "display_name": display_name,
            "lat": lat, "lon": lon,
            "data": live_conditions,
            "note": "No ML forecast model covers this location yet. Showing the nearest live reading found.",
        }

    # No live reading anywhere reasonable either — fall back to general
    # country-level context instead of a dead end, when we know the country.
    reference = COUNTRY_REFERENCE_PM25.get(country) if country else None
    return {
        "mode": "no_data",
        "display_name": display_name,
        "lat": lat, "lon": lon,
        "country": country,
        "reference_note": reference,
        "note": ("No live air quality station data could be found anywhere near this location. "
                 + (f"As general context, air quality in {country} is {reference}." if reference
                    else "No general reference is available for this region either.")),
    }


if __name__ == "__main__":
    print("=== Manual Location Entry Demo ===\n")

    features_path = Path(__file__).parent.parent.parent / "data" / "processed" / "features.csv"
    if features_path.exists():
        live_df = pd.read_csv(features_path)
        print(f"Loaded real live data: {len(live_df)} rows\n")
    else:
        live_df = pd.DataFrame(columns=["station", "latitude", "longitude"])
        print("No real features.csv found, using empty test data\n")

    test_queries = [
        "Connaught Place, Delhi",
        "Hisar, Haryana",
        "Reykjavik, Iceland",
        "Times Square, New York",
    ]

    for q in test_queries:
        print(f"--- Query: '{q}' ---")
        result = resolve_location(q, live_df)
        print(f"  Mode: {result['mode']}")
        print(f"  {result.get('note', result.get('message', ''))}")
        print()
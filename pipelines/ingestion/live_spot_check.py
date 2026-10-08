"""
Live spot-check: a fast, single-city aqicn.org read used to refresh "current
PM2.5" at request time, even when the full CPCB-based features.csv is stale.
This is separate from the historical model pipeline — the forecast's lag
features still come from features.csv, but the *current reading* shown to
the user can be minutes old instead of weeks old.
"""

import os
from pathlib import Path

import requests

AQICN_TOKEN = os.environ.get("AQICN_TOKEN", "demo")

CITY_SLUGS = {
    "Delhi": "delhi", "Gurugram": "gurugram", "Ghaziabad": "ghaziabad",
    "Noida": "noida", "Faridabad": "faridabad",
    "Bengaluru": "bengaluru", "Hyderabad": "hyderabad", "Lucknow": "lucknow",
    "Chennai": "chennai", "Mumbai": "mumbai", "Kolkata": "kolkata",
    "Jaipur": "jaipur", "Patna": "patna",
}


def get_live_spot(city, timeout=6):
    """Fresh PM2.5 (+ other pollutants) for a city from aqicn.org.
    Returns None on any failure — caller should fall back to CSV data."""
    slug = CITY_SLUGS.get(city)
    if not slug:
        return None
    try:
        resp = requests.get(
            f"https://api.waqi.info/feed/{slug}/",
            params={"token": AQICN_TOKEN}, timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, ValueError):
        return None

    if payload.get("status") != "ok":
        return None
    d = payload.get("data", {})
    iaqi = d.get("iaqi", {})
    pm25 = iaqi.get("pm25", {}).get("v")
    if pm25 is None:
        return None

    import pandas as pd
    try:
        timestamp = pd.to_datetime(d["time"]["s"])
    except (KeyError, ValueError):
        return None

    return {
        "pm25": float(pm25),
        "pm10": iaqi.get("pm10", {}).get("v"),
        "no2": iaqi.get("no2", {}).get("v"),
        "o3": iaqi.get("o3", {}).get("v"),
        "co": iaqi.get("co", {}).get("v"),
        "station_name": d.get("city", {}).get("name"),
        "timestamp": timestamp,
    }
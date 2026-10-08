"""
Shared helpers for baseline lookup + city features, used identically by both
training scripts and all inference code (dashboard, location comparison,
manual entry) — single source of truth so train/inference never diverge.
"""

import json
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent.parent / "models"
BASELINES_PATH = MODELS_DIR / "station_baselines.json"

KNOWN_CITIES = [
    "Delhi", "Mumbai", "Bengaluru", "Kolkata", "Chennai", "Hyderabad", "Pune",
    "Noida", "Gurugram", "Faridabad", "Ghaziabad", "Greater Noida",
    "Bahadurgarh", "Sonipat",
]

# Minimum rows of accumulated data before we trust an ML forecast for a city
MIN_ROWS_FOR_ML_FORECAST = 100


def load_baselines() -> dict:
    with open(BASELINES_PATH) as f:
        return json.load(f)


def get_station_baseline(station: str, city: str, baselines: dict) -> float:
    """Station-specific baseline if known, else city-level, else global."""
    if station in baselines["station_baselines"]:
        return baselines["station_baselines"][station]
    if city in baselines["city_baselines"]:
        return baselines["city_baselines"][city]
    return baselines["global_baseline"]


def city_has_sufficient_data(city: str, baselines: dict) -> bool:
    return baselines["city_row_counts"].get(city, 0) >= MIN_ROWS_FOR_ML_FORECAST


def build_city_dummies(city: str) -> dict:
    """One-hot city features. Unknown cities get all zeros (model treats as
    'other' — falls back to whatever general pattern it learned)."""
    return {f"city_{c}": (1 if c == city else 0) for c in KNOWN_CITIES}


CITY_DUMMY_COLS = [f"city_{c}" for c in KNOWN_CITIES]
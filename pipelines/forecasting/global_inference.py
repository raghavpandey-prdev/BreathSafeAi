"""
Serving layer for the GLOBAL model (train_global_model.py) -- the fallback
tier for any location on Earth that isn't one of the 10 India-specific
trained cities, but does have a live current reading available.

Honesty constraint this module exists to handle: a brand-new location has
only ONE live reading, never real lag/rolling history (we only have polled
history for the 24 OpenAQ training locations). Rather than silently fail or
fabricate fake history, this does a labeled COLD-START approximation: treat
the current reading as a stand-in for "no known recent change," which is a
standard, defensible production pattern -- and marks every such response
with is_cold_start=True so nothing downstream can present it as more
confident than it is.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xgboost as xgb

MODELS_DIR = Path(__file__).parent.parent.parent / "models"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_VARS = {
    "temperature_2m": "temp_c",
    "relative_humidity_2m": "humidity_pct",
    "wind_speed_10m": "wind_speed_ms",
    "wind_direction_10m": "wind_deg",
    "surface_pressure": "pressure_hpa",
    "precipitation": "precip_mm",
}

RISK_BANDS = [
    (30, "Good"), (60, "Satisfactory"), (90, "Moderate"),
    (120, "Poor"), (250, "Very Poor"), (float("inf"), "Severe"),
]


def risk_label(pm25):
    for upper, label in RISK_BANDS:
        if pm25 <= upper:
            return label
    return "Severe"


class GlobalForecaster:
    COLD_START_TRUST = {1: 0.9, 3: 0.75, 6: 0.6, 12: 0.4, 24: 0.25}

    def __init__(self):
        with open(MODELS_DIR / "global_manifest.json") as f:
            self.manifest = json.load(f)
        self.horizons = self.manifest["horizons"]
        self.regions = self.manifest["regions"]

        self.models, self.params = {}, {}
        for h in self.horizons:
            model = xgb.XGBRegressor()
            model.load_model(str(MODELS_DIR / f"global_model_{h}h.json"))
            self.models[h] = model
            with open(MODELS_DIR / f"global_conformal_{h}h.json") as f:
                self.params[h] = json.load(f)

    def fetch_current_weather(self, lat, lon):
        params = {
            "latitude": lat, "longitude": lon,
            "current": ",".join(WEATHER_VARS),
            "timezone": "UTC", "wind_speed_unit": "ms",
        }
        resp = requests.get(FORECAST_URL, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        current = {WEATHER_VARS.get(k, k): v for k, v in data.get("current", {}).items()
                   if k in WEATHER_VARS}
        return current

    def guess_region(self, lat, lon):
        """Very coarse lat/lon -> region bucket, matching the labels used in
        training, so conformal intervals pick a sensible region_q_hat even
        for a location that was never in the training set."""
        if lat > 5 and 60 <= lon <= 100:
            return "South Asia"
        if lat > 15 and 100 < lon <= 150:
            return "East Asia"
        if -10 <= lat <= 20 and 90 <= lon <= 150:
            return "Southeast Asia"
        if 35 <= lat <= 72 and -15 <= lon <= 45:
            return "Europe"
        if 15 <= lat <= 72 and -170 <= lon <= -50:
            return "North America"
        if -60 <= lat < 15 and -90 <= lon <= -30:
            return "South America"
        if -5 <= lat <= 40 and -20 <= lon <= 65:
            return "Africa / Middle East"
        if lat < -5 and lon > 110:
            return "Oceania"
        return None  # unknown -- caller falls back to the global q_hat

    def _build_cold_start_feature_dict(self, h, current_pm25, current_weather, lat, lon, now):
        """No lag/rolling history available for a brand-new location -- every
        PM2.5-history feature is approximated from the single current
        reading (a 'nothing is known to be changing' assumption). This is
        what is_cold_start=True in the response is warning about."""
        feat = {
            "hour_sin": np.sin(2 * np.pi * now.hour / 24), "hour_cos": np.cos(2 * np.pi * now.hour / 24),
            "dow_sin": np.sin(2 * np.pi * now.dayofweek / 7), "dow_cos": np.cos(2 * np.pi * now.dayofweek / 7),
            "month_sin": np.sin(2 * np.pi * now.month / 12), "month_cos": np.cos(2 * np.pi * now.month / 12),
            "is_weekend": int(now.dayofweek >= 5),
            "latitude": lat, "longitude": lon,
            "PM2.5": current_pm25,
        }
        for lag in [1, 2, 3, 6, 12, 24]:
            feat[f"PM2.5_lag{lag}h"] = current_pm25
        feat["PM2.5_change_1h"] = 0.0
        feat["PM2.5_change_24h"] = 0.0
        for w_label in ["PM2.5_roll3h_mean", "PM2.5_roll6h_mean", "PM2.5_roll24h_mean", "PM2.5_roll7d_mean"]:
            feat[w_label] = current_pm25
        feat["PM2.5_roll24h_std"] = 0.0
        feat.update(current_weather)
        return feat
    
    COLD_START_TRUST = {1: 0.9, 3: 0.75, 6: 0.6, 12: 0.4, 24: 0.25}


        # How much to trust the model's own extrapolation at each horizon, for a
    # COLD-START location specifically. We have zero real trend information
    # here -- just one reading -- so a 24h-ahead guess deserves far less
    # trust than a 1h-ahead one. This decays independently of (and on top
    # of) the region blend weight learned from real data.
    def predict_cold_start(self, current_pm25, lat, lon, now_utc=None):
        """Forecast for a location with NO prior polling history -- just one
        live reading. Every result is marked is_cold_start=True."""
        now_utc = now_utc or pd.Timestamp.now(tz="UTC")
        current_weather = self.fetch_current_weather(lat, lon)
        region = self.guess_region(lat, lon)

        results = []
        for h in self.horizons:
            params = self.params[h]
            feature_cols = params["feature_cols"]
            feat = self._build_cold_start_feature_dict(h, current_pm25, current_weather, lat, lon, now_utc)

            X = pd.DataFrame([feat]).reindex(columns=feature_cols)
            raw_pred = float(self.models[h].predict(X)[0])

            base_weight = params.get("blend_weights", {}).get(region, params.get("default_blend_weight", 1.0)) \
                if params.get("use_blend") and region else 1.0
            weight = base_weight * self.COLD_START_TRUST.get(h, 0.5)
            pred = weight * raw_pred + (1 - weight) * current_pm25

            q_hat = params.get("region_q_hats", {}).get(region, params["global_q_hat"]) if region \
                else params["global_q_hat"]

            results.append({
                "horizon_hours": h,
                "point": round(pred, 1),
                "lower": round(max(0, pred - q_hat), 1),
                "upper": round(pred + q_hat, 1),
                "risk_level": risk_label(pred),
            })

        return {
            "current_pm25": current_pm25,
            "current_weather": current_weather,
            "region_used": region,
            "is_cold_start": True,
            "confidence_note": ("No prior history for this exact location -- forecast approximated "
                                 "from the current reading only (cold start), weighted more toward "
                                 "the current reading at longer horizons where there's less trend "
                                 "information to go on. Treat with extra caution."),
            "horizons": results,
        }
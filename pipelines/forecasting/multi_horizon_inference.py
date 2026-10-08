"""
Serving-layer multi-horizon inference.

Loads every trained horizon model, fetches real hourly forecast weather from
Open-Meteo (free, no API key, 16-day hourly forecast), builds each horizon's
exact feature vector, applies the per-city blend weight learned at training
time, and returns a full prediction (point + interval) per horizon plus an
interpolated hourly curve for a "best time to go outside" recommendation.

predict_all() returns every trained horizon (used by /forecast/multi).
predict_single() returns just one horizon plus its engineered feature row,
for callers that need the feature row too (e.g. /forecast's explanation
layer), without paying for every horizon's weather lookup and prediction.
Both share the same feature-building and prediction internals so they can
never silently diverge.
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
FUTURE_WEATHER_VARS = ["temp_c", "humidity_pct", "wind_speed_ms", "precip_mm"]
FUTURE_WEATHER_FOR_HORIZON = {1: 6, 3: 6, 6: 6, 12: 12, 24: 24}

RISK_BANDS = [  # (upper bound, label) — Indian CPCB AQI PM2.5 breakpoints
    (30, "Good"), (60, "Satisfactory"), (90, "Moderate"),
    (120, "Poor"), (250, "Very Poor"), (float("inf"), "Severe"),
]


def risk_label(pm25):
    for upper, label in RISK_BANDS:
        if pm25 <= upper:
            return label
    return "Severe"


class MultiHorizonForecaster:
    def __init__(self):
        with open(MODELS_DIR / "model_manifest.json") as f:
            self.manifest = json.load(f)
        with open(MODELS_DIR / "station_baselines_v2.json") as f:
            self.baselines = json.load(f)
        try:
            with open(MODELS_DIR / "station_name_map.json") as f:
                self.station_map = json.load(f)
        except FileNotFoundError:
            self.station_map = {}

        self.horizons = self.manifest["horizons"]
        self.cities = self.manifest["cities"]
        self.models, self.params = {}, {}
        for h in self.horizons:
            model = xgb.XGBRegressor()
            model.load_model(str(MODELS_DIR / f"pm25_model_{h}h.json"))
            self.models[h] = model
            with open(MODELS_DIR / f"pm25_conformal_{h}h.json") as f:
                self.params[h] = json.load(f)

    # ------------------------------------------------------------ weather --

    def fetch_forecast_weather(self, lat, lon):
        """Hourly forecast weather for the next 24h, in IST, plus current."""
        params = {
            "latitude": lat, "longitude": lon,
            "hourly": ",".join(WEATHER_VARS),
            "current": ",".join(WEATHER_VARS),
            "wind_speed_unit": "ms",
            "timezone": "Asia/Kolkata",
            "forecast_days": 2,
        }
        resp = requests.get(FORECAST_URL, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        hourly = pd.DataFrame(data["hourly"])
        hourly["time"] = pd.to_datetime(hourly["time"])
        hourly = hourly.rename(columns=WEATHER_VARS)

        current = {WEATHER_VARS.get(k, k): v for k, v in data.get("current", {}).items()
                   if k in WEATHER_VARS}
        return current, hourly

    def weather_at_offset(self, hourly_df, now_ist, offset_hours):
        target = (now_ist + pd.Timedelta(hours=offset_hours)).floor("h")
        row = hourly_df.iloc[(hourly_df["time"] - target).abs().argsort()[:1]]
        if row.empty:
            return {}
        row = row.iloc[0]
        return {v: float(row[v]) for v in FUTURE_WEATHER_VARS if v in row}

    # --------------------------------------------------------- station id --

    def station_baseline(self, station, city):
        matched = self.station_map.get(station, {}).get("matched_station")
        stations = self.baselines["stations"]
        if matched and matched in stations:
            return stations[matched]
        cities = self.baselines["cities"]
        if city in cities:
            return cities[city]
        return self.baselines["global"]

    # ------------------------------------------------------- feature build --

    def _build_feature_dict(self, h, row, current_weather, future_weather, city, station, now_ist):
        """Pure feature engineering for one horizon — no model call. Shared
        by predict_horizon (all-horizons path) and predict_single (one-off)."""
        params = self.params[h]
        feature_cols = params["feature_cols"]
        fw_h = FUTURE_WEATHER_FOR_HORIZON[h]

        feat = {}
        feat["hour_sin"] = np.sin(2 * np.pi * now_ist.hour / 24)
        feat["hour_cos"] = np.cos(2 * np.pi * now_ist.hour / 24)
        feat["dow_sin"] = np.sin(2 * np.pi * now_ist.dayofweek / 7)
        feat["dow_cos"] = np.cos(2 * np.pi * now_ist.dayofweek / 7)
        feat["month_sin"] = np.sin(2 * np.pi * now_ist.month / 12)
        feat["month_cos"] = np.cos(2 * np.pi * now_ist.month / 12)
        feat["is_weekend"] = int(now_ist.dayofweek >= 5)

        feat["PM2.5"] = row.get("PM2.5", np.nan)
        for lag in [1, 2, 3, 6, 12, 24]:
            feat[f"PM2.5_lag{lag}h"] = row.get(f"PM2.5_lag{lag}h", np.nan)
        feat["PM2.5_change_1h"] = row.get("PM2.5_change_1h", np.nan)
        feat["PM2.5_change_24h"] = row.get("PM2.5_change_24h", np.nan)
        for w in [3, 6, 24]:
            feat[f"PM2.5_roll{w}h_mean"] = row.get(f"PM2.5_roll{w}h_mean", np.nan)
        feat["PM2.5_roll24h_std"] = row.get("PM2.5_roll24h_std", np.nan)
        for col in ["PM10", "NO2", "OZONE", "CO"]:
            feat[f"{col}_lag1h"] = row.get(f"{col}_lag1h", np.nan)

        feat.update(current_weather)
        for v in FUTURE_WEATHER_VARS:
            key = f"{v}_fut{fw_h}h"
            if key in feature_cols:
                feat[key] = future_weather.get(v, current_weather.get(v, np.nan))

        for c in self.cities:
            feat[f"city_{c.replace(' ', '_')}"] = 1 if c == city else 0

        feat["station_baseline_pm25"] = self.station_baseline(station, city)
        return feat

    def _predict_from_features(self, h, feat, city, now_ist):
        """Model call + blend + conformal interval, given an already-built
        feature dict. No weather fetching — pure prediction step."""
        params = self.params[h]
        feature_cols = params["feature_cols"]

        X = pd.DataFrame([feat]).reindex(columns=feature_cols)
        raw_pred = float(self.models[h].predict(X)[0])

        weight = params.get("blend_weights", {}).get(city, params.get("default_blend_weight", 1.0))
        persistence = feat["PM2.5"] if not np.isnan(feat["PM2.5"]) else raw_pred
        blended = weight * raw_pred + (1 - weight) * persistence

        season = "winter" if now_ist.month in (12, 1, 2) else \
                 "summer" if now_ist.month in (3, 4, 5) else "monsoon"
        q_hat = params["season_q_hats"].get(season, params["global_q_hat"])

        return {
            "horizon_hours": h,
            "point": round(blended, 1),
            "lower": round(max(0, blended - q_hat), 1),
            "upper": round(blended + q_hat, 1),
            "risk_level": risk_label(blended + q_hat),
        }

    def predict_horizon(self, h, row, current_weather, future_weather, city, station, now_ist):
        feat = self._build_feature_dict(h, row, current_weather, future_weather, city, station, now_ist)
        return self._predict_from_features(h, feat, city, now_ist)

    # ------------------------------------------------------------ predict --

    def predict_all(self, row, city, station, lat, lon, now_ist=None):
        now_ist = now_ist or pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)
        current_weather, hourly = self.fetch_forecast_weather(lat, lon)

        results = []
        for h in self.horizons:
            fw_h = FUTURE_WEATHER_FOR_HORIZON[h]
            future_weather = self.weather_at_offset(hourly, now_ist, fw_h)
            results.append(self.predict_horizon(
                h, row, current_weather, future_weather, city, station, now_ist
            ))

        curve = self.hourly_curve(row.get("PM2.5", np.nan), results)
        best = min(curve, key=lambda p: p["pm25"])

        return {
            "current_pm25": row.get("PM2.5"),
            "current_weather": current_weather,
            "horizons": results,
            "hourly_curve": curve,
            "best_time_next_24h": best,
        }

    def predict_single(self, row, city, station, lat, lon, horizon=6, now_ist=None):
        """One horizon's prediction plus the exact engineered feature row
        used to produce it — the feature row is what an explanation layer
        needs (combined with the model's feature_importances_) to say *why*
        this particular forecast came out the way it did."""
        if horizon not in self.horizons:
            raise ValueError(f"No trained model for horizon={horizon}h. Available: {self.horizons}")

        now_ist = now_ist or pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)
        current_weather, hourly = self.fetch_forecast_weather(lat, lon)
        fw_h = FUTURE_WEATHER_FOR_HORIZON[horizon]
        future_weather = self.weather_at_offset(hourly, now_ist, fw_h)

        feat = self._build_feature_dict(horizon, row, current_weather, future_weather, city, station, now_ist)
        result = self._predict_from_features(horizon, feat, city, now_ist)
        return result, feat, current_weather

    def feature_importance(self, horizon=6):
        """{feature_name: importance} for one horizon's model — used by the
        explanation layer and the /model/info model card."""
        if horizon not in self.horizons:
            raise ValueError(f"No trained model for horizon={horizon}h. Available: {self.horizons}")
        return dict(zip(self.params[horizon]["feature_cols"], self.models[horizon].feature_importances_))

    @staticmethod
    def hourly_curve(current_pm25, horizon_results):
        """Monotone-ish hourly interpolation across the trained anchor
        horizons (0, 1, 3, 6, 12, 24h) — an approximation between them, not a
        model in its own right, but enough for a 'best time today' chart."""
        anchor_h = [0] + [r["horizon_hours"] for r in horizon_results]
        anchor_v = [current_pm25] + [r["point"] for r in horizon_results]
        hours = np.arange(0, 25)
        values = np.interp(hours, anchor_h, anchor_v)
        return [{"hour_offset": int(h), "pm25": round(float(v), 1), "risk_level": risk_label(v)}
                for h, v in zip(hours, values)]
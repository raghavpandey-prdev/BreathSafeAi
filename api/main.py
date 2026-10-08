"""
FastAPI backend — exposes the full pipeline as clean JSON endpoints, so ANY
frontend (custom HTML, React, mobile app) can consume it.

Usage: uvicorn api.main:app --reload --port 8000
"""

import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "pipelines" / "personalization"))
sys.path.append(str(Path(__file__).parent.parent / "pipelines" / "advisory"))
sys.path.append(str(Path(__file__).parent.parent / "pipelines" / "location"))
sys.path.append(str(Path(__file__).parent.parent / "pipelines" / "explainability"))
sys.path.append(str(Path(__file__).parent.parent / "pipelines" / "features"))
sys.path.append(str(Path(__file__).parent.parent / "pipelines" / "forecasting"))
sys.path.append(str(Path(__file__).parent.parent / "pipelines" / "ingestion"))

from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
import json

from risk_engine import Forecast, UserProfile, generate_advisory
from generate_advisory import generate_natural_advisory
from location_intelligence import get_latest_row_for_station
from manual_location import resolve_location, geocode_location, find_nearest_live_station
from explain_forecast import compute_explanation
from multi_horizon_inference import MultiHorizonForecaster, risk_label
from live_spot_check import get_live_spot
from global_inference import GlobalForecaster

DATA_DIR = Path(__file__).parent.parent / "data"
FEATURES_CSV = DATA_DIR / "processed" / "features.csv"
MODELS_DIR = Path(__file__).parent.parent / "models"
REPORTS_DIR = Path(__file__).parent.parent / "reports"

GENERIC_ADVICE = {
    "Good": "Air quality is good. No precautions needed.",
    "Satisfactory": "Air quality is acceptable for most people.",
    "Moderate": "Sensitive groups (asthma, heart conditions, children, elderly) should limit prolonged outdoor exertion.",
    "Poor": "Everyone may begin to notice effects. Limit prolonged outdoor exertion, especially sensitive groups.",
    "Very Poor": "Avoid prolonged outdoor exertion. Sensitive groups should stay indoors where possible.",
    "Severe": "Avoid all outdoor exertion. This is a health emergency level for sensitive groups.",
}

app = FastAPI(title="BreatheSafe AI API", version="1.2.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"mode": "error", "message": f"Server error: {exc}"},
    )


_df_cache = None
_forecaster_cache = None


def get_data():
    global _df_cache
    if _df_cache is None:
        _df_cache = pd.read_csv(FEATURES_CSV, parse_dates=["timestamp"])
    return _df_cache


def get_forecaster():
    global _forecaster_cache
    if _forecaster_cache is None:
        _forecaster_cache = MultiHorizonForecaster()
    return _forecaster_cache

_global_forecaster_cache = None

def get_global_forecaster():
    global _global_forecaster_cache
    if _global_forecaster_cache is None:
        _global_forecaster_cache = GlobalForecaster()
    return _global_forecaster_cache


def _attach_global_forecast(result, pm25, lat, lon):
    """Best-effort: never let a global-model failure break the live reading
    that's already being returned."""
    try:
        gf = get_global_forecaster()
        result["global_forecast"] = gf.predict_cold_start(pm25, lat, lon)
    except Exception as e:
        result["global_forecast"] = None
    return result

def build_profile(asthma, heart_condition, pregnant, elderly, child, outdoor_job, activity) -> UserProfile:
    return UserProfile(
        has_asthma_or_copd=asthma, has_heart_condition=heart_condition,
        is_pregnant=pregnant, is_elderly=elderly, is_child=child,
        outdoor_occupation=outdoor_job, planned_activity=activity,
    )


def _clean(obj):
    """Recursively convert numpy scalar types to native Python types (FastAPI's
    JSON encoder can't serialize numpy.int64/float64 at all) and replace
    NaN/inf with None, so the response is always valid, serializable JSON."""
    if isinstance(obj, np.floating):
        obj = float(obj)
    elif isinstance(obj, np.integer):
        obj = int(obj)
    elif isinstance(obj, np.bool_):
        obj = bool(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


def _forecast_one(location: str, df: pd.DataFrame, profile: UserProfile,
                   language: str = "english", tone: str = "informative", with_explanation: bool = True):
    resolved = resolve_location(location, df)

    if resolved["mode"] == "error":
        return {"mode": "error", "message": resolved["message"], "query": location}

    if resolved["mode"] == "live_only":
        live_data = resolved["data"]
        pm25 = live_data.get("pm25")
        risk = risk_label(pm25) if pm25 is not None else None
        result = {
            "mode": "live_only",
            "query": location,
            "display_name": resolved["display_name"],
            "note": resolved["note"],
            "live_data": live_data,
            "risk_level": risk,
            "advice": GENERIC_ADVICE.get(risk, "No model-based forecast is available here yet "
                                                 "— this is the live reading only.") if risk else None,
        }
        if pm25 is not None:
            result = _attach_global_forecast(result, pm25, resolved["lat"], resolved["lon"])
        return _clean(result)

    if resolved["mode"] == "no_data":
        return {
            "mode": "no_data",
            "query": location,
            "display_name": resolved["display_name"],
            "note": resolved["note"],
            "country": resolved.get("country"),
        }

    station = resolved["station"]
    row = get_latest_row_for_station(station, df)
    if row is None:
        return {"mode": "error", "message": f"No recent data for station '{station}'.", "query": location}

    city = row["city"]
    forecaster = get_forecaster()

    data_as_of = row["timestamp"]
    data_source = "cpcb"
    current_pm25 = float(row["PM2.5"])
    pollutants = {
        "pm10": row.get("PM10"), "no2": row.get("NO2"),
        "ozone": row.get("OZONE"), "co": row.get("CO"),
    }

    age_hours = (pd.Timestamp.now() - data_as_of).total_seconds() / 3600
    if age_hours > 3:
        spot = get_live_spot(city)
        if spot:
            current_pm25 = spot["pm25"]
            data_as_of = spot["timestamp"]
            data_source = "live_spot_check"
            pollutants = {"pm10": spot.get("pm10"), "no2": spot.get("no2"),
                          "ozone": spot.get("o3"), "co": spot.get("co")}

    if city not in forecaster.cities:
        risk = risk_label(current_pm25)
        result = {
            "mode": "live_only",
            "query": location,
            "display_name": resolved["display_name"],
            "note": f"No trained forecast model covers {city} yet. Showing the live reading only.",
            "live_data": {"pm25": current_pm25, "station_name": station, **pollutants},
            "risk_level": risk,
            "advice": GENERIC_ADVICE.get(risk),
        }
        result = _attach_global_forecast(result, current_pm25, row["latitude"], row["longitude"])
        return _clean(result)

    row_dict = row.to_dict()
    row_dict["PM2.5"] = current_pm25  # use the freshened reading, not the possibly-stale CSV one
    horizon_result, feat, current_weather = forecaster.predict_single(
        row_dict, city=city, station=station, lat=row["latitude"], lon=row["longitude"], horizon=6,
    )

    forecast = Forecast(
        point_estimate=horizon_result["point"],
        lower_bound=horizon_result["lower"],
        upper_bound=horizon_result["upper"],
    )
    advisory = generate_advisory(forecast, profile)

    result = {
        "mode": "forecast",
        "query": location,
        "display_name": resolved["display_name"],
        "station": station,
        "distance_km": round(resolved["distance_km"], 1),
        "confidence_note": resolved.get("confidence_note", ""),
        "current_pm25": current_pm25,
        "data_source": data_source,
        "data_as_of": data_as_of.isoformat(),
        "data_age_hours": round((pd.Timestamp.now() - data_as_of).total_seconds() / 3600, 1),
        "pollutants": _clean(pollutants),
        "forecast_point": horizon_result["point"],
        "forecast_lower": horizon_result["lower"],
        "forecast_upper": horizon_result["upper"],
        "risk_level": advisory["personal_risk_level"],
        "advice": advisory["advice"],
    }

    if with_explanation:
        message = generate_natural_advisory(advisory, language=language, tone=tone, city=city)
        importance = forecaster.feature_importance(horizon=6)
        importance = {k: v for k, v in importance.items() if not k.startswith("city_")}
        explanation = compute_explanation(pd.Series(feat), importance)
        result["message"] = message
        result["explanation"] = explanation["explanation_text"]

    return _clean(result)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/forecast")
def get_forecast(
    location: str = Query(..., description="Any address/area name"),
    asthma: bool = False,
    heart_condition: bool = False,
    pregnant: bool = False,
    elderly: bool = False,
    child: bool = False,
    outdoor_job: bool = False,
    activity: str = "general_outdoor",
    language: str = "english",
    tone: str = "informative",
):
    df = get_data()
    profile = build_profile(asthma, heart_condition, pregnant, elderly, child, outdoor_job, activity)
    return _forecast_one(location, df, profile, language, tone)


@app.get("/compare")
def compare_locations(
    locations: str = Query(..., description="Comma-separated list of addresses/areas, e.g. 'Connaught Place, Delhi;Sector 62, Noida'"),
    asthma: bool = False,
    heart_condition: bool = False,
    pregnant: bool = False,
    elderly: bool = False,
    child: bool = False,
    outdoor_job: bool = False,
    activity: str = "general_outdoor",
):
    """Forecast + risk for several locations at once, ranked best-to-worst."""
    df = get_data()
    profile = build_profile(asthma, heart_condition, pregnant, elderly, child, outdoor_job, activity)

    queries = [q.strip() for q in locations.split(";") if q.strip()]
    if not queries:
        raise HTTPException(status_code=400, detail="Provide at least one location, separated by ';'")

    risk_order = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    results = []
    for q in queries:
        r = _forecast_one(q, df, profile, with_explanation=False)
        results.append(r)

    forecastable = [r for r in results if r["mode"] == "forecast"]
    excluded = [r for r in results if r["mode"] != "forecast"]
    ranked = sorted(forecastable, key=lambda r: risk_order.index(r["risk_level"]))

    return {
        "results": results,
        "ranked_count": len(ranked),
        "excluded_count": len(excluded),
        "best": ranked[0] if len(ranked) >= 2 else None,
        "worst": ranked[-1] if len(ranked) >= 2 else None,
    }


@app.get("/trend/{station}")
def get_trend(station: str, hours: int = Query(72, ge=1, le=24 * 30)):
    """Recent PM2.5 history for a station — powers a trend chart on the frontend."""
    df = get_data()
    station_rows = df[df["station"] == station].sort_values("timestamp")
    if station_rows.empty:
        raise HTTPException(status_code=404, detail=f"No data for station '{station}'")

    cutoff = station_rows["timestamp"].max() - pd.Timedelta(hours=hours)
    recent = station_rows[station_rows["timestamp"] >= cutoff]

    return {
        "station": station,
        "points": [
            {"timestamp": ts.isoformat(), "pm25": float(v)}
            for ts, v in zip(recent["timestamp"], recent["PM2.5"])
            if pd.notna(v)
        ],
    }


@app.get("/model/info")
def model_info():
    """A lightweight model card — what's deployed, how it was validated."""
    forecaster = get_forecaster()
    params = forecaster.params[6]
    importance = forecaster.feature_importance(horizon=6)
    top_features = dict(sorted(importance.items(), key=lambda kv: -kv[1])[:8])

    return {
        "model_type": "XGBoost Regressor, multi-horizon PM2.5 (1/3/6/12/24h), city-blended",
        "uncertainty_method": "Split conformal prediction, season-stratified",
        "target_confidence": params.get("target_confidence"),
        "empirical_coverage": params.get("empirical_coverage"),
        "season_interval_widths": params.get("season_q_hats"),
        "top_features": top_features,
        "cities_covered": sorted(forecaster.cities),
    }


@app.get("/model/evaluation")
def model_evaluation():
    """Real model-vs-baseline numbers from the last training run, for the
    dashboard's Model Health tab — not hand-typed, read from the actual report."""
    path = REPORTS_DIR / "multi_horizon_evaluation.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No evaluation report found. Run train_multi_horizon.py first.")

    df = pd.read_csv(path)
    all_df = df[df["city"] == "ALL"]

    def model_row_for(hdf, label, exclude=()):
        m = hdf["model"].str.contains(label, case=False, na=False)
        for e in exclude:
            m &= ~hdf["model"].str.contains(e, case=False, na=False)
        sub = hdf[m]
        return sub.iloc[0] if len(sub) else None

    horizons = []
    for h in sorted(all_df["horizon_h"].unique()):
        hdf = all_df[all_df["horizon_h"] == h]
        persistence = model_row_for(hdf, "persistence")
        climatology = model_row_for(hdf, "climatology")
        model = model_row_for(hdf, "xgboost", exclude=("raw",))
        if model is None:
            model = model_row_for(hdf, "xgboost")
        horizons.append({
            "horizon_h": int(h),
            "persistence_mae": float(persistence["MAE"]) if persistence is not None else None,
            "climatology_mae": float(climatology["MAE"]) if climatology is not None else None,
            "model_mae": float(model["MAE"]) if model is not None else None,
        })

    return {"horizons": horizons}


@app.get("/forecast/multi")
def forecast_multi(location: str = Query(..., description="Any address/area name")):
    df = get_data()
    forecaster = get_forecaster()

    lat, lon, display_name, country = geocode_location(location)
    if lat is None:
        raise HTTPException(status_code=422, detail=display_name)

    station, distance = find_nearest_live_station(lat, lon, df)
    if station is None or distance > 50:
        raise HTTPException(status_code=422, detail="No live monitoring station found nearby for a multi-horizon forecast.")

    city = df[df["station"] == station]["city"].iloc[0]
    if city not in forecaster.cities:
        raise HTTPException(
            status_code=422,
            detail=f"Multi-horizon forecast isn't trained for {city} yet. Covered cities: {', '.join(sorted(forecaster.cities))}."
        )

    row = get_latest_row_for_station(station, df)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No recent data for station '{station}'")

    result = forecaster.predict_all(
        row.to_dict(), city=city, station=station,
        lat=row["latitude"], lon=row["longitude"],
    )
    result["display_name"] = display_name
    result["station"] = station
    return _clean(result)


@app.get("/best-time")
def best_time(location: str = Query(...)):
    """Convenience wrapper: just the single best window in the next 24h."""
    full = forecast_multi(location)
    return {
        "display_name": full["display_name"],
        "current_pm25": full["current_pm25"],
        "best_time_next_24h": full["best_time_next_24h"],
        "hourly_curve": full["hourly_curve"],
    }
from manual_location import resolve_location, geocode_location, find_nearest_live_station, NOMINATIM_HEADERS
import requests as _requests

@app.get("/geocode/suggest")
def geocode_suggest(q: str = Query(..., min_length=2)):
    try:
        resp = _requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": q, "format": "json", "limit": 5, "addressdetails": 1},
            headers=NOMINATIM_HEADERS, timeout=8,
        )
        resp.raise_for_status()
        results = resp.json()
    except _requests.RequestException:
        return {"suggestions": []}
    return {"suggestions": [
        {"display_name": r["display_name"], "lat": float(r["lat"]), "lon": float(r["lon"])}
        for r in results
    ]}

# Serve the frontend directly from FastAPI
app.mount("/", StaticFiles(directory=str(Path(__file__).parent.parent / "web"), html=True), name="web")
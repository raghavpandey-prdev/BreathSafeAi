"""
Day 12 — Explainability Layer.

For a given forecast, surfaces WHICH specific feature(s) drove it, using the
model's own feature importances combined with this specific row's actual
values — not just a generic importance ranking, but "why THIS forecast."

Usage: import compute_explanation() into location_intelligence.py / dashboard,
or run this file directly for a demo against your latest live data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
import json

DATA_DIR = Path(__file__).parent.parent.parent / "data"
FEATURES_CSV = DATA_DIR / "processed" / "features.csv"
MODELS_DIR = Path(__file__).parent.parent.parent / "models"

FEATURE_COLS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend",
    "PM2.5_lag1h", "PM2.5_lag2h", "PM2.5_lag3h", "PM2.5_lag6h", "PM2.5_lag24h",
    "PM2.5_trend_24h",
    "PM2.5_roll3h_mean", "PM2.5_roll6h_mean",
    "PM10_lag1h", "NO2_lag1h", "OZONE_lag1h", "CO_lag1h",
    "temp_c", "humidity_pct", "wind_speed_ms", "wind_deg",
    "station_baseline_pm25",
]

# Human-readable labels for technical feature names — this is what actually
# gets shown to a user, not the raw column name
FEATURE_LABELS = {
    "PM2.5": "the current PM2.5 reading",
    "PM2.5_lag12h": "PM2.5 twelve hours ago",
    "PM2.5_change_1h": "how much PM2.5 has changed in the last hour",
    "PM2.5_change_24h": "how much PM2.5 has changed since this time yesterday",
    "PM2.5_roll24h_mean": "average PM2.5 over the last 24 hours",
    "PM2.5_roll24h_std": "how much PM2.5 has been fluctuating over the last 24 hours",
    "precip_mm": "current rainfall",
    "pressure_hpa": "current air pressure",
    "PM2.5_lag1h": "PM2.5 an hour ago",
    "PM2.5_lag24h": "PM2.5 at this time yesterday",
    "PM2.5_trend_24h": "whether pollution is rising or falling vs. yesterday",
    "PM2.5_roll3h_mean": "average PM2.5 over the last 3 hours",
    "PM2.5_roll6h_mean": "average PM2.5 over the last 6 hours",
    "PM10_lag1h": "PM10 (larger dust particles) an hour ago",
    "NO2_lag1h": "nitrogen dioxide (traffic-related) an hour ago",
    "OZONE_lag1h": "ozone levels an hour ago",
    "CO_lag1h": "carbon monoxide an hour ago",
    "temp_c": "current temperature",
    "humidity_pct": "current humidity",
    "wind_speed_ms": "current wind speed",
    "wind_deg": "current wind direction",
    "month_sin": "time of year (season)",
    "month_cos": "time of year (season)",
    "hour_sin": "time of day",
    "hour_cos": "time of day",
    "dow_sin": "day of the week",
    "dow_cos": "day of the week",
    "is_weekend": "whether it's a weekend",
    "station_baseline_pm25": "this station's typical pollution level",
    "PM2.5_lag6h": "PM2.5 six hours ago",
}


def load_model_importance():
    model = xgb.XGBRegressor()
    model.load_model(str(MODELS_DIR / "pm25_model_6h_conformal.json"))
    importance = dict(zip(FEATURE_COLS, model.feature_importances_))
    return model, importance


def compute_explanation(row: pd.Series, importance: dict, top_n: int = 3) -> dict:
    """Combine GLOBAL feature importance with THIS row's actual value to produce
    a row-specific explanation. Cyclical features (month_sin/cos, hour_sin/cos)
    are merged into one concept each, since sin/cos pairs represent a single
    underlying idea (season, time of day), not two separate factors."""

    # Merge cyclical sin/cos pairs into single conceptual features first
    MERGE_GROUPS = {
        "season": ["month_sin", "month_cos"],
        "time_of_day": ["hour_sin", "hour_cos"],
        "day_of_week": ["dow_sin", "dow_cos"],
    }
    merged_importance = dict(importance)
    for concept, members in MERGE_GROUPS.items():
        present = [m for m in members if m in merged_importance]
        if present:
            merged_importance[concept] = sum(merged_importance.pop(m) for m in present)

    candidates = []
    for feat, imp in merged_importance.items():
        if imp < 0.01:
            continue
        # For merged concepts, there's no single row value — handle separately
        if feat in ("season", "time_of_day", "day_of_week"):
            candidates.append((feat, imp, None))
            continue
        if feat in ("season", "time_of_day", "day_of_week"):
            candidates.append((feat, imp, None))
            continue
        if feat not in row.index or pd.isna(row[feat]):
            continue
        candidates.append((feat, imp, row[feat]))

    candidates.sort(key=lambda x: -x[1])
    top_features = candidates[:top_n]

    explanations = []
    for feat, imp, value in top_features:
        if feat == "season":
            explanations.append("the time of year (seasonal pollution pattern)")
        elif feat == "time_of_day":
            explanations.append("the time of day")
        elif feat == "day_of_week":
            explanations.append("the day of the week")
        elif "lag" in feat or "roll" in feat:
            label = FEATURE_LABELS.get(feat, feat.replace("_", " "))
            explanations.append(f"{label} was {value:.0f} µg/m³")
        elif feat == "PM2.5_trend_24h":
            direction = "rising" if value > 0 else "falling" if value < 0 else "steady"
            explanations.append(f"pollution is currently {direction} compared to yesterday ({value:+.0f} µg/m³)")
        elif feat in ("temp_c", "humidity_pct", "wind_speed_ms"):
            label = FEATURE_LABELS.get(feat, feat)
            explanations.append(f"{label} is {value:.1f}")
        else:
            label = FEATURE_LABELS.get(feat, feat.replace("_", " "))
            explanations.append(label)

    return {
        "top_factors": [f for f, _, _ in top_features],
        "explanation_text": "Main factors: " + "; ".join(explanations) + "." if explanations else
                             "No single dominant factor — this forecast relies on a broad mix of signals.",
    }


if __name__ == "__main__":
    print("=== Day 12: Explainability Demo ===\n")

    if not FEATURES_CSV.exists():
        print("No features.csv found.")
    else:
        df = pd.read_csv(FEATURES_CSV, parse_dates=["timestamp"])
        model, importance = load_model_importance()

        print("=== Global Feature Importance (model-wide, merged) ===")
        MERGE_GROUPS = {
            "Season": ["month_sin", "month_cos"],
            "Time of day": ["hour_sin", "hour_cos"],
            "Day of week": ["dow_sin", "dow_cos"],
        }
        merged = dict(importance)
        for label, members in MERGE_GROUPS.items():
            present = [m for m in members if m in merged]
            if present:
                merged[label] = sum(merged.pop(m) for m in present)
            else:
                continue

        for feat, imp in sorted(merged.items(), key=lambda x: -x[1])[:8]:
            display_label = feat if feat in MERGE_GROUPS else FEATURE_LABELS.get(feat, feat)
            print(f"  {display_label}: {imp:.3f}")
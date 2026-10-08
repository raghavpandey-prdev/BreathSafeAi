"""
Day 9 — Personalization Engine.

Maps a PM2.5 forecast (point estimate + conformal prediction interval) plus a
user health profile into a personalized risk level and recommended action.
Deliberately rule-based, not another ML model — health advisories need to be
explainable and defensible ("why did it say this?"), not a black box.

Uses official CPCB PM2.5 concentration breakpoints (µg/m³, 24-hr basis):
  Good: 0-30 | Satisfactory: 31-60 | Moderate: 61-90 |
  Poor: 91-120 | Very Poor: 121-250 | Severe: 250+

IMPORTANT DESIGN CHOICE: risk classification uses the UPPER bound of the
conformal interval, not the point estimate. Given real uncertainty, an
advisory should err cautious rather than assume the best case.

Usage: import into the dashboard, or run this file directly for a demo.
"""

from dataclasses import dataclass, field
from typing import Optional


# --- CPCB PM2.5 breakpoints (µg/m³) ---
AQI_BREAKPOINTS = [
    (0, 30, "Good"),
    (30, 60, "Satisfactory"),
    (60, 90, "Moderate"),
    (90, 120, "Poor"),
    (120, 250, "Very Poor"),
    (250, float("inf"), "Severe"),
]

# Tier order for shifting sensitive groups earlier
TIER_ORDER = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]


@dataclass
class UserProfile:
    """A person's health/activity profile — drives how cautious the advisory is."""
    has_asthma_or_copd: bool = False
    has_heart_condition: bool = False
    is_pregnant: bool = False
    is_elderly: bool = False  # 60+
    is_child: bool = False  # under 12
    outdoor_occupation: bool = False  # e.g. delivery, construction, traffic police
    planned_activity: str = "general_outdoor"  # "outdoor_exercise" | "commute" | "general_outdoor" | "indoor"


@dataclass
class Forecast:
    """Output from the conformal-calibrated 6h model."""
    point_estimate: float
    lower_bound: float
    upper_bound: float
    dominant_pollutant: str = "PM2.5"


def classify_pm25(value: float) -> str:
    for lo, hi, label in AQI_BREAKPOINTS:
        if lo <= value < hi:
            return label
    return "Severe"


def sensitivity_factor(profile: UserProfile) -> float:
    """Multiplicative adjustment simulating lower tolerance — not a category jump,
    so it stays meaningful whether the base air quality is mild or already severe."""
    factor = 1.0
    if profile.has_asthma_or_copd:
        factor += 0.30
    if profile.has_heart_condition:
        factor += 0.30
    if profile.is_pregnant:
        factor += 0.15
    if profile.is_elderly:
        factor += 0.15
    if profile.is_child:
        factor += 0.15
    if profile.outdoor_occupation:
        factor += 0.15
    return factor


ACTIVITY_ADVICE = {
    "Good": {
        "outdoor_exercise": "Great day for outdoor exercise.",
        "commute": "No air quality concerns for your commute.",
        "general_outdoor": "Enjoy time outdoors freely.",
        "indoor": "No air quality concerns either way.",
    },
    "Satisfactory": {
        "outdoor_exercise": "Fine for most people; sensitive individuals may notice minor discomfort.",
        "commute": "No significant concern for a normal commute.",
        "general_outdoor": "Generally fine for outdoor time.",
        "indoor": "No concern.",
    },
    "Moderate": {
        "outdoor_exercise": "Consider lighter exertion, or exercise indoors if you're sensitive to air quality.",
        "commute": "Fine for most people; consider a mask if you're sensitive.",
        "general_outdoor": "Okay for most people; take breaks if you notice irritation.",
        "indoor": "No concern, but consider keeping windows closed during peak hours.",
    },
    "Poor": {
        "outdoor_exercise": "Avoid outdoor exercise — move it indoors today.",
        "commute": "Consider a well-fitted mask (N95) for your commute.",
        "general_outdoor": "Limit prolonged outdoor time where possible.",
        "indoor": "Keep windows closed; consider an air purifier if available.",
    },
    "Very Poor": {
        "outdoor_exercise": "Skip outdoor exercise entirely today.",
        "commute": "Wear a well-fitted N95 mask; minimize time outdoors.",
        "general_outdoor": "Avoid non-essential time outdoors.",
        "indoor": "Keep windows closed; run an air purifier if you have one.",
    },
    "Severe": {
        "outdoor_exercise": "Do not exercise outdoors — genuine health risk today.",
        "commute": "Avoid non-essential travel. If you must go out, wear an N95 mask.",
        "general_outdoor": "Stay indoors as much as possible.",
        "indoor": "Keep windows sealed; air purifier strongly recommended if available.",
    },
}


def generate_advisory(forecast: Forecast, profile: UserProfile) -> dict:
    """Main entry point: combine forecast + profile into a personalized advisory."""
    base_tier = classify_pm25(forecast.upper_bound)
    factor = sensitivity_factor(profile)
    adjusted_value = forecast.upper_bound * factor
    personal_tier = classify_pm25(adjusted_value)
    advice = ACTIVITY_ADVICE[personal_tier][profile.planned_activity]

    explanation_parts = [
        f"Forecast: {forecast.point_estimate:.0f} µg/m³ "
        f"(range {forecast.lower_bound:.0f}-{forecast.upper_bound:.0f} µg/m³)."
    ]
    explanation_parts.append(
        f"Using the upper end of that range ({forecast.upper_bound:.0f}) for caution, "
        f"this falls in the '{base_tier}' category."
    )
    if factor > 1.0:
        reasons = []
        if profile.has_asthma_or_copd:
            reasons.append("asthma/COPD")
        if profile.has_heart_condition:
            reasons.append("a heart condition")
        if profile.is_pregnant:
            reasons.append("pregnancy")
        if profile.is_elderly:
            reasons.append("age 60+")
        if profile.is_child:
            reasons.append("being a child")
        if profile.outdoor_occupation:
            reasons.append("an outdoor occupation")
        explanation_parts.append(
            f"Because of {', '.join(reasons)}, your personal risk level is raised "
            f"to '{personal_tier}'."    
        )
        explanation_parts.append(f"Adjusted for your sensitivity, this is treated as ~{adjusted_value:.0f} µg/m³.")        
    else:
        explanation_parts.append("No sensitivity adjustments apply to your profile.")

    return {
        "base_aqi_category": base_tier,
        "personal_risk_level": personal_tier,
        "advice": advice,
        "explanation": " ".join(explanation_parts),
    }


# --- Demo / manual test ---
if __name__ == "__main__":
    print("=== Day 9: Personalization Engine Demo ===\n")

    example_forecast = Forecast(point_estimate=95, lower_bound=60, upper_bound=140)

    profiles = {
        "Healthy adult, planning a run": UserProfile(planned_activity="outdoor_exercise"),
        "Person with asthma, planning a run": UserProfile(
            has_asthma_or_copd=True, planned_activity="outdoor_exercise"
        ),
        "Elderly person, general outdoor errands": UserProfile(
            is_elderly=True, planned_activity="general_outdoor"
        ),
        "Delivery worker (outdoor occupation), commuting": UserProfile(
            outdoor_occupation=True, planned_activity="commute"
        ),
        "Pregnant, staying mostly indoors": UserProfile(
            is_pregnant=True, planned_activity="indoor"
        ),
    }
    print("\n=== Second example: forecast near a category boundary ===\n")
    boundary_forecast = Forecast(point_estimate=70, lower_bound=55, upper_bound=85)

    for label, profile in profiles.items():
        result = generate_advisory(boundary_forecast, profile)
        print(f"--- {label} ---")
        print(f"  Personal risk level: {result['personal_risk_level']}")
        print(f"  Advice: {result['advice']}\n")

    for label, profile in profiles.items():
        result = generate_advisory(example_forecast, profile)
        print(f"--- {label} ---")
        print(f"  Personal risk level: {result['personal_risk_level']}")
        print(f"  Advice: {result['advice']}")
        print(f"  Explanation: {result['explanation']}\n")
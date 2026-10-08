"""
Day 13 — Dashboard.

Ties together the risk engine (Day 9), natural-language advisory (Day 10),
location comparison (Day 11), and explainability (Day 12) into one interface.

Usage: streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "pipelines" / "personalization"))
sys.path.append(str(Path(__file__).parent.parent / "pipelines" / "advisory"))
sys.path.append(str(Path(__file__).parent.parent / "pipelines" / "location"))
sys.path.append(str(Path(__file__).parent.parent / "pipelines" / "explainability"))

import streamlit as st
import pandas as pd

from risk_engine import Forecast, UserProfile, generate_advisory
from generate_advisory import generate_natural_advisory
from location_intelligence import (
    get_latest_row_for_station, forecast_for_row, load_model_and_params,
)
from explain_forecast import compute_explanation, FEATURE_COLS as EXPLAIN_FEATURE_COLS

DATA_DIR = Path(__file__).parent.parent / "data"
FEATURES_CSV = DATA_DIR / "processed" / "features.csv"

st.set_page_config(page_title="BreatheSafe AI", page_icon="🌬️", layout="wide")


@st.cache_data(ttl=1800)
def load_features():
    if not FEATURES_CSV.exists():
        return None
    return pd.read_csv(FEATURES_CSV, parse_dates=["timestamp"])


@st.cache_resource
def load_model():
    return load_model_and_params()


def main():
    st.title("🌬️ BreatheSafe AI")
    st.caption("Personalized air quality forecasting for Delhi/NCR")

    df = load_features()
    latest_data_time = df["timestamp"].max()
    st.caption(f"Data last updated: {latest_data_time.strftime('%Y-%m-%d %H:%M')}")
    if df is None:
        st.error("No feature data found. Run the data pipeline first (build_features.py).")
        return

    model, params = load_model()
    all_stations = sorted(df["station"].dropna().unique().tolist())

    st.sidebar.header("Your Profile")
    has_asthma = st.sidebar.checkbox("Asthma / COPD")
    has_heart = st.sidebar.checkbox("Heart condition")
    is_pregnant = st.sidebar.checkbox("Pregnant")
    is_elderly = st.sidebar.checkbox("Age 60+")
    is_child = st.sidebar.checkbox("Child (under 12)")
    outdoor_job = st.sidebar.checkbox("Outdoor occupation (delivery, construction, etc.)")

    st.sidebar.header("Planned Activity")
    activity = st.sidebar.selectbox(
        "What are you planning?",
        ["outdoor_exercise", "commute", "general_outdoor", "indoor"],
        format_func=lambda x: {
            "outdoor_exercise": "Outdoor exercise",
            "commute": "Commute",
            "general_outdoor": "General outdoor time",
            "indoor": "Staying indoors",
        }[x],
    )

    st.sidebar.header("Message Style")
    language = st.sidebar.selectbox("Language", ["hinglish", "hindi", "english"])
    tone = st.sidebar.selectbox("Tone", ["informative", "gentle", "direct"])

    profile = UserProfile(
        has_asthma_or_copd=has_asthma,
        has_heart_condition=has_heart,
        is_pregnant=is_pregnant,
        is_elderly=is_elderly,
        is_child=is_child,
        outdoor_occupation=outdoor_job,
        planned_activity=activity,
    )

    st.header("Your Locations")
    col1, col2 = st.columns(2)
    with col1:
        home_query = st.text_input("Home (type any address/area)", "Connaught Place, Delhi")
    with col2:
        work_query = st.text_input("Work (type any address/area)", "")

    from manual_location import resolve_location
    locations_input = {"Home": home_query, "Work": work_query} if work_query else {"Home": home_query}

    resolved_locations = {}
    for label, query in locations_input.items():
        result = resolve_location(query, df)
        if result["mode"] == "error":
            st.error(f"{label}: {result['message']}")
        elif result["mode"] == "forecast":
            resolved_locations[label] = result["station"]
            if "confidence_note" in result and "Full confidence" not in result["confidence_note"]:
                st.caption(f"{label}: {result['confidence_note']}")
        else:
            st.warning(f"{label}: {result['note']}")

    locations = resolved_locations

    st.divider()

    cols = st.columns(len(locations))
    for col, (label, station) in zip(cols, locations.items()):
        with col:
            st.subheader(label)
            st.caption(station)

            row = get_latest_row_for_station(station, df)
            if row is None:
                st.warning("No recent data for this station.")
                continue

            forecast = forecast_for_row(row, model, params)
            advisory = generate_advisory(forecast, profile)

            risk_colors = {
                "Good": "🟢", "Satisfactory": "🟢", "Moderate": "🟡",
                "Poor": "🟠", "Very Poor": "🔴", "Severe": "🟣",
            }
            emoji = risk_colors.get(advisory["personal_risk_level"], "")

            st.metric("Current PM2.5", f"{row['PM2.5']:.0f} µg/m³")
            st.metric("6h Forecast", f"{forecast.point_estimate:.0f} µg/m³")
            st.caption(f"Likely range: {forecast.lower_bound:.0f}-{forecast.upper_bound:.0f} µg/m³ "
                       f"(risk level below uses the upper end, to stay on the safe side)")
            st.markdown(f"**Risk level:** {emoji} {advisory['personal_risk_level']}")

            with st.spinner("Generating advisory..."):
                message = generate_natural_advisory(advisory, language=language, tone=tone)
            st.info(message)

            available_features = [c for c in EXPLAIN_FEATURE_COLS if c in row.index]
            importance = dict(zip(available_features, model.feature_importances_[:len(available_features)]))
            explanation = compute_explanation(row, importance)
            st.caption(explanation["explanation_text"])

    st.divider()

    st.header("Comparison")
    rows = []
    for label, station in locations.items():
        row = get_latest_row_for_station(station, df)
        if row is not None:
            forecast = forecast_for_row(row, model, params)
            advisory = generate_advisory(forecast, profile)
            rows.append({"Location": label, "Risk Level": advisory["personal_risk_level"],
                         "Forecast": f"{forecast.point_estimate:.0f} µg/m³"})
    if rows:
        st.table(pd.DataFrame(rows))


if __name__ == "__main__":
    main()
"""
Day 4 — Exploratory Data Analysis on accumulated AQI + weather data.

Produces:
1. Missing-data summary per pollutant/station (the NA,NA,NA rows we've been seeing)
2. Station ranking by average pollution level
3. Diurnal pattern (how each pollutant varies by hour of day)
4. Basic weather correlation check
5. Saved plots + a summary CSV you can actually look at

Usage: python pipelines/eda/run_eda.py
Outputs land in data/processed/
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed, just save files
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "processed"
PLOTS_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

AQI_CSV = RAW_DIR / "aqi_history.csv"
WEATHER_CSV = RAW_DIR / "weather_history.csv"


def load_aqi():
    df = pd.read_csv(AQI_CSV)

    # Two different timestamp formats depending on source — parse each separately
    cpcb_mask = df["source"] == "cpcb_data_gov_in"
    df["timestamp"] = pd.NaT
    df.loc[cpcb_mask, "timestamp"] = pd.to_datetime(
        df.loc[cpcb_mask, "last_update"], format="%d-%m-%Y %H:%M:%S", errors="coerce"
    )
    df.loc[~cpcb_mask, "timestamp"] = pd.to_datetime(
        df.loc[~cpcb_mask, "last_update"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )

    df["avg_value"] = pd.to_numeric(df["avg_value"], errors="coerce")  # 'NA' strings -> NaN
    df["hour"] = df["timestamp"].dt.hour
    return df


def load_weather():
    if not WEATHER_CSV.exists():
        return None
    df = pd.read_csv(WEATHER_CSV)
    df["timestamp"] = pd.to_datetime(df["collected_at_utc"], errors="coerce")
    df["hour"] = df["timestamp"].dt.hour
    return df


def missing_data_summary(df):
    print("\n=== Missing Data Summary (CPCB records only) ===")
    cpcb = df[df["source"] == "cpcb_data_gov_in"]
    summary = cpcb.groupby("pollutant_id")["avg_value"].apply(
        lambda x: f"{x.isna().sum()} missing / {len(x)} total ({100 * x.isna().mean():.1f}%)"
    )
    print(summary.to_string())

    station_missing = (
        cpcb.groupby("station")["avg_value"]
        .apply(lambda x: x.isna().mean())
        .sort_values(ascending=False)
        .head(10)
    )
    print("\nTop 10 stations by missing-data rate:")
    print(station_missing.to_string())

    summary.to_csv(OUT_DIR / "missing_data_by_pollutant.csv")


def station_ranking(df):
    print("\n=== Station Ranking by Average PM2.5 ===")
    cpcb = df[(df["source"] == "cpcb_data_gov_in") & (df["pollutant_id"] == "PM2.5")]
    ranking = cpcb.groupby("station")["avg_value"].mean().sort_values(ascending=False)
    print(ranking.head(15).to_string())
    ranking.to_csv(OUT_DIR / "station_pm25_ranking.csv")

    fig, ax = plt.subplots(figsize=(10, 6))
    ranking.head(15).plot(kind="barh", ax=ax)
    ax.set_xlabel("Average PM2.5")
    ax.set_title("Top 15 Most Polluted Stations (PM2.5)")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "station_pm25_ranking.png", dpi=120)
    plt.close()


def diurnal_pattern(df):
    print("\n=== Diurnal Pattern (avg by hour of day) ===")
    cpcb = df[df["source"] == "cpcb_data_gov_in"]
    pollutants = ["PM2.5", "PM10", "NO2", "OZONE", "CO", "SO2", "NH3"]

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()

    for i, pol in enumerate(pollutants):
        sub = cpcb[cpcb["pollutant_id"] == pol]
        if sub.empty or sub["hour"].nunique() < 2:
            axes[i].set_title(f"{pol} (not enough hourly spread yet)")
            continue
        hourly = sub.groupby("hour")["avg_value"].mean()
        hourly.plot(ax=axes[i], marker="o")
        axes[i].set_title(f"{pol} by hour of day")
        axes[i].set_xlabel("Hour")
        axes[i].set_ylabel("Avg value")

    for j in range(len(pollutants), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "diurnal_patterns.png", dpi=120)
    plt.close()
    print(f"Saved diurnal pattern plots to {PLOTS_DIR / 'diurnal_patterns.png'}")
    print("NOTE: with only a few hours of data so far, this will look sparse — "
          "re-run this script again once you have several days of history for a "
          "meaningful pattern.")


def weather_check(weather_df):
    if weather_df is None:
        print("\nNo weather_history.csv found — skipping weather check.")
        return
    print("\n=== Weather Snapshot ===")
    latest = weather_df.sort_values("timestamp").groupby("city").tail(1)
    print(latest[["city", "temp_c", "humidity_pct", "wind_speed_ms", "weather_desc"]].to_string(index=False))


def run():
    print("Loading data...")
    aqi_df = load_aqi()
    weather_df = load_weather()

    print(f"Total AQI records: {len(aqi_df)}")
    print(f"Date range: {aqi_df['timestamp'].min()} to {aqi_df['timestamp'].max()}")
    print(f"Unique hours of data: {aqi_df['timestamp'].dt.floor('h').nunique()}")

    missing_data_summary(aqi_df)
    station_ranking(aqi_df)
    diurnal_pattern(aqi_df)
    weather_check(weather_df)

    print(f"\n=== Done. Outputs saved to {OUT_DIR} ===")
    print("Check the plots/ subfolder for visual patterns, and the CSV summaries "
          "for exact numbers.")


if __name__ == "__main__":
    run()

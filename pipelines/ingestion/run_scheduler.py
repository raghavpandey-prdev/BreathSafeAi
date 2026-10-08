"""
Runs poll_aqi.py automatically every N minutes, for as long as this terminal window
stays open. Simple option for now — for something that survives reboots/closed
terminals, set up Windows Task Scheduler instead (see docs/scheduling.md).

Usage: python pipelines/ingestion/run_scheduler.py
Stop with Ctrl+C.
"""

import schedule
import time
from poll_aqi import run as run_aqi
from poll_weather import run as run_weather

POLL_INTERVAL_MINUTES = 60  # CPCB data updates roughly hourly, no need to poll more often


def job():
    try:
        run_aqi()
    except Exception as e:
        print(f"AQI poll failed with error: {e}. Will retry next scheduled run.")
    try:
        run_weather()
    except Exception as e:
        print(f"Weather poll failed with error: {e}. Will retry next scheduled run.")


if __name__ == "__main__":
    print(f"Starting scheduler — polling every {POLL_INTERVAL_MINUTES} minutes.")
    print("Keep this terminal window open. Press Ctrl+C to stop.\n")

    job()  # run once immediately on startup, then wait for the schedule
    schedule.every(POLL_INTERVAL_MINUTES).minutes.do(job)

    while True:
        schedule.run_pending()
        time.sleep(30)

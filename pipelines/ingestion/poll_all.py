"""
Runs one AQI poll + one weather poll, then exits. Designed to be called by Windows
Task Scheduler on an hourly trigger (see docs/scheduling.md) — NOT meant to be run
as a long-lived loop like run_scheduler.py is.
"""

from poll_aqi import run as run_aqi
from poll_weather import run as run_weather

if __name__ == "__main__":
    try:
        run_aqi()
    except Exception as e:
        print(f"AQI poll failed: {e}")

    try:
        run_weather()
    except Exception as e:
        print(f"Weather poll failed: {e}")

import subprocess
subprocess.run(["python", "pipelines/features/build_features.py"])        

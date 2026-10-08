"""
Day 1 sanity check — verify both data sources actually work before building
anything on top of them. Run this FIRST.

Usage: python pipelines/ingestion/test_api_access.py
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

DATA_GOV_IN_KEY = os.environ.get("DATA_GOV_IN_API_KEY", "")
AQICN_TOKEN = os.environ.get("AQICN_TOKEN", "demo")  # falls back to public demo token

DATA_GOV_IN_RESOURCE_ID = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"


def test_data_gov_in():
    print("Testing data.gov.in (CPCB real-time AQI)...")
    if not DATA_GOV_IN_KEY:
        print("  SKIPPED — DATA_GOV_IN_API_KEY not set in .env yet.")
        return False

    url = (
        f"https://api.data.gov.in/resource/{DATA_GOV_IN_RESOURCE_ID}"
        f"?api-key={DATA_GOV_IN_KEY}&format=json&limit=5"
    )

    # Python's requests hangs on this specific host for some users (server-side
    # bot mitigation stalling non-browser TLS clients). Windows' built-in curl.exe
    # (or PowerShell's stack) reaches it fine, so use that as the actual method here.
    import subprocess
    import json as jsonlib

    try:
        result = subprocess.run(
            ["curl.exe", "-s", "-m", "20", url],
            capture_output=True, text=True, timeout=25
        )
    except FileNotFoundError:
        print("  FAILED — curl.exe not found on this system. Falling back to requests (may hang)...")
        return _test_data_gov_in_via_requests(url)
    except subprocess.TimeoutExpired:
        print("  FAILED — curl.exe also timed out. This may be a genuine server-side issue right now.")
        return False

    if result.returncode != 0:
        print(f"  FAILED — curl returned error code {result.returncode}: {result.stderr[:300]}")
        return False

    try:
        data = jsonlib.loads(result.stdout)
    except ValueError:
        print(f"  FAILED — response was not valid JSON: {result.stdout[:300]}")
        return False

    records = data.get("records", [])
    if not records:
        print("  WARNING — got a valid response but no records. Check your key/resource ID.")
        return False

    print(f"  SUCCESS (via curl.exe) — got {len(records)} records. Sample:")
    print(f"  {records[0]}")
    return True


def _test_data_gov_in_via_requests(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=30)
    except requests.RequestException as e:
        print(f"  FAILED — could not reach data.gov.in: {type(e).__name__}: {e}")
        return False

    if resp.status_code != 200:
        print(f"  FAILED — status code {resp.status_code}")
        print(f"  Response: {resp.text[:300]}")
        return False

    try:
        data = resp.json()
    except ValueError:
        print("  FAILED — response was not valid JSON")
        return False

    records = data.get("records", [])
    if not records:
        print("  WARNING — got a valid response but no records. Check your key/resource ID.")
        return False

    print(f"  SUCCESS — got {len(records)} records. Sample:")
    print(f"  {records[0]}")
    return True


def test_aqicn():
    print("\nTesting aqicn.org (backup source)...")
    url = f"https://api.waqi.info/feed/delhi/?token={AQICN_TOKEN}"
    try:
        resp = requests.get(url, timeout=10)
    except requests.RequestException as e:
        print(f"  FAILED — could not reach aqicn.org: {e}")
        return False

    try:
        data = resp.json()
    except ValueError:
        print("  FAILED — response was not valid JSON")
        return False

    if data.get("status") != "ok":
        print(f"  FAILED — API returned status: {data.get('status')}, message: {data.get('data')}")
        return False

    aqi = data.get("data", {}).get("aqi")
    city = data.get("data", {}).get("city", {}).get("name")
    print(f"  SUCCESS — {city}: current AQI = {aqi}")
    if AQICN_TOKEN == "demo":
        print("  NOTE: this used the public 'demo' token. Get your own real token for actual use "
              "(https://aqicn.org/data-platform/token/) — demo has limited/shared quota.")
    return True


if __name__ == "__main__":
    r1 = test_data_gov_in()
    r2 = test_aqicn()

    print("\n--- Summary ---")
    print(f"data.gov.in: {'WORKING' if r1 else 'NOT WORKING YET'}")
    print(f"aqicn.org:   {'WORKING' if r2 else 'NOT WORKING YET'}")

    if r1 or r2:
        print("\nAt least one source works — you can proceed to Day 2.")
    else:
        print("\nNeither source is working yet. Fix this before proceeding — "
              "check your .env file and API keys.")

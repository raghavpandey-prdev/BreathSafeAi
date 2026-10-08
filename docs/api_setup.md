# Day 1 — API Access Setup

## 1. data.gov.in (primary source — CPCB real-time AQI)

1. Go to https://data.gov.in
2. Click "Register" (top right) — sign up with email, or via DigiLocker SSO
3. Once logged in, go to your profile → "My Account" → you'll find an API key section, or
   go directly to https://data.gov.in/user/register and after registering, your API key
   is available from your account dashboard
4. The dataset we need is: **"Real Time Air Quality Index from Various Locations"**
   - Resource ID: `3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69`
   - You can find/confirm this yourself at:
     https://www.data.gov.in/resource/real-time-air-quality-index-various-locations
5. Example API call once you have your key:
```
https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69?api-key=YOUR_KEY&format=json&limit=100
```

## 2. aqicn.org (backup/supplementary source)

1. Go to https://aqicn.org/data-platform/token/
2. Enter your email, submit — token arrives instantly, no approval wait
3. Test with the public demo token first (no signup needed) to confirm the format works:
```
https://api.waqi.info/feed/delhi/?token=demo
```
4. Once you have your real token, replace `demo` with it for actual use.

## 3. Store both keys safely

Create a `.env` file in the project root (this is already in `.gitignore`, so it won't
get committed):
```
DATA_GOV_IN_API_KEY=your_key_here
AQICN_TOKEN=your_token_here
```

## 4. Verify before building anything else

Run `python pipelines/ingestion/test_api_access.py` (created below) — this is the step
we skipped last time and paid for. Confirm both APIs actually return real data before
writing a single line of pipeline code on top of them.

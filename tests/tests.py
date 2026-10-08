"""
Smoke tests for the FastAPI backend. Run with: pytest tests/
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_forecast_valid_location():
    resp = client.get("/forecast", params={"location": "Connaught Place, Delhi"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] in ("forecast", "live_only", "error")
    if body["mode"] == "forecast":
        assert "forecast_point" in body
        assert body["forecast_lower"] <= body["forecast_point"] <= body["forecast_upper"]
        assert body["risk_level"] in (
            "Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"
        )


def test_compare_requires_locations():
    resp = client.get("/compare", params={"locations": ""})
    assert resp.status_code == 400


def test_model_info():
    resp = client.get("/model/info")
    assert resp.status_code == 200
    body = resp.json()
    assert "empirical_coverage" in body
    assert "top_features" in body


def test_trend_unknown_station():
    resp = client.get("/trend/Nonexistent Station XYZ")
    assert resp.status_code == 404
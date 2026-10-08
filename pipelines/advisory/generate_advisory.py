"""
Day 10 — Health Advisory Generation.

Takes the structured output from Day 9's risk engine (personal_risk_level,
advice, explanation) and turns it into natural, warm advisory text via Groq —
in English, Hindi, or Hinglish. This is the module that actually talks TO the
person, rather than printing a rules-engine result.

Requires GROQ_API_KEY in your .env file.

Usage: python pipelines/advisory/generate_advisory.py
"""

import os
from groq import Groq
from dotenv import load_dotenv

# Import the Day 9 risk engine directly
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "personalization"))
from risk_engine import Forecast, UserProfile, generate_advisory

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL = "openai/gpt-oss-20b"

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


LANGUAGE_INSTRUCTIONS = {
    "english": "Write in plain, natural English.",
    "hindi": "Write in Hindi (Devanagari script).",
    "hinglish": "Write in Hinglish — Hindi mixed with English, written in Roman "
                "script, the way people actually text each other in Delhi/NCR "
                "(e.g. 'Aaj bahar mat jao, air bahut kharab hai'). Not overly "
                "formal Hindi, not pure English.",
}


def build_prompt(risk_result: dict, language: str, tone: str, city: str = "your area") -> str:
    tone_instructions = {
        "informative": "Calm and informative — state the facts and the recommendation clearly, "
                       "without being alarmist or preachy.",
        "gentle": "Warm and gentle — like a caring friend giving a heads-up, not a warning label.",
        "direct": "Short and direct — get straight to the point, minimal explanation.",
    }

    return f"""You are writing a short air quality health advisory message for someone in {city}, India.

Structured data to base the message on:
- Personal risk level: {risk_result['personal_risk_level']}
- Recommended action: {risk_result['advice']}
- Reasoning: {risk_result['explanation']}

Instructions:
- {LANGUAGE_INSTRUCTIONS[language]}
- Tone: {tone_instructions[tone]}
- Keep it to 2-3 sentences maximum.
- Do not invent any numbers, health claims, or advice beyond what's given above.
- Do not use the words "AQI" or technical jargon unless naturally necessary — write like a person, not a report.
- Do not start with a greeting like "Hi" or "Hello" — just give the message directly.

Write only the message itself, nothing else (no preamble, no quotation marks around it).
"""


def generate_natural_advisory(risk_result: dict, language: str = "hinglish", tone: str = "informative", city: str = "your area") -> str:
    if client is None:
        return ("[GROQ_API_KEY not set — cannot generate natural-language advisory. "
                "Falling back to structured advice: " + risk_result["advice"] + "]")

    prompt = build_prompt(risk_result, language, tone, city)# --- Demo ---
if __name__ == "__main__":
    print("=== Day 10: Health Advisory Generation Demo (full risk spectrum) ===\n")

    # Forecasts chosen to land in each CPCB category, using a healthy adult profile
    # planning general outdoor activity, so we see all 6 risk levels distinctly
    spectrum_tests = [
        ("Good", Forecast(point_estimate=15, lower_bound=10, upper_bound=25)),
        ("Satisfactory", Forecast(point_estimate=40, lower_bound=30, upper_bound=50)),
        ("Moderate", Forecast(point_estimate=70, lower_bound=60, upper_bound=80)),
        ("Poor", Forecast(point_estimate=100, lower_bound=90, upper_bound=110)),
        ("Very Poor", Forecast(point_estimate=180, lower_bound=150, upper_bound=200)),
        ("Severe", Forecast(point_estimate=300, lower_bound=270, upper_bound=330)),
    ]

    healthy_adult = UserProfile(planned_activity="general_outdoor")

    for expected_label, forecast in spectrum_tests:
        risk_result = generate_advisory(forecast, healthy_adult)
        message = generate_natural_advisory(risk_result, language="hinglish", tone="informative")
        print(f"--- Expected: {expected_label} ---")
        print(f"  Actual risk level: {risk_result['personal_risk_level']}")
        print(f"  Message: {message}\n")
import { useState } from "react";
import { getForecast, getForecastMulti } from "../api.js";
import RiskBadge from "./RiskBadge.jsx";
import HorizonCards from "./HorizonCards.jsx";
import HourlyCurveChart from "./HourlyCurveChart.jsx";
import SpikeAlertBanner from "./SpikeAlertBanner.jsx";
import TrendSparkline from "./TrendSparkline.jsx";
import LocationAutocomplete from "./LocationAutocomplete.jsx";

const DEFAULT_PROFILE = {
  asthma: false, heart_condition: false, pregnant: false,
  elderly: false, child: false, outdoor_job: false,
  activity: "general_outdoor", language: "english", tone: "informative",
};

export default function ForecastPanel() {
  const [location, setLocation] = useState("Connaught Place, Delhi");
  const [profile, setProfile] = useState(DEFAULT_PROFILE);
  const [advisory, setAdvisory] = useState(null);
  const [multi, setMulti] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const toggle = (key) => setProfile((p) => ({ ...p, [key]: !p[key] }));
  const setField = (key, value) => setProfile((p) => ({ ...p, [key]: value }));

  async function handleSearch() {
    setLoading(true);
    setError(null);
    setAdvisory(null);
    setMulti(null);
    try {
      const [adv, m] = await Promise.all([
        getForecast(location, profile),
        getForecastMulti(location).catch(() => null),
      ]);
      setAdvisory(adv);
      setMulti(m);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-2xl shadow-sm p-6">
        <h2 className="font-semibold text-slate-700 mb-3">Check a location</h2>
        <div className="flex gap-2 mb-4">
          <LocationAutocomplete
            value={location}
            onChange={setLocation}
            onEnter={handleSearch}
            placeholder="Type any address or area..."
          />
          <button
            onClick={handleSearch}
            disabled={loading}
            className="bg-slate-800 text-white px-5 rounded-lg hover:bg-slate-900 disabled:opacity-50"
          >
            {loading ? "Checking..." : "Check"}
          </button>
        </div>

        <div className="flex flex-wrap gap-3 text-sm mb-4">
          {[
            ["asthma", "Asthma/COPD"], ["heart_condition", "Heart patient"],
            ["pregnant", "Pregnant"], ["elderly", "60+"],
            ["child", "Child"], ["outdoor_job", "Outdoor job"],
          ].map(([key, label]) => (
            <label key={key} className="flex items-center gap-1.5 text-slate-600 whitespace-nowrap">
              <input type="checkbox" checked={profile[key]} onChange={() => toggle(key)} />
              {label}
            </label>
          ))}
        </div>

        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          <div>
            <label className="block text-xs text-slate-400 mb-1">Activity</label>
            <select
              value={profile.activity}
              onChange={(e) => setField("activity", e.target.value)}
              className="border rounded-lg p-1.5 w-full text-sm"
            >
              <option value="outdoor_exercise">Outdoor exercise</option>
              <option value="commute">Commute</option>
              <option value="general_outdoor">General outdoor time</option>
              <option value="indoor">Staying indoors</option>
            </select>
          </div>
          <div>
            <label className="block text-xs text-slate-400 mb-1">Message language</label>
            <select
              value={profile.language}
              onChange={(e) => setField("language", e.target.value)}
              className="border rounded-lg p-1.5 w-full text-sm"
            >
              <option value="english">English</option>
              <option value="hindi">Hindi</option>
              <option value="hinglish">Hinglish</option>
            </select>
          </div>
          <div>
            <label className="block text-xs text-slate-400 mb-1">Tone</label>
            <select
              value={profile.tone}
              onChange={(e) => setField("tone", e.target.value)}
              className="border rounded-lg p-1.5 w-full text-sm"
            >
              <option value="informative">Informative</option>
              <option value="gentle">Gentle</option>
              <option value="direct">Direct</option>
            </select>
          </div>
        </div>
      </div>

      {error && (
        <div className="bg-red-50 text-red-600 text-sm rounded-lg p-4">{error}</div>
      )}

      {advisory && advisory.mode === "live_only" && (
        <div className="bg-white rounded-2xl shadow-sm p-6">
          <div className="flex items-start justify-between mb-1">
            <h3 className="font-bold text-lg">{advisory.display_name}</h3>
            {advisory.risk_level && <RiskBadge level={advisory.risk_level} />}
          </div>
          <p className="text-amber-600 text-sm mt-1 mb-3">{advisory.note}</p>

          {advisory.live_data?.pm25 != null ? (
            <>
              <div className="text-3xl font-bold mb-1">{advisory.live_data.pm25} µg/m³</div>
              {advisory.live_data.confidence_note && (
                <p className="text-xs text-amber-600 mb-2">{advisory.live_data.confidence_note}</p>
              )}
            </>
          ) : (
            <div className="text-slate-400 text-sm mb-2">No reading available</div>
          )}

          {advisory.live_data?.station_name && (
            <p className="text-xs text-slate-400 mb-3">
              Live from {advisory.live_data.station_name}
              {advisory.live_data.distance_km != null && ` (${Math.round(advisory.live_data.distance_km)} km away)`}
            </p>
          )}
          {advisory.advice && (
            <div className="bg-amber-50 rounded-lg p-3 text-sm text-amber-900">
              {advisory.advice}
            </div>
          )}

          {advisory.global_forecast && (
            <div className="mt-3 pt-3 border-t border-slate-100">
              <p className="text-xs text-amber-600 mb-2">{advisory.global_forecast.confidence_note}</p>
              <h4 className="text-sm font-semibold text-slate-600 mb-2">Approximate forecast (global model)</h4>
              <HorizonCards horizons={advisory.global_forecast.horizons} />
            </div>
          )}
        </div>
      )}

      {advisory && advisory.mode === "no_data" && (
        <div className="bg-white rounded-2xl shadow-sm p-6">
          <h3 className="font-bold text-lg mb-2">{advisory.display_name}</h3>
          <p className="text-slate-500 text-sm">{advisory.note}</p>
        </div>
      )}

      {advisory && advisory.mode === "forecast" && (
        <div className="bg-white rounded-2xl shadow-sm p-6 space-y-5">
          <div className="flex items-start justify-between">
            <div>
              <h3 className="font-bold text-lg">{advisory.display_name}</h3>
              <p className="text-xs text-slate-400">
                Nearest station: {advisory.station} ({advisory.distance_km} km)
              </p>
            </div>
            <RiskBadge level={advisory.risk_level} />
          </div>

          {multi && <SpikeAlertBanner horizons={multi.horizons} />}

          <div className="flex items-center gap-2 text-xs">
            <span className={`px-2 py-0.5 rounded-full font-medium ${
              advisory.data_source === "live_spot_check"
                ? "bg-emerald-100 text-emerald-700"
                : "bg-slate-100 text-slate-500"
            }`}>
              {advisory.data_source === "live_spot_check" ? "● Live" : "Archived"}
            </span>
            <span className="text-slate-400">
              {advisory.data_age_hours < 1
                ? "updated just now"
                : advisory.data_age_hours >= 24
                  ? `${Math.round(advisory.data_age_hours / 24)} day(s) old`
                  : `${Math.round(advisory.data_age_hours)}h old`}
            </span>
          </div>

          <div className="bg-indigo-50 rounded-lg p-3 text-sm text-indigo-900">
            {advisory.message}
          </div>
          <p className="text-xs text-slate-400">{advisory.explanation}</p>

          {advisory.pollutants && (
            <div className="grid grid-cols-4 gap-2 text-center text-xs">
              {[
                ["pm10", "PM10"], ["no2", "NO₂"], ["ozone", "O₃"], ["co", "CO"],
              ].map(([key, label]) => (
                <div key={key} className="bg-slate-50 rounded-lg py-2">
                  <div className="text-slate-400">{label}</div>
                  <div className="font-semibold text-slate-700">
                    {advisory.pollutants[key] ?? "–"}
                  </div>
                </div>
              ))}
            </div>
          )}

          {multi?.current_weather && (
            <div className="grid grid-cols-4 gap-2 text-center text-xs">
              <div className="bg-slate-50 rounded-lg py-2">
                <div className="text-slate-400">Temp</div>
                <div className="font-semibold text-slate-700">{multi.current_weather.temp_c}°C</div>
              </div>
              <div className="bg-slate-50 rounded-lg py-2">
                <div className="text-slate-400">Humidity</div>
                <div className="font-semibold text-slate-700">{multi.current_weather.humidity_pct}%</div>
              </div>
              <div className="bg-slate-50 rounded-lg py-2">
                <div className="text-slate-400">Wind</div>
                <div className="font-semibold text-slate-700">{multi.current_weather.wind_speed_ms} m/s</div>
              </div>
              <div className="bg-slate-50 rounded-lg py-2">
                <div className="text-slate-400">Rain</div>
                <div className="font-semibold text-slate-700">{multi.current_weather.precip_mm ?? 0} mm</div>
              </div>
            </div>
          )}

          {multi && (
            <>
              <div>
                <h4 className="text-sm font-semibold text-slate-600 mb-2">Multi-horizon forecast</h4>
                <HorizonCards horizons={multi.horizons} />
              </div>
              <div>
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-sm font-semibold text-slate-600">Next 24 hours</h4>
                  {multi.best_time_next_24h && (
                    <span className="text-xs text-emerald-600 font-medium">
                      Best window: +{multi.best_time_next_24h.hour_offset}h ({multi.best_time_next_24h.pm25} µg/m³)
                    </span>
                  )}
                </div>
                <HourlyCurveChart
                  curve={multi.hourly_curve}
                  bestHour={multi.best_time_next_24h?.hour_offset}
                />
              </div>
            </>
          )}

          <TrendSparkline station={advisory.station} />
        </div>
      )}
    </div>
  );
}
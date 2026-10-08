const BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

async function getJSON(path) {
  const res = await fetch(`${BASE}${path}`);
  const text = await res.text();
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    throw new Error(
      res.ok
        ? "Server returned a non-JSON response."
        : `Server error (${res.status}): ${text.slice(0, 150)}`
    );
  }
  if (!res.ok && !data.mode) {
    throw new Error(data.detail || data.message || `Request failed (${res.status})`);
  }
  return data;
}

export function getForecast(location, profile) {
  const params = new URLSearchParams({ location, ...profile });
  return getJSON(`/forecast?${params}`);
}

export function getForecastMulti(location) {
  const params = new URLSearchParams({ location });
  return getJSON(`/forecast/multi?${params}`);
}

export function getBestTime(location) {
  const params = new URLSearchParams({ location });
  return getJSON(`/best-time?${params}`);
}

export function compareLocations(locations) {
  const params = new URLSearchParams({ locations: locations.join(";") });
  return getJSON(`/compare?${params}`);
}

export function getModelInfo() {
  return getJSON(`/model/info`);
}

export function getModelEvaluation() {
  return getJSON(`/model/evaluation`);
}

export function getTrend(station, hours = 168) {
  const params = new URLSearchParams({ hours });
  return getJSON(`/trend/${encodeURIComponent(station)}?${params}`);
}

export function getSuggestions(query) {
  return getJSON(`/geocode/suggest?q=${encodeURIComponent(query)}`);
}
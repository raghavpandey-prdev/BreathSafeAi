import { useState } from "react";
import { compareLocations } from "../api.js";
import RiskBadge from "./RiskBadge.jsx";

export default function CompareView() {
  const [rows, setRows] = useState([
    { label: "Home", query: "Connaught Place, Delhi" },
    { label: "Work", query: "Bandra Kurla Complex, Mumbai" },
  ]);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const updateRow = (i, field, value) => {
    setRows((r) => r.map((row, idx) => (idx === i ? { ...row, [field]: value } : row)));
  };
  const addRow = () => setRows((r) => [...r, { label: "", query: "" }]);
  const removeRow = (i) => setRows((r) => r.filter((_, idx) => idx !== i));

  async function handleCompare() {
    setLoading(true);
    setError(null);
    try {
      const queries = rows.map((r) => r.query).filter(Boolean);
      const data = await compareLocations(queries);
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-2xl shadow-sm p-6">
        <h2 className="font-semibold text-slate-700 mb-3">Compare locations</h2>
        <div className="space-y-2 mb-4">
          {rows.map((row, i) => (
            <div key={i} className="flex gap-2">
              <input
                value={row.label}
                onChange={(e) => updateRow(i, "label", e.target.value)}
                placeholder="Label"
                className="border rounded-lg p-2 w-28"
              />
              <input
                value={row.query}
                onChange={(e) => updateRow(i, "query", e.target.value)}
                placeholder="Address or area"
                className="border rounded-lg p-2 flex-1"
              />
              <button onClick={() => removeRow(i)} className="px-3 text-slate-400 hover:text-red-500">
                &times;
              </button>
            </div>
          ))}
        </div>
        <div className="flex gap-2">
          <button onClick={addRow} className="text-sm bg-slate-100 px-3 py-1.5 rounded-lg hover:bg-slate-200">
            + Add location
          </button>
          <button
            onClick={handleCompare}
            disabled={loading}
            className="text-sm bg-slate-800 text-white px-4 py-1.5 rounded-lg hover:bg-slate-900 disabled:opacity-50"
          >
            {loading ? "Comparing..." : "Compare"}
          </button>
        </div>
      </div>

      {error && <div className="bg-red-50 text-red-600 text-sm rounded-lg p-4">{error}</div>}

      {result && result.excluded_count > 0 && (
        <div className="bg-amber-50 text-amber-700 text-xs rounded-lg p-3">
          {result.excluded_count} of {result.results.length} location(s) have no ML forecast coverage
          {result.ranked_count < 2
            ? " — not enough locations with a forecast to pick a best option."
            : " and were left out of the ranking."}
        </div>
      )}

      {result && (
        <div className="bg-white rounded-2xl shadow-sm p-6">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-slate-400 border-b">
                <th className="pb-2">Location</th>
                <th className="pb-2">Current</th>
                <th className="pb-2">6h Forecast</th>
                <th className="pb-2">Risk</th>
              </tr>
            </thead>
            <tbody>
              {result.results.map((r, i) => (
                <tr key={i} className="border-b last:border-0">
                  <td className="py-2">
                    {rows[i]?.label || r.query}
                    {result.best?.query === r.query && (
                      <span className="ml-2 text-xs text-emerald-600 font-medium">Best</span>
                    )}
                  </td>
                  {r.mode === "forecast" ? (
                    <>
                      <td className="py-2">{r.current_pm25} µg/m³</td>
                      <td className="py-2">{r.forecast_point} µg/m³</td>
                      <td className="py-2"><RiskBadge level={r.risk_level} /></td>
                    </>
                  ) : r.mode === "live_only" ? (
                    <>
                      <td className="py-2">{r.live_data?.pm25 ?? "N/A"} µg/m³</td>
                      <td className="py-2 text-slate-400 text-xs">No forecast coverage</td>
                      <td className="py-2">
                        {r.risk_level ? <RiskBadge level={r.risk_level} /> : <span className="text-slate-400 text-xs">–</span>}
                      </td>
                    </>
                  ) : (
                    <td className="py-2 text-slate-400" colSpan={3}>
                      {r.message || r.note}
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
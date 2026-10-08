import { useState } from "react";
import ForecastPanel from "./components/ForecastPanel.jsx";
import CompareView from "./components/CompareView.jsx";
import ModelHealth from "./components/ModelHealth.jsx";

const TABS = [
  { id: "forecast", label: "Forecast" },
  { id: "compare", label: "Compare" },
  { id: "model", label: "Model Health" },
];

export default function App() {
  const [tab, setTab] = useState("forecast");

  return (
    <div className="min-h-screen bg-gradient-to-br from-sky-50 to-indigo-100">
      <div className="max-w-4xl mx-auto p-6">
        <div className="flex items-center gap-3 mb-1">
          <svg xmlns="http://www.w3.org/2000/svg" className="w-8 h-8 text-indigo-600" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <path d="M4 8h9a3 3 0 1 0-2.5-4.7" />
            <path d="M2 13h13a3 3 0 1 1-2.5 4.7" />
            <path d="M4 18h7" />
          </svg>
          <h1 className="text-2xl font-bold text-slate-800">BreatheSafe AI</h1>
        </div>
        <p className="text-slate-500 mb-6 text-sm">Multi-horizon air quality forecasting, evaluated against real baselines</p>

        <div className="flex gap-1 mb-6 bg-white/60 rounded-xl p-1 w-fit">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition ${
                tab === t.id ? "bg-white shadow text-slate-800" : "text-slate-500 hover:text-slate-700"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>

        {tab === "forecast" && <ForecastPanel />}
        {tab === "compare" && <CompareView />}
        {tab === "model" && <ModelHealth />}
      </div>
    </div>
  );
}
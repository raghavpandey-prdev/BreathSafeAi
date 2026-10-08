import { useEffect, useState } from "react";
import { getModelInfo, getModelEvaluation } from "../api.js";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from "recharts";

export default function ModelHealth() {
  const [info, setInfo] = useState(null);
  const [evaluation, setEvaluation] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([getModelInfo(), getModelEvaluation()])
      .then(([i, e]) => {
        setInfo(i);
        setEvaluation(e);
      })
      .catch((e) => setError(e.message));
  }, []);

  if (error) return <div className="bg-red-50 text-red-600 text-sm rounded-lg p-4">{error}</div>;
  if (!info || !evaluation) return <div className="text-slate-400 text-sm">Loading model info...</div>;

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-2xl shadow-sm p-6">
        <h2 className="font-semibold text-slate-700 mb-4">Model vs. baselines</h2>
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={evaluation.horizons}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="horizon_h" tickFormatter={(h) => `${h}h`} tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} label={{ value: "MAE (µg/m³)", angle: -90, position: "insideLeft", fontSize: 11 }} />
              <Tooltip />
              <Legend />
              <Bar dataKey="persistence_mae" name="Persistence" fill="#94a3b8" radius={[4, 4, 0, 0]} />
              <Bar dataKey="climatology_mae" name="Climatology" fill="#c7d2fe" radius={[4, 4, 0, 0]} />
              <Bar dataKey="model_mae" name="Model (deployed)" fill="#6366f1" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <p className="text-xs text-slate-400 mt-2">
          Evaluated on 2020 data the model never trained or calibrated on. Lower is better.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white rounded-2xl shadow-sm p-6">
          <h3 className="font-semibold text-slate-700 mb-3">Model card</h3>
          <dl className="text-sm space-y-2">
            <div className="flex justify-between"><dt className="text-slate-400">Type</dt><dd>{info.model_type}</dd></div>
            <div className="flex justify-between"><dt className="text-slate-400">Uncertainty method</dt><dd className="text-right">{info.uncertainty_method}</dd></div>
            <div className="flex justify-between"><dt className="text-slate-400">Target confidence</dt><dd>{Math.round((info.target_confidence ?? 0) * 100)}%</dd></div>
            <div className="flex justify-between"><dt className="text-slate-400">Empirical coverage</dt><dd>{Math.round((info.empirical_coverage ?? 0) * 100)}%</dd></div>
          </dl>
        </div>
        <div className="bg-white rounded-2xl shadow-sm p-6">
          <h3 className="font-semibold text-slate-700 mb-3">Top features</h3>
          <ul className="text-sm space-y-1.5">
            {Object.entries(info.top_features || {}).map(([name, value]) => (
              <li key={name} className="flex justify-between">
                <span className="text-slate-600">{name}</span>
                <span className="text-slate-400">{(value * 100).toFixed(1)}%</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
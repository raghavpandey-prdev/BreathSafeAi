import { useEffect, useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
import { getTrend } from "../api.js";

export default function TrendSparkline({ station }) {
  const [points, setPoints] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!station) return;
    setPoints(null);
    setError(null);
    getTrend(station, 168)
      .then((data) => setPoints(data.points || []))
      .catch((e) => setError(e.message));
  }, [station]);

  if (error) return null; // non-critical widget; fail quietly
  if (!points) return <div className="text-xs text-slate-400">Loading recent trend...</div>;
  if (points.length === 0) return null;

  const data = points.map((p) => ({
    time: new Date(p.timestamp).getTime(),
    pm25: p.pm25,
  }));

  const first = new Date(data[0].time);
  const last = new Date(data[data.length - 1].time);
  const fmt = (d) => d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  const daysSinceLast = (Date.now() - last.getTime()) / (1000 * 60 * 60 * 24);
  const isStale = daysSinceLast > 1;

  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-sm font-semibold text-slate-600">
          Recent actuals ({fmt(first)} – {fmt(last)})
        </h4>
        {isStale && (
          <span className="text-xs text-amber-600">
            ends {Math.round(daysSinceLast)} day(s) ago
          </span>
        )}
      </div>
      <div className="h-32 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              dataKey="time"
              type="number"
              domain={["dataMin", "dataMax"]}
              tickFormatter={(t) => new Date(t).toLocaleDateString(undefined, { month: "short", day: "numeric" })}
              tick={{ fontSize: 11, fill: "#94a3b8" }}
            />
            <YAxis tick={{ fontSize: 11, fill: "#94a3b8" }} width={32} />
            <Tooltip
              labelFormatter={(t) => new Date(t).toLocaleString()}
              formatter={(v) => [`${v} µg/m³`, "PM2.5"]}
            />
            <Line type="monotone" dataKey="pm25" stroke="#94a3b8" strokeWidth={1.5} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
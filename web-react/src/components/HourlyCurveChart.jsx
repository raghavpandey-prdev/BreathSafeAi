import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceDot,
} from "recharts";

export default function HourlyCurveChart({ curve, bestHour }) {
  if (!curve || curve.length === 0) return null;

  const data = curve.map((p) => ({ hour: p.hour_offset, pm25: p.pm25, risk: p.risk_level }));

  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 10, right: 20, left: -10, bottom: 0 }}>
          <defs>
            <linearGradient id="pmGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#6366f1" stopOpacity={0.35} />
              <stop offset="95%" stopColor="#6366f1" stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis
            dataKey="hour"
            tickFormatter={(h) => `+${h}h`}
            tick={{ fontSize: 12, fill: "#64748b" }}
          />
          <YAxis tick={{ fontSize: 12, fill: "#64748b" }} width={40} />
          <Tooltip
            formatter={(value, name, props) => [`${value} µg/m³ (${props.payload.risk})`, "PM2.5"]}
            labelFormatter={(h) => `${h} hour${h === 1 ? "" : "s"} from now`}
          />
          <Area type="monotone" dataKey="pm25" stroke="#6366f1" strokeWidth={2} fill="url(#pmGradient)" />
          {bestHour != null && (
            <ReferenceDot x={bestHour} y={data.find((d) => d.hour === bestHour)?.pm25} r={5} fill="#16a34a" stroke="white" strokeWidth={2} />
          )}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
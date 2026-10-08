import RiskBadge from "./RiskBadge.jsx";

export default function HorizonCards({ horizons }) {
  if (!horizons || horizons.length === 0) return null;

  return (
    <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
      {horizons.map((h) => (
        <div key={h.horizon_hours} className="bg-white rounded-xl border border-slate-200 p-3 text-center">
          <div className="text-xs text-slate-400 mb-1">+{h.horizon_hours}h</div>
          <div className="text-xl font-bold text-slate-800">{h.point}</div>
          <div className="text-[11px] text-slate-400 mb-2">{h.lower}–{h.upper} µg/m³</div>
          <RiskBadge level={h.risk_level} />
        </div>
      ))}
    </div>
  );
}
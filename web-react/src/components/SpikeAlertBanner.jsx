const ALERT_LEVELS = ["Poor", "Very Poor", "Severe"];

export default function SpikeAlertBanner({ horizons }) {
  if (!horizons || horizons.length === 0) return null;

  const spike = horizons.find(
    (h) => h.horizon_hours <= 12 && ALERT_LEVELS.includes(h.risk_level)
  );
  if (!spike) return null;

  return (
    <div className="bg-red-50 border border-red-200 text-red-800 rounded-xl p-4 flex items-start gap-3">
      <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5 mt-0.5 flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M12 9v4M12 17h.01M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z" />
      </svg>
      <div className="text-sm">
        <span className="font-semibold">Air quality is expected to worsen soon:</span>{" "}
        predicted to reach <span className="font-semibold">{spike.risk_level}</span> within
        {" "}{spike.horizon_hours}h ({spike.point} µg/m³). Consider planning outdoor time around this.
      </div>
    </div>
  );
}
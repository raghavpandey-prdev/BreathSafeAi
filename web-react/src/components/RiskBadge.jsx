const STYLES = {
  Good: "bg-emerald-100 text-emerald-800",
  Satisfactory: "bg-green-100 text-green-800",
  Moderate: "bg-yellow-100 text-yellow-800",
  Poor: "bg-orange-100 text-orange-800",
  "Very Poor": "bg-red-100 text-red-800",
  Severe: "bg-purple-100 text-purple-800",
};

export default function RiskBadge({ level, className = "" }) {
  const style = STYLES[level] || "bg-slate-100 text-slate-700";
  return (
    <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold ${style} ${className}`}>
      {level || "Unknown"}
    </span>
  );
}
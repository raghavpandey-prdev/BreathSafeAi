import { useState, useEffect, useRef } from "react";
import { getSuggestions } from "../api.js";

export default function LocationAutocomplete({ value, onChange, onEnter, placeholder }) {
  const [suggestions, setSuggestions] = useState([]);
  const [open, setOpen] = useState(false);
  const debounceRef = useRef(null);

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    if (!value || value.length < 3) {
      setSuggestions([]);
      return;
    }
    debounceRef.current = setTimeout(async () => {
      try {
        const data = await getSuggestions(value);
        setSuggestions(data.suggestions || []);
        setOpen(true);
      } catch {
        setSuggestions([]);
      }
    }, 450); // respects Nominatim's ~1 req/sec rate limit
    return () => clearTimeout(debounceRef.current);
  }, [value]);

  return (
    <div className="relative flex-1">
      <input
        value={value}
        onChange={(e) => { onChange(e.target.value); setOpen(true); }}
        onKeyDown={(e) => e.key === "Enter" && onEnter && onEnter()}
        onFocus={() => suggestions.length && setOpen(true)}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
        placeholder={placeholder}
        className="border rounded-lg p-2 w-full"
      />
      {open && suggestions.length > 0 && (
        <ul className="absolute z-10 bg-white border rounded-lg shadow-lg mt-1 w-full max-h-56 overflow-auto">
          {suggestions.map((s, i) => (
            <li
              key={i}
              onMouseDown={() => { onChange(s.display_name); setOpen(false); }}
              className="px-3 py-2 text-sm hover:bg-slate-50 cursor-pointer"
            >
              {s.display_name}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
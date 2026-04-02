import { useState } from "react";
import { Search, Bell, RefreshCw } from "lucide-react";
import { useStore } from "../../store/useStore";

const SUGGESTIONS = [
  "AAPL","MSFT","GOOGL","AMZN","NVDA","TSLA","META","NFLX","AMD","JPM",
  "BAC","GS","V","MA","SPY","QQQ","BRK-B","JNJ","UNH","INTC",
];

export default function Topbar({ onRefresh }) {
  const [query, setQuery]   = useState("");
  const [open, setOpen]     = useState(false);
  const setSymbol = useStore((s) => s.setActiveSymbol);
  const notifs    = useStore((s) => s.notifications);

  const filtered = query.length > 0
    ? SUGGESTIONS.filter((s) => s.includes(query.toUpperCase()))
    : [];

  function pick(sym) {
    setSymbol(sym);
    setQuery("");
    setOpen(false);
  }

  return (
    <header className="
      fixed top-0 left-60 right-0 h-14 z-30
      bg-bg-secondary/80 backdrop-blur border-b border-border
      flex items-center px-6 gap-4
    ">
      {/* Search */}
      <div className="relative flex-1 max-w-xs">
        <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted" />
        <input
          className="input-primary w-full pl-9 pr-3 h-8 text-xs"
          placeholder="Search symbol… (AAPL, TSLA…)"
          value={query}
          onChange={(e) => { setQuery(e.target.value); setOpen(true); }}
          onBlur={() => setTimeout(() => setOpen(false), 150)}
          onKeyDown={(e) => { if (e.key === "Enter" && query) pick(query.toUpperCase()); }}
        />
        {open && filtered.length > 0 && (
          <div className="absolute top-full mt-1 left-0 right-0 bg-bg-card border border-border
                          rounded-lg shadow-2xl z-50 overflow-hidden">
            {filtered.slice(0, 6).map((sym) => (
              <button
                key={sym}
                onMouseDown={() => pick(sym)}
                className="w-full text-left px-4 py-2 text-sm font-mono text-text-secondary
                           hover:bg-bg-hover hover:text-accent-cyan transition-colors"
              >
                {sym}
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="flex-1" />

      {/* Refresh */}
      {onRefresh && (
        <button onClick={onRefresh} className="btn-ghost p-2 rounded-lg">
          <RefreshCw size={14} />
        </button>
      )}

      {/* Notifications bell */}
      <button className="relative btn-ghost p-2 rounded-lg">
        <Bell size={14} />
        {notifs.length > 0 && (
          <span className="absolute top-1 right-1 w-1.5 h-1.5 rounded-full bg-accent-red" />
        )}
      </button>

      {/* Time */}
      <div className="text-text-muted font-mono text-[11px]">
        {new Date().toLocaleTimeString("en-US", { hour12: false })}
      </div>
    </header>
  );
}

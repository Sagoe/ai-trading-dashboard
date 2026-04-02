import { useEffect, useState, useRef } from "react";
import { TrendingUp, TrendingDown, Wifi } from "lucide-react";
import { getStockPrice } from "../../utils/api";
import { useStore } from "../../store/useStore";

const TICKER_SYMBOLS = [
  "AAPL","MSFT","GOOGL","AMZN","NVDA","TSLA","META","NFLX","AMD","JPM","V","GS","SPY","QQQ"
];

const REFRESH_MS = 30000; // 30 seconds

function TickerItem({ sym, onClick }) {
  const [data, setData]   = useState(null);
  const [flash, setFlash] = useState(null); // "up" | "down"
  const prevPrice         = useRef(null);

  async function load() {
    try {
      const d = await getStockPrice(sym);
      if (prevPrice.current !== null && d.price !== prevPrice.current) {
        setFlash(d.price > prevPrice.current ? "up" : "down");
        setTimeout(() => setFlash(null), 800);
      }
      prevPrice.current = d.price;
      setData(d);
    } catch (_) {}
  }

  useEffect(() => {
    load();
    const id = setInterval(load, REFRESH_MS);
    return () => clearInterval(id);
  }, [sym]);

  const up = data?.change_pct >= 0;

  return (
    <button
      onClick={() => onClick(sym)}
      className={`
        flex items-center gap-2 px-4 py-1.5 rounded-lg shrink-0 transition-all duration-300
        hover:bg-bg-hover cursor-pointer border
        ${flash === "up"   ? "border-accent-green/60 bg-accent-green/10" :
          flash === "down" ? "border-accent-red/60 bg-accent-red/10"   :
          "border-transparent"}
      `}
    >
      <span className="font-display font-bold text-xs text-text-primary">{sym}</span>
      {data ? (
        <>
          <span className="font-mono text-sm font-bold text-text-primary">
            ${data.price}
          </span>
          <span className={`flex items-center gap-0.5 font-mono text-xs font-semibold ${up ? "text-accent-green" : "text-accent-red"}`}>
            {up ? <TrendingUp size={10} /> : <TrendingDown size={10} />}
            {up ? "+" : ""}{data.change_pct}%
          </span>
        </>
      ) : (
        <span className="w-16 h-3 skeleton rounded" />
      )}
    </button>
  );
}

export default function LiveTicker() {
  const setSymbol   = useStore((s) => s.setActiveSymbol);
  const [time, setTime] = useState(new Date());
  const trackRef    = useRef(null);

  // Clock
  useEffect(() => {
    const id = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(id);
  }, []);

  // Auto-scroll
  useEffect(() => {
    const el = trackRef.current;
    if (!el) return;
    let pos = 0;
    const id = setInterval(() => {
      pos += 0.5;
      if (pos >= el.scrollWidth / 2) pos = 0;
      el.scrollLeft = pos;
    }, 30);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="
      fixed top-0 left-60 right-0 z-50
      bg-bg-primary/95 backdrop-blur border-b border-border
      flex items-center h-10 overflow-hidden gap-3
    ">
      {/* Live badge */}
      <div className="flex items-center gap-1.5 px-3 shrink-0 border-r border-border h-full">
        <div className="live-dot" />
        <span className="text-accent-green font-mono text-[10px] font-bold uppercase tracking-widest">Live</span>
      </div>

      {/* Scrolling ticker */}
      <div
        ref={trackRef}
        className="flex-1 flex items-center gap-1 overflow-x-hidden"
        style={{ scrollBehavior: "auto" }}
        onMouseEnter={() => trackRef.current && (trackRef.current._paused = true)}
        onMouseLeave={() => trackRef.current && (trackRef.current._paused = false)}
      >
        {/* Duplicate symbols for infinite scroll illusion */}
        {[...TICKER_SYMBOLS, ...TICKER_SYMBOLS].map((sym, i) => (
          <TickerItem key={`${sym}-${i}`} sym={sym} onClick={setSymbol} />
        ))}
      </div>

      {/* Clock + connection */}
      <div className="flex items-center gap-2 px-3 shrink-0 border-l border-border h-full">
        <Wifi size={11} className="text-accent-green" />
        <span className="font-mono text-[11px] text-text-muted">
          {time.toLocaleTimeString("en-US", { hour12: false })}
        </span>
      </div>
    </div>
  );
}

import { useEffect, useState, useRef } from "react";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";
import { getStockPrice } from "../../utils/api";
import { useStore } from "../../store/useStore";

const FEED_SYMBOLS = ["AAPL","MSFT","GOOGL","NVDA","TSLA","AMZN","META","AMD","JPM","V"];
const INTERVAL_MS  = 30000;

function PriceRow({ sym }) {
  const [data, setData]     = useState(null);
  const [changed, setChanged] = useState(null);
  const prevRef             = useRef(null);
  const setSymbol           = useStore((s) => s.setActiveSymbol);

  // Use a module-level ref per symbol
  useEffect(() => {
    let prev = null;
    async function load() {
      try {
        const d = await getStockPrice(sym);
        if (prev !== null && d.price !== prev) {
          setChanged(d.price > prev ? "up" : "down");
          setTimeout(() => setChanged(null), 1000);
        }
        prev = d.price;
        setData(d);
      } catch (_) {}
    }
    load();
    const id = setInterval(load, INTERVAL_MS);
    return () => clearInterval(id);
  }, [sym]);

  const up = (data?.change_pct ?? 0) >= 0;

  return (
    <div
      onClick={() => setSymbol(sym)}
      className={`
        flex items-center gap-3 px-3 py-2.5 rounded-lg cursor-pointer
        transition-all duration-300 border
        ${changed === "up"   ? "bg-accent-green/10 border-accent-green/30" :
          changed === "down" ? "bg-accent-red/10 border-accent-red/30"     :
          "border-transparent hover:bg-bg-hover hover:border-border"}
      `}
    >
      <div className="w-8 h-8 rounded-lg bg-bg-primary border border-border
                      flex items-center justify-center shrink-0">
        <span className="font-display font-bold text-[9px] text-accent-cyan">{sym.slice(0,3)}</span>
      </div>
      <div className="flex-1 min-w-0">
        <p className="font-display font-bold text-xs text-text-primary">{sym}</p>
        {data ? (
          <p className="font-mono text-[10px] text-text-muted">
            Vol {(data.volume / 1_000_000).toFixed(1)}M
          </p>
        ) : <div className="skeleton h-2 w-12 mt-1 rounded" />}
      </div>
      {data ? (
        <div className="text-right">
          <p className={`font-mono font-bold text-sm transition-colors ${
            changed === "up" ? "text-accent-green" : changed === "down" ? "text-accent-red" : "text-text-primary"
          }`}>${data.price}</p>
          <div className={`flex items-center justify-end gap-0.5 font-mono text-[10px] ${up ? "text-accent-green" : "text-accent-red"}`}>
            {up ? <TrendingUp size={9} /> : <TrendingDown size={9} />}
            {up ? "+" : ""}{data.change_pct}%
          </div>
        </div>
      ) : (
        <div className="text-right space-y-1">
          <div className="skeleton h-4 w-16 rounded" />
          <div className="skeleton h-2 w-10 rounded ml-auto" />
        </div>
      )}
    </div>
  );
}

export default function LivePriceFeed() {
  const [lastUpdate, setLastUpdate] = useState(new Date());

  useEffect(() => {
    const id = setInterval(() => setLastUpdate(new Date()), INTERVAL_MS);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="card h-full flex flex-col">
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 className="font-display font-semibold text-text-primary text-sm">Live Prices</h3>
          <p className="text-text-muted text-[10px] font-mono mt-0.5">
            Updates every 30s · click to analyse
          </p>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="live-dot" />
          <span className="text-[10px] font-mono text-text-muted">
            {lastUpdate.toLocaleTimeString("en-US", { hour12: false })}
          </span>
        </div>
      </div>
      <div className="flex-1 space-y-0.5 overflow-y-auto pr-1">
        {FEED_SYMBOLS.map((sym) => (
          <PriceRow key={sym} sym={sym} />
        ))}
      </div>
    </div>
  );
}

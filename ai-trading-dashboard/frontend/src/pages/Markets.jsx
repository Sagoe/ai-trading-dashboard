import { useState, useEffect } from "react";
import Layout from "../components/layout/Layout";
import { ChangePill, SectionHeader, ErrorBox } from "../components/ui";
import { getStockPrice } from "../utils/api";
import { useStore } from "../store/useStore";
import PriceChart from "../components/charts/PriceChart";
import { useHistory } from "../hooks/useData";

const TICKERS = ["AAPL","MSFT","GOOGL","AMZN","NVDA","TSLA","META","NFLX","AMD","JPM","BAC","V"];

function MiniTicker({ sym, onSelect, selected }) {
  const [data, setData] = useState(null);
  useEffect(() => {
    getStockPrice(sym).then(setData).catch(() => {});
  }, [sym]);

  return (
    <div
      onClick={() => onSelect(sym)}
      className={`card-hover cursor-pointer transition-all ${selected === sym ? "border-accent-cyan/50 bg-accent-cyan/5" : ""}`}
    >
      <div className="flex items-center justify-between">
        <span className="font-display font-semibold text-sm text-text-primary">{sym}</span>
        {data ? <ChangePill value={data.change_pct} /> : <div className="skeleton h-4 w-12" />}
      </div>
      {data ? (
        <div className="mt-1">
          <p className="font-mono text-text-primary text-base font-bold">${data.price}</p>
          <p className={`font-mono text-xs ${data.change >= 0 ? "text-accent-green" : "text-accent-red"}`}>
            {data.change >= 0 ? "+" : ""}{data.change}
          </p>
        </div>
      ) : (
        <div className="mt-2 space-y-1">
          <div className="skeleton h-5 w-20" />
          <div className="skeleton h-3 w-12" />
        </div>
      )}
    </div>
  );
}

export default function Markets() {
  const [selected, setSelected] = useState("AAPL");
  const { data: histData, loading } = useHistory(selected, "6mo");
  const setActiveSymbol = useStore((s) => s.setActiveSymbol);

  function pickSymbol(sym) {
    setSelected(sym);
    setActiveSymbol(sym);
  }

  return (
    <Layout>
      <div className="mb-6">
        <h1 className="font-display font-bold text-2xl text-text-primary">Markets</h1>
        <p className="text-text-muted text-sm mt-1">Live prices for major stocks</p>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-3 mb-6">
        {TICKERS.map((sym) => (
          <MiniTicker key={sym} sym={sym} onSelect={pickSymbol} selected={selected} />
        ))}
      </div>

      <div className="card">
        <SectionHeader title={`${selected} — 6 Month Chart`} sub="OHLCV with technical indicators" />
        {loading
          ? <div className="skeleton h-80 rounded-lg" />
          : histData?.data
          ? <PriceChart data={histData.data} period="6mo" showVolume />
          : <ErrorBox message="Chart data unavailable." />
        }
      </div>
    </Layout>
  );
}

import { useState, useRef, useEffect } from "react";
import {
  DollarSign, Activity, BarChart2, Cpu,
  UploadCloud, X, CheckCircle, FileText, TrendingUp
} from "lucide-react";
import Layout from "../components/layout/Layout";
import PriceChart from "../components/charts/PriceChart";
import ForecastChart from "../components/charts/ForecastChart";
import LivePriceFeed from "../components/ui/LivePriceFeed";
import {
  StatCard, SignalBadge, SentimentGauge,
  SkeletonCard, ErrorBox, SectionHeader, ChangePill
} from "../components/ui";
import { useHistory, usePrediction, useSentiment, useStockPrice } from "../hooks/useData";
import { useStore } from "../store/useStore";
import { getStockPrice } from "../utils/api";
import axios from "axios";

const REFRESH_MS  = 300000;
const BACKEND_URL = import.meta.env.PROD
  ? "https://ai-trading-dashboard-sotg.onrender.com"
  : "";

// ── Live stat card with auto-refresh ─────────────────────
function LiveStatCard({ label, icon: Icon, accent, getValue, formatValue, sub }) {
  const [val, setVal]     = useState(null);
  const [flash, setFlash] = useState(false);
  const prev              = useRef(null);

  useEffect(() => {
    async function load() {
      try {
        const v = await getValue();
        if (prev.current !== null && v !== prev.current) {
          setFlash(true);
          setTimeout(() => setFlash(false), 600);
        }
        prev.current = v;
        setVal(v);
      } catch (_) {}
    }
    load();
    const id = setInterval(load, REFRESH_MS);
    return () => clearInterval(id);
  }, []);

  const colors = {
    cyan:   "text-accent-cyan",
    green:  "text-accent-green",
    red:    "text-accent-red",
    yellow: "text-accent-yellow",
  };

  return (
    <div className={`card flex flex-col gap-2 transition-all duration-300 ${flash ? "border-accent-cyan/50 bg-accent-cyan/5" : ""}`}>
      <div className="flex items-center justify-between">
        <p className="text-text-muted text-xs font-mono uppercase tracking-widest">{label}</p>
        {Icon && (
          <div className="w-7 h-7 rounded-lg border flex items-center justify-center"
               style={{ background: "rgba(0,212,255,0.05)", borderColor: "rgba(0,212,255,0.2)" }}>
            <Icon size={13} className={colors[accent]} />
          </div>
        )}
      </div>
      {val !== null ? (
        <p className={`font-display font-bold text-2xl ${colors[accent]} transition-transform`}>
          {formatValue(val)}
        </p>
      ) : (
        <div className="skeleton h-8 w-28 rounded" />
      )}
      {sub && <p className="text-text-muted text-xs">{sub}</p>}
    </div>
  );
}

// ── Upload Panel ──────────────────────────────────────────
function UploadPanel({ onResult, onClear, hasResult }) {
  const [dragging, setDragging] = useState(false);
  const [file, setFile]         = useState(null);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState(null);
  const inputRef                = useRef();

  function pickFile(f) {
    if (!f.name.toLowerCase().endsWith(".csv")) {
      setError("Only CSV files supported."); return;
    }
    setError(null); setFile(f);
  }

  async function run() {
    if (!file) return;
    setLoading(true); setError(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const uploadUrl = `${BACKEND_URL}/upload/`;
      const res = await axios.post(uploadUrl, form, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 120000,
      });
      onResult(res.data);
    } catch (e) {
      setError(e?.response?.data?.detail || e.message || "Upload failed");
    } finally {
      setLoading(false);
    }
  }

  function clear() { setFile(null); setError(null); onClear(); }

  return (
    <div className="card border-dashed border-2 border-accent-cyan/20 hover:border-accent-cyan/40 transition-all">
      <SectionHeader title="Upload Your Data" sub="Drop a CSV for AI predictions" />
      <input ref={inputRef} type="file" accept=".csv" className="hidden"
             onChange={(e) => e.target.files[0] && pickFile(e.target.files[0])} />
      {!file ? (
        <div
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => { e.preventDefault(); setDragging(false); pickFile(e.dataTransfer.files[0]); }}
          onClick={() => inputRef.current?.click()}
          className={`flex flex-col items-center justify-center py-8 rounded-lg cursor-pointer transition-all
            ${dragging ? "bg-accent-cyan/10" : "hover:bg-bg-hover"}`}
        >
          <UploadCloud size={32} className="text-text-muted mb-2" />
          <p className="text-text-secondary text-sm">
            Drop CSV or <span className="text-accent-cyan">click to browse</span>
          </p>
          <p className="text-text-muted text-xs mt-1">Close column · 60+ rows</p>
        </div>
      ) : (
        <div className="space-y-3">
          <div className="flex items-center gap-3 bg-bg-secondary rounded-lg p-3">
            <CheckCircle size={16} className="text-accent-green shrink-0" />
            <div className="flex-1 min-w-0">
              <p className="text-text-primary text-sm font-semibold truncate">{file.name}</p>
              <p className="text-text-muted text-xs">{(file.size / 1024).toFixed(1)} KB</p>
            </div>
            <button onClick={clear} className="text-text-muted hover:text-accent-red transition-colors">
              <X size={14} />
            </button>
          </div>
          {error && (
            <p className="text-accent-red text-xs bg-accent-red/10 rounded p-2 font-mono">{error}</p>
          )}
          <div className="flex gap-2">
            <button onClick={run} disabled={loading}
                    className="btn-primary flex items-center gap-2 flex-1 justify-center">
              {loading
                ? <><div className="w-3 h-3 border-2 border-accent-cyan/30 border-t-accent-cyan rounded-full animate-spin" />Analysing…</>
                : <><TrendingUp size={13} />Run AI Prediction</>}
            </button>
            {hasResult && (
              <button onClick={clear} className="btn-ghost text-xs">Clear</button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Upload Results ────────────────────────────────────────
function UploadResults({ result }) {
  const signal = result?.signal;
  const s      = result?.summary || {};
  const fc     = result?.forecast || {};

  return (
    <div className="space-y-4 animate-fade-in-up">
      {/* Header */}
      <div className="card flex items-center gap-3 flex-wrap border-accent-green/20 bg-accent-green/5">
        <FileText size={15} className="text-accent-green shrink-0" />
        <div className="flex-1">
          <p className="font-display font-semibold text-text-primary text-sm">{result.filename}</p>
          <p className="text-text-muted text-xs font-mono">{result.rows_used} rows analysed</p>
        </div>
        {signal && <SignalBadge signal={signal.signal} confidence={signal.confidence} />}
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        {[
          { label: "Last Close",   val: `$${result.current_price}`,  accent: "cyan" },
          { label: "Min",          val: `$${s.min}`,                  accent: "yellow" },
          { label: "Max",          val: `$${s.max}`,                  accent: "yellow" },
          { label: "Average",      val: `$${s.mean}`,                 accent: "cyan" },
          { label: "Total Return",
            val: `${(s.return_total_pct ?? 0) >= 0 ? "+" : ""}${s.return_total_pct}%`,
            accent: (s.return_total_pct ?? 0) >= 0 ? "green" : "red" },
          { label: "RSI",
            val: result.rsi ?? "—",
            accent: (result.rsi ?? 50) > 70 ? "red" : (result.rsi ?? 50) < 30 ? "green" : "yellow" },
        ].map(({ label, val, accent }) => (
          <div key={label} className="card text-center py-3">
            <p className="text-text-muted text-[10px] font-mono uppercase tracking-wider mb-1">{label}</p>
            <p className={`font-display font-bold text-sm ${
              accent === "cyan"   ? "text-accent-cyan"
            : accent === "green"  ? "text-accent-green"
            : accent === "red"    ? "text-accent-red"
            : "text-accent-yellow"}`}>{val}</p>
          </div>
        ))}
      </div>

      {/* Historical chart */}
      <div className="card">
        <SectionHeader title="Historical Price (Your Data)" sub="With technical indicators" />
        <PriceChart data={result.history || []} showVolume={(result.history?.[0]?.Volume ?? 0) > 0} />
      </div>

      {/* Forecast chart */}
      <div className="card">
        <SectionHeader title="AI Forecast — Next 10 Periods" sub="Ensemble · ARIMA · SVR">
          {signal && <SignalBadge signal={signal.signal} confidence={signal.confidence} />}
        </SectionHeader>
        <ForecastChart history={result.history || []} prediction={fc} />
      </div>

      {/* Forecast table */}
      <div className="card overflow-x-auto">
        <SectionHeader title="Forecast Values" sub="Predicted closing prices" />
        <table className="w-full text-xs font-mono">
          <thead>
            <tr className="text-text-muted border-b border-border text-left">
              {["Period","Ensemble","ARIMA","SVR","vs Last Close"].map(h => (
                <th key={h} className="py-2 px-3 font-normal">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {(fc.dates || []).map((date, i) => {
              const ens  = fc.ensemble?.[i];
              const diff = ens ? ((ens - result.current_price) / result.current_price * 100) : null;
              return (
                <tr key={date} className="border-b border-border/40 hover:bg-bg-hover">
                  <td className="py-2 px-3 text-text-muted">{date}</td>
                  <td className="py-2 px-3 text-accent-green font-semibold">{ens ? `$${ens}` : "—"}</td>
                  <td className="py-2 px-3 text-accent-purple">{fc.arima?.[i] ? `$${fc.arima[i]}` : "—"}</td>
                  <td className="py-2 px-3 text-accent-red">{fc.svr?.[i] ? `$${fc.svr[i]}` : "—"}</td>
                  <td className={`py-2 px-3 font-semibold ${diff >= 0 ? "text-accent-green" : "text-accent-red"}`}>
                    {diff !== null ? `${diff >= 0 ? "+" : ""}${diff.toFixed(2)}%` : "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Model metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {["arima","svr"].map((m) => (
          <div key={m} className="card">
            <SectionHeader title={`${m.toUpperCase()} Metrics`} />
            <div className="grid grid-cols-4 gap-2">
              {["rmse","mae","r2","mape"].map((k) => (
                <div key={k} className="bg-bg-secondary rounded-lg p-2 text-center border border-border">
                  <p className="text-text-muted text-[10px] font-mono uppercase">{k}</p>
                  <p className="font-mono text-accent-cyan text-sm font-bold mt-1">
                    {result.metrics?.[m]?.[k] !== undefined
                      ? Number(result.metrics[m][k]).toFixed(4) : "—"}
                  </p>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Main Dashboard ────────────────────────────────────────
export default function Dashboard() {
  const symbol = useStore((s) => s.activeSymbol);
  const [period, setPeriod]             = useState("1y");
  const [uploadResult, setUploadResult] = useState(null);

  const { data: priceData, loading: priceLoading }                       = useStockPrice(symbol);
  const { data: histData,  loading: histLoading,  refetch: refetchHist } = useHistory(symbol, period);
  const { data: predData,  loading: predLoading }                        = usePrediction(symbol, 10);
  const { data: sentData,  loading: sentLoading }                        = useSentiment(symbol);

  const hist         = histData?.data || [];
  const pred         = predData?.forecast;
  const signal       = predData?.signal;
  const sent         = sentData;
  const forecastLast = Array.isArray(pred?.ensemble) && pred.ensemble.length > 0
    ? pred.ensemble[pred.ensemble.length - 1] : null;

  return (
    <Layout onRefresh={refetchHist}>

      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="font-display font-bold text-2xl text-text-primary">{symbol}</h1>
          <p className="text-text-muted text-sm mt-0.5">Live Market · AI Analysis</p>
        </div>
        <div className="flex items-center gap-3">
          {signal && <SignalBadge signal={signal.signal} confidence={signal.confidence} />}
          {priceData?.change_pct !== undefined && <ChangePill value={priceData.change_pct} />}
        </div>
      </div>

      {/* Live Stat Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {(priceLoading || !priceData) ? (
          [1,2,3,4].map(i => <SkeletonCard key={i} />)
        ) : (
          <>
            <LiveStatCard
              label="Current Price"
              icon={DollarSign}
              accent="cyan"
              getValue={() => getStockPrice(symbol).then(d => d.price)}
              formatValue={(v) => `$${v}`}
              sub={`Open $${priceData.open}`}
            />
            <LiveStatCard
              label="Day Change"
              icon={Activity}
              accent={priceData.change >= 0 ? "green" : "red"}
              getValue={() => getStockPrice(symbol).then(d => d.change)}
              formatValue={(v) => `${v >= 0 ? "+" : ""}$${v}`}
              sub={`${priceData.change_pct >= 0 ? "+" : ""}${priceData.change_pct}%`}
            />
            <LiveStatCard
              label="Day High / Low"
              icon={BarChart2}
              accent="yellow"
              getValue={() => getStockPrice(symbol).then(d => d.high)}
              formatValue={(v) => `$${v}`}
              sub={`Low $${priceData.low}`}
            />
            <StatCard
              label="AI Forecast"
              value={forecastLast ? `$${forecastLast}` : predLoading ? "…" : "—"}
              sub="10-day ensemble"
              icon={Cpu}
              accent="cyan"
            />
          </>
        )}
      </div>

      {/* Chart + Live Feed */}
      <div className="grid grid-cols-1 xl:grid-cols-4 gap-4 mb-4">
        <div className="xl:col-span-3 card">
          <SectionHeader title="Price & Indicators" sub={`${symbol} · ${period}`} />
          {histLoading
            ? <div className="skeleton h-72 rounded-lg" />
            : <PriceChart data={hist} period={period} onPeriodChange={setPeriod} />}
        </div>
        <div className="xl:col-span-1">
          <LivePriceFeed />
        </div>
      </div>

      {/* Sentiment + Forecast */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4 mb-4">
        <div className="card space-y-4">
          <SectionHeader title="Sentiment" sub="FinBERT analysis" />
          {sentLoading ? (
            <div className="skeleton h-48 rounded-lg" />
          ) : sent ? (
            <>
              <SentimentGauge overall={sent.overall} score={sent.score} />
              <div className="grid grid-cols-3 gap-2 text-center text-xs font-mono">
                <div className="bg-accent-green/10 rounded p-2">
                  <p className="text-accent-green font-bold">{sent.positive_count ?? 0}</p>
                  <p className="text-text-muted">Positive</p>
                </div>
                <div className="bg-accent-yellow/10 rounded p-2">
                  <p className="text-accent-yellow font-bold">{sent.neutral_count ?? 0}</p>
                  <p className="text-text-muted">Neutral</p>
                </div>
                <div className="bg-accent-red/10 rounded p-2">
                  <p className="text-accent-red font-bold">{sent.negative_count ?? 0}</p>
                  <p className="text-text-muted">Negative</p>
                </div>
              </div>
              <div className="space-y-2 max-h-40 overflow-y-auto pr-1">
                {sent.articles?.slice(0, 5).map((a, i) => (
                  <div key={i} className="border border-border rounded p-2 text-xs">
                    <p className="text-text-secondary line-clamp-2">{a.headline}</p>
                    <div className="flex items-center justify-between mt-1">
                      <span className="text-text-muted">{a.source}</span>
                      <span className={
                        a.label === "positive" ? "text-accent-green"
                      : a.label === "negative" ? "text-accent-red"
                      : "text-accent-yellow"}>{a.label}</span>
                    </div>
                  </div>
                ))}
              </div>
            </>
          ) : <ErrorBox message="Sentiment unavailable." />}
        </div>

        <div className="xl:col-span-2 card">
          <SectionHeader title="AI Forecast" sub="Ensemble · LSTM · ARIMA · SVR" />
          {predLoading
            ? <div className="skeleton h-64 rounded-lg" />
            : pred
            ? <ForecastChart history={hist} prediction={pred} />
            : <ErrorBox message="Forecast unavailable." />}
        </div>
      </div>

      {/* Upload Section */}
      <div className="border-t border-border pt-6 mt-2">
        <div className="flex items-center gap-2 mb-4">
          <UploadCloud size={18} className="text-accent-cyan" />
          <h2 className="font-display font-bold text-lg text-text-primary">Upload Your Own Data</h2>
        </div>
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-4 mb-6">
          <div className="xl:col-span-1">
            <UploadPanel
              onResult={setUploadResult}
              onClear={() => setUploadResult(null)}
              hasResult={!!uploadResult}
            />
          </div>
          {!uploadResult && (
            <div className="xl:col-span-2 card flex flex-col justify-center items-center py-12
                            text-center gap-3 border-dashed border-2 border-border">
              <BarChart2 size={36} className="text-text-muted" />
              <p className="font-display font-semibold text-text-secondary">
                Upload a CSV to see predictions here
              </p>
              <p className="text-text-muted text-sm max-w-sm">
                Export from Yahoo Finance or TradingView. Must have a{" "}
                <code className="font-mono bg-bg-primary px-1 rounded text-accent-cyan">Close</code>
                {" "}column and at least 60 rows.
              </p>
              <div className="flex flex-wrap justify-center gap-4 mt-2 text-xs font-mono text-text-muted">
                <span>✓ Historical chart</span>
                <span>✓ AI forecast</span>
                <span>✓ Buy/Sell signal</span>
                <span>✓ Model metrics</span>
              </div>
            </div>
          )}
        </div>
        {uploadResult && <UploadResults result={uploadResult} />}
      </div>

    </Layout>
  );
}

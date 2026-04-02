import { useState, useRef, useCallback } from "react";
import Layout from "../components/layout/Layout";
import ForecastChart from "../components/charts/ForecastChart";
import PriceChart from "../components/charts/PriceChart";
import { SignalBadge, SectionHeader, ErrorBox, SentimentGauge } from "../components/ui";
import { Upload, FileText, TrendingUp, BarChart2, Activity, AlertCircle, CheckCircle, X } from "lucide-react";
import axios from "axios";

function StatBox({ label, value, accent = "cyan" }) {
  const colors = {
    cyan:   "text-accent-cyan",
    green:  "text-accent-green",
    red:    "text-accent-red",
    yellow: "text-accent-yellow",
  };
  return (
    <div className="bg-bg-secondary border border-border rounded-lg p-3 text-center">
      <p className="text-text-muted text-[10px] font-mono uppercase tracking-wider mb-1">{label}</p>
      <p className={`font-display font-bold text-base ${colors[accent]}`}>{value ?? "—"}</p>
    </div>
  );
}

function MetricBox({ label, value }) {
  return (
    <div className="bg-bg-secondary border border-border rounded-lg p-3 text-center">
      <p className="text-text-muted text-[10px] font-mono uppercase tracking-wider mb-1">{label}</p>
      <p className="font-mono text-sm text-text-primary font-semibold">
        {value !== undefined && value !== null ? Number(value).toFixed(4) : "—"}
      </p>
    </div>
  );
}

export default function UploadPredict() {
  const [dragging, setDragging]   = useState(false);
  const [file, setFile]           = useState(null);
  const [loading, setLoading]     = useState(false);
  const [result, setResult]       = useState(null);
  const [error, setError]         = useState(null);
  const inputRef                  = useRef();

  // ── Drag & Drop ─────────────────────────────────────────
  const onDragOver  = (e) => { e.preventDefault(); setDragging(true);  };
  const onDragLeave = ()  => setDragging(false);
  const onDrop      = (e) => {
    e.preventDefault(); setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) pickFile(f);
  };

  function pickFile(f) {
    if (!f.name.endsWith(".csv")) {
      setError("Only CSV files are supported.");
      return;
    }
    setFile(f);
    setError(null);
    setResult(null);
  }

  async function runPrediction() {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const form = new FormData();
      form.append("file", file);
      const res = await axios.post("/upload/", form, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 120000,
      });
      setResult(res.data);
    } catch (e) {
      setError(e?.response?.data?.detail || e.message || "Upload failed");
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setFile(null); setResult(null); setError(null);
  }

  const signal = result?.signal;

  return (
    <Layout>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="font-display font-bold text-2xl text-text-primary">Upload & Predict</h1>
          <p className="text-text-muted text-sm mt-1">
            Upload your own CSV data to analyse trends and get AI forecasts
          </p>
        </div>
        {signal && <SignalBadge signal={signal.signal} confidence={signal.confidence} />}
      </div>

      {/* CSV format hint */}
      <div className="card border-accent-cyan/20 bg-accent-cyan/5 mb-4 flex gap-3 items-start">
        <AlertCircle size={15} className="text-accent-cyan mt-0.5 shrink-0" />
        <div className="text-xs font-body text-text-secondary space-y-0.5">
          <p className="font-semibold text-accent-cyan">Expected CSV format</p>
          <p>Must include a <code className="font-mono bg-bg-primary px-1 rounded">Close</code> column.
            Optional: <code className="font-mono bg-bg-primary px-1 rounded">Date, Open, High, Low, Volume</code>
          </p>
          <p className="text-text-muted">Minimum 60 rows. Works with Yahoo Finance exports, TradingView exports, or any OHLCV CSV.</p>
        </div>
      </div>

      {/* Drop zone */}
      {!result && (
        <div
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
          onClick={() => !file && inputRef.current?.click()}
          className={`
            card border-2 border-dashed transition-all duration-200 mb-4
            ${dragging
              ? "border-accent-cyan bg-accent-cyan/10 scale-[1.01]"
              : file
              ? "border-accent-green/40 bg-accent-green/5"
              : "border-border hover:border-accent-cyan/40 hover:bg-bg-hover cursor-pointer"
            }
          `}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".csv"
            className="hidden"
            onChange={(e) => e.target.files[0] && pickFile(e.target.files[0])}
          />

          <div className="flex flex-col items-center justify-center py-10 gap-3">
            {file ? (
              <>
                <CheckCircle size={36} className="text-accent-green" />
                <div className="text-center">
                  <p className="font-display font-semibold text-text-primary">{file.name}</p>
                  <p className="text-text-muted text-sm mt-1">
                    {(file.size / 1024).toFixed(1)} KB — ready to analyse
                  </p>
                </div>
                <div className="flex gap-3 mt-2">
                  <button
                    onClick={(e) => { e.stopPropagation(); runPrediction(); }}
                    disabled={loading}
                    className="btn-primary flex items-center gap-2 px-6 py-2"
                  >
                    {loading ? (
                      <>
                        <div className="w-3 h-3 border-2 border-accent-cyan/30 border-t-accent-cyan rounded-full animate-spin" />
                        Analysing…
                      </>
                    ) : (
                      <><TrendingUp size={14} /> Run AI Prediction</>
                    )}
                  </button>
                  <button
                    onClick={(e) => { e.stopPropagation(); reset(); }}
                    className="btn-ghost flex items-center gap-1"
                  >
                    <X size={13} /> Remove
                  </button>
                </div>
              </>
            ) : (
              <>
                <Upload size={36} className="text-text-muted" />
                <div className="text-center">
                  <p className="font-display font-semibold text-text-secondary">
                    Drop your CSV here or click to browse
                  </p>
                  <p className="text-text-muted text-sm mt-1">Supports any OHLCV CSV file</p>
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {/* Loading spinner */}
      {loading && (
        <div className="card flex flex-col items-center justify-center py-16 gap-4 mb-4">
          <div className="w-10 h-10 border-4 border-accent-cyan/20 border-t-accent-cyan rounded-full animate-spin" />
          <div className="text-center">
            <p className="font-display font-semibold text-text-primary">Running AI models…</p>
            <p className="text-text-muted text-sm mt-1">ARIMA · SVR · Ensemble — this may take 15–30 seconds</p>
          </div>
        </div>
      )}

      {error && <ErrorBox message={error} />}

      {/* ── RESULTS ──────────────────────────────────────── */}
      {result && (
        <div className="space-y-4 animate-fade-in-up">

          {/* File info bar */}
          <div className="card flex items-center gap-4 flex-wrap">
            <FileText size={16} className="text-accent-cyan shrink-0" />
            <div>
              <p className="font-display font-semibold text-text-primary text-sm">{result.filename}</p>
              <p className="text-text-muted text-xs font-mono">
                {result.rows_uploaded} rows uploaded · {result.rows_used} rows used after processing
              </p>
            </div>
            <div className="ml-auto flex gap-2">
              {signal && <SignalBadge signal={signal.signal} confidence={signal.confidence} />}
              <button onClick={reset} className="btn-ghost flex items-center gap-1 text-xs">
                <Upload size={12} /> New Upload
              </button>
            </div>
          </div>

          {/* Key stats */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
            <StatBox label="Current Price"    value={`$${result.current_price}`}               accent="cyan"   />
            <StatBox label="Min Price"        value={`$${result.summary.min}`}                  accent="yellow" />
            <StatBox label="Max Price"        value={`$${result.summary.max}`}                  accent="yellow" />
            <StatBox label="Mean Price"       value={`$${result.summary.mean}`}                 accent="cyan"   />
            <StatBox label="Total Return"
              value={`${result.summary.return_total_pct >= 0 ? "+" : ""}${result.summary.return_total_pct}%`}
              accent={result.summary.return_total_pct >= 0 ? "green" : "red"}
            />
            <StatBox label="RSI"
              value={result.rsi ?? "—"}
              accent={result.rsi > 70 ? "red" : result.rsi < 30 ? "green" : "yellow"}
            />
          </div>

          {/* Price + indicators chart */}
          <div className="card">
            <SectionHeader
              title="Historical Price & Indicators"
              sub={`Last ${result.history.length} data points from your file`}
            />
            <PriceChart data={result.history} showVolume={result.history[0]?.Volume > 0} />
          </div>

          {/* Forecast chart */}
          <div className="card">
            <SectionHeader
              title="AI Forecast — Next 10 Periods"
              sub="Ensemble · ARIMA · SVR based on your uploaded data"
            />
            <ForecastChart history={result.history} prediction={result.forecast} />
          </div>

          {/* Forecast table */}
          <div className="card">
            <SectionHeader title="Forecast Values" sub="Predicted closing prices per period" />
            <div className="overflow-x-auto">
              <table className="w-full text-xs font-mono">
                <thead>
                  <tr className="text-text-muted border-b border-border text-left">
                    <th className="py-2 px-3 font-normal">Period</th>
                    <th className="py-2 px-3 font-normal text-accent-green">Ensemble</th>
                    <th className="py-2 px-3 font-normal text-accent-purple">ARIMA</th>
                    <th className="py-2 px-3 font-normal text-accent-red">SVR</th>
                    <th className="py-2 px-3 font-normal">vs Current</th>
                  </tr>
                </thead>
                <tbody>
                  {result.forecast.dates.map((date, i) => {
                    const ens = result.forecast.ensemble[i];
                    const diff = ens ? ((ens - result.current_price) / result.current_price * 100) : null;
                    return (
                      <tr key={date} className="border-b border-border/40 hover:bg-bg-hover">
                        <td className="py-2 px-3 text-text-muted">{date}</td>
                        <td className="py-2 px-3 text-accent-green font-semibold">
                          {ens ? `$${ens}` : "—"}
                        </td>
                        <td className="py-2 px-3 text-accent-purple">
                          {result.forecast.arima[i] ? `$${result.forecast.arima[i]}` : "—"}
                        </td>
                        <td className="py-2 px-3 text-accent-red">
                          {result.forecast.svr[i] ? `$${result.forecast.svr[i]}` : "—"}
                        </td>
                        <td className={`py-2 px-3 ${diff >= 0 ? "text-accent-green" : "text-accent-red"}`}>
                          {diff !== null ? `${diff >= 0 ? "+" : ""}${diff.toFixed(2)}%` : "—"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Model metrics */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {["arima", "svr"].map((m) => (
              <div key={m} className="card">
                <SectionHeader title={`${m.toUpperCase()} Model Metrics`} />
                <div className="grid grid-cols-4 gap-2">
                  {["rmse","mae","r2","mape"].map((k) => (
                    <MetricBox key={k} label={k.toUpperCase()} value={result.metrics?.[m]?.[k]} />
                  ))}
                </div>
              </div>
            ))}
          </div>

        </div>
      )}
    </Layout>
  );
}

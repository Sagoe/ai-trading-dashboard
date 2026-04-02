import { useState } from "react";
import Layout from "../components/layout/Layout";
import ForecastChart from "../components/charts/ForecastChart";
import { SignalBadge, SectionHeader, ErrorBox, SkeletonCard } from "../components/ui";
import { usePrediction, useHistory } from "../hooks/useData";
import { useStore } from "../store/useStore";
import { Brain, Target, BarChart2, Activity } from "lucide-react";

const HORIZONS = [5, 10, 20, 30];
const MODELS   = ["ensemble", "lstm", "arima", "svr"];

function MetricBadge({ label, value }) {
  return (
    <div className="bg-bg-secondary rounded-lg p-3 text-center border border-border">
      <p className="text-text-muted text-[10px] font-mono uppercase tracking-wider">{label}</p>
      <p className="font-display font-bold text-accent-cyan text-base mt-1">
        {value !== undefined && value !== null ? Number(value).toFixed(4) : "—"}
      </p>
    </div>
  );
}

export default function Predictions() {
  const symbol  = useStore((s) => s.activeSymbol);
  const [horizon, setHorizon] = useState(10);
  const [model, setModel]     = useState("ensemble");

  const { data: pred, loading, error } = usePrediction(symbol, horizon, model);
  const { data: hist }                  = useHistory(symbol, "1y");

  const signal  = pred?.signal;
  const metrics = pred?.metrics;
  const forecast = pred?.forecast;

  return (
    <Layout>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="font-display font-bold text-2xl text-text-primary">Predictions</h1>
          <p className="text-text-muted text-sm mt-1">AI model forecasts for {symbol}</p>
        </div>
        {signal && <SignalBadge signal={signal.signal} confidence={signal.confidence} />}
      </div>

      {/* Controls */}
      <div className="card mb-4 flex items-center gap-6 flex-wrap">
        <div>
          <p className="text-text-muted text-[10px] font-mono uppercase tracking-wider mb-2">Horizon (days)</p>
          <div className="flex gap-2">
            {HORIZONS.map((h) => (
              <button
                key={h}
                onClick={() => setHorizon(h)}
                className={`px-3 py-1.5 rounded text-xs font-mono transition-all ${
                  horizon === h
                    ? "bg-accent-cyan/10 text-accent-cyan border border-accent-cyan/30"
                    : "text-text-muted border border-border hover:text-text-secondary"
                }`}
              >{h}d</button>
            ))}
          </div>
        </div>
        <div>
          <p className="text-text-muted text-[10px] font-mono uppercase tracking-wider mb-2">Model</p>
          <div className="flex gap-2 flex-wrap">
            {MODELS.map((m) => (
              <button
                key={m}
                onClick={() => setModel(m)}
                className={`px-3 py-1.5 rounded text-xs font-mono uppercase transition-all ${
                  model === m
                    ? "bg-accent-green/10 text-accent-green border border-accent-green/30"
                    : "text-text-muted border border-border hover:text-text-secondary"
                }`}
              >{m}</button>
            ))}
          </div>
        </div>
      </div>

      {/* Signal cards */}
      {loading ? (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
          {[1,2,3,4].map(i => <SkeletonCard key={i} />)}
        </div>
      ) : pred ? (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
          <div className="card">
            <p className="text-text-muted text-xs font-mono mb-1">Signal</p>
            <div className="mt-2">{signal && <SignalBadge signal={signal.signal} confidence={signal.confidence} />}</div>
          </div>
          <div className="card">
            <p className="text-text-muted text-xs font-mono mb-1">Current Price</p>
            <p className="font-display font-bold text-xl text-accent-cyan">${pred.current_price}</p>
          </div>
          <div className="card">
            <p className="text-text-muted text-xs font-mono mb-1">Predicted ({horizon}d)</p>
            <p className="font-display font-bold text-xl text-accent-green">
              ${forecast?.ensemble?.[forecast.ensemble.length - 1] ?? "—"}
            </p>
          </div>
          <div className="card">
            <p className="text-text-muted text-xs font-mono mb-1">RSI</p>
            <p className={`font-display font-bold text-xl ${
              pred.rsi > 70 ? "text-accent-red"
            : pred.rsi < 30 ? "text-accent-green"
            : "text-accent-yellow"}`}>
              {pred.rsi?.toFixed(1) ?? "—"}
            </p>
          </div>
        </div>
      ) : error ? (
        <ErrorBox message={error} />
      ) : null}

      {/* Forecast chart */}
      <div className="card mb-4">
        <SectionHeader title="Forecast Chart" sub={`${model.toUpperCase()} · ${horizon} days`} />
        {loading
          ? <div className="skeleton h-72 rounded-lg" />
          : forecast
          ? <ForecastChart history={hist?.data || []} prediction={forecast} />
          : <ErrorBox message="No forecast data." />}
      </div>

      {/* Model metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {["arima","svr"].map((m) => (
          <div key={m} className="card">
            <SectionHeader title={`${m.toUpperCase()} Metrics`} />
            <div className="grid grid-cols-4 gap-2">
              {["rmse","mae","r2","mape"].map((k) => (
                <MetricBadge key={k} label={k.toUpperCase()} value={metrics?.[m]?.[k]} />
              ))}
            </div>
          </div>
        ))}
      </div>
    </Layout>
  );
}

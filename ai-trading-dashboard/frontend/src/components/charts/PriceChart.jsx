import {
  ResponsiveContainer, ComposedChart, Area, Line, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ReferenceLine,
} from "recharts";
import { useState } from "react";
import clsx from "clsx";

const PERIODS = ["1mo","3mo","6mo","1y","2y","5y"];

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-bg-card border border-border rounded-lg p-3 shadow-2xl text-xs font-mono">
      <p className="text-text-muted mb-1">{label}</p>
      {payload.map((p) => (
        <div key={p.dataKey} className="flex gap-3 justify-between">
          <span style={{ color: p.color }}>{p.name}</span>
          <span className="text-text-primary">{Number(p.value).toFixed(2)}</span>
        </div>
      ))}
    </div>
  );
};

export default function PriceChart({ data = [], period, onPeriodChange, showVolume = true }) {
  const [activeLines, setActiveLines] = useState({
    Close: true, SMA_20: true, EMA_12: false, BB_Upper: false, BB_Lower: false,
  });

  const toggle = (key) => setActiveLines((p) => ({ ...p, [key]: !p[key] }));

  // Downsample for perf
  const step  = Math.max(1, Math.floor(data.length / 300));
  const chart = data.filter((_, i) => i % step === 0).map((d) => ({
    ...d,
    Date: d.Date?.slice(0, 10),
  }));

  return (
    <div className="space-y-3">
      {/* Controls */}
      <div className="flex items-center gap-2 flex-wrap">
        {PERIODS.map((p) => (
          <button
            key={p}
            onClick={() => onPeriodChange?.(p)}
            className={clsx(
              "px-2.5 py-1 rounded text-xs font-mono transition-all",
              period === p
                ? "bg-accent-cyan/10 text-accent-cyan border border-accent-cyan/30"
                : "text-text-muted hover:text-text-secondary"
            )}
          >{p}</button>
        ))}
        <div className="ml-auto flex gap-2 flex-wrap">
          {Object.keys(activeLines).map((k) => (
            <button
              key={k}
              onClick={() => toggle(k)}
              className={clsx(
                "px-2 py-0.5 rounded text-[10px] font-mono border transition-all",
                activeLines[k]
                  ? "border-accent-cyan/40 text-accent-cyan bg-accent-cyan/10"
                  : "border-border text-text-muted"
              )}
            >{k}</button>
          ))}
        </div>
      </div>

      {/* Price chart */}
      <ResponsiveContainer width="100%" height={280}>
        <ComposedChart data={chart} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
          <defs>
            <linearGradient id="closeGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor="#00d4ff" stopOpacity={0.15} />
              <stop offset="95%" stopColor="#00d4ff" stopOpacity={0}    />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e2d44" />
          <XAxis dataKey="Date" tick={{ fill: "#8899aa", fontSize: 10, fontFamily: "JetBrains Mono" }}
                 tickLine={false} axisLine={false} interval="preserveStartEnd" />
          <YAxis tick={{ fill: "#8899aa", fontSize: 10, fontFamily: "JetBrains Mono" }}
                 tickLine={false} axisLine={false} domain={["auto","auto"]}
                 tickFormatter={(v) => `$${v.toFixed(0)}`} width={55} />
          <Tooltip content={<CustomTooltip />} />

          {activeLines.BB_Upper && <Line type="monotone" dataKey="BB_Upper" stroke="#7c3aed" strokeWidth={1} dot={false} name="BB Upper" strokeDasharray="4 2" />}
          {activeLines.BB_Lower && <Line type="monotone" dataKey="BB_Lower" stroke="#7c3aed" strokeWidth={1} dot={false} name="BB Lower" strokeDasharray="4 2" />}
          {activeLines.SMA_20   && <Line type="monotone" dataKey="SMA_20"   stroke="#ffd60a" strokeWidth={1} dot={false} name="SMA20" />}
          {activeLines.EMA_12   && <Line type="monotone" dataKey="EMA_12"   stroke="#ff3b5c" strokeWidth={1} dot={false} name="EMA12" />}
          {activeLines.Close && (
            <Area type="monotone" dataKey="Close" stroke="#00d4ff" strokeWidth={2}
                  fill="url(#closeGrad)" dot={false} name="Close" />
          )}
        </ComposedChart>
      </ResponsiveContainer>

      {/* Volume chart */}
      {showVolume && (
        <ResponsiveContainer width="100%" height={70}>
          <ComposedChart data={chart} margin={{ top: 0, right: 4, bottom: 0, left: 0 }}>
            <XAxis dataKey="Date" hide />
            <YAxis hide domain={["auto","auto"]} />
            <Tooltip content={<CustomTooltip />} />
            <Bar dataKey="Volume" fill="#00d4ff" opacity={0.25} name="Volume" radius={[1,1,0,0]} />
          </ComposedChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}

import {
  ResponsiveContainer, ComposedChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, ReferenceLine,
} from "recharts";

const Tip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-bg-card border border-border rounded-lg p-3 shadow-2xl text-xs font-mono">
      <p className="text-text-muted mb-1">{label}</p>
      {payload.map((p) => (
        <div key={p.dataKey} className="flex justify-between gap-4">
          <span style={{ color: p.color }}>{p.name}</span>
          <span className="text-text-primary">${Number(p.value).toFixed(2)}</span>
        </div>
      ))}
    </div>
  );
};

export default function ForecastChart({ history = [], prediction = {} }) {
  const { dates = [], ensemble = [], lstm = [], arima = [], svr = [] } = prediction;

  // Last 60 history points
  const hist = history.slice(-60).map((d) => ({
    Date: d.Date?.slice(0, 10),
    Actual: d.Close,
  }));

  // Future points
  const future = dates.map((d, i) => ({
    Date: d,
    Ensemble: ensemble[i],
    LSTM:     lstm[i],
    ARIMA:    arima[i],
    SVR:      svr[i],
  }));

  const combined = [
    ...hist,
    // Bridge: last hist point repeated as first forecast point
    hist.length > 0
      ? { Date: hist[hist.length - 1].Date, Actual: hist[hist.length - 1].Actual,
          Ensemble: ensemble[0], LSTM: lstm[0], ARIMA: arima[0], SVR: svr[0] }
      : null,
    ...future,
  ].filter(Boolean);

  const splitDate = hist.length > 0 ? hist[hist.length - 1].Date : null;

  return (
    <ResponsiveContainer width="100%" height={300}>
      <ComposedChart data={combined} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e2d44" />
        <XAxis dataKey="Date" tick={{ fill: "#8899aa", fontSize: 10, fontFamily: "JetBrains Mono" }}
               tickLine={false} axisLine={false} interval="preserveStartEnd" />
        <YAxis tick={{ fill: "#8899aa", fontSize: 10, fontFamily: "JetBrains Mono" }}
               tickLine={false} axisLine={false} tickFormatter={(v) => `$${v.toFixed(0)}`} width={55} />
        <Tooltip content={<Tip />} />
        <Legend wrapperStyle={{ fontSize: 11, fontFamily: "JetBrains Mono", color: "#8899aa" }} />

        {splitDate && (
          <ReferenceLine x={splitDate} stroke="#1e2d44" strokeDasharray="4 2"
                         label={{ value: "Today", fill: "#8899aa", fontSize: 10 }} />
        )}

        <Line type="monotone" dataKey="Actual"   stroke="#00d4ff" strokeWidth={2} dot={false} name="Actual" />
        <Line type="monotone" dataKey="Ensemble" stroke="#00ff88" strokeWidth={2} dot={false} name="Ensemble" strokeDasharray="0" />
        <Line type="monotone" dataKey="LSTM"     stroke="#ffd60a" strokeWidth={1} dot={false} name="LSTM"    strokeDasharray="4 2" />
        <Line type="monotone" dataKey="ARIMA"    stroke="#7c3aed" strokeWidth={1} dot={false} name="ARIMA"   strokeDasharray="4 2" />
        <Line type="monotone" dataKey="SVR"      stroke="#ff3b5c" strokeWidth={1} dot={false} name="SVR"     strokeDasharray="4 2" />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

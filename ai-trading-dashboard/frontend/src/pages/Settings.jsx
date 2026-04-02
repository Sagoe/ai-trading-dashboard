import { useEffect } from "react";
import Layout from "../components/layout/Layout";
import { SectionHeader } from "../components/ui";
import { useStore } from "../store/useStore";
import { Check, Sun, Moon } from "lucide-react";
import { create } from "zustand";
import { persist } from "zustand/middleware";

// ── Persisted settings store ──────────────────────────────
export const useSettings = create(
  persist(
    (set) => ({
      darkMode:       true,
      compactLayout:  false,
      colorTheme:     "#00d4ff",
      priceAlerts:    true,
      signalAlerts:   true,
      newsAlerts:     false,
      autoRefresh:    false,
      set:    (key, val) => set({ [key]: val }),
      toggle: (key)      => set((s) => ({ [key]: !s[key] })),
    }),
    { name: "ai-trading-settings" }
  )
);

// ── Apply theme to <html> ─────────────────────────────────
function applyTheme(dark, color, compact) {
  const root = document.documentElement;

  // Dark / Light
  if (dark) {
    root.classList.add("dark");
    root.style.setProperty("--bg-primary",   "#0a0e1a");
    root.style.setProperty("--bg-secondary", "#0f1628");
    root.style.setProperty("--bg-card",      "#131c30");
    root.style.setProperty("--text-primary", "#e8f0fe");
    root.style.setProperty("--text-muted",   "#8899aa");
    root.style.setProperty("--border-color", "#1e2d44");
    document.body.style.background = "#0a0e1a";
    document.body.style.color      = "#e8f0fe";
  } else {
    root.classList.remove("dark");
    root.style.setProperty("--bg-primary",   "#f0f4f8");
    root.style.setProperty("--bg-secondary", "#ffffff");
    root.style.setProperty("--bg-card",      "#ffffff");
    root.style.setProperty("--text-primary", "#0f1628");
    root.style.setProperty("--text-muted",   "#64748b");
    root.style.setProperty("--border-color", "#e2e8f0");
    document.body.style.background = "#f0f4f8";
    document.body.style.color      = "#0f1628";
  }

  // Accent color
  root.style.setProperty("--color-accent", color);

  // Compact
  root.style.setProperty("--card-padding", compact ? "0.5rem" : "1rem");
}

// ── Toggle ────────────────────────────────────────────────
function Toggle({ checked, onChange }) {
  return (
    <button
      onClick={() => onChange(!checked)}
      className={`relative w-12 h-6 rounded-full transition-all duration-300 focus:outline-none
        ${checked ? "bg-accent-cyan" : "bg-gray-400"}`}
    >
      <span className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full shadow-md
                        transition-transform duration-300 flex items-center justify-center
                        ${checked ? "translate-x-6" : ""}`}>
        {checked
          ? <Moon size={10} className="text-gray-700" />
          : <Sun  size={10} className="text-yellow-500" />}
      </span>
    </button>
  );
}

// ── Color Swatch ──────────────────────────────────────────
function ColorSwatch({ color, active, onClick, label }) {
  return (
    <button
      onClick={onClick}
      title={label}
      className="relative w-9 h-9 rounded-full transition-all duration-150
                 hover:scale-110 active:scale-95 focus:outline-none"
      style={{ background: color, boxShadow: active ? `0 0 0 3px white, 0 0 0 5px ${color}` : "none" }}
    >
      {active && (
        <span className="absolute inset-0 flex items-center justify-center">
          <Check size={14} className="text-white drop-shadow font-bold" />
        </span>
      )}
    </button>
  );
}

// ── Row ───────────────────────────────────────────────────
function Row({ label, sub, children }) {
  return (
    <div className="flex items-center justify-between py-3.5 border-b border-border/50 last:border-0">
      <div>
        <p className="text-text-primary text-sm">{label}</p>
        {sub && <p className="text-text-muted text-xs mt-0.5">{sub}</p>}
      </div>
      <div className="ml-4 shrink-0">{children}</div>
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────
export default function Settings() {
  const s                  = useSettings();
  const watchlist          = useStore((w) => w.watchlist);
  const removeFromWatchlist = useStore((w) => w.removeFromWatchlist);

  const COLORS = [
    { hex: "#00d4ff", label: "Cyan"   },
    { hex: "#00ff88", label: "Green"  },
    { hex: "#7c3aed", label: "Purple" },
    { hex: "#ffd60a", label: "Yellow" },
    { hex: "#ff3b5c", label: "Red"    },
    { hex: "#f97316", label: "Orange" },
  ];

  // Apply theme whenever settings change
  useEffect(() => {
    applyTheme(s.darkMode, s.colorTheme, s.compactLayout);
  }, [s.darkMode, s.colorTheme, s.compactLayout]);

  // Apply on first load
  useEffect(() => {
    applyTheme(s.darkMode, s.colorTheme, s.compactLayout);
  }, []);

  function handleDarkMode(val) {
    s.set("darkMode", val);
    applyTheme(val, s.colorTheme, s.compactLayout);
  }

  function handleColor(hex) {
    s.set("colorTheme", hex);
    applyTheme(s.darkMode, hex, s.compactLayout);
  }

  function handleCompact(val) {
    s.set("compactLayout", val);
    applyTheme(s.darkMode, s.colorTheme, val);
  }

  return (
    <Layout>
      <div className="mb-6">
        <h1 className="font-display font-bold text-2xl text-text-primary">Settings</h1>
        <p className="text-text-muted text-sm mt-1">Preferences and configuration</p>
      </div>

      <div className="max-w-2xl space-y-4">

        {/* Appearance */}
        <div className="card">
          <SectionHeader title="Appearance" />

          <Row
            label={s.darkMode ? "Dark Mode" : "Light Mode"}
            sub={s.darkMode ? "Professional dark trading theme" : "Clean light theme"}
          >
            <div className="flex items-center gap-2">
              <Sun size={14} className={s.darkMode ? "text-text-muted" : "text-yellow-500"} />
              <Toggle checked={s.darkMode} onChange={handleDarkMode} />
              <Moon size={14} className={s.darkMode ? "text-accent-cyan" : "text-text-muted"} />
            </div>
          </Row>

          <Row label="Compact Layout" sub="Reduce card padding and spacing">
            <Toggle checked={s.compactLayout} onChange={handleCompact} />
          </Row>

          <Row label="Color Theme" sub="Accent colour across the dashboard">
            <div className="flex gap-2">
              {COLORS.map(({ hex, label }) => (
                <ColorSwatch
                  key={hex}
                  color={hex}
                  label={label}
                  active={s.colorTheme === hex}
                  onClick={() => handleColor(hex)}
                />
              ))}
            </div>
          </Row>
        </div>

        {/* Notifications */}
        <div className="card">
          <SectionHeader title="Notifications" />
          <Row label="Price Alerts" sub="Notify on significant price moves">
            <Toggle checked={s.priceAlerts} onChange={(v) => s.set("priceAlerts", v)} />
          </Row>
          <Row label="Signal Alerts" sub="BUY / SELL / HOLD changes">
            <Toggle checked={s.signalAlerts} onChange={(v) => s.set("signalAlerts", v)} />
          </Row>
          <Row label="News Alerts" sub="Major sentiment shifts">
            <Toggle checked={s.newsAlerts} onChange={(v) => s.set("newsAlerts", v)} />
          </Row>
        </div>

        {/* Data */}
        <div className="card">
          <SectionHeader title="Data & API" />
          <Row label="Auto-refresh" sub="Refresh live prices every 30s">
            <Toggle checked={s.autoRefresh} onChange={(v) => s.set("autoRefresh", v)} />
          </Row>
          <Row label="Data Source" sub="Live market data provider">
            <span className="font-mono text-xs text-accent-cyan bg-accent-cyan/10
                             border border-accent-cyan/20 px-2 py-0.5 rounded">YFinance</span>
          </Row>
          <Row label="Sentiment Model" sub="NLP model for news analysis">
            <span className="font-mono text-xs text-accent-purple bg-accent-purple/10
                             border border-accent-purple/20 px-2 py-0.5 rounded">FinBERT</span>
          </Row>
        </div>

        {/* Active config summary */}
        <div className="card">
          <SectionHeader title="Active Configuration" />
          <div className="grid grid-cols-2 gap-2 text-xs font-mono">
            {[
              ["Theme",          s.darkMode      ? "🌙 Dark"  : "☀️ Light"],
              ["Compact Layout", s.compactLayout ? "✓ On"     : "✗ Off"  ],
              ["Accent Color",   s.colorTheme                             ],
              ["Price Alerts",   s.priceAlerts   ? "✓ On"     : "✗ Off"  ],
              ["Signal Alerts",  s.signalAlerts  ? "✓ On"     : "✗ Off"  ],
              ["News Alerts",    s.newsAlerts    ? "✓ On"     : "✗ Off"  ],
              ["Auto Refresh",   s.autoRefresh   ? "✓ On"     : "✗ Off"  ],
            ].map(([k, v]) => (
              <div key={k} className="flex justify-between items-center
                                      bg-bg-secondary rounded px-3 py-2 border border-border">
                <span className="text-text-muted">{k}</span>
                <span className="flex items-center gap-1">
                  {k === "Accent Color" && (
                    <span className="w-3 h-3 rounded-full inline-block"
                          style={{ background: v }} />
                  )}
                  <span className={
                    String(v).startsWith("✓") ? "text-accent-green"
                  : String(v).startsWith("✗") ? "text-text-muted"
                  : "text-text-primary"
                  }>{v}</span>
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Watchlist */}
        <div className="card">
          <SectionHeader title="Watchlist" sub={`${watchlist.length} symbols`} />
          <div className="flex flex-wrap gap-2">
            {watchlist.map((sym) => (
              <div key={sym} className="flex items-center gap-1 bg-bg-secondary border border-border
                                        rounded px-2.5 py-1 text-xs font-mono group">
                <span className="text-accent-cyan">{sym}</span>
                <button onClick={() => removeFromWatchlist(sym)}
                        className="text-text-muted hover:text-accent-red ml-1 transition-colors">×</button>
              </div>
            ))}
            {watchlist.length === 0 && (
              <p className="text-text-muted text-xs font-mono">No symbols in watchlist.</p>
            )}
          </div>
        </div>

        {/* About */}
        <div className="card border-border/50">
          <SectionHeader title="About" />
          <div className="text-text-muted text-xs font-mono space-y-1">
            <p>AI Trading Dashboard · v1.0.0</p>
            <p>Models: LSTM · ARIMA · SVR · Ensemble</p>
            <p>Sentiment: FinBERT (ProsusAI)</p>
            <p>Data: Yahoo Finance · NewsAPI</p>
            <p className="pt-2 border-t border-border mt-2 text-text-muted/50">
              ⚠ For educational purposes only. Not financial advice.
            </p>
          </div>
        </div>

      </div>
    </Layout>
  );
}

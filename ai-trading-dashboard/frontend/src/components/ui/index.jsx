import clsx from "clsx";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";

// ── Stat Card ─────────────────────────────────────────────
export function StatCard({ label, value, sub, trend, icon: Icon, accent = "cyan" }) {
  const colors = {
    cyan:   "text-accent-cyan border-accent-cyan/20 bg-accent-cyan/5",
    green:  "text-accent-green border-accent-green/20 bg-accent-green/5",
    red:    "text-accent-red border-accent-red/20 bg-accent-red/5",
    yellow: "text-accent-yellow border-accent-yellow/20 bg-accent-yellow/5",
    purple: "text-accent-purple border-accent-purple/20 bg-accent-purple/5",
  };
  const safeAccent = colors[accent] ? accent : "cyan";
  return (
    <div className="card flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <p className="text-text-muted text-xs font-mono uppercase tracking-widest">{label}</p>
        {Icon && (
          <div className={clsx("w-7 h-7 rounded-lg border flex items-center justify-center", colors[safeAccent])}>
            <Icon size={13} />
          </div>
        )}
      </div>
      <p className={clsx("font-display font-bold text-2xl", colors[safeAccent].split(" ")[0])}>
        {value}
      </p>
      {sub && (
        <p className="text-text-muted text-xs font-body">{sub}</p>
      )}
      {trend !== undefined && (
        <div className={trend >= 0 ? "stat-up" : "stat-down"}>
          {trend >= 0 ? "+" : ""}{trend}%
        </div>
      )}
    </div>
  );
}

// ── Signal Badge ──────────────────────────────────────────
export function SignalBadge({ signal, confidence }) {
  const map = {
    BUY:  { cls: "badge-buy",  Icon: TrendingUp   },
    SELL: { cls: "badge-sell", Icon: TrendingDown  },
    HOLD: { cls: "badge-hold", Icon: Minus         },
  };
  const { cls, Icon } = map[signal] || map.HOLD;
  return (
    <span className={cls}>
      <Icon size={11} />
      {signal}
      {confidence && <span className="opacity-70">· {confidence}%</span>}
    </span>
  );
}

// ── Change Pill ───────────────────────────────────────────
export function ChangePill({ value }) {
  const pos = value >= 0;
  return (
    <span className={clsx(
      "inline-flex items-center gap-0.5 font-mono text-xs px-1.5 py-0.5 rounded",
      pos ? "bg-accent-green/10 text-accent-green" : "bg-accent-red/10 text-accent-red"
    )}>
      {pos ? <TrendingUp size={10} /> : <TrendingDown size={10} />}
      {pos ? "+" : ""}{value?.toFixed(2)}%
    </span>
  );
}

// ── Skeleton ──────────────────────────────────────────────
export function Skeleton({ className }) {
  return <div className={clsx("skeleton", className)} />;
}

export function SkeletonCard() {
  return (
    <div className="card space-y-3">
      <Skeleton className="h-3 w-24" />
      <Skeleton className="h-8 w-32" />
      <Skeleton className="h-3 w-16" />
    </div>
  );
}

// ── Section Header ────────────────────────────────────────
export function SectionHeader({ title, sub, children }) {
  return (
    <div className="flex items-center justify-between mb-4">
      <div>
        <h2 className="font-display font-semibold text-text-primary text-base">{title}</h2>
        {sub && <p className="text-text-muted text-xs mt-0.5">{sub}</p>}
      </div>
      {children}
    </div>
  );
}

// ── Error Box ─────────────────────────────────────────────
export function ErrorBox({ message }) {
  return (
    <div className="card border-accent-red/20 bg-accent-red/5 text-accent-red text-sm font-body p-4">
      ⚠ {message}
    </div>
  );
}

// ── Sentiment Indicator ───────────────────────────────────
export function SentimentGauge({ overall, score }) {
  const color = overall === "positive"
    ? "text-accent-green" : overall === "negative"
    ? "text-accent-red" : "text-accent-yellow";
  const pct = Math.round((score + 1) / 2 * 100);
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className={clsx("font-display font-semibold uppercase text-sm", color)}>
          {overall}
        </span>
        <span className="font-mono text-xs text-text-muted">{score?.toFixed(3)}</span>
      </div>
      <div className="h-1.5 w-full bg-bg-primary rounded-full overflow-hidden">
        <div
          className={clsx("h-full rounded-full transition-all duration-700",
            overall === "positive" ? "bg-accent-green"
          : overall === "negative" ? "bg-accent-red"
          : "bg-accent-yellow")}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

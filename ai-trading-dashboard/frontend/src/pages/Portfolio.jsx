import { useState } from "react";
import Layout from "../components/layout/Layout";
import { SectionHeader, ErrorBox, ChangePill, SignalBadge } from "../components/ui";
import { usePortfolio } from "../hooks/useData";
import { addPosition, removePosition } from "../utils/api";
import { PlusCircle, Trash2, TrendingUp, TrendingDown, DollarSign } from "lucide-react";
import {
  ResponsiveContainer, PieChart, Pie, Cell, Tooltip, BarChart, Bar, XAxis, YAxis, CartesianGrid
} from "recharts";

const COLORS = ["#00d4ff","#00ff88","#ffd60a","#7c3aed","#ff3b5c","#f97316","#06b6d4","#8b5cf6"];

export default function Portfolio() {
  const { data, loading, error, refetch } = usePortfolio();
  const [form, setForm] = useState({ symbol: "", shares: "", avg_cost: "" });
  const [adding, setAdding] = useState(false);
  const [formOpen, setFormOpen] = useState(false);

  async function handleAdd() {
    if (!form.symbol || !form.shares || !form.avg_cost) return;
    setAdding(true);
    try {
      await addPosition({ symbol: form.symbol, shares: +form.shares, avg_cost: +form.avg_cost });
      setForm({ symbol: "", shares: "", avg_cost: "" });
      setFormOpen(false);
      refetch();
    } catch (e) { alert(e); }
    finally { setAdding(false); }
  }

  async function handleRemove(sym) {
    if (!confirm(`Remove ${sym}?`)) return;
    await removePosition(sym);
    refetch();
  }

  const holdings = data?.holdings || [];
  const pieData  = holdings.map((h) => ({ name: h.symbol, value: h.value }));
  const barData  = holdings.map((h) => ({
    symbol: h.symbol, pl: h.profit_loss, fill: h.profit_loss >= 0 ? "#00ff88" : "#ff3b5c"
  }));

  return (
    <Layout onRefresh={refetch}>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="font-display font-bold text-2xl text-text-primary">Portfolio</h1>
          <p className="text-text-muted text-sm mt-1">Track your positions and P&L</p>
        </div>
        <button onClick={() => setFormOpen(!formOpen)} className="btn-primary flex items-center gap-2">
          <PlusCircle size={14} /> Add Position
        </button>
      </div>

      {/* Add form */}
      {formOpen && (
        <div className="card mb-4 border-accent-cyan/20">
          <SectionHeader title="Add Position" />
          <div className="flex gap-3 flex-wrap">
            <input className="input-primary w-28" placeholder="Symbol" value={form.symbol}
                   onChange={(e) => setForm({ ...form, symbol: e.target.value.toUpperCase() })} />
            <input className="input-primary w-28" placeholder="Shares" type="number" value={form.shares}
                   onChange={(e) => setForm({ ...form, shares: e.target.value })} />
            <input className="input-primary w-36" placeholder="Avg Cost ($)" type="number" value={form.avg_cost}
                   onChange={(e) => setForm({ ...form, avg_cost: e.target.value })} />
            <button onClick={handleAdd} disabled={adding} className="btn-primary">
              {adding ? "Adding…" : "Add"}
            </button>
          </div>
        </div>
      )}

      {/* Summary cards */}
      {data && data.total_value !== undefined && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          {[
            { label: "Total Value",   val: `$${(data.total_value || 0).toLocaleString()}`, accent: "cyan"  },
            { label: "Cost Basis",    val: `$${(data.total_cost  || 0).toLocaleString()}`, accent: "yellow"},
            { label: "Total P&L",     val: `${(data.total_pl || 0) >= 0 ? "+" : ""}$${(data.total_pl || 0).toLocaleString()}`,
              accent: (data.total_pl || 0) >= 0 ? "green" : "red" },
            { label: "Return",        val: `${(data.total_pl_pct || 0) >= 0 ? "+" : ""}${data.total_pl_pct || 0}%`,
              accent: (data.total_pl_pct || 0) >= 0 ? "green" : "red" },
          ].map(({ label, val, accent }) => (
            <div key={label} className="card">
              <p className="text-text-muted text-xs font-mono mb-1">{label}</p>
              <p className={`font-display font-bold text-xl ${
                accent === "cyan"   ? "text-accent-cyan"
              : accent === "green"  ? "text-accent-green"
              : accent === "red"    ? "text-accent-red"
              : "text-accent-yellow"}`}>{val}</p>
            </div>
          ))}
        </div>
      )}

      {loading && <div className="skeleton h-32 rounded-xl mb-4" />}
      {error   && <ErrorBox message={error} />}

      {/* Charts row */}
      {holdings.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
          {/* Pie */}
          <div className="card">
            <SectionHeader title="Allocation" />
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={pieData} cx="50%" cy="50%" outerRadius={80} dataKey="value"
                     label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                     labelLine={{ stroke: "#8899aa" }}>
                  {pieData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip formatter={(v) => `$${Number(v).toFixed(2)}`}
                         contentStyle={{ background: "#131c30", border: "1px solid #1e2d44", borderRadius: 8, fontSize: 12 }} />
              </PieChart>
            </ResponsiveContainer>
          </div>

          {/* P&L bar */}
          <div className="card">
            <SectionHeader title="Profit / Loss by Position" />
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={barData} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e2d44" />
                <XAxis dataKey="symbol" tick={{ fill: "#8899aa", fontSize: 10, fontFamily: "JetBrains Mono" }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: "#8899aa", fontSize: 10, fontFamily: "JetBrains Mono" }} axisLine={false} tickLine={false}
                       tickFormatter={(v) => `$${v.toFixed(0)}`} />
                <Tooltip formatter={(v) => `$${Number(v).toFixed(2)}`}
                         contentStyle={{ background: "#131c30", border: "1px solid #1e2d44", borderRadius: 8, fontSize: 12 }} />
                <Bar dataKey="pl" name="P&L" radius={[3,3,0,0]}>
                  {barData.map((entry, i) => (
                    <Cell key={i} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Holdings table */}
      {holdings.length > 0 && (
        <div className="card overflow-x-auto">
          <SectionHeader title="Holdings" sub={`${holdings.length} positions`} />
          <table className="w-full text-sm font-mono">
            <thead>
              <tr className="text-text-muted text-[11px] uppercase tracking-wider border-b border-border">
                {["Symbol","Shares","Avg Cost","Price","Value","P&L","P&L %","Change",""].map((h) => (
                  <th key={h} className="text-left py-2 px-2 font-normal">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {holdings.map((h) => (
                <tr key={h.symbol} className="border-b border-border/50 hover:bg-bg-hover transition-colors">
                  <td className="py-3 px-2 font-display font-semibold text-accent-cyan">{h.symbol}</td>
                  <td className="py-3 px-2 text-text-secondary">{h.shares}</td>
                  <td className="py-3 px-2 text-text-secondary">${h.avg_cost}</td>
                  <td className="py-3 px-2 text-text-primary">${h.current_price}</td>
                  <td className="py-3 px-2 text-text-primary">${h.value.toLocaleString()}</td>
                  <td className={`py-3 px-2 ${h.profit_loss >= 0 ? "text-accent-green" : "text-accent-red"}`}>
                    {h.profit_loss >= 0 ? "+" : ""}${h.profit_loss}
                  </td>
                  <td className="py-3 px-2"><ChangePill value={h.profit_loss_pct} /></td>
                  <td className="py-3 px-2"><ChangePill value={h.change_pct} /></td>
                  <td className="py-3 px-2">
                    <button onClick={() => handleRemove(h.symbol)}
                            className="text-text-muted hover:text-accent-red transition-colors">
                      <Trash2 size={13} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {!loading && holdings.length === 0 && (
        <div className="card text-center py-16 text-text-muted">
          <p className="font-display text-lg mb-2">No positions yet</p>
          <p className="text-sm">Click "Add Position" to track your first stock.</p>
        </div>
      )}
    </Layout>
  );
}

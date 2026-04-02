import { NavLink } from "react-router-dom";
import {
  LayoutDashboard, TrendingUp, Brain, Briefcase,
  Settings, Zap, ChevronRight, UploadCloud,
} from "lucide-react";
import { useStore } from "../../store/useStore";
import clsx from "clsx";

const NAV = [
  { to: "/",            icon: LayoutDashboard, label: "Dashboard"   },
  { to: "/markets",     icon: TrendingUp,      label: "Markets"     },
  { to: "/predictions", icon: Brain,           label: "Predictions" },
  { to: "/portfolio",   icon: Briefcase,       label: "Portfolio"   },
  { to: "/upload",      icon: UploadCloud,     label: "Upload & AI" },
  { to: "/settings",    icon: Settings,        label: "Settings"    },
];

export default function Sidebar() {
  const activeSymbol = useStore((s) => s.activeSymbol);

  return (
    <aside className="
      fixed left-0 top-0 h-screen w-60 z-40
      bg-bg-secondary border-r border-border
      flex flex-col
    ">
      {/* Logo */}
      <div className="px-5 py-5 border-b border-border">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-accent-cyan/10 border border-accent-cyan/30
                          flex items-center justify-center">
            <Zap size={16} className="text-accent-cyan" />
          </div>
          <div>
            <p className="font-display font-bold text-sm text-text-primary leading-none">
              AlphaAI
            </p>
            <p className="text-text-muted text-[10px] font-mono mt-0.5">TRADING SYSTEM</p>
          </div>
        </div>
      </div>

      {/* Active symbol pill */}
      <div className="px-4 py-3 border-b border-border">
        <p className="text-text-muted text-[10px] font-mono uppercase tracking-widest mb-1">
          Active Symbol
        </p>
        <div className="flex items-center gap-2">
          <div className="live-dot" />
          <span className="font-display font-semibold text-accent-cyan text-sm">
            {activeSymbol}
          </span>
        </div>
      </div>

      {/* Nav links */}
      <nav className="flex-1 px-3 py-4 space-y-0.5">
        {NAV.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/"}
            className={({ isActive }) =>
              clsx(
                "flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all duration-150 group",
                isActive
                  ? "bg-accent-cyan/10 text-accent-cyan border border-accent-cyan/20"
                  : "text-text-secondary hover:text-text-primary hover:bg-bg-hover"
              )
            }
          >
            {({ isActive }) => (
              <>
                <Icon size={16} className={isActive ? "text-accent-cyan" : "text-text-muted group-hover:text-text-secondary"} />
                <span className="font-body">{label}</span>
                {isActive && <ChevronRight size={12} className="ml-auto text-accent-cyan/60" />}
              </>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div className="px-4 py-3 border-t border-border">
        <p className="text-text-muted text-[10px] font-mono text-center">
          v1.0.0 • AI Trading Dashboard
        </p>
      </div>
    </aside>
  );
}

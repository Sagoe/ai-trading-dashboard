import { create } from "zustand";

export const useStore = create((set, get) => ({
  // Active symbol
  activeSymbol: "AAPL",
  setActiveSymbol: (sym) => set({ activeSymbol: sym.toUpperCase() }),

  // Price cache  { SYM: { price, change, change_pct, ... } }
  prices: {},
  setPrice: (sym, data) =>
    set((s) => ({ prices: { ...s.prices, [sym]: data } })),

  // Theme
  theme: "dark",
  toggleTheme: () =>
    set((s) => ({ theme: s.theme === "dark" ? "light" : "dark" })),

  // Watchlist
  watchlist: ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"],
  addToWatchlist: (sym) =>
    set((s) =>
      s.watchlist.includes(sym)
        ? s
        : { watchlist: [...s.watchlist, sym] }
    ),
  removeFromWatchlist: (sym) =>
    set((s) => ({ watchlist: s.watchlist.filter((w) => w !== sym) })),

  // Notifications
  notifications: [],
  addNotification: (msg, type = "info") =>
    set((s) => ({
      notifications: [
        ...s.notifications,
        { id: Date.now(), msg, type, ts: new Date() },
      ].slice(-20),
    })),
  clearNotifications: () => set({ notifications: [] }),
}));

/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        bg: {
          primary:   "#0a0e1a",
          secondary: "#0f1628",
          card:      "#131c30",
          hover:     "#1a2540",
        },
        accent: {
          cyan:   "#00d4ff",
          green:  "#00ff88",
          red:    "#ff3b5c",
          yellow: "#ffd60a",
          purple: "#7c3aed",
        },
        text: {
          primary:   "#e8f0fe",
          secondary: "#8899aa",
          muted:     "#4a5568",
        },
        border: "#1e2d44",
      },
      fontFamily: {
        display: ["'Space Grotesk'", "sans-serif"],
        mono:    ["'JetBrains Mono'", "monospace"],
        body:    ["'DM Sans'", "sans-serif"],
      },
      backgroundImage: {
        "grid-pattern": "linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px)",
      },
      backgroundSize: {
        "grid": "40px 40px",
      },
    },
  },
  plugins: [],
};

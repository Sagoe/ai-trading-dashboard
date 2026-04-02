import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: "/ai-trading-dashboard/",
  build: {
    outDir: "dist",
  },
  server: {
    port: 5173,
    proxy: {
      "/stocks":    { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/history":   { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/predict":   { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/sentiment": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/portfolio": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/upload":    { target: "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
});

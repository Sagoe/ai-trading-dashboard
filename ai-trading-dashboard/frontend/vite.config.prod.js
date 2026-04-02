import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// IMPORTANT: Replace YOUR_GITHUB_USERNAME with your actual GitHub username
// IMPORTANT: Replace YOUR_RENDER_BACKEND_URL with your Render backend URL
//            e.g. https://ai-trading-backend.onrender.com

const BACKEND_URL = "https://YOUR_RENDER_BACKEND_URL.onrender.com";
const REPO_NAME   = "ai-trading-dashboard"; // your GitHub repo name

export default defineConfig({
  plugins: [react()],
  base: `/${REPO_NAME}/`,   // required for GitHub Pages
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
  build: {
    outDir: "dist",
    define: {
      __BACKEND_URL__: JSON.stringify(BACKEND_URL),
    },
  },
});

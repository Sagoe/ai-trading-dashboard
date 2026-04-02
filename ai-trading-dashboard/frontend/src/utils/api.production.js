import axios from "axios";

// Production: points to Render backend
// Development: Vite proxy routes to localhost:8000
const BACKEND = import.meta.env.PROD
  ? "https://ai-trading-dashboard-sotg.onrender.com"
  : "";

const api = axios.create({
  baseURL: BACKEND,
  timeout: 30000,
});

api.interceptors.response.use(
  (r) => r.data,
  (e) => Promise.reject(e?.response?.data?.detail || e.message || "API error")
);

export const getStockList      = ()      => api.get("/stocks/");
export const getStockInfo      = (sym)   => api.get(`/stocks/${sym}/info`);
export const getStockPrice     = (sym)   => api.get(`/stocks/${sym}/price`);
export const getMarketOverview = ()      => api.get("/stocks/market/overview");

export const getHistory = (sym, period = "1y", indicators = true) =>
  api.get(`/history/${sym}`, { params: { period, indicators } });

export const getPrediction = (sym, horizon = 10, model = "ensemble") =>
  api.get(`/predict/${sym}`, { params: { horizon, model } });

export const getSentiment = (sym) => api.get(`/sentiment/${sym}`);

export const getPortfolio   = ()     => api.get("/portfolio/");
export const addPosition    = (data) => api.post("/portfolio/add", data);
export const removePosition = (sym)  => api.delete(`/portfolio/${sym}`);

export const uploadCSV = (file) => {
  const form = new FormData();
  form.append("file", file);
  return api.post("/upload/", form, {
    headers: { "Content-Type": "multipart/form-data" },
    timeout: 120000,
  });
};

export default api;

# 🤖 AI Trading Dashboard

A full-stack AI-powered stock prediction and trading dashboard built with **React + Vite** (frontend) and **FastAPI + PyTorch/TensorFlow** (backend).

---

## 🗂 Project Structure

```
ai-trading-dashboard/
├── backend/                  ← FastAPI server (open in PyCharm)
│   ├── main.py               ← App entry point
│   ├── requirements.txt      ← Python dependencies
│   ├── .env                  ← API keys (copy from .env.example)
│   ├── routers/
│   │   ├── stocks.py         ← GET /stocks/*
│   │   ├── history.py        ← GET /history/{symbol}
│   │   ├── predict.py        ← GET /predict/{symbol}
│   │   ├── sentiment.py      ← GET /sentiment/{symbol}
│   │   └── portfolio.py      ← GET/POST/DELETE /portfolio/*
│   └── services/
│       ├── data_service.py      ← Yahoo Finance + indicators
│       ├── prediction_service.py← LSTM / ARIMA / SVR / Ensemble
│       └── sentiment_service.py ← FinBERT + NewsAPI
│
├── frontend/                 ← React + Vite (open in VS Code)
│   ├── index.html
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── src/
│       ├── App.jsx
│       ├── main.jsx
│       ├── index.css
│       ├── utils/api.js
│       ├── store/useStore.js
│       ├── hooks/useData.js
│       ├── components/
│       │   ├── layout/      ← Sidebar, Topbar, Layout
│       │   ├── charts/      ← PriceChart, ForecastChart
│       │   └── ui/          ← StatCard, SignalBadge, etc.
│       └── pages/
│           ├── Dashboard.jsx
│           ├── Markets.jsx
│           ├── Predictions.jsx
│           ├── Portfolio.jsx
│           └── Settings.jsx
│
└── ml/
    └── train.py              ← Pre-train models (run in PyCharm)
```

---

## ⚡ Quick Start

### 1. Backend Setup (PyCharm)

```bash
# 1. Open the `backend/` folder in PyCharm
# 2. Create a virtual environment (Python 3.10+)
cd backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy and edit .env
cp .env .env.local
# Add your NEWS_API_KEY (free at https://newsapi.org)

# 5. Start the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**PyCharm tip:** Go to `Run → Edit Configurations → Add → Python` and set:
- Script: `main.py`
- Parameters: *(blank)*
- Working dir: `backend/`
- Then right-click `main.py` → Run with uvicorn, or use the terminal.

Or use the PyCharm built-in terminal:
```bash
uvicorn main:app --reload
```

API docs available at: **http://localhost:8000/docs**

---

### 2. Pre-train Models (optional but faster predictions)

Run this in PyCharm terminal from the **project root**:
```bash
cd backend
python ../ml/train.py --symbols AAPL MSFT GOOGL NVDA TSLA --epochs 50
```

This will save trained models to `backend/ml/saved_models/`.

---

### 3. Frontend Setup (VS Code)

```bash
# 1. Open the `frontend/` folder in VS Code
cd frontend

# 2. Install Node.js dependencies (Node 18+ required)
npm install

# 3. Start development server
npm run dev
```

App will be running at: **http://localhost:5173**

**VS Code tip:** Install these extensions:
- ESLint
- Prettier
- Tailwind CSS IntelliSense
- ES7+ React/Redux/React-Native snippets

---

## 🔑 API Keys

| Service    | Key Name       | Where to get                          | Required? |
|------------|----------------|---------------------------------------|-----------|
| NewsAPI    | `NEWS_API_KEY` | https://newsapi.org/register          | Optional  |
| Yahoo Finance | (none)      | Free, built-in via `yfinance`         | ✅ Free   |

If `NEWS_API_KEY` is not set, the app uses mock news articles for sentiment demo.

---

## 📡 API Endpoints

| Method | Endpoint                         | Description                        |
|--------|----------------------------------|------------------------------------|
| GET    | `/stocks/`                       | List all available symbols         |
| GET    | `/stocks/{symbol}/price`         | Current price + change             |
| GET    | `/stocks/{symbol}/info`          | Company metadata                   |
| GET    | `/stocks/market/overview`        | SPY, QQQ, DIA, IWM indices        |
| GET    | `/history/{symbol}?period=1y`   | OHLCV + all technical indicators   |
| GET    | `/predict/{symbol}?horizon=10`  | AI forecast + signal + metrics     |
| GET    | `/sentiment/{symbol}`            | FinBERT sentiment analysis         |
| GET    | `/portfolio/`                    | Get all holdings with P&L          |
| POST   | `/portfolio/add`                 | Add a position                     |
| DELETE | `/portfolio/{symbol}`            | Remove a position                  |

---

## 🧠 AI Models

| Model    | Type        | Library          | Notes                          |
|----------|-------------|------------------|--------------------------------|
| LSTM     | Deep Learning | TensorFlow/Keras | 2-layer with BatchNorm + Dropout |
| ARIMA    | Statistical | statsmodels      | Order (5,1,0) auto-fitted     |
| SVR      | ML          | scikit-learn     | RBF kernel, lag features       |
| Ensemble | Combined    | —                | Weighted avg: 60/20/20         |

**Metrics tracked:** RMSE · MAE · MAPE · R²

---

## 📊 Technical Indicators

RSI · MACD · EMA (12/26/50) · SMA (20/50/200) · Bollinger Bands · OBV · Log Returns

---

## 🎨 Frontend Pages

| Page        | Route          | Description                              |
|-------------|----------------|------------------------------------------|
| Dashboard   | `/`            | Price, chart, AI forecast, sentiment     |
| Markets     | `/markets`     | Ticker grid + selected chart             |
| Predictions | `/predictions` | Model comparison, signals, metrics       |
| Portfolio   | `/portfolio`   | Holdings, P&L, charts                    |
| Settings    | `/settings`    | Theme, notifications, watchlist          |

---

## 🚨 Disclaimer

> This project is for **educational purposes only**. It does not constitute financial advice.
> Never trade based solely on AI predictions. Past performance does not guarantee future results.

---

## 🛠 Tech Stack

**Frontend:** React 18 · Vite · Tailwind CSS · Recharts · Zustand · React Router · Axios

**Backend:** FastAPI · Uvicorn · Pydantic · yfinance · pandas · scikit-learn · statsmodels

**AI/ML:** TensorFlow/Keras (LSTM) · scikit-learn (SVR) · statsmodels (ARIMA) · HuggingFace Transformers (FinBERT)

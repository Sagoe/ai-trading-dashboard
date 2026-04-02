"""
Prediction Service
Trains / loads LSTM, ARIMA, SVR models and generates forecasts.
"""

import numpy as np
import pandas as pd
import os, joblib, logging
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

MODEL_DIR = "./ml/saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLS = [
    "Close", "Open", "High", "Low", "Volume",
    "RSI", "MACD", "MACD_Signal",
    "EMA_12", "EMA_26", "EMA_50",
    "SMA_20", "SMA_50",
    "BB_Upper", "BB_Lower", "BB_Width",
    "OBV", "Returns",
]


def _metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100)
    return {"rmse": round(rmse, 4), "mae": round(mae, 4),
            "r2": round(r2, 4), "mape": round(mape, 4)}


# ──────────────────────────────────────────
# LSTM MODEL
# ──────────────────────────────────────────
def build_lstm(input_shape, units=64):
    """Build LSTM model using Keras."""
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization

        model = Sequential([
            LSTM(units, return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.2),
            LSTM(units // 2, return_sequences=False),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="huber", metrics=["mae"])
        return model
    except ImportError:
        logger.warning("TensorFlow not available — LSTM disabled")
        return None


def train_lstm(symbol: str, df: pd.DataFrame, window: int = 60, epochs: int = 50):
    """Train LSTM on the given enriched DataFrame."""
    try:
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        cols = [c for c in FEATURE_COLS if c in df.columns]
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[cols])

        X, y = [], []
        for i in range(window, len(scaled)):
            X.append(scaled[i - window:i])
            y.append(scaled[i, cols.index("Close")])
        X, y = np.array(X), np.array(y)

        split = int(len(X) * 0.85)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        model = build_lstm(input_shape=(window, len(cols)))
        if model is None:
            return None, None, {}

        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6),
        ]
        model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  epochs=epochs, batch_size=32, callbacks=callbacks, verbose=0)

        # Save
        model_path = os.path.join(MODEL_DIR, f"{symbol}_lstm.keras")
        scaler_path = os.path.join(MODEL_DIR, f"{symbol}_scaler.pkl")
        model.save(model_path)
        joblib.dump({"scaler": scaler, "cols": cols}, scaler_path)

        # Evaluate
        y_pred = model.predict(X_val, verbose=0).flatten()
        # Inverse scale
        dummy = np.zeros((len(y_val), len(cols)))
        dummy[:, cols.index("Close")] = y_val
        y_true_inv = scaler.inverse_transform(dummy)[:, cols.index("Close")]
        dummy[:, cols.index("Close")] = y_pred
        y_pred_inv = scaler.inverse_transform(dummy)[:, cols.index("Close")]

        return model, scaler, _metrics(y_true_inv, y_pred_inv)

    except Exception as e:
        logger.error(f"LSTM training error: {e}")
        return None, None, {}


def predict_lstm(symbol: str, df: pd.DataFrame, horizon: int = 10, window: int = 60):
    """Load saved LSTM and predict next `horizon` closes."""
    try:
        import tensorflow as tf
        model_path = os.path.join(MODEL_DIR, f"{symbol}_lstm.keras")
        scaler_path = os.path.join(MODEL_DIR, f"{symbol}_scaler.pkl")

        if not os.path.exists(model_path):
            return train_lstm(symbol, df)  # auto-train first time

        model = tf.keras.models.load_model(model_path)
        bundle = joblib.load(scaler_path)
        scaler, cols = bundle["scaler"], bundle["cols"]

        scaled = scaler.transform(df[cols].tail(window + horizon))
        preds = []
        seq = scaled[:window].copy()

        for _ in range(horizon):
            x = seq[-window:].reshape(1, window, len(cols))
            p = model.predict(x, verbose=0)[0, 0]
            next_row = seq[-1].copy()
            next_row[cols.index("Close")] = p
            seq = np.vstack([seq, next_row])
            preds.append(p)

        dummy = np.zeros((horizon, len(cols)))
        dummy[:, cols.index("Close")] = preds
        inv = scaler.inverse_transform(dummy)[:, cols.index("Close")]
        return inv.tolist()

    except Exception as e:
        logger.error(f"LSTM predict error: {e}")
        return []


# ──────────────────────────────────────────
# ARIMA MODEL
# ──────────────────────────────────────────
def train_predict_arima(df: pd.DataFrame, horizon: int = 10) -> dict:
    """Fit ARIMA(5,1,0) on Close prices and forecast."""
    try:
        close = df["Close"].values
        train = close[:-20]

        model = ARIMA(train, order=(5, 1, 0))
        fit = model.fit()
        forecast = fit.forecast(steps=horizon)
        in_sample = fit.fittedvalues

        # Align lengths — differencing makes fittedvalues 1 shorter than train
        min_len = min(len(train) - 1, len(in_sample))
        metrics = _metrics(train[1:min_len + 1], in_sample[:min_len])

        return {
            "forecast": [round(float(v), 2) for v in forecast],
            "metrics": metrics,
        }
    except Exception as e:
        logger.error(f"ARIMA error: {e}")
        return {"forecast": [], "metrics": {}}


# ──────────────────────────────────────────
# SVR MODEL
# ──────────────────────────────────────────
def train_predict_svr(df: pd.DataFrame, horizon: int = 10) -> dict:
    """SVR on lag features of Close price."""
    try:
        close = df["Close"].values
        if len(close) < 35:
            logger.warning("SVR: not enough data (need 35+ rows)")
            return {"forecast": [], "metrics": {}}

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(close.reshape(-1, 1)).flatten()

        window = min(30, len(close) // 3)
        X, y = [], []
        for i in range(window, len(scaled)):
            X.append(scaled[i - window:i])
            y.append(scaled[i])
        X, y = np.array(X), np.array(y)

        if len(X) < 10:
            logger.warning("SVR: too few sequences after windowing")
            return {"forecast": [], "metrics": {}}

        split = max(1, int(len(X) * 0.85))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        if len(X_train) < 5:
            X_train, y_train = X, y
            X_val,   y_val   = X[-5:], y[-5:]

        svr = SVR(kernel="rbf", C=100, gamma=0.01, epsilon=0.001)
        svr.fit(X_train, y_train)

        y_pred_val = svr.predict(X_val)
        y_true_inv = scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
        y_pred_inv = scaler.inverse_transform(y_pred_val.reshape(-1, 1)).flatten()

        # Future predictions
        seq = list(scaled[-window:])
        forecasts = []
        for _ in range(horizon):
            p = svr.predict([seq[-window:]])[0]
            seq.append(p)
            forecasts.append(p)

        inv_forecast = scaler.inverse_transform(
            np.array(forecasts).reshape(-1, 1)
        ).flatten()

        return {
            "forecast": [round(float(v), 2) for v in inv_forecast],
            "metrics": _metrics(y_true_inv, y_pred_inv),
        }
    except Exception as e:
        logger.error(f"SVR error: {e}")
        return {"forecast": [], "metrics": {}}


# ──────────────────────────────────────────
# ENSEMBLE
# ──────────────────────────────────────────
def ensemble_forecast(lstm_preds, arima_preds, svr_preds,
                      weights=(0.6, 0.2, 0.2)) -> list:
    """Weighted average of three model forecasts."""
    try:
        length = min(len(lstm_preds), len(arima_preds), len(svr_preds))
        result = []
        for i in range(length):
            val = (weights[0] * lstm_preds[i] +
                   weights[1] * arima_preds[i] +
                   weights[2] * svr_preds[i])
            result.append(round(val, 2))
        return result
    except Exception:
        return lstm_preds or arima_preds or svr_preds


def generate_signal(current_price: float, predicted_price: float,
                    rsi: float = None) -> dict:
    """Generate Buy / Sell / Hold signal with confidence."""
    change_pct = ((predicted_price - current_price) / current_price) * 100

    if change_pct > 3:
        signal = "BUY"
        confidence = min(95, 60 + change_pct * 5)
    elif change_pct < -3:
        signal = "SELL"
        confidence = min(95, 60 + abs(change_pct) * 5)
    else:
        signal = "HOLD"
        confidence = 50 + abs(change_pct) * 3

    # RSI override
    if rsi is not None:
        if rsi > 75:
            signal = "SELL"
            confidence = max(confidence, 70)
        elif rsi < 30:
            signal = "BUY"
            confidence = max(confidence, 70)

    return {
        "signal": signal,
        "confidence": round(float(confidence), 1),
        "change_pct": round(float(change_pct), 2),
    }

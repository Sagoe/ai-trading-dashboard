"""
upload router — accepts CSV file, runs predictions, returns forecast + chart data
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
import numpy as np
import io
import logging
from services.prediction_service import (
    train_predict_arima, train_predict_svr,
    ensemble_forecast, generate_signal
)
from services.data_service import add_technical_indicators

router = APIRouter()
logger = logging.getLogger(__name__)


def _normalise_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names to Title case and ensure Close exists."""
    df.columns = [c.strip().title() for c in df.columns]

    # Accept 'Adj Close' as 'Close'
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    # Accept 'Adj. Close'
    if "Close" not in df.columns:
        for c in df.columns:
            if "close" in c.lower():
                df["Close"] = df[c]
                break

    if "Close" not in df.columns:
        raise ValueError(
            f"CSV must contain a 'Close' column. "
            f"Columns found: {list(df.columns)}"
        )

    for col in ["Open", "High", "Low"]:
        if col not in df.columns:
            df[col] = df["Close"]
    if "Volume" not in df.columns:
        df["Volume"] = 1  # avoid zero-volume OBV crash

    return df


def _parse_date_index(df: pd.DataFrame) -> pd.DataFrame:
    """Try to set a datetime index."""
    date_candidates = ["Date", "Datetime", "Timestamp", "Time",
                       "date", "datetime", "timestamp"]
    for col in date_candidates:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True)
                df.set_index(col, inplace=True)
                df.index.name = "Date"
                return df
            except Exception:
                pass
    try:
        df.index = pd.to_datetime(df.index, infer_datetime_format=True)
        df.index.name = "Date"
    except Exception:
        df.index = pd.bdate_range(
            end=pd.Timestamp.today(), periods=len(df), freq="B"
        )
        df.index.name = "Date"
    return df


@router.post("/")
async def upload_csv(file: UploadFile = File(...)):
    """
    Accept a CSV with at least a 'Close' column (60+ rows).
    Returns historical data + indicators + AI forecast + signal.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    # ── Parse ────────────────────────────────────────────
    try:
        contents = await file.read()
        df_raw = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    if df_raw.empty:
        raise HTTPException(status_code=422, detail="CSV file is empty.")

    # ── Normalise ────────────────────────────────────────
    try:
        df_raw = _normalise_df(df_raw)
        df_raw = _parse_date_index(df_raw)
        df_raw = df_raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        df_raw = df_raw.apply(pd.to_numeric, errors="coerce")
        df_raw.dropna(subset=["Close"], inplace=True)
        df_raw.sort_index(inplace=True)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if len(df_raw) < 30:
        raise HTTPException(
            status_code=422,
            detail=f"Need at least 30 rows. Your file has {len(df_raw)} usable rows after cleaning."
        )

    # ── Technical indicators (need 30+ rows) ────────────
    try:
        df = add_technical_indicators(df_raw)
        if len(df) < 10:
            raise ValueError("Too few rows remain after indicator calculation.")
    except Exception as e:
        logger.warning(f"Indicators failed ({e}) — using raw OHLCV only")
        df = df_raw.copy()

    if df.empty or "Close" not in df.columns:
        raise HTTPException(status_code=422, detail="DataFrame is empty after processing.")

    # ── Models ───────────────────────────────────────────
    horizon = 10

    arima_result = {"forecast": [], "metrics": {}}
    svr_result   = {"forecast": [], "metrics": {}}

    if len(df) >= 30:
        try:
            arima_result = train_predict_arima(df, horizon=horizon)
        except Exception as e:
            logger.error(f"ARIMA failed: {e}")

        try:
            svr_result = train_predict_svr(df, horizon=horizon)
        except Exception as e:
            logger.error(f"SVR failed: {e}")

    arima_preds = arima_result.get("forecast", [])
    svr_preds   = svr_result.get("forecast", [])

    # Build ensemble — use whatever models worked
    if arima_preds and svr_preds:
        ens = ensemble_forecast(svr_preds, arima_preds, svr_preds,
                                weights=(0.5, 0.3, 0.2))
    elif arima_preds:
        ens = arima_preds
    elif svr_preds:
        ens = svr_preds
    else:
        # Naive forecast: last price repeated
        last = float(df["Close"].iloc[-1])
        ens = [round(last, 2)] * horizon

    # ── Signal ───────────────────────────────────────────
    current_price = float(df["Close"].iloc[-1])
    rsi_val = float(df["RSI"].iloc[-1]) if "RSI" in df.columns and not df["RSI"].isna().all() else None

    signal = generate_signal(
        current_price=current_price,
        predicted_price=ens[-1] if ens else current_price,
        rsi=rsi_val,
    )

    # ── Future dates ─────────────────────────────────────
    last_date    = df.index[-1]
    future_dates = pd.bdate_range(start=last_date, periods=horizon + 1)[1:]
    date_labels  = [d.strftime("%Y-%m-%d") for d in future_dates]

    # ── History for chart (last 200 rows) ────────────────
    hist_df = df.reset_index()
    hist_df["Date"] = hist_df["Date"].astype(str)
    float_cols = hist_df.select_dtypes(include=["float32","float64"]).columns
    hist_df[float_cols] = hist_df[float_cols].round(4)
    history = hist_df.tail(200).to_dict(orient="records")

    # ── Summary stats ────────────────────────────────────
    returns = df["Close"].pct_change().dropna()
    first_price = float(df["Close"].iloc[0])
    total_return = ((current_price - first_price) / first_price * 100) if first_price else 0

    return {
        "filename":      file.filename,
        "rows_uploaded": len(df_raw),
        "rows_used":     len(df),
        "columns":       list(df_raw.columns),
        "current_price": round(current_price, 2),
        "signal":        signal,
        "rsi":           round(rsi_val, 2) if rsi_val is not None else None,
        "summary": {
            "min":               round(float(df["Close"].min()), 2),
            "max":               round(float(df["Close"].max()), 2),
            "mean":              round(float(df["Close"].mean()), 2),
            "std":               round(float(df["Close"].std()), 2),
            "return_total_pct":  round(total_return, 2),
            "volatility_daily":  round(float(returns.std()), 4),
        },
        "forecast": {
            "dates":    date_labels,
            "ensemble": ens,
            "arima":    arima_preds,
            "svr":      svr_preds,
        },
        "metrics": {
            "arima": arima_result.get("metrics", {}),
            "svr":   svr_result.get("metrics", {}),
        },
        "history": history,
    }

"""predict router — runs all models and returns ensemble forecast"""
from fastapi import APIRouter, HTTPException, Query
from services.data_service import fetch_ohlcv, add_technical_indicators, get_current_price
from services.prediction_service import (
    predict_lstm, train_predict_arima, train_predict_svr,
    ensemble_forecast, generate_signal
)
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/{symbol}")
async def predict_stock(
    symbol: str,
    horizon: int = Query(10, ge=1, le=30, description="Days to forecast"),
    model: str = Query("ensemble", description="lstm | arima | svr | ensemble"),
):
    try:
        sym = symbol.upper()
        df_raw = fetch_ohlcv(sym, period="5y")
        df = add_technical_indicators(df_raw)
        current = get_current_price(sym)
        current_price = current["price"]

        arima_result = train_predict_arima(df, horizon=horizon)
        svr_result = train_predict_svr(df, horizon=horizon)
        lstm_preds = predict_lstm(sym, df, horizon=horizon)

        # Fallback: if LSTM unavailable, use SVR
        if not lstm_preds:
            lstm_preds = svr_result["forecast"]

        arima_preds = arima_result["forecast"]
        svr_preds = svr_result["forecast"]

        ens = ensemble_forecast(lstm_preds, arima_preds, svr_preds)
        target_map = {"lstm": lstm_preds, "arima": arima_preds,
                      "svr": svr_preds, "ensemble": ens}
        selected = target_map.get(model, ens)

        rsi = float(df["RSI"].iloc[-1]) if "RSI" in df.columns else None
        signal_data = generate_signal(
            current_price=current_price,
            predicted_price=selected[-1] if selected else current_price,
            rsi=rsi,
        )

        # Build date labels
        last_date = df.index[-1]
        import pandas as pd
        future_dates = pd.bdate_range(start=last_date, periods=horizon + 1)[1:]
        date_labels = [d.strftime("%Y-%m-%d") for d in future_dates]

        return {
            "symbol": sym,
            "current_price": current_price,
            "horizon_days": horizon,
            "model_used": model,
            "forecast": {
                "dates": date_labels,
                "ensemble": ens,
                "lstm": lstm_preds,
                "arima": arima_preds,
                "svr": svr_preds,
            },
            "signal": signal_data,
            "rsi": rsi,
            "metrics": {
                "arima": arima_result.get("metrics", {}),
                "svr": svr_result.get("metrics", {}),
            },
        }
    except Exception as e:
        logger.error(f"Prediction error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

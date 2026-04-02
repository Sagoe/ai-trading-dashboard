"""history router — returns OHLCV + indicators as JSON"""
from fastapi import APIRouter, HTTPException, Query
from services.data_service import fetch_ohlcv, add_technical_indicators
import pandas as pd

router = APIRouter()


@router.get("/{symbol}")
async def get_history(
    symbol: str,
    period: str = Query("1y", description="1mo 3mo 6mo 1y 2y 5y"),
    indicators: bool = Query(True),
):
    try:
        df = fetch_ohlcv(symbol.upper(), period=period)
        if indicators:
            df = add_technical_indicators(df)

        df = df.reset_index()
        df["Date"] = df["Date"].astype(str)

        # Round floats
        float_cols = df.select_dtypes("float64").columns
        df[float_cols] = df[float_cols].round(4)

        return {
            "symbol": symbol.upper(),
            "period": period,
            "rows": len(df),
            "columns": list(df.columns),
            "data": df.to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

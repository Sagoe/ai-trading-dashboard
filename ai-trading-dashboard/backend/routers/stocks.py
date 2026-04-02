"""stocks router"""
from fastapi import APIRouter, HTTPException
from services.data_service import POPULAR_SYMBOLS, get_stock_info, get_current_price
import asyncio

router = APIRouter()


@router.get("/")
async def list_stocks():
    return {"symbols": POPULAR_SYMBOLS}


@router.get("/{symbol}/info")
async def stock_info(symbol: str):
    try:
        return get_stock_info(symbol.upper())
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{symbol}/price")
async def stock_price(symbol: str):
    try:
        return get_current_price(symbol.upper())
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/market/overview")
async def market_overview():
    """Return prices for key market indices."""
    indices = ["SPY", "QQQ", "DIA", "IWM"]
    results = []
    for sym in indices:
        try:
            results.append(get_current_price(sym))
        except Exception:
            pass
    return {"indices": results}

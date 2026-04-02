"""
portfolio router
In-memory portfolio for demo. In production swap with a real DB.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from services.data_service import get_current_price

router = APIRouter()

# In-memory store (resets on restart — use a DB in production)
_portfolio: dict = {}   # { symbol: { shares, avg_cost } }


class AddPosition(BaseModel):
    symbol: str
    shares: float
    avg_cost: float


@router.get("/")
async def get_portfolio():
    holdings = []
    total_value = 0.0
    total_cost = 0.0

    for sym, pos in _portfolio.items():
        try:
            price_data = get_current_price(sym)
            current_price = price_data["price"]
            value = pos["shares"] * current_price
            cost = pos["shares"] * pos["avg_cost"]
            pl = value - cost
            pl_pct = (pl / cost * 100) if cost else 0

            holdings.append({
                "symbol": sym,
                "shares": pos["shares"],
                "avg_cost": pos["avg_cost"],
                "current_price": current_price,
                "value": round(value, 2),
                "cost_basis": round(cost, 2),
                "profit_loss": round(pl, 2),
                "profit_loss_pct": round(pl_pct, 2),
                "change": price_data["change"],
                "change_pct": price_data["change_pct"],
            })
            total_value += value
            total_cost += cost
        except Exception:
            pass

    total_pl = total_value - total_cost
    return {
        "holdings": holdings,
        "total_value": round(total_value, 2),
        "total_cost": round(total_cost, 2),
        "total_pl": round(total_pl, 2),
        "total_pl_pct": round((total_pl / total_cost * 100) if total_cost else 0, 2),
    }


@router.post("/add")
async def add_position(pos: AddPosition):
    sym = pos.symbol.upper()
    _portfolio[sym] = {"shares": pos.shares, "avg_cost": pos.avg_cost}
    return {"message": f"Added {pos.shares} shares of {sym}", "portfolio_size": len(_portfolio)}


@router.delete("/{symbol}")
async def remove_position(symbol: str):
    sym = symbol.upper()
    if sym not in _portfolio:
        raise HTTPException(status_code=404, detail=f"{sym} not in portfolio")
    del _portfolio[sym]
    return {"message": f"Removed {sym}"}

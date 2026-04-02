"""sentiment router"""
from fastapi import APIRouter, HTTPException
from services.sentiment_service import analyze_sentiment
from utils.config import settings

router = APIRouter()


@router.get("/{symbol}")
async def get_sentiment(symbol: str):
    try:
        return analyze_sentiment(symbol.upper(), api_key=settings.NEWS_API_KEY)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

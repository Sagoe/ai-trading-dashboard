"""
AI Trading Dashboard — FastAPI Backend
Entry point: uvicorn main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from routers import predict, history, sentiment, stocks, portfolio, upload
from utils.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 AI Trading Dashboard backend starting...")
    yield
    logger.info("🛑 Backend shutting down.")


app = FastAPI(
    title="AI Trading Dashboard API",
    description="ML-powered stock prediction, sentiment analysis, and portfolio tracking.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(stocks.router,    prefix="/stocks",    tags=["Stocks"])
app.include_router(history.router,   prefix="/history",   tags=["History"])
app.include_router(predict.router,   prefix="/predict",   tags=["Predictions"])
app.include_router(sentiment.router, prefix="/sentiment", tags=["Sentiment"])
app.include_router(portfolio.router, prefix="/portfolio", tags=["Portfolio"])
app.include_router(upload.router,    prefix="/upload",    tags=["Upload"])


@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "message": "AI Trading Dashboard API is running"}


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return {}


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy", "version": "1.0.0"}

"""
Data Service — fetches OHLCV from Yahoo Finance,
computes all technical indicators, and caches results.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator
from sklearn.preprocessing import MinMaxScaler
import logging, os, time, random

logger = logging.getLogger(__name__)

CACHE_DIR = "./data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

POPULAR_SYMBOLS = [
    "AAPL","MSFT","GOOGL","AMZN","NVDA",
    "TSLA","META","NFLX","AMD","INTC",
    "JPM","BAC","GS","V","MA",
    "SPY","QQQ","BRK-B","JNJ","UNH",
]

_last_fetch: dict = {}
MIN_FETCH_GAP = 2.0


def _rate_limit(symbol: str):
    now  = time.time()
    last = _last_fetch.get(symbol, 0)
    gap  = now - last
    if gap < MIN_FETCH_GAP:
        time.sleep(MIN_FETCH_GAP - gap + random.uniform(0.1, 0.5))
    _last_fetch[symbol] = time.time()


def fetch_ohlcv(symbol: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    cache_file = os.path.join(CACHE_DIR, f"{symbol}_{period}.csv")
    if os.path.exists(cache_file):
        age_hours = (pd.Timestamp.now().timestamp() - os.path.getmtime(cache_file)) / 3600
        if age_hours < 1:
            logger.info(f"Cache hit for {symbol}")
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    _rate_limit(symbol)
    logger.info(f"Fetching {symbol} from Yahoo Finance...")
    df     = pd.DataFrame()
    errors = []

    for attempt, fn in enumerate([
        lambda: yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=True),
        lambda: yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False, threads=False),
        lambda: yf.Ticker(symbol).history(period="2y", interval=interval, auto_adjust=True),
    ]):
        try:
            if attempt > 0:
                time.sleep(attempt * 1.5)
            df = fn()
            if df is not None and not df.empty:
                break
        except Exception as e:
            errors.append(str(e))

    if df is None or df.empty:
        raise ValueError(f"No data for '{symbol}'. Try again in 60s. Errors: {'; '.join(errors)}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    cols = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    df   = df[cols].copy()
    df.dropna(inplace=True)
    df.to_csv(cache_file)
    logger.info(f"Cached {len(df)} rows for {symbol}")
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df     = df.copy()
    close  = df["Close"]
    volume = df["Volume"]

    df["RSI"]         = RSIIndicator(close=close, window=14).rsi()
    macd              = MACD(close=close)
    df["MACD"]        = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"]   = macd.macd_diff()
    df["EMA_12"]      = EMAIndicator(close=close, window=12).ema_indicator()
    df["EMA_26"]      = EMAIndicator(close=close, window=26).ema_indicator()
    df["EMA_50"]      = EMAIndicator(close=close, window=50).ema_indicator()
    df["SMA_20"]      = SMAIndicator(close=close, window=20).sma_indicator()
    df["SMA_50"]      = SMAIndicator(close=close, window=50).sma_indicator()
    df["SMA_200"]     = SMAIndicator(close=close, window=200).sma_indicator()

    bb = BollingerBands(close=close, window=20, window_dev=2)
    df["BB_Upper"]  = bb.bollinger_hband()
    df["BB_Middle"] = bb.bollinger_mavg()
    df["BB_Lower"]  = bb.bollinger_lband()
    df["BB_Width"]  = bb.bollinger_wband()

    df["OBV"]         = OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
    df["Returns"]     = close.pct_change()
    df["Log_Returns"] = np.log(close / close.shift(1))
    df.dropna(inplace=True)
    return df


def prepare_sequences(df, feature_cols, target_col="Close", window=60):
    scaler    = MinMaxScaler()
    scaled    = scaler.fit_transform(df[feature_cols])
    scaled_df = pd.DataFrame(scaled, columns=feature_cols, index=df.index)
    X, y      = [], []
    for i in range(window, len(scaled_df)):
        X.append(scaled_df.iloc[i - window:i].values)
        y.append(scaled_df.iloc[i][target_col])
    return np.array(X), np.array(y), scaler, feature_cols.index(target_col)


def get_stock_info(symbol: str) -> dict:
    try:
        _rate_limit(symbol)
        info = yf.Ticker(symbol).info
        return {
            "symbol": symbol, "name": info.get("longName", symbol),
            "sector": info.get("sector","N/A"), "industry": info.get("industry","N/A"),
            "market_cap": info.get("marketCap"), "pe_ratio": info.get("trailingPE"),
            "52w_high": info.get("fiftyTwoWeekHigh"), "52w_low": info.get("fiftyTwoWeekLow"),
            "avg_volume": info.get("averageVolume"), "beta": info.get("beta"),
            "dividend_yield": info.get("dividendYield"),
            "description": info.get("longBusinessSummary",""),
        }
    except Exception as e:
        logger.warning(f"Could not fetch info for {symbol}: {e}")
        return {"symbol": symbol, "name": symbol}


def get_current_price(symbol: str) -> dict:
    # Price cache — 5 min
    cache_file = os.path.join(CACHE_DIR, f"{symbol}_price.csv")
    if os.path.exists(cache_file):
        age_mins = (pd.Timestamp.now().timestamp() - os.path.getmtime(cache_file)) / 60
        if age_mins < 5:
            try:
                return pd.read_csv(cache_file).iloc[0].to_dict()
            except Exception:
                pass

    _rate_limit(symbol)

    try:
        ticker = yf.Ticker(symbol)

        # fast_info path
        try:
            fi    = ticker.fast_info
            price = float(fi.last_price)
            if price and price > 0:
                prev   = float(fi.previous_close or price)
                change = price - prev
                result = {
                    "symbol":     symbol,
                    "price":      round(price, 2),
                    "open":       round(float(fi.open or price), 2),
                    "high":       round(float(fi.day_high or price), 2),
                    "low":        round(float(fi.day_low  or price), 2),
                    "volume":     int(fi.last_volume or 0),
                    "change":     round(change, 2),
                    "change_pct": round((change / prev * 100) if prev else 0, 2),
                }
                pd.DataFrame([result]).to_csv(cache_file, index=False)
                return result
        except Exception:
            pass

        # history fallback
        hist = ticker.history(period="5d", interval="1d", auto_adjust=True)
        if hist is None or hist.empty:
            raise ValueError(f"Empty history for {symbol}")

        hist.dropna(subset=["Close"], inplace=True)
        latest = hist.iloc[-1]
        prev   = float(hist.iloc[-2]["Close"]) if len(hist) > 1 else float(latest["Close"])
        price  = float(latest["Close"])
        change = price - prev
        result = {
            "symbol":     symbol,
            "price":      round(price, 2),
            "open":       round(float(latest.get("Open", price)), 2),
            "high":       round(float(latest.get("High", price)), 2),
            "low":        round(float(latest.get("Low",  price)), 2),
            "volume":     int(latest.get("Volume", 0)),
            "change":     round(change, 2),
            "change_pct": round((change / prev * 100) if prev else 0, 2),
        }
        pd.DataFrame([result]).to_csv(cache_file, index=False)
        return result

    except Exception as e:
        logger.error(f"get_current_price failed for {symbol}: {e}")
        raise ValueError(f"Could not fetch price for {symbol}: {e}")
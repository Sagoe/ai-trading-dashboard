"""
ml/train.py
─────────────────────────────────────────────────────────────
Run this script in PyCharm to pre-train all models for a set
of symbols before starting the backend server.

Usage:
    python ml/train.py --symbols AAPL MSFT GOOGL --epochs 60
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.services.data_service import fetch_ohlcv, add_technical_indicators
from backend.services.prediction_service import (
    train_lstm, train_predict_arima, train_predict_svr
)
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def train_all(symbols: list, epochs: int = 50):
    results = {}
    for sym in symbols:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training models for {sym}...")
        logger.info(f"{'='*50}")

        try:
            df_raw = fetch_ohlcv(sym, period="5y")
            df     = add_technical_indicators(df_raw)
            logger.info(f"  Data loaded: {len(df)} rows")

            # ARIMA
            arima  = train_predict_arima(df, horizon=10)
            logger.info(f"  ARIMA metrics: {arima['metrics']}")

            # SVR
            svr    = train_predict_svr(df, horizon=10)
            logger.info(f"  SVR metrics: {svr['metrics']}")

            # LSTM
            model, scaler, metrics = train_lstm(sym, df, epochs=epochs)
            if model:
                logger.info(f"  LSTM metrics: {metrics}")
            else:
                logger.warning(f"  LSTM skipped (TensorFlow unavailable)")

            results[sym] = {"arima": arima["metrics"], "svr": svr["metrics"], "lstm": metrics}

        except Exception as e:
            logger.error(f"  Failed for {sym}: {e}")
            results[sym] = {"error": str(e)}

    logger.info(f"\n{'='*50}")
    logger.info("Training complete. Summary:")
    for sym, r in results.items():
        logger.info(f"  {sym}: {r}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train AI trading models")
    parser.add_argument("--symbols", nargs="+", default=["AAPL","MSFT","GOOGL"],
                        help="Stock symbols to train on")
    parser.add_argument("--epochs", type=int, default=50,
                        help="LSTM training epochs")
    args = parser.parse_args()
    train_all(args.symbols, args.epochs)

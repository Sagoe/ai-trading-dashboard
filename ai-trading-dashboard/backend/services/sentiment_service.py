"""
Sentiment Service
Fetches financial news via NewsAPI and scores with FinBERT.
Falls back to VADER lexicon if transformers unavailable.
"""

import os, logging, re
from datetime import datetime, timedelta
from typing import List, Dict

logger = logging.getLogger(__name__)

# ── FinBERT loader (lazy) ─────────────────────────────────
_finbert = None
_tokenizer = None

def _load_finbert():
    global _finbert, _tokenizer
    if _finbert is None:
        try:
            from transformers import pipeline
            logger.info("Loading FinBERT...")
            _finbert = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                return_all_scores=True,
                truncation=True,
                max_length=512,
            )
            logger.info("FinBERT loaded.")
        except Exception as e:
            logger.warning(f"FinBERT unavailable: {e}. Using fallback scorer.")
    return _finbert


# ── News fetching ─────────────────────────────────────────
def fetch_news(symbol: str, api_key: str, days_back: int = 7) -> List[Dict]:
    """Fetch recent news headlines for `symbol` using NewsAPI."""
    if not api_key:
        logger.warning("No NEWS_API_KEY set — returning mock news.")
        return _mock_news(symbol)

    try:
        from newsapi import NewsApiClient
        client = NewsApiClient(api_key=api_key)
        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        # Search by company symbol + name
        articles = client.get_everything(
            q=f"{symbol} stock",
            from_param=from_date,
            language="en",
            sort_by="relevancy",
            page_size=20,
        )
        return articles.get("articles", [])
    except Exception as e:
        logger.error(f"NewsAPI error: {e}")
        return _mock_news(symbol)


def _mock_news(symbol: str) -> List[Dict]:
    """Return deterministic mock articles when API key is absent."""
    return [
        {"title": f"{symbol} beats earnings expectations by 8%",
         "description": "Strong quarterly results driven by cloud revenue growth.",
         "publishedAt": datetime.now().isoformat(), "source": {"name": "Mock News"}},
        {"title": f"Analysts raise {symbol} price target",
         "description": "Multiple banks upgrade stock amid positive macro outlook.",
         "publishedAt": datetime.now().isoformat(), "source": {"name": "Mock Finance"}},
        {"title": f"{symbol} faces regulatory scrutiny over data practices",
         "description": "Antitrust investigation may weigh on near-term outlook.",
         "publishedAt": datetime.now().isoformat(), "source": {"name": "Mock Wire"}},
    ]


# ── Scoring ───────────────────────────────────────────────
def _finbert_score(text: str) -> Dict:
    pipe = _load_finbert()
    if pipe is None:
        return _vader_score(text)

    results = pipe(text[:512])[0]
    scores = {r["label"].lower(): r["score"] for r in results}
    compound = scores.get("positive", 0) - scores.get("negative", 0)
    label = "positive" if compound > 0.1 else "negative" if compound < -0.1 else "neutral"
    return {"label": label, "compound": round(compound, 4),
            "positive": round(scores.get("positive", 0), 4),
            "negative": round(scores.get("negative", 0), 4),
            "neutral": round(scores.get("neutral", 0), 4)}


def _vader_score(text: str) -> Dict:
    """Minimal keyword-based fallback when FinBERT is unavailable."""
    text_l = text.lower()
    pos_words = ["beat", "surge", "strong", "growth", "upgrade",
                 "profit", "gain", "record", "bullish", "positive"]
    neg_words = ["miss", "fall", "weak", "loss", "downgrade",
                 "cut", "risk", "concern", "bearish", "negative"]

    pos = sum(1 for w in pos_words if w in text_l)
    neg = sum(1 for w in neg_words if w in text_l)
    total = pos + neg or 1
    compound = (pos - neg) / total
    label = "positive" if compound > 0.1 else "negative" if compound < -0.1 else "neutral"
    return {"label": label, "compound": round(compound, 4),
            "positive": round(pos / total, 4),
            "negative": round(neg / total, 4),
            "neutral": round(1 - abs(compound), 4)}


def analyze_sentiment(symbol: str, api_key: str = "") -> Dict:
    """Full pipeline: fetch news → score each → aggregate."""
    articles = fetch_news(symbol, api_key)

    scored = []
    for art in articles[:15]:
        text = f"{art.get('title', '')}. {art.get('description', '')}"
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        score = _finbert_score(text)
        scored.append({
            "headline": art.get("title", ""),
            "source": art.get("source", {}).get("name", ""),
            "published": art.get("publishedAt", ""),
            **score,
        })

    if not scored:
        return {"symbol": symbol, "overall": "neutral", "score": 0.0, "articles": []}

    compounds = [s["compound"] for s in scored]
    avg = sum(compounds) / len(compounds)
    pos_count = sum(1 for s in scored if s["label"] == "positive")
    neg_count = sum(1 for s in scored if s["label"] == "negative")
    overall = "positive" if avg > 0.05 else "negative" if avg < -0.05 else "neutral"

    return {
        "symbol": symbol,
        "overall": overall,
        "score": round(float(avg), 4),
        "articles_analyzed": len(scored),
        "positive_count": pos_count,
        "negative_count": neg_count,
        "neutral_count": len(scored) - pos_count - neg_count,
        "articles": scored[:10],
    }

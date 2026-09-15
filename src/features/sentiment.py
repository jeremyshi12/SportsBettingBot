"""Real-time sentiment and context feature extractor.

POINT-IN-TIME WARNING
---------------------
This engine queries a live news search at the moment `get_team_sentiment` is
called. That is correct for live trading and **invalid for backtesting**: a
backtest replaying March 2026 markets would score them against whatever the
web says today, which is a look-ahead leak and also makes the backtest
non-reproducible and network-dependent.

`FeatureEngine` therefore defaults to `allow_live_sentiment=False` and returns
0.0 for these features during research. Enable it only in the live runner, or
supply a historical sentiment table keyed by (team, timestamp).

Both third-party dependencies are optional; the module degrades to neutral
sentiment when they are absent so that the research pipeline never depends on
a scraping package being installed.
"""

import logging
from typing import Dict
import time

try:
    from duckduckgo_search import DDGS
except ImportError:  # optional dependency
    DDGS = None
try:
    from textblob import TextBlob
except ImportError:  # optional dependency
    TextBlob = None

logger = logging.getLogger("trading.features.sentiment")

class SentimentEngine:
    """Fetches news and computes sentiment scores for sports teams."""

    def __init__(self):
        self.cache: Dict[str, Dict[str, float]] = {}
        self.cache_ttl = 3600 * 4  # 4 hours
        if DDGS is None or TextBlob is None:
            logger.info(
                "duckduckgo-search / textblob not installed -- sentiment "
                "features will be neutral (0.0)."
            )
            self.ddgs = None
            return
        try:
            self.ddgs = DDGS()
        except BaseException as e:
            logger.warning(f"Failed to initialize DDGS: {e}")
            self.ddgs = None

    def get_team_sentiment(self, team_name: str) -> float:
        """Fetch recent news and compute average sentiment score (-1.0 to 1.0)."""
        if not self.ddgs:
            return 0.0

        now = time.time()

        # Simplify common non-team names or generic markets
        if team_name.lower() in ("yes", "no", "over", "under"):
            return 0.0

        if team_name in self.cache:
            entry = self.cache[team_name]
            if now - entry["ts"] < self.cache_ttl:
                return entry["score"]

        try:
            # Query the news (sleep slightly to avoid rate-limiting if calling many)
            time.sleep(0.5)
            results = list(self.ddgs.news(f"{team_name} basketball", max_results=5))
            if not results:
                self.cache[team_name] = {"score": 0.0, "ts": now}
                return 0.0
                
            total_sentiment = 0.0
            
            for res in results:
                title = res.get("title", "")
                body = res.get("body", "")
                text = f"{title}. {body}"
                blob = TextBlob(text)
                total_sentiment += blob.sentiment.polarity
                
            avg_sentiment = total_sentiment / max(1, len(results))
            
            self.cache[team_name] = {"score": avg_sentiment, "ts": now}
            return float(avg_sentiment)
            
        except Exception as e:
            logger.warning(f"Failed to fetch sentiment for {team_name}: {e}")
            return 0.0

# Global instance for easy access
sentiment_engine = SentimentEngine()

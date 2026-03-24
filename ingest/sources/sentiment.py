"""News and social media sentiment source using NewsAPI."""

from __future__ import annotations

import random
from datetime import date
from pathlib import Path

import pandas as pd
import structlog

log = structlog.get_logger()

_SAMPLE_HEADLINES = [
    "Company beats earnings expectations",
    "Revenue growth slows amid macro headwinds",
    "Management raises full-year guidance",
    "Supply chain disruptions impact margins",
    "Strong demand drives record quarterly results",
]


class SentimentSource:
    """News and social media sentiment. Uses NewsAPI.

    In production, requires NEWS_API_KEY environment variable and the
    ``newsapi-python`` package. This implementation returns sample data
    when the API key is unavailable, enabling offline development and testing.
    """

    def __init__(self, news_api_key: str, parquet_dir: str) -> None:
        self._api_key = news_api_key
        self._parquet_dir = Path(parquet_dir)
        self._parquet_dir.mkdir(parents=True, exist_ok=True)
        self._degraded = not bool(news_api_key)
        if self._degraded:
            log.warning("SentimentSource: no API key provided, using sample data")

    def fetch_sentiment(
        self,
        tickers: list[str],
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Fetch sentiment scores for tickers between start and end.

        Returns a DataFrame with columns:
            symbol, date, sentiment_score, source, headline_count

        When running in degraded mode (no API key), returns deterministic
        sample data suitable for testing.
        """
        log.info(
            "Fetching sentiment",
            ticker_count=len(tickers),
            start=str(start),
            end=str(end),
            degraded=self._degraded,
        )

        if self._degraded:
            return self._sample_sentiment(tickers, start, end)

        return self._fetch_from_api(tickers, start, end)

    def _fetch_from_api(
        self,
        tickers: list[str],
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Fetch from NewsAPI and compute simple sentiment scores."""
        try:
            from newsapi import NewsApiClient  # type: ignore[import]
        except ImportError:
            log.warning("newsapi-python not installed; falling back to sample data")
            return self._sample_sentiment(tickers, start, end)

        client = NewsApiClient(api_key=self._api_key)
        rows: list[dict] = []

        for ticker in tickers:
            try:
                response = client.get_everything(
                    q=ticker,
                    from_param=start.isoformat(),
                    to=end.isoformat(),
                    language="en",
                    sort_by="relevancy",
                    page_size=100,
                )
                articles = response.get("articles", [])
                headline_count = len(articles)

                if headline_count == 0:
                    continue

                # Simple keyword-based sentiment scoring
                positive_words = {"beat", "record", "growth", "raised", "strong", "surpassed"}
                negative_words = {"miss", "decline", "cut", "below", "weak", "disappoints"}

                pos_count = 0
                neg_count = 0
                for article in articles:
                    title = (article.get("title") or "").lower()
                    for w in positive_words:
                        if w in title:
                            pos_count += 1
                    for w in negative_words:
                        if w in title:
                            neg_count += 1

                total = pos_count + neg_count
                if total > 0:
                    score = (pos_count - neg_count) / total
                else:
                    score = 0.0

                rows.append({
                    "symbol": ticker,
                    "date": end,
                    "sentiment_score": round(score, 4),
                    "source": "newsapi",
                    "headline_count": headline_count,
                })

            except Exception as exc:
                log.warning("Failed to fetch sentiment for ticker", ticker=ticker, error=str(exc))
                continue

        if not rows:
            return pd.DataFrame(
                columns=["symbol", "date", "sentiment_score", "source", "headline_count"]
            )

        df = pd.DataFrame(rows)
        log.info("Sentiment fetch complete", rows=len(df))
        return df

    def _sample_sentiment(
        self,
        tickers: list[str],
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Generate deterministic sample sentiment data for testing."""
        rng = random.Random(hash((str(start), str(end))))
        rows = []
        for ticker in tickers:
            score = round(rng.uniform(-0.5, 0.8), 4)
            count = rng.randint(3, 25)
            rows.append({
                "symbol": ticker,
                "date": end,
                "sentiment_score": score,
                "source": "sample",
                "headline_count": count,
            })
        return pd.DataFrame(rows)

    def write_parquet(self, df: pd.DataFrame, batch_date: date) -> Path:
        """Write sentiment DataFrame to a dated Parquet file and return the path."""
        path = self._parquet_dir / f"sentiment_{batch_date.isoformat()}.parquet"
        df.to_parquet(path, index=False)
        log.info("Wrote sentiment rows to parquet", rows=len(df), path=str(path))
        return path

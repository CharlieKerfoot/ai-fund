"""Earnings transcript source using Financial Modeling Prep API."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

_MAX_TRANSCRIPT_CHARS = 150_000


class EarningsSource:
    """Downloads earnings call transcripts from Financial Modeling Prep."""

    FMP_BASE = "https://financialmodelingprep.com/api/v3"

    def __init__(self, fmp_api_key: str, parquet_dir: str) -> None:
        self._api_key = fmp_api_key
        self._parquet_dir = Path(parquet_dir)
        self._parquet_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _get(self, url: str, params: dict | None = None) -> requests.Response:
        """Make a GET request with retry logic."""
        all_params = {"apikey": self._api_key}
        if params:
            all_params.update(params)
        response = self._session.get(url, params=all_params, timeout=30)
        response.raise_for_status()
        return response

    def fetch_transcripts(self, ticker: str, year: int, quarter: int) -> list[dict]:
        """Return list of transcript dicts with: symbol, date, quarter, year, content."""
        url = f"{self.FMP_BASE}/earning_call_transcript/{ticker.upper()}"
        try:
            resp = self._get(url, params={"year": year, "quarter": quarter})
            data = resp.json()
            if not data or not isinstance(data, list):
                return []

            results = []
            for item in data:
                content = item.get("content", "")
                if len(content) > _MAX_TRANSCRIPT_CHARS:
                    content = content[:_MAX_TRANSCRIPT_CHARS]
                results.append({
                    "symbol": item.get("symbol", ticker),
                    "date": item.get("date", ""),
                    "quarter": item.get("quarter", quarter),
                    "year": item.get("year", year),
                    "content": content,
                })
            return results
        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else None
            if code == 404:
                logger.debug("No transcript found for %s Q%d %d", ticker, quarter, year)
            else:
                logger.warning("HTTP error fetching transcript for %s Q%d %d: %s", ticker, quarter, year, exc)
            return []
        except Exception as exc:
            logger.warning("Error fetching transcript for %s Q%d %d: %s", ticker, quarter, year, exc)
            return []

    def fetch_recent_transcripts(
        self,
        tickers: list[str],
        lookback_quarters: int = 4,
    ) -> list[dict]:
        """Fetch recent N quarters of transcripts for all tickers."""
        from datetime import datetime

        now = datetime.utcnow()
        current_year = now.year
        current_quarter = (now.month - 1) // 3 + 1

        # Build list of (year, quarter) pairs going back lookback_quarters
        quarters: list[tuple[int, int]] = []
        year = current_year
        quarter = current_quarter
        for _ in range(lookback_quarters):
            quarters.append((year, quarter))
            quarter -= 1
            if quarter == 0:
                quarter = 4
                year -= 1

        all_transcripts: list[dict] = []
        for ticker in tickers:
            for year, quarter in quarters:
                transcripts = self.fetch_transcripts(ticker, year, quarter)
                all_transcripts.extend(transcripts)

        return all_transcripts

    def write_parquet(self, transcripts: list[dict], batch_date: date) -> Path:
        """Write transcript records to a dated Parquet file."""
        df = pd.DataFrame(transcripts)
        path = self._parquet_dir / f"earnings_{batch_date.isoformat()}.parquet"
        df.to_parquet(path, index=False)
        logger.info("Wrote %d transcript records to %s", len(df), path)
        return path

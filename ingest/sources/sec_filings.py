"""SEC EDGAR filings source."""

from __future__ import annotations

import logging
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class SECFilingsSource:
    """Downloads SEC filings metadata from EDGAR and stages to Parquet."""

    BASE_URL = "https://efts.sec.gov/LATEST/search-index"
    SUBMISSIONS_URL = "https://data.sec.gov/submissions"

    def __init__(self, user_agent: str, parquet_dir: str, rate_limit: float = 8.0) -> None:
        self._user_agent = user_agent
        self._parquet_dir = Path(parquet_dir)
        self._parquet_dir.mkdir(parents=True, exist_ok=True)
        self._rate_limit = rate_limit
        self._min_interval = 1.0 / rate_limit
        self._last_request_time: float = 0.0
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": user_agent})

    def _throttle(self) -> None:
        """Enforce rate limiting between requests."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.monotonic()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    def _get(self, url: str, params: dict | None = None) -> requests.Response:
        """Make a GET request with rate limiting and retry logic."""
        self._throttle()
        response = self._session.get(url, params=params, timeout=30)
        if response.status_code == 429:
            logger.warning("SEC rate limit hit (429), will retry after backoff")
            response.raise_for_status()
        return response

    def get_cik(self, ticker: str) -> str | None:
        """Get SEC CIK number for a ticker symbol."""
        url = f"{self.SUBMISSIONS_URL}/CIK{ticker.upper()}.json"
        try:
            resp = self._get(url)
            if resp.status_code == 404:
                logger.warning("No CIK found for ticker %s", ticker)
                return None
            resp.raise_for_status()
            data = resp.json()
            cik = str(data.get("cik", "")).zfill(10)
            return cik if cik.strip("0") else None
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                logger.warning("No CIK found for ticker %s", ticker)
                return None
            logger.warning("HTTP error fetching CIK for %s: %s", ticker, exc)
            return None
        except Exception as exc:
            logger.warning("Error fetching CIK for %s: %s", ticker, exc)
            return None

    def fetch_filings(
        self,
        ticker: str,
        filing_types: list[str],
        start: date,
        end: date,
    ) -> list[dict]:
        """Return list of filing metadata dicts.

        Each dict has keys: accession_number, filing_type, filed_date, description, url, cik.
        """
        cik = self.get_cik(ticker)
        if cik is None:
            logger.warning("Cannot fetch filings for %s: no CIK found", ticker)
            return []

        filings: list[dict] = []
        try:
            url = f"{self.SUBMISSIONS_URL}/CIK{cik}.json"
            resp = self._get(url)
            if resp.status_code == 404:
                return []
            resp.raise_for_status()
            data = resp.json()

            recent = data.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            accessions = recent.get("accessionNumber", [])
            dates = recent.get("filingDate", [])
            descriptions = recent.get("primaryDocument", [])

            for form, acc, filed, doc in zip(forms, accessions, dates, descriptions):
                if form not in filing_types:
                    continue
                filed_date = date.fromisoformat(filed)
                if not (start <= filed_date <= end):
                    continue
                acc_clean = acc.replace("-", "")
                doc_url = (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{int(cik)}/{acc_clean}/{doc}"
                )
                filings.append({
                    "accession_number": acc,
                    "filing_type": form,
                    "filed_date": filed_date,
                    "description": doc,
                    "url": doc_url,
                    "cik": cik,
                })
        except Exception as exc:
            logger.warning("Error fetching filings for %s: %s", ticker, exc)

        return filings

    def download_filing_text(self, accession_number: str, cik: str) -> str:
        """Download and return the primary document text (truncated to 200K chars)."""
        acc_clean = accession_number.replace("-", "")
        index_url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{int(cik)}/{acc_clean}/{accession_number}-index.htm"
        )
        try:
            resp = self._get(index_url)
            if resp.status_code == 404:
                logger.warning("Filing not found: %s", accession_number)
                return ""
            resp.raise_for_status()
            text = resp.text
            return text[:200_000]
        except Exception as exc:
            logger.warning("Error downloading filing %s: %s", accession_number, exc)
            return ""

    def write_parquet(self, filings: list[dict], batch_date: date) -> Path:
        """Write filing metadata to Parquet. Text content stored separately."""
        df = pd.DataFrame(filings)
        if "filed_date" in df.columns:
            df["filed_date"] = pd.to_datetime(df["filed_date"]).dt.date
        path = self._parquet_dir / f"sec_filings_{batch_date.isoformat()}.parquet"
        df.to_parquet(path, index=False)
        logger.info("Wrote %d filing records to %s", len(df), path)
        return path

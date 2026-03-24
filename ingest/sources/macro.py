"""Macro data source using the FRED API."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import structlog
from fredapi import Fred
from tenacity import retry, stop_after_attempt, wait_exponential

log = structlog.get_logger()


class MacroDataSource:
    """Downloads macroeconomic time series from FRED and stages to Parquet."""

    SERIES: list[str] = [
        "GDP",
        "CPIAUCSL",
        "UNRATE",
        "FEDFUNDS",
        "DGS10",
        "DGS2",
        "BAA10Y",
        "VIXCLS",
        "M2SL",
        "DCOILWTICO",
    ]

    def __init__(self, fred_api_key: str, parquet_dir: str) -> None:
        self._fred = Fred(api_key=fred_api_key)
        self._parquet_dir = Path(parquet_dir)
        self._parquet_dir.mkdir(parents=True, exist_ok=True)

    def fetch_series(
        self,
        series_ids: list[str],
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Fetch FRED series between start and end.

        Returns a DataFrame with columns:
            series_id, date, release_date, value

        release_date is approximated as today's date (ingestion timestamp) because
        FRED vintage data is not available in the free tier.
        """
        log.info("Fetching FRED macro series", series_count=len(series_ids), start=str(start), end=str(end))
        today = date.today()
        results: list[pd.DataFrame] = []

        for sid in series_ids:
            try:
                series = self._fetch_one(sid, start.isoformat(), end.isoformat())
                if series is None or series.empty:
                    log.warning("No data returned for FRED series", series_id=sid)
                    continue

                frame = series.reset_index()
                frame.columns = ["date", "value"]
                frame["series_id"] = sid
                frame["release_date"] = today
                frame["date"] = pd.to_datetime(frame["date"]).dt.date
                results.append(frame[["series_id", "date", "release_date", "value"]])
                log.info("Fetched FRED series", series_id=sid, rows=len(frame))

            except Exception as exc:
                log.warning("Failed to fetch FRED series", series_id=sid, error=str(exc))
                continue

        if not results:
            return pd.DataFrame(columns=["series_id", "date", "release_date", "value"])

        combined = pd.concat(results, ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"]).dt.date
        combined["release_date"] = pd.to_datetime(combined["release_date"]).dt.date
        log.info("Macro fetch complete", series_fetched=len(results), total_rows=len(combined))
        return combined

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _fetch_one(self, series_id: str, start: str, end: str) -> pd.Series:
        """Fetch a single FRED series with retry logic."""
        return self._fred.get_series(series_id, observation_start=start, observation_end=end)

    def write_parquet(self, df: pd.DataFrame, batch_date: date) -> Path:
        """Write macro DataFrame to a dated Parquet file and return the path."""
        path = self._parquet_dir / f"macro_{batch_date.isoformat()}.parquet"
        df.to_parquet(path, index=False)
        log.info("Wrote macro rows to parquet", rows=len(df), path=str(path))
        return path

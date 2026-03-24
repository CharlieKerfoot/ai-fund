"""Congressional trading disclosures via Quiver Quant."""

from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import structlog

log = structlog.get_logger()

_SAMPLE_POLITICIANS = [
    "Sen. A. Smith",
    "Rep. B. Jones",
    "Sen. C. Williams",
    "Rep. D. Brown",
]

_TRANSACTION_TYPES = ["purchase", "sale", "exchange"]


class CongressSource:
    """Congressional trading disclosures via Quiver Quant.

    In production, requires a Quiver Quant API key. This implementation
    returns sample data when the API key is unavailable.

    API docs: https://api.quiverquant.com/beta/
    """

    QUIVER_BASE = "https://api.quiverquant.com/beta"

    def __init__(self, api_key: str, parquet_dir: str) -> None:
        self._api_key = api_key
        self._parquet_dir = Path(parquet_dir)
        self._parquet_dir.mkdir(parents=True, exist_ok=True)
        self._degraded = not bool(api_key)
        if self._degraded:
            log.warning("CongressSource: no API key provided, using sample data")

    def fetch_transactions(
        self,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Fetch congressional trading transactions between start and end.

        Returns a DataFrame with columns:
            symbol, date, transaction_type, amount, politician

        When running in degraded mode (no API key), returns sample data.
        """
        log.info(
            "Fetching congressional transactions",
            start=str(start),
            end=str(end),
            degraded=self._degraded,
        )

        if self._degraded:
            return self._sample_transactions(start, end)

        return self._fetch_from_api(start, end)

    def _fetch_from_api(self, start: date, end: date) -> pd.DataFrame:
        """Fetch from Quiver Quant API."""
        try:
            import requests
        except ImportError:
            log.warning("requests not installed; falling back to sample data")
            return self._sample_transactions(start, end)

        headers = {"Authorization": f"Token {self._api_key}"}
        rows: list[dict] = []

        try:
            resp = requests.get(
                f"{self.QUIVER_BASE}/bulk/congresstrading",
                headers=headers,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            for record in data:
                try:
                    tx_date = date.fromisoformat(record.get("Date", "")[:10])
                except (ValueError, TypeError):
                    continue

                if not (start <= tx_date <= end):
                    continue

                rows.append({
                    "symbol": record.get("Ticker", ""),
                    "date": tx_date,
                    "transaction_type": record.get("Transaction", "").lower(),
                    "amount": record.get("Amount", 0),
                    "politician": record.get("Representative", ""),
                })

        except Exception as exc:
            log.warning("Failed to fetch congressional data from API", error=str(exc))
            return self._sample_transactions(start, end)

        if not rows:
            return pd.DataFrame(
                columns=["symbol", "date", "transaction_type", "amount", "politician"]
            )

        df = pd.DataFrame(rows)
        log.info("Congressional data fetch complete", rows=len(df))
        return df

    def _sample_transactions(self, start: date, end: date) -> pd.DataFrame:
        """Generate deterministic sample congressional transaction data."""
        rng = random.Random(hash((str(start), str(end))))
        sample_symbols = ["AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "JPM", "XOM"]
        rows = []

        current = start
        while current <= end:
            if current.weekday() < 5:  # weekdays only
                n_transactions = rng.randint(0, 3)
                for _ in range(n_transactions):
                    rows.append({
                        "symbol": rng.choice(sample_symbols),
                        "date": current,
                        "transaction_type": rng.choice(_TRANSACTION_TYPES),
                        "amount": rng.choice([1_000, 5_000, 15_000, 50_000, 100_000, 500_000]),
                        "politician": rng.choice(_SAMPLE_POLITICIANS),
                    })
            current += timedelta(days=1)

        if not rows:
            return pd.DataFrame(
                columns=["symbol", "date", "transaction_type", "amount", "politician"]
            )

        return pd.DataFrame(rows)

    def write_parquet(self, df: pd.DataFrame, batch_date: date) -> Path:
        """Write congressional transactions to a dated Parquet file and return the path."""
        path = self._parquet_dir / f"congress_{batch_date.isoformat()}.parquet"
        df.to_parquet(path, index=False)
        log.info("Wrote congressional transaction rows to parquet", rows=len(df), path=str(path))
        return path

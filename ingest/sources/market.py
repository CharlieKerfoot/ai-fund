"""Market data source using yfinance."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import structlog
import yfinance as yf

log = structlog.get_logger()


class MarketDataSource:
    """Downloads OHLCV market data from Yahoo Finance and stages to Parquet."""

    def __init__(self, parquet_dir: str) -> None:
        self._parquet_dir = Path(parquet_dir)
        self._parquet_dir.mkdir(parents=True, exist_ok=True)

    def fetch_daily(
        self,
        symbols: list[str],
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Download daily OHLCV + adj_close for symbols between start and end (inclusive).

        Returns a DataFrame with columns:
            symbol, date, open, high, low, close, adj_close, volume
        Missing or invalid symbols are skipped with a warning.
        """
        log.info("Fetching market data", symbol_count=len(symbols), start=str(start), end=str(end))
        results: list[pd.DataFrame] = []

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                raw = ticker.history(
                    start=start.isoformat(),
                    end=end.isoformat(),
                    auto_adjust=False,
                    actions=False,
                )

                if raw.empty:
                    log.warning("No data returned for symbol", symbol=symbol)
                    continue

                raw = raw.reset_index()
                raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]

                # Normalize column names: yfinance may return 'adj_close' or 'adj close'
                rename_map: dict[str, str] = {}
                for col in raw.columns:
                    normalized = col.lower().replace(" ", "_")
                    if normalized != col:
                        rename_map[col] = normalized
                if rename_map:
                    raw = raw.rename(columns=rename_map)

                # Ensure adj_close column exists
                if "adj_close" not in raw.columns and "adjclose" in raw.columns:
                    raw = raw.rename(columns={"adjclose": "adj_close"})
                elif "adj_close" not in raw.columns:
                    raw["adj_close"] = raw.get("close", pd.Series(dtype=float))

                raw["symbol"] = symbol
                raw["date"] = pd.to_datetime(raw["date"]).dt.date

                keep = ["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"]
                available = [c for c in keep if c in raw.columns]
                results.append(raw[available])

            except Exception as exc:
                log.warning("Failed to fetch data for symbol", symbol=symbol, error=str(exc))
                continue

        if not results:
            return pd.DataFrame(
                columns=["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"]
            )

        combined = pd.concat(results, ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"]).dt.date
        log.info("Market fetch complete", rows=len(combined), symbols_fetched=len(results))
        return combined

    def write_parquet(self, df: pd.DataFrame, batch_date: date) -> Path:
        """Write price DataFrame to a dated Parquet file and return the path."""
        path = self._parquet_dir / f"prices_{batch_date.isoformat()}.parquet"
        df.to_parquet(path, index=False)
        log.info("Wrote price rows to parquet", rows=len(df), path=str(path))
        return path

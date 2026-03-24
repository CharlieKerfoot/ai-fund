"""Shared pytest fixtures for the AI Fund test suite."""

from __future__ import annotations

from datetime import UTC, date, datetime

import duckdb
import pandas as pd
import pytest

from ingest.store.feature_store import FeatureStore
from ingest.store.schema import create_schema


@pytest.fixture
def in_memory_db() -> duckdb.DuckDBPyConnection:
    """Create an in-memory DuckDB connection with the full schema applied."""
    conn = duckdb.connect(":memory:")
    create_schema(conn)
    return conn


@pytest.fixture
def feature_store(tmp_path) -> FeatureStore:
    """Create a FeatureStore backed by a temp directory."""
    db_path = str(tmp_path / "test.duckdb")
    parquet_dir = str(tmp_path / "parquet")
    store = FeatureStore(db_path=db_path, parquet_dir=parquet_dir)
    yield store
    store.close()


@pytest.fixture
def sample_prices_df() -> pd.DataFrame:
    """Return a 20-row DataFrame with AAPL and MSFT price data."""
    rows = []
    aapl_base = 170.0
    msft_base = 330.0
    now = datetime.now(tz=UTC)

    for i in range(10):
        day = date(2024, 1, i + 2)  # 2024-01-02 through 2024-01-11
        rows.append({
            "symbol": "AAPL",
            "date": day,
            "open": round(aapl_base + i * 0.5, 2),
            "high": round(aapl_base + i * 0.5 + 1.0, 2),
            "low": round(aapl_base + i * 0.5 - 0.5, 2),
            "close": round(aapl_base + i * 0.5 + 0.25, 2),
            "adj_close": round(aapl_base + i * 0.5 + 0.25, 2),
            "volume": 60_000_000 + i * 500_000,
            "ingested_at": now,
        })
        rows.append({
            "symbol": "MSFT",
            "date": day,
            "open": round(msft_base + i * 0.8, 2),
            "high": round(msft_base + i * 0.8 + 1.5, 2),
            "low": round(msft_base + i * 0.8 - 0.7, 2),
            "close": round(msft_base + i * 0.8 + 0.4, 2),
            "adj_close": round(msft_base + i * 0.8 + 0.4, 2),
            "volume": 25_000_000 + i * 300_000,
            "ingested_at": now,
        })

    return pd.DataFrame(rows)


@pytest.fixture
def sample_macro_df() -> pd.DataFrame:
    """Return a 10-row DataFrame with FEDFUNDS and DGS10 macro data."""
    today = date.today()
    rows = []
    for i in range(5):
        obs_date = date(2024, 1, i + 1)
        rows.append({
            "series_id": "FEDFUNDS",
            "date": obs_date,
            "release_date": today,
            "value": round(5.25 + i * 0.01, 4),
        })
        rows.append({
            "series_id": "DGS10",
            "date": obs_date,
            "release_date": today,
            "value": round(4.10 + i * 0.02, 4),
        })
    return pd.DataFrame(rows)

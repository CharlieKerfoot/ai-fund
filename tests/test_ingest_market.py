"""Tests for market data ingestion: validation, deduplication, and gap detection."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import pytest

from ingest.quality.validate import (
    check_gaps,
    dedup_prices,
    validate_prices,
)

# ---------------------------------------------------------------------------
# validate_prices
# ---------------------------------------------------------------------------

def _make_prices(overrides: dict | None = None) -> pd.DataFrame:
    """Build a minimal valid prices DataFrame with optional column overrides."""
    base = pd.DataFrame([
        {
            "symbol": "AAPL",
            "date": date(2024, 1, 2),
            "open": 170.0,
            "high": 172.0,
            "low": 169.0,
            "close": 171.0,
            "adj_close": 171.0,
            "volume": 60_000_000,
        },
        {
            "symbol": "MSFT",
            "date": date(2024, 1, 2),
            "open": 330.0,
            "high": 333.0,
            "low": 329.0,
            "close": 331.5,
            "adj_close": 331.5,
            "volume": 25_000_000,
        },
    ])
    if overrides:
        for col, val in overrides.items():
            base[col] = val
    return base


def test_validate_prices_valid():
    """A valid prices DataFrame should return no errors."""
    df = _make_prices()
    errors = validate_prices(df)
    assert errors == []


def test_validate_prices_null_close():
    """A null close value should produce an error."""
    df = _make_prices()
    df.loc[0, "close"] = None
    errors = validate_prices(df)
    assert any("null" in e.lower() and "close" in e for e in errors)


def test_validate_prices_negative():
    """A negative close price should produce an error."""
    df = _make_prices()
    df.loc[0, "close"] = -5.0
    errors = validate_prices(df)
    assert any("non-positive" in e.lower() or "negative" in e.lower() for e in errors)


def test_validate_prices_negative_volume():
    """Negative volume should produce an error."""
    df = _make_prices()
    df.loc[0, "volume"] = -100
    errors = validate_prices(df)
    assert any("volume" in e.lower() for e in errors)


def test_validate_prices_missing_column():
    """Missing a required column should produce an error."""
    df = _make_prices().drop(columns=["close"])
    errors = validate_prices(df)
    assert any("close" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# dedup_prices
# ---------------------------------------------------------------------------

def test_dedup_prices():
    """Duplicate (symbol, date) rows should be reduced to one, keeping latest ingested_at."""
    early = datetime(2024, 1, 5, tzinfo=UTC)
    late = datetime(2024, 3, 1, tzinfo=UTC)

    df = pd.DataFrame([
        {
            "symbol": "AAPL",
            "date": date(2024, 1, 3),
            "close": 170.0,
            "ingested_at": early,
        },
        {
            "symbol": "AAPL",
            "date": date(2024, 1, 3),
            "close": 171.5,  # updated close
            "ingested_at": late,
        },
        {
            "symbol": "MSFT",
            "date": date(2024, 1, 3),
            "close": 330.0,
            "ingested_at": early,
        },
    ])

    result = dedup_prices(df)

    assert len(result) == 2
    aapl_row = result[result["symbol"] == "AAPL"]
    assert len(aapl_row) == 1
    assert float(aapl_row.iloc[0]["close"]) == pytest.approx(171.5)


def test_dedup_prices_no_ingested_at():
    """dedup_prices should still work when ingested_at is absent (keep='last')."""
    df = pd.DataFrame([
        {"symbol": "AAPL", "date": date(2024, 1, 3), "close": 170.0},
        {"symbol": "AAPL", "date": date(2024, 1, 3), "close": 171.5},
    ])

    result = dedup_prices(df)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# check_gaps
# ---------------------------------------------------------------------------

def test_check_gaps_no_gap():
    """A contiguous business-day series should produce no errors."""
    dates = pd.bdate_range(start="2024-01-02", end="2024-01-12")
    df = pd.DataFrame({
        "symbol": "AAPL",
        "date": dates,
        "close": 170.0,
    })
    errors = check_gaps(df, freq="B")
    assert errors == []


def test_check_gaps():
    """A symbol missing 6+ consecutive business days should trigger an error."""
    # Jan 2 through Jan 5 (4 days), then skip Jan 8–15 (6 business days), resume Jan 16
    early = pd.bdate_range(start="2024-01-02", end="2024-01-05")
    late = pd.bdate_range(start="2024-01-16", end="2024-01-19")
    all_dates = list(early) + list(late)

    df = pd.DataFrame({
        "symbol": "AAPL",
        "date": all_dates,
        "close": 170.0,
    })

    errors = check_gaps(df, freq="B")
    assert len(errors) >= 1
    assert any("AAPL" in e for e in errors)


def test_check_gaps_small_gap_ok():
    """A gap of exactly 5 consecutive business days should not exceed the threshold."""
    # Jan 2, Jan 3, then skip Jan 4-8 (5 business days Mon-Fri), resume Jan 9
    early = pd.bdate_range(start="2024-01-02", end="2024-01-03")
    late = pd.bdate_range(start="2024-01-09", end="2024-01-12")
    all_dates = list(early) + list(late)

    df = pd.DataFrame({
        "symbol": "AAPL",
        "date": all_dates,
        "close": 170.0,
    })

    errors = check_gaps(df, freq="B")
    # 5 consecutive days is exactly the threshold — should NOT be an error
    assert errors == []

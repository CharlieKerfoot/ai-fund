"""Tests for macro data ingestion: validation and source fetching."""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pandas as pd

from ingest.quality.validate import validate_macro

# ---------------------------------------------------------------------------
# validate_macro
# ---------------------------------------------------------------------------

def _make_macro(overrides: dict | None = None) -> pd.DataFrame:
    """Build a minimal valid macro DataFrame with optional column overrides."""
    base = pd.DataFrame([
        {
            "series_id": "FEDFUNDS",
            "date": date(2024, 1, 1),
            "release_date": date(2024, 1, 10),
            "value": 5.25,
        },
        {
            "series_id": "DGS10",
            "date": date(2024, 1, 1),
            "release_date": date(2024, 1, 10),
            "value": 4.10,
        },
    ])
    if overrides:
        for col, val in overrides.items():
            base[col] = val
    return base


def test_validate_macro_valid():
    """A valid macro DataFrame should return no errors."""
    df = _make_macro()
    errors = validate_macro(df)
    assert errors == []


def test_validate_macro_null_value():
    """A null value in the 'value' column should produce an error."""
    df = _make_macro()
    df.loc[0, "value"] = None
    errors = validate_macro(df)
    assert any("null" in e.lower() and "value" in e for e in errors)


def test_validate_macro_null_series_id():
    """A null series_id should produce an error."""
    df = _make_macro()
    df.loc[0, "series_id"] = None
    errors = validate_macro(df)
    assert any("series_id" in e.lower() for e in errors)


def test_validate_macro_missing_column():
    """A missing required column should produce an error."""
    df = _make_macro().drop(columns=["value"])
    errors = validate_macro(df)
    assert any("value" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# MacroDataSource (mocked FRED)
# ---------------------------------------------------------------------------

def test_macro_source_fetch_series():
    """MacroDataSource.fetch_series should return correctly shaped DataFrame."""
    from ingest.sources.macro import MacroDataSource

    mock_series = pd.Series(
        [5.25, 5.30, 5.35],
        index=pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
        name="FEDFUNDS",
    )

    with patch("ingest.sources.macro.Fred") as mock_fred_cls:
        instance = mock_fred_cls.return_value
        instance.get_series.return_value = mock_series

        source = MacroDataSource(fred_api_key="test_key", parquet_dir="/tmp/test_macro")
        result = source.fetch_series(
            series_ids=["FEDFUNDS"],
            start=date(2024, 1, 1),
            end=date(2024, 3, 31),
        )

    assert not result.empty
    assert set(result.columns) >= {"series_id", "date", "release_date", "value"}
    assert (result["series_id"] == "FEDFUNDS").all()
    assert len(result) == 3


def test_macro_source_skips_failed_series():
    """MacroDataSource.fetch_series should skip series that raise exceptions."""
    from ingest.sources.macro import MacroDataSource

    with patch("ingest.sources.macro.Fred") as mock_fred_cls2:
        instance = mock_fred_cls2.return_value
        instance.get_series.side_effect = Exception("API unavailable")

        source = MacroDataSource(fred_api_key="test_key", parquet_dir="/tmp/test_macro_skip")
        result = source.fetch_series(
            series_ids=["FEDFUNDS"],
            start=date(2024, 1, 1),
            end=date(2024, 3, 31),
        )

    assert result.empty


def test_macro_source_write_parquet(tmp_path):
    """MacroDataSource.write_parquet should write a readable Parquet file."""
    from ingest.sources.macro import MacroDataSource

    with patch("ingest.sources.macro.Fred"):
        source = MacroDataSource(fred_api_key="test_key", parquet_dir=str(tmp_path))

    df = _make_macro()
    batch_date = date(2024, 1, 15)
    path = source.write_parquet(df, batch_date)

    assert path.exists()
    loaded = pd.read_parquet(path)
    assert len(loaded) == len(df)

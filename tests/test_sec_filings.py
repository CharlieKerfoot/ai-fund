"""Tests for SECFilingsSource."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest
import requests

from ingest.sources.sec_filings import SECFilingsSource

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sec_source(tmp_path):
    """SECFilingsSource with a temp parquet directory."""
    return SECFilingsSource(
        user_agent="Test Fund test@example.com",
        parquet_dir=str(tmp_path / "parquet"),
        rate_limit=100.0,  # High rate limit for tests
    )


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _mock_response(status_code: int, data: dict | list | None = None, text: str = "") -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.text = text
    if data is not None:
        resp.json.return_value = data
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(
            response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


_SUBMISSION_RESPONSE = {
    "cik": "320193",
    "name": "Apple Inc.",
    "filings": {
        "recent": {
            "form": ["10-K", "10-Q", "8-K"],
            "accessionNumber": ["0000320193-24-000001", "0000320193-24-000002", "0000320193-24-000003"],
            "filingDate": ["2024-01-15", "2024-04-20", "2024-02-01"],
            "primaryDocument": ["aapl10k.htm", "aapl10q.htm", "aapl8k.htm"],
        }
    },
}

_CIK_RESPONSE = {
    "cik": "320193",
    "name": "Apple Inc.",
    "filings": {"recent": {"form": [], "accessionNumber": [], "filingDate": [], "primaryDocument": []}},
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_get_cik_mock(sec_source: SECFilingsSource):
    """Mock requests; verify CIK parsing."""
    with patch.object(sec_source, "_get") as mock_get:
        mock_get.return_value = _mock_response(200, _CIK_RESPONSE)
        cik = sec_source.get_cik("AAPL")

    assert cik is not None
    assert cik == "0000320193"
    # CIK should be zero-padded to 10 digits
    assert len(cik) == 10


def test_get_cik_not_found(sec_source: SECFilingsSource):
    """404 response returns None."""
    with patch.object(sec_source, "_get") as mock_get:
        mock_get.return_value = _mock_response(404)
        cik = sec_source.get_cik("FAKEXYZ")

    assert cik is None


def test_get_cik_error(sec_source: SECFilingsSource):
    """Network error returns None (graceful degradation)."""
    with patch.object(sec_source, "_get") as mock_get:
        mock_get.side_effect = Exception("Connection refused")
        cik = sec_source.get_cik("AAPL")

    assert cik is None


def test_fetch_filings_mock(sec_source: SECFilingsSource):
    """Mock EDGAR API response; verify filing list parsed correctly."""
    with patch.object(sec_source, "_get") as mock_get:
        # First call: get_cik; second call: fetch_filings (submissions)
        mock_get.side_effect = [
            _mock_response(200, _CIK_RESPONSE),  # get_cik
            _mock_response(200, _SUBMISSION_RESPONSE),  # fetch_filings
        ]

        filings = sec_source.fetch_filings(
            ticker="AAPL",
            filing_types=["10-K", "10-Q"],
            start=date(2024, 1, 1),
            end=date(2024, 12, 31),
        )

    assert len(filings) == 2  # 10-K and 10-Q, not the 8-K

    forms = {f["filing_type"] for f in filings}
    assert "10-K" in forms
    assert "10-Q" in forms
    assert "8-K" not in forms

    # Verify required keys
    for filing in filings:
        assert "accession_number" in filing
        assert "filing_type" in filing
        assert "filed_date" in filing
        assert "description" in filing
        assert "url" in filing
        assert "cik" in filing


def test_fetch_filings_date_filter(sec_source: SECFilingsSource):
    """Only filings within the date range are returned."""
    with patch.object(sec_source, "_get") as mock_get:
        mock_get.side_effect = [
            _mock_response(200, _CIK_RESPONSE),
            _mock_response(200, _SUBMISSION_RESPONSE),
        ]

        filings = sec_source.fetch_filings(
            ticker="AAPL",
            filing_types=["10-K", "10-Q", "8-K"],
            start=date(2024, 4, 1),  # After the 10-K filing date
            end=date(2024, 12, 31),
        )

    # Only filings from April onwards
    for filing in filings:
        assert filing["filed_date"] >= date(2024, 4, 1)


def test_fetch_filings_no_cik(sec_source: SECFilingsSource):
    """Returns empty list when CIK not found."""
    with patch.object(sec_source, "get_cik") as mock_cik:
        mock_cik.return_value = None
        filings = sec_source.fetch_filings(
            ticker="FAKEXYZ",
            filing_types=["10-K"],
            start=date(2024, 1, 1),
            end=date(2024, 12, 31),
        )

    assert filings == []


def test_write_parquet(sec_source: SECFilingsSource, tmp_path):
    """write_parquet creates a valid Parquet file."""
    import pandas as pd

    filings = [
        {
            "accession_number": "0000320193-24-000001",
            "filing_type": "10-K",
            "filed_date": date(2024, 1, 15),
            "description": "aapl10k.htm",
            "url": "https://www.sec.gov/Archives/edgar/data/320193/...",
            "cik": "0000320193",
        }
    ]

    path = sec_source.write_parquet(filings, date(2024, 1, 15))
    assert path.exists()

    df = pd.read_parquet(path)
    assert len(df) == 1
    assert df.iloc[0]["filing_type"] == "10-K"

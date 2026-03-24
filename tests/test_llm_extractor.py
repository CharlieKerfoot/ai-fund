"""Tests for LLMExtractor."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from ingest.extractors.llm_extractor import EarningsFeatures, FilingFeatures, LLMExtractor

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_extractor(daily_budget: float = 5.0) -> LLMExtractor:
    return LLMExtractor(api_key="test-key", model="claude-opus-4-6", daily_budget_usd=daily_budget)


def _make_message_response(content: str, input_tokens: int = 100, output_tokens: int = 50):
    """Build a mock anthropic Messages response."""
    msg = MagicMock()
    msg.content = [MagicMock(text=content)]
    msg.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    return msg


_EARNINGS_JSON = json.dumps({
    "sentiment_score": 0.6,
    "guidance_direction": "up",
    "management_confidence": 0.8,
    "key_risks": ["supply chain", "competition"],
    "hedging_language_count": 3,
    "forward_looking_positive": 5,
    "forward_looking_negative": 1,
})

_FILING_JSON = json.dumps({
    "risk_factor_delta": "increased",
    "new_risk_factors": ["cybersecurity breach", "regulatory change"],
    "accounting_changes": ["adopted ASC 842"],
    "litigation_exposure": "medium",
    "going_concern": False,
    "revenue_trend": "growing",
})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_extract_earnings_success():
    """Mock Claude API response with valid JSON; verify EarningsFeatures parsed correctly."""
    extractor = _make_extractor()

    mock_response = _make_message_response(_EARNINGS_JSON)

    with patch.object(extractor, "_get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = extractor.extract_earnings(
            text="We had a great quarter with strong revenue growth.",
            ticker="AAPL",
            quarter="Q1 2024",
        )

    assert result is not None
    assert isinstance(result, EarningsFeatures)
    assert result.sentiment_score == pytest.approx(0.6)
    assert result.guidance_direction == "up"
    assert result.management_confidence == pytest.approx(0.8)
    assert "supply chain" in result.key_risks
    assert result.hedging_language_count == 3
    assert result.forward_looking_positive == 5
    assert result.forward_looking_negative == 1


def test_extract_earnings_api_error():
    """Mock API raising exception; verify returns None (degraded mode)."""
    extractor = _make_extractor()

    with patch.object(extractor, "_get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("Connection error")
        mock_get_client.return_value = mock_client

        result = extractor.extract_earnings(
            text="Revenue grew significantly this quarter.",
            ticker="MSFT",
            quarter="Q2 2024",
        )

    assert result is None


def test_budget_tracking():
    """Verify cost accumulates and check_budget() returns False when exceeded."""
    # Very small budget: $0.001
    extractor = _make_extractor(daily_budget=0.001)

    mock_response = _make_message_response(
        _EARNINGS_JSON,
        input_tokens=100_000,  # 100K input tokens
        output_tokens=10_000,   # 10K output tokens
    )

    assert extractor.check_budget() is True

    with patch.object(extractor, "_get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        extractor.extract_earnings(
            text="Large text that will cost a lot.",
            ticker="GOOG",
            quarter="Q3 2024",
        )

    # After 100K input + 10K output tokens:
    # cost = (100_000 / 1_000_000) * 15.0 + (10_000 / 1_000_000) * 75.0
    # cost = 1.5 + 0.75 = 2.25 USD >> 0.001 USD budget
    assert extractor.get_cost_today() > 0.001
    assert extractor.check_budget() is False


def test_chunk_text():
    """Verify long text gets chunked at paragraph boundaries."""
    extractor = _make_extractor()

    # Create text with clear paragraph boundaries
    paragraphs = [f"Paragraph {i}.\n\nSome content here for paragraph {i}." for i in range(20)]
    long_text = "\n\n".join(paragraphs)

    chunks = extractor._chunk_text(long_text, max_chars=200)

    assert len(chunks) > 1
    # Each chunk should be <= max_chars (with some tolerance for boundary logic)
    for chunk in chunks:
        assert len(chunk) <= 200 + 50  # small tolerance for boundary overshoot

    # Reassembled content should cover the original text
    reassembled = " ".join(chunks)
    for i in range(20):
        assert f"Paragraph {i}" in reassembled


def test_chunk_text_short():
    """Short text should return single chunk."""
    extractor = _make_extractor()
    text = "Short text."
    chunks = extractor._chunk_text(text, max_chars=80000)
    assert chunks == [text]


def test_extract_filing_success():
    """Mock Claude API for filing extraction; verify FilingFeatures parsed correctly."""
    extractor = _make_extractor()

    mock_response = _make_message_response(_FILING_JSON)

    with patch.object(extractor, "_get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = extractor.extract_filing(
            text="This annual report describes our business risks.",
            ticker="TSLA",
            filing_type="10-K",
        )

    assert result is not None
    assert isinstance(result, FilingFeatures)
    assert result.risk_factor_delta == "increased"
    assert "cybersecurity breach" in result.new_risk_factors
    assert "adopted ASC 842" in result.accounting_changes
    assert result.litigation_exposure == "medium"
    assert result.going_concern is False
    assert result.revenue_trend == "growing"

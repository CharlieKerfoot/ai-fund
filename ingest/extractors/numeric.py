"""Numeric feature extraction from financial filings and earnings text."""

from __future__ import annotations

import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)

# Regex patterns for financial figures
_EPS_PATTERNS = [
    # "$1.23 per share", "$1.23 per diluted share"
    r"\$\s*([\d,]+\.?\d*)\s+per\s+(?:diluted\s+)?share",
    # "EPS of $1.23", "EPS: $1.23"
    r"\bEPS\s*(?:of|:)?\s*\$\s*([\d,]+\.?\d*)",
    # "earnings per share of $1.23"
    r"earnings\s+per\s+(?:diluted\s+)?share\s+(?:of\s+|was\s+|were\s+)?\$?\s*([\d,]+\.?\d*)",
    # "(loss) per share of ($0.45)" -- negative
    r"loss\s+per\s+(?:diluted\s+)?share\s+(?:of\s+)?\(\$?\s*([\d,]+\.?\d*)\)",
]

_REVENUE_PATTERNS = [
    # "revenue of $1.23 billion", "revenues of $456 million"
    r"revenues?\s+(?:of\s+|were\s+|was\s+|totaled?\s+)?\$\s*([\d,]+\.?\d*)\s*(billion|million|B|M)\b",
    # "$1.23 billion in revenue"
    r"\$\s*([\d,]+\.?\d*)\s*(billion|million|B|M)\s+in\s+revenues?",
    # "net revenue: $456M"
    r"net\s+revenues?\s*[:\-]\s*\$\s*([\d,]+\.?\d*)\s*(billion|million|B|M)\b",
]


def parse_financial_tables(html: str) -> dict[str, pd.DataFrame]:
    """Extract financial tables from HTML filing.

    Returns a dict mapping a table index label to a DataFrame.
    Tables are labeled as "table_0", "table_1", etc.
    """
    try:
        tables = pd.read_html(html, flavor="lxml")
    except Exception:
        try:
            tables = pd.read_html(html)
        except Exception as exc:
            logger.warning("Failed to parse HTML tables: %s", exc)
            return {}

    result: dict[str, pd.DataFrame] = {}
    for i, df in enumerate(tables):
        if df.empty:
            continue
        # Clean up column names
        df.columns = [str(c).strip() for c in df.columns]
        result[f"table_{i}"] = df
    return result


def extract_eps(text: str) -> float | None:
    """Extract EPS value from earnings text using regex.

    Returns the first matched EPS value as a float (may be negative for losses).
    Returns None if no match found.
    """
    text_lower = text.lower()

    # Check for loss patterns first
    loss_match = re.search(
        r"loss\s+per\s+(?:diluted\s+)?share\s+(?:of\s+)?\(?\$?\s*([\d,]+\.?\d*)\)?",
        text_lower,
        re.IGNORECASE,
    )
    if loss_match:
        try:
            value = float(loss_match.group(1).replace(",", ""))
            return -value
        except (ValueError, AttributeError):
            pass

    for pattern in _EPS_PATTERNS[:-1]:  # skip the loss pattern already handled
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                value = float(match.group(1).replace(",", ""))
                return value
            except (ValueError, AttributeError):
                continue

    return None


def extract_revenue(text: str) -> float | None:
    """Extract revenue figure (in millions) from text using regex.

    Returns the value in millions (e.g., $1.2B -> 1200.0).
    Returns None if no match found.
    """
    for pattern in _REVENUE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                value = float(match.group(1).replace(",", ""))
                unit = match.group(2).lower()
                if unit in ("billion", "b"):
                    value *= 1000.0
                # "million" or "m" -> already in millions
                return value
            except (ValueError, AttributeError, IndexError):
                continue

    return None

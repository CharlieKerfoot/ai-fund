"""Data quality validation helpers for ingested price and macro data."""

from __future__ import annotations

import pandas as pd


def validate_prices(df: pd.DataFrame) -> list[str]:
    """Validate a prices DataFrame. Returns a list of error messages (empty = valid)."""
    errors: list[str] = []

    required_cols = ["symbol", "date", "close", "volume"]
    for col in required_cols:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")

    if errors:
        # Cannot proceed with further checks if columns are missing
        return errors

    null_close = df["close"].isna().sum()
    if null_close > 0:
        errors.append(f"Found {null_close} null value(s) in 'close' column")

    null_symbol = df["symbol"].isna().sum()
    if null_symbol > 0:
        errors.append(f"Found {null_symbol} null value(s) in 'symbol' column")

    null_date = df["date"].isna().sum()
    if null_date > 0:
        errors.append(f"Found {null_date} null value(s) in 'date' column")

    non_positive_close = (df["close"].dropna() <= 0).sum()
    if non_positive_close > 0:
        errors.append(
            f"Found {non_positive_close} row(s) with non-positive 'close' price"
        )

    negative_volume = (df["volume"].dropna() < 0).sum()
    if negative_volume > 0:
        errors.append(f"Found {negative_volume} row(s) with negative 'volume'")

    return errors


def validate_macro(df: pd.DataFrame) -> list[str]:
    """Validate a macro DataFrame. Returns a list of error messages (empty = valid)."""
    errors: list[str] = []

    required_cols = ["series_id", "date", "value"]
    for col in required_cols:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")

    if errors:
        return errors

    null_value = df["value"].isna().sum()
    if null_value > 0:
        errors.append(f"Found {null_value} null value(s) in 'value' column")

    null_series = df["series_id"].isna().sum()
    if null_series > 0:
        errors.append(f"Found {null_series} null value(s) in 'series_id' column")

    null_date = df["date"].isna().sum()
    if null_date > 0:
        errors.append(f"Found {null_date} null value(s) in 'date' column")

    return errors


def check_gaps(df: pd.DataFrame, freq: str = "B") -> list[str]:
    """Check for unexpected gaps in time series data.

    Flags any symbol with more than 5 consecutive missing business days.
    Returns a list of error messages.
    """
    errors: list[str] = []

    if df.empty or "date" not in df.columns:
        return errors

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if "symbol" not in df.columns:
        # Treat as a single series
        _check_series_gaps(df["date"], "series", freq, errors, max_consecutive=5)
        return errors

    for symbol, group in df.groupby("symbol"):
        _check_series_gaps(group["date"], str(symbol), freq, errors, max_consecutive=5)

    return errors


def _check_series_gaps(
    dates: pd.Series,
    label: str,
    freq: str,
    errors: list[str],
    max_consecutive: int,
) -> None:
    """Check a series of dates for gaps exceeding max_consecutive business days."""
    if dates.empty:
        return

    dates_sorted = dates.sort_values().drop_duplicates()
    if len(dates_sorted) < 2:
        return

    full_range = pd.bdate_range(start=dates_sorted.iloc[0], end=dates_sorted.iloc[-1], freq=freq)
    if full_range.empty:
        return

    present = set(dates_sorted.dt.normalize())
    consecutive = 0
    max_gap = 0

    for bday in full_range:
        if bday not in present:
            consecutive += 1
            max_gap = max(max_gap, consecutive)
        else:
            consecutive = 0

    if max_gap > max_consecutive:
        errors.append(
            f"Symbol '{label}' has a gap of {max_gap} consecutive missing business days"
            f" (threshold: {max_consecutive})"
        )


def dedup_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate (symbol, date) rows, keeping the row with the latest ingested_at."""
    if df.empty:
        return df

    if "ingested_at" not in df.columns:
        return df.drop_duplicates(subset=["symbol", "date"], keep="last").reset_index(drop=True)

    df = df.copy()
    df["ingested_at"] = pd.to_datetime(df["ingested_at"])
    df = df.sort_values("ingested_at", ascending=True)
    df = df.drop_duplicates(subset=["symbol", "date"], keep="last")
    return df.reset_index(drop=True)

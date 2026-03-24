"""Point-in-time query helpers for the DuckDB feature store."""

from __future__ import annotations

from datetime import date
from typing import Any

import duckdb
import pandas as pd


def as_of_prices(
    conn: duckdb.DuckDBPyConnection,
    symbols: list[str],
    start: date,
    end: date,
    as_of_date: date,
) -> pd.DataFrame:
    """Return price rows for symbols in [start, end] that were ingested on or before as_of_date.

    Enforces point-in-time correctness by filtering on ingested_at.
    """
    placeholders = ", ".join("?" for _ in symbols)
    sql = f"""
        SELECT symbol, date, open, high, low, close, adj_close, volume, source, ingested_at
        FROM prices
        WHERE symbol IN ({placeholders})
          AND date >= ?
          AND date <= ?
          AND ingested_at <= CAST(? AS TIMESTAMP)
        ORDER BY symbol, date
    """
    params = symbols + [start, end, as_of_date]
    return conn.execute(sql, params).df()


def as_of_macro(
    conn: duckdb.DuckDBPyConnection,
    series_ids: list[str],
    start: date,
    end: date,
    as_of_date: date,
) -> pd.DataFrame:
    """Return macro rows for series_ids in [start, end] released on or before as_of_date.

    Enforces point-in-time correctness via release_date.
    """
    placeholders = ", ".join("?" for _ in series_ids)
    sql = f"""
        SELECT series_id, date, release_date, value
        FROM macro
        WHERE series_id IN ({placeholders})
          AND date >= ?
          AND date <= ?
          AND release_date <= CAST(? AS DATE)
        ORDER BY series_id, date
    """
    params = series_ids + [start, end, as_of_date]
    return conn.execute(sql, params).df()


def latest_value(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    key_col: str,
    value_col: str,
    as_of_date: date,
) -> dict[str, Any]:
    """Return a mapping of key -> most-recent value for each key as of as_of_date.

    Assumes the table has a `date` column. Looks for the latest date <= as_of_date
    per distinct key value.
    """
    sql = f"""
        WITH ranked AS (
            SELECT
                {key_col},
                {value_col},
                date,
                ROW_NUMBER() OVER (
                    PARTITION BY {key_col}
                    ORDER BY date DESC
                ) AS rn
            FROM {table}
            WHERE date <= CAST(? AS DATE)
        )
        SELECT {key_col}, {value_col}
        FROM ranked
        WHERE rn = 1
        ORDER BY {key_col}
    """
    rows = conn.execute(sql, [as_of_date]).fetchall()
    return {row[0]: row[1] for row in rows}

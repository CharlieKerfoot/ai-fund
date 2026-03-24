"""Feature store backed by DuckDB with Parquet staging."""

from __future__ import annotations

import logging
from datetime import UTC, date
from pathlib import Path

import duckdb
import pandas as pd

from ingest.store.schema import create_schema

logger = logging.getLogger(__name__)


class FeatureStore:
    """Persistent feature store using DuckDB with Parquet staging."""

    def __init__(self, db_path: str, parquet_dir: str) -> None:
        self._db_path = db_path
        self._parquet_dir = Path(parquet_dir)
        self._parquet_dir.mkdir(parents=True, exist_ok=True)

        if db_path == ":memory:":
            self._conn = duckdb.connect(":memory:")
        else:
            db_file = Path(db_path)
            db_file.parent.mkdir(parents=True, exist_ok=True)
            self._conn = duckdb.connect(str(db_file))

        create_schema(self._conn)

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def query_prices(
        self,
        symbols: list[str],
        start: date,
        end: date,
        as_of: date,
    ) -> pd.DataFrame:
        """Return OHLCV prices for symbols between start/end, as available as of as_of."""
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
        params = symbols + [start, end, as_of]
        return self._conn.execute(sql, params).df()

    def query_macro(
        self,
        series_ids: list[str],
        start: date,
        end: date,
        as_of: date,
    ) -> pd.DataFrame:
        """Return macro series between start/end, as available as of as_of (release_date)."""
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
        params = series_ids + [start, end, as_of]
        return self._conn.execute(sql, params).df()

    # ------------------------------------------------------------------
    # Insert methods
    # ------------------------------------------------------------------

    def insert_prices(self, df: pd.DataFrame, source: str) -> None:
        """Insert or replace price rows from df into the prices table."""
        if df.empty:
            return

        work = df.copy()
        work["source"] = source
        if "ingested_at" not in work.columns:
            work["ingested_at"] = pd.Timestamp.utcnow()

        # Ensure correct dtypes
        work["date"] = pd.to_datetime(work["date"]).dt.date
        work["ingested_at"] = pd.to_datetime(work["ingested_at"])

        # Register the DataFrame and upsert
        self._conn.register("_prices_staging", work)
        self._conn.execute("""
            INSERT OR REPLACE INTO prices
                (symbol, date, open, high, low, close, adj_close, volume, source, ingested_at)
            SELECT symbol, date, open, high, low, close, adj_close, volume, source, ingested_at
            FROM _prices_staging
        """)
        self._conn.unregister("_prices_staging")

    def insert_macro(self, df: pd.DataFrame) -> None:
        """Insert or replace macro rows from df into the macro table."""
        if df.empty:
            return

        work = df.copy()
        work["date"] = pd.to_datetime(work["date"]).dt.date
        work["release_date"] = pd.to_datetime(work["release_date"]).dt.date

        self._conn.register("_macro_staging", work)
        self._conn.execute("""
            INSERT OR REPLACE INTO macro (series_id, date, release_date, value)
            SELECT series_id, date, release_date, value
            FROM _macro_staging
        """)
        self._conn.unregister("_macro_staging")

    def insert_signals(
        self,
        signal_id: str,
        df: pd.DataFrame,
        version: str,
        metadata: dict,
    ) -> None:
        """Insert or replace signal rows for a given signal_id."""
        if df.empty:
            return

        import json

        work = df.copy()
        work["signal_id"] = signal_id
        work["signal_version"] = version
        work["metadata"] = json.dumps(metadata)
        work["date"] = pd.to_datetime(work["date"]).dt.date

        self._conn.register("_signals_staging", work)
        self._conn.execute("""
            INSERT OR REPLACE INTO signals
                (signal_id, symbol, date, value, confidence, signal_version, metadata)
            SELECT signal_id, symbol, date, value, confidence, signal_version, metadata
            FROM _signals_staging
        """)
        self._conn.unregister("_signals_staging")

    # ------------------------------------------------------------------
    # Text feature insert / update methods
    # ------------------------------------------------------------------

    def insert_text_features(self, records: list[dict]) -> None:
        """Batch insert records into text_features table."""
        if not records:
            return

        import json as _json

        for record in records:
            feature_json = record.get("feature_json")
            if feature_json is not None and not isinstance(feature_json, str):
                feature_json = _json.dumps(feature_json)

            self._conn.execute(
                """
                INSERT OR REPLACE INTO text_features
                    (id, symbol, date, source_type, source_id, feature_type,
                     feature_value, feature_json, raw_text_hash, extraction_status,
                     retry_count, extracted_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    record.get("id"),
                    record.get("symbol"),
                    record.get("date"),
                    record.get("source_type"),
                    record.get("source_id"),
                    record.get("feature_type"),
                    record.get("feature_value"),
                    feature_json,
                    record.get("raw_text_hash"),
                    record.get("extraction_status", "pending"),
                    record.get("retry_count", 0),
                    record.get("extracted_at"),
                ],
            )

    def get_pending_extractions(self, source_type: str, limit: int = 50) -> list[dict]:
        """Return pending or retryable text_features rows for source_type."""
        sql = """
            SELECT *
            FROM text_features
            WHERE source_type = ?
              AND (extraction_status = 'pending'
                   OR (extraction_status = 'failed' AND retry_count < 3))
            ORDER BY date
            LIMIT ?
        """
        result = self._conn.execute(sql, [source_type, limit]).fetchdf()
        return result.to_dict(orient="records")

    def mark_extraction_complete(self, record_id: str, features: dict) -> None:
        """Update text_features row to complete with extracted feature data."""
        import json as _json
        from datetime import datetime

        feature_json = _json.dumps(features)
        extracted_at = datetime.now(tz=UTC)

        self._conn.execute(
            """
            UPDATE text_features
            SET extraction_status = 'complete',
                feature_json = ?,
                extracted_at = ?
            WHERE id = ?
            """,
            [feature_json, extracted_at, record_id],
        )

    def mark_extraction_failed(self, record_id: str, error: str) -> None:
        """Increment retry_count; set status to 'failed' if retry_count >= 3."""
        row = self._conn.execute(
            "SELECT retry_count FROM text_features WHERE id = ?",
            [record_id],
        ).fetchone()

        if row is None:
            logger.warning("mark_extraction_failed: record_id %s not found", record_id)
            return

        new_count = (row[0] or 0) + 1
        new_status = "failed" if new_count >= 3 else "pending"

        self._conn.execute(
            """
            UPDATE text_features
            SET retry_count = ?,
                extraction_status = ?
            WHERE id = ?
            """,
            [new_count, new_status, record_id],
        )
        logger.warning(
            "Extraction failed for record %s (attempt %d): %s",
            record_id,
            new_count,
            error,
        )

    # ------------------------------------------------------------------
    # Unprocessed text features (legacy)
    # ------------------------------------------------------------------

    def get_unprocessed(self, source_type: str) -> list[dict]:
        """Return text_features rows with extraction_status='pending' for source_type."""
        sql = """
            SELECT *
            FROM text_features
            WHERE source_type = ?
              AND extraction_status = 'pending'
            ORDER BY date
        """
        result = self._conn.execute(sql, [source_type]).fetchdf()
        return result.to_dict(orient="records")

    # ------------------------------------------------------------------
    # Parquet merge
    # ------------------------------------------------------------------

    def merge_parquet(self, table: str, parquet_path: str) -> None:
        """Load a Parquet file into a DuckDB table via INSERT OR REPLACE."""
        sql = f"""
            INSERT OR REPLACE INTO {table}
            SELECT * FROM read_parquet(?)
        """
        self._conn.execute(sql, [str(parquet_path)])

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying DuckDB connection."""
        self._conn.close()

"""Pipeline monitoring via the pipeline_runs DuckDB table."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, date, datetime
from typing import Any

import duckdb

logger = logging.getLogger(__name__)


class PipelineMonitor:
    """Records and queries pipeline execution state in the pipeline_runs table."""

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._conn = conn

    def start_stage(
        self,
        stage: str,
        run_date: date,
        metadata: dict | None = None,
    ) -> str:
        """Insert a new pipeline_runs row with status='running'. Returns the run_id."""
        run_id = str(uuid.uuid4())
        started_at = datetime.now(tz=UTC)
        meta_json = json.dumps(metadata or {})

        self._conn.execute(
            """
            INSERT INTO pipeline_runs
                (run_id, date, stage, status, started_at, completed_at, error_message, metadata)
            VALUES (?, ?, ?, 'running', ?, NULL, NULL, ?)
            """,
            [run_id, run_date, stage, started_at, meta_json],
        )
        logger.info("Pipeline stage '%s' started — run_id=%s", stage, run_id)
        return run_id

    def complete_stage(self, run_id: str) -> None:
        """Mark a pipeline run as completed."""
        completed_at = datetime.now(tz=UTC)
        self._conn.execute(
            """
            UPDATE pipeline_runs
            SET status = 'complete', completed_at = ?
            WHERE run_id = ?
            """,
            [completed_at, run_id],
        )
        logger.info("Pipeline run %s marked complete", run_id)

    def fail_stage(self, run_id: str, error: str) -> None:
        """Mark a pipeline run as failed with an error message."""
        completed_at = datetime.now(tz=UTC)
        self._conn.execute(
            """
            UPDATE pipeline_runs
            SET status = 'failed', completed_at = ?, error_message = ?
            WHERE run_id = ?
            """,
            [completed_at, error, run_id],
        )
        logger.warning("Pipeline run %s marked failed: %s", run_id, error)

    def is_complete(self, stage: str, run_date: date) -> bool:
        """Return True if stage completed successfully for the given date."""
        row = self._conn.execute(
            """
            SELECT COUNT(*) FROM pipeline_runs
            WHERE stage = ?
              AND date = ?
              AND status = 'complete'
            """,
            [stage, run_date],
        ).fetchone()
        return bool(row and row[0] > 0)

    def get_last_run(self, stage: str) -> dict[str, Any] | None:
        """Return the most recent pipeline_runs row for a stage, or None."""
        row = self._conn.execute(
            """
            SELECT run_id, date, stage, status, started_at, completed_at,
                   error_message, metadata
            FROM pipeline_runs
            WHERE stage = ?
            ORDER BY started_at DESC
            LIMIT 1
            """,
            [stage],
        ).fetchone()

        if row is None:
            return None

        keys = [
            "run_id", "date", "stage", "status", "started_at",
            "completed_at", "error_message", "metadata",
        ]
        return dict(zip(keys, row))

    def cleanup_stale_runs(self, older_than_date: date) -> None:
        """Mark any 'running' stages from before older_than_date as 'failed'."""
        now = datetime.now(tz=UTC)
        self._conn.execute(
            """
            UPDATE pipeline_runs
            SET status = 'failed',
                completed_at = ?,
                error_message = 'Marked stale by cleanup'
            WHERE status = 'running'
              AND date < ?
            """,
            [now, older_than_date],
        )
        logger.info("Cleaned up stale pipeline runs older than %s", older_than_date)

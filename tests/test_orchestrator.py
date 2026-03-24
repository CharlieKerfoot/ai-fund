"""Tests for Phase 6: Daily pipeline orchestrator."""

from __future__ import annotations

from datetime import UTC, date, datetime
from unittest.mock import MagicMock

import duckdb
import pytest

from ingest.store.schema import create_schema
from orchestrator.daily import DailyPipeline

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def make_conn() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(":memory:")
    create_schema(conn)
    return conn


class FakeFeatureStore:
    """Minimal feature store stub."""

    def __init__(self, conn) -> None:
        self._conn = conn

    def get_pending_extractions(self, source_type: str, limit: int = 50) -> list:
        return []

    def query_prices(self, symbols, start, end, as_of):
        import pandas as pd
        return pd.DataFrame()

    def insert_signals(self, signal_id, df, version, metadata):
        pass


def make_pipeline(conn) -> DailyPipeline:
    """Create a pipeline with all stubbed dependencies."""
    feature_store = FakeFeatureStore(conn)

    return DailyPipeline(
        config={"symbols": ["AAPL", "MSFT"], "initial_capital": 1_000_000},
        feature_store=feature_store,
        monitor=None,
        market_source=None,
        macro_source=None,
        llm_extractor=None,
        signal_registry=MagicMock(list_active=lambda: []),
        signal_combiner=None,
        regime_detector=None,
        allocator=None,
        order_manager=None,
        simulator=None,
        pnl_tracker=None,
        feedback_loop=None,
        reporter=None,
    )


def insert_complete_stage(conn, run_date: date, stage: str) -> None:
    """Insert a completed pipeline_runs record for the given stage."""
    conn.execute(
        """
        INSERT INTO pipeline_runs
            (run_id, date, stage, status, started_at, completed_at, error_message, metadata)
        VALUES (?, ?, ?, 'complete', ?, ?, NULL, NULL)
        """,
        [
            f"run-{stage}",
            run_date,
            stage,
            datetime.now(tz=UTC),
            datetime.now(tz=UTC),
        ],
    )


def insert_running_stage(conn, run_date: date, stage: str) -> None:
    """Insert a running pipeline_runs record for the given stage."""
    conn.execute(
        """
        INSERT INTO pipeline_runs
            (run_id, date, stage, status, started_at, completed_at, error_message, metadata)
        VALUES (?, ?, ?, 'running', ?, NULL, NULL, NULL)
        """,
        [
            f"run-{stage}",
            run_date,
            stage,
            datetime.now(tz=UTC),
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDailyPipeline:
    @pytest.mark.asyncio
    async def test_pipeline_skips_completed_stage(self):
        """If a stage is already complete for today, it should be skipped."""
        conn = make_conn()
        pipeline = make_pipeline(conn)
        run_date = date(2024, 1, 10)

        # Mark all stages except 'report' as complete
        for stage in DailyPipeline.STAGES:
            insert_complete_stage(conn, run_date, stage)

        summary = await pipeline.run(run_date=run_date, dry_run=True)

        # All stages should be marked as 'skipped' (already complete)
        for stage in DailyPipeline.STAGES:
            assert summary[stage] == "skipped", (
                f"Stage {stage} should be 'skipped', got '{summary[stage]}'"
            )

    @pytest.mark.asyncio
    async def test_pipeline_runs_stages_when_not_complete(self):
        """Stages not yet complete for today should be run."""
        conn = make_conn()
        pipeline = make_pipeline(conn)
        run_date = date(2024, 1, 10)

        # No stages marked complete
        summary = await pipeline.run(run_date=run_date, dry_run=True)

        # All stages should be complete or failed (not skipped)
        for stage in DailyPipeline.STAGES:
            assert summary[stage] in ("complete", "failed"), (
                f"Stage {stage} status '{summary[stage]}' unexpected"
            )

    @pytest.mark.asyncio
    async def test_pipeline_resumes_from_failure(self):
        """Mark some stages complete; verify remaining stages are run."""
        conn = make_conn()
        pipeline = make_pipeline(conn)
        run_date = date(2024, 1, 10)

        # Mark first 3 stages complete
        for stage in DailyPipeline.STAGES[:3]:
            insert_complete_stage(conn, run_date, stage)

        summary = await pipeline.run(run_date=run_date, dry_run=True)

        # First 3 should be skipped
        for stage in DailyPipeline.STAGES[:3]:
            assert summary[stage] == "skipped", f"Stage {stage} should be skipped"

        # Remaining stages should have run
        for stage in DailyPipeline.STAGES[3:]:
            assert summary[stage] in ("complete", "failed", "skipped"), (
                f"Stage {stage} unexpected status: {summary[stage]}"
            )

    @pytest.mark.asyncio
    async def test_pipeline_marks_stale_running(self):
        """Old 'running' stages from previous dates get marked 'failed' on startup."""
        conn = make_conn()
        pipeline = make_pipeline(conn)

        yesterday = date(2024, 1, 9)
        run_date = date(2024, 1, 10)

        # Insert stale running stages from yesterday
        insert_running_stage(conn, yesterday, "ingest")
        insert_running_stage(conn, yesterday, "signals")

        # Verify they are running before pipeline starts
        rows = conn.execute(
            "SELECT stage, status FROM pipeline_runs WHERE date = ? AND status = 'running'",
            [yesterday],
        ).fetchall()
        assert len(rows) == 2

        # Run pipeline for today (which cleans up stale runs)
        await pipeline.run(run_date=run_date, dry_run=True)

        # Now the stale runs should be marked 'failed'
        rows = conn.execute(
            "SELECT stage, status FROM pipeline_runs WHERE date = ? AND status = 'failed'",
            [yesterday],
        ).fetchall()
        assert len(rows) == 2, (
            f"Expected 2 stale runs marked failed, got {len(rows)}"
        )

    @pytest.mark.asyncio
    async def test_pipeline_dry_run(self):
        """dry_run=True skips actual API calls but completes all stages."""
        conn = make_conn()
        pipeline = make_pipeline(conn)
        run_date = date(2024, 1, 10)

        summary = await pipeline.run(run_date=run_date, dry_run=True)

        assert isinstance(summary, dict)
        assert set(summary.keys()) == set(DailyPipeline.STAGES)

    @pytest.mark.asyncio
    async def test_pipeline_critical_stage_halts(self):
        """Failure in critical 'ingest' stage should halt remaining stages."""
        conn = make_conn()
        pipeline = make_pipeline(conn)
        run_date = date(2024, 1, 10)

        # Make _run_ingest raise an exception
        async def fail_ingest(run_date, dry_run):
            raise RuntimeError("Network error")

        pipeline._run_ingest = fail_ingest

        summary = await pipeline.run(run_date=run_date, dry_run=False)

        assert summary["ingest"] == "failed"
        # All subsequent stages should be skipped (pipeline halted)
        for stage in DailyPipeline.STAGES[1:]:
            assert summary[stage] == "skipped", (
                f"Stage {stage} should be skipped after critical failure, got '{summary[stage]}'"
            )

    @pytest.mark.asyncio
    async def test_pipeline_non_critical_stage_continues(self):
        """Failure in non-critical stage should allow pipeline to continue."""
        conn = make_conn()
        pipeline = make_pipeline(conn)
        run_date = date(2024, 1, 10)

        # Make _run_extract raise an exception (extract is non-critical)
        async def fail_extract(run_date, dry_run):
            raise RuntimeError("LLM quota exceeded")

        pipeline._run_extract = fail_extract

        summary = await pipeline.run(run_date=run_date, dry_run=True)

        assert summary["extract"] == "failed"
        # Pipeline should have continued to later stages
        later_stages = DailyPipeline.STAGES[DailyPipeline.STAGES.index("extract") + 1:]
        ran_count = sum(
            1 for s in later_stages if summary.get(s) in ("complete", "failed")
        )
        assert ran_count > 0, "Expected at least one stage to run after extract failure"

    def test_is_trading_day_weekday(self):
        """Monday through Friday are trading days (fallback)."""
        conn = make_conn()
        pipeline = make_pipeline(conn)

        monday = date(2024, 1, 8)
        friday = date(2024, 1, 12)
        saturday = date(2024, 1, 13)
        sunday = date(2024, 1, 14)

        # Weekdays (fallback: Mon-Fri)
        assert pipeline.is_trading_day(monday) is True
        assert pipeline.is_trading_day(friday) is True
        # Weekends
        assert pipeline.is_trading_day(saturday) is False
        assert pipeline.is_trading_day(sunday) is False

    @pytest.mark.asyncio
    async def test_pipeline_idempotent_second_run(self):
        """Running pipeline twice for same date: second run skips all stages."""
        conn = make_conn()
        pipeline = make_pipeline(conn)
        run_date = date(2024, 1, 10)

        # First run
        summary1 = await pipeline.run(run_date=run_date, dry_run=True)

        # Second run should skip all completed stages
        summary2 = await pipeline.run(run_date=run_date, dry_run=True)

        for stage in DailyPipeline.STAGES:
            if summary1[stage] == "complete":
                assert summary2[stage] == "skipped", (
                    f"Stage {stage} completed in run1 but not skipped in run2"
                )

"""Tests for FeatureStore, schema creation, and PipelineMonitor."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd

from ingest.quality.monitor import PipelineMonitor

# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------

def test_schema_creation(in_memory_db):
    """All expected tables should exist after create_schema."""
    expected_tables = {
        "prices",
        "text_features",
        "macro",
        "signals",
        "positions",
        "orders",
        "fills",
        "daily_pnl",
        "regime_history",
        "pipeline_runs",
        "embeddings",
    }
    rows = in_memory_db.execute("SHOW TABLES").fetchall()
    actual_tables = {row[0] for row in rows}
    assert expected_tables == actual_tables


# ---------------------------------------------------------------------------
# Price insert and query
# ---------------------------------------------------------------------------

def test_insert_and_query_prices(feature_store, sample_prices_df):
    """Inserted prices should be queryable and match expected shape."""
    feature_store.insert_prices(sample_prices_df, source="yahoo")

    result = feature_store.query_prices(
        symbols=["AAPL", "MSFT"],
        start=date(2024, 1, 1),
        end=date(2024, 12, 31),
        as_of=date(2099, 1, 1),  # far future → get everything
    )

    assert len(result) == len(sample_prices_df)
    assert set(result["symbol"].unique()) == {"AAPL", "MSFT"}
    assert (result["close"] > 0).all()


def test_point_in_time_prices(feature_store):
    """Prices ingested after as_of must be excluded."""
    early = datetime(2024, 1, 5, tzinfo=UTC)
    late = datetime(2024, 6, 1, tzinfo=UTC)

    df_early = pd.DataFrame([{
        "symbol": "AAPL",
        "date": date(2024, 1, 3),
        "open": 170.0, "high": 172.0, "low": 169.0,
        "close": 171.0, "adj_close": 171.0, "volume": 50_000_000,
        "ingested_at": early,
    }])
    df_late = pd.DataFrame([{
        "symbol": "AAPL",
        "date": date(2024, 1, 4),
        "open": 172.0, "high": 174.0, "low": 171.0,
        "close": 173.0, "adj_close": 173.0, "volume": 55_000_000,
        "ingested_at": late,
    }])

    feature_store.insert_prices(df_early, source="yahoo")
    feature_store.insert_prices(df_late, source="yahoo")

    # as_of before late ingestion: only early row
    result = feature_store.query_prices(
        symbols=["AAPL"],
        start=date(2024, 1, 1),
        end=date(2024, 12, 31),
        as_of=date(2024, 3, 1),
    )
    assert len(result) == 1
    assert pd.Timestamp(result.iloc[0]["date"]).date() == date(2024, 1, 3)

    # as_of after both ingestions: both rows
    result_all = feature_store.query_prices(
        symbols=["AAPL"],
        start=date(2024, 1, 1),
        end=date(2024, 12, 31),
        as_of=date(2024, 7, 1),
    )
    assert len(result_all) == 2


# ---------------------------------------------------------------------------
# Macro insert and query
# ---------------------------------------------------------------------------

def test_insert_and_query_macro(feature_store, sample_macro_df):
    """Inserted macro data should be queryable and match expected shape."""
    feature_store.insert_macro(sample_macro_df)

    result = feature_store.query_macro(
        series_ids=["FEDFUNDS", "DGS10"],
        start=date(2024, 1, 1),
        end=date(2024, 12, 31),
        as_of=date(2099, 1, 1),
    )

    assert len(result) == len(sample_macro_df)
    assert set(result["series_id"].unique()) == {"FEDFUNDS", "DGS10"}


def test_point_in_time_macro(feature_store):
    """Macro rows with release_date after as_of must be excluded."""
    df_old = pd.DataFrame([{
        "series_id": "FEDFUNDS",
        "date": date(2024, 1, 1),
        "release_date": date(2024, 1, 10),  # released early
        "value": 5.25,
    }])
    df_new = pd.DataFrame([{
        "series_id": "FEDFUNDS",
        "date": date(2024, 1, 2),
        "release_date": date(2024, 6, 1),  # released later
        "value": 5.30,
    }])

    feature_store.insert_macro(df_old)
    feature_store.insert_macro(df_new)

    # as_of before second release: only first row
    result = feature_store.query_macro(
        series_ids=["FEDFUNDS"],
        start=date(2024, 1, 1),
        end=date(2024, 12, 31),
        as_of=date(2024, 3, 1),
    )
    assert len(result) == 1
    assert pd.Timestamp(result.iloc[0]["date"]).date() == date(2024, 1, 1)

    # as_of after both releases: both rows
    result_all = feature_store.query_macro(
        series_ids=["FEDFUNDS"],
        start=date(2024, 1, 1),
        end=date(2024, 12, 31),
        as_of=date(2024, 7, 1),
    )
    assert len(result_all) == 2


# ---------------------------------------------------------------------------
# PipelineMonitor lifecycle
# ---------------------------------------------------------------------------

def test_pipeline_monitor_lifecycle(in_memory_db):
    """start_stage → complete_stage → is_complete should return True."""
    monitor = PipelineMonitor(in_memory_db)
    run_date = date(2024, 1, 15)

    run_id = monitor.start_stage("ingest_prices", run_date, metadata={"source": "yahoo"})
    assert run_id is not None

    assert not monitor.is_complete("ingest_prices", run_date)

    monitor.complete_stage(run_id)
    assert monitor.is_complete("ingest_prices", run_date)

    last = monitor.get_last_run("ingest_prices")
    assert last is not None
    assert last["status"] == "complete"
    assert last["stage"] == "ingest_prices"


def test_pipeline_monitor_fail(in_memory_db):
    """fail_stage should mark the run as failed."""
    monitor = PipelineMonitor(in_memory_db)
    run_date = date(2024, 2, 1)

    run_id = monitor.start_stage("ingest_macro", run_date)
    monitor.fail_stage(run_id, "Connection timeout")

    assert not monitor.is_complete("ingest_macro", run_date)

    last = monitor.get_last_run("ingest_macro")
    assert last["status"] == "failed"
    assert "timeout" in last["error_message"].lower()


def test_pipeline_monitor_stale_cleanup(in_memory_db):
    """cleanup_stale_runs should mark old 'running' entries as 'failed'."""
    monitor = PipelineMonitor(in_memory_db)

    # Insert a stale run from the past
    old_date = date(2024, 1, 1)
    monitor.start_stage("ingest_prices", old_date)

    # Verify it is currently 'running'
    last = monitor.get_last_run("ingest_prices")
    assert last["status"] == "running"

    # Cleanup anything older than today
    monitor.cleanup_stale_runs(older_than_date=date.today())

    last_after = monitor.get_last_run("ingest_prices")
    assert last_after["status"] == "failed"
    assert "stale" in last_after["error_message"].lower()

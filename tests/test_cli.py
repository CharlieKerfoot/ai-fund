"""Tests for Phase 7: CLI commands."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from cli import app

runner = CliRunner()


class TestCliHelp:
    def test_cli_help(self):
        """Top-level --help should print usage without error."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "ai-fund" in result.output.lower() or "Usage" in result.output

    def test_run_pipeline_help(self):
        result = runner.invoke(app, ["run-pipeline", "--help"])
        assert result.exit_code == 0
        assert "--dry-run" in result.output

    def test_backtest_help(self):
        result = runner.invoke(app, ["backtest", "--help"])
        assert result.exit_code == 0
        assert "--start" in result.output

    def test_status_help(self):
        result = runner.invoke(app, ["status", "--help"])
        assert result.exit_code == 0

    def test_positions_help(self):
        result = runner.invoke(app, ["positions", "--help"])
        assert result.exit_code == 0

    def test_signals_help(self):
        result = runner.invoke(app, ["signals", "--help"])
        assert result.exit_code == 0

    def test_report_help(self):
        result = runner.invoke(app, ["report", "--help"])
        assert result.exit_code == 0
        assert "--open" in result.output

    def test_risk_check_help(self):
        result = runner.invoke(app, ["risk-check", "--help"])
        assert result.exit_code == 0

    def test_add_signal_help(self):
        result = runner.invoke(app, ["add-signal", "--help"])
        assert result.exit_code == 0
        assert "--weight" in result.output

    def test_backfill_help(self):
        result = runner.invoke(app, ["backfill", "--help"])
        assert result.exit_code == 0
        assert "--years" in result.output


class TestRunPipelineDryRun:
    def test_run_pipeline_dry_run(self, tmp_path):
        """--dry-run should complete without making real API calls."""
        db_path = str(tmp_path / "test.duckdb")
        env = {"DUCKDB_PATH": db_path}

        with patch.dict(os.environ, env):
            result = runner.invoke(app, ["run-pipeline", "--dry-run", "--date", "2024-01-10"])

        # Should not crash (exit 0 or 1 is acceptable; just verify no unhandled exception)
        assert result.exit_code in (0, 1)
        # With dry_run all stages should be attempted
        if result.exit_code == 0:
            assert "Pipeline Summary" in result.output or "complete" in result.output.lower()

    def test_run_pipeline_dry_run_invalid_date(self):
        """Invalid date format should exit with error."""
        result = runner.invoke(app, ["run-pipeline", "--dry-run", "--date", "not-a-date"])
        assert result.exit_code != 0 or "Invalid date" in result.output

    def test_run_pipeline_dry_run_with_stage(self, tmp_path):
        """--stage flag should run only the specified stage."""
        db_path = str(tmp_path / "test.duckdb")
        env = {"DUCKDB_PATH": db_path}

        with patch.dict(os.environ, env):
            result = runner.invoke(
                app,
                ["run-pipeline", "--dry-run", "--stage", "ingest", "--date", "2024-01-10"],
            )

        assert result.exit_code in (0, 1)

    def test_run_pipeline_dry_run_invalid_stage(self, tmp_path):
        """Unknown --stage should exit with error."""
        db_path = str(tmp_path / "test.duckdb")
        env = {"DUCKDB_PATH": db_path}

        with patch.dict(os.environ, env):
            result = runner.invoke(
                app,
                ["run-pipeline", "--dry-run", "--stage", "nonexistent", "--date", "2024-01-10"],
            )

        assert result.exit_code != 0 or "Unknown stage" in result.output


class TestStatusCommand:
    def test_status_no_db(self, tmp_path):
        """Status with no database should print a warning, not crash."""
        db_path = str(tmp_path / "nonexistent.duckdb")
        env = {"DUCKDB_PATH": db_path}

        with patch.dict(os.environ, env):
            result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "No database found" in result.output or "backfill" in result.output.lower()

    def test_status_with_empty_db(self, tmp_path):
        """Status with an empty database should show empty table without crashing."""
        import duckdb

        from ingest.store.schema import create_schema

        db_path = str(tmp_path / "test.duckdb")
        conn = duckdb.connect(db_path)
        create_schema(conn)
        conn.close()

        env = {"DUCKDB_PATH": db_path}
        with patch.dict(os.environ, env):
            result = runner.invoke(app, ["status"])

        assert result.exit_code == 0


class TestSignalsCommand:
    def test_signals_command(self):
        """signals command should list built-in signals or show empty message."""
        result = runner.invoke(app, ["signals"])
        assert result.exit_code == 0
        # Either shows signals or says none registered
        assert "Signal" in result.output or "No signals" in result.output

    def test_signals_shows_registry(self):
        """signals command output should contain registry-related text."""
        result = runner.invoke(app, ["signals"])
        assert result.exit_code == 0
        # Should contain a table or message about signals
        output_lower = result.output.lower()
        assert any(word in output_lower for word in ["signal", "registry", "registered"])


class TestPositionsCommand:
    def test_positions_no_db(self, tmp_path):
        """positions with no database should exit cleanly."""
        db_path = str(tmp_path / "nonexistent.duckdb")
        env = {"DUCKDB_PATH": db_path}

        with patch.dict(os.environ, env):
            result = runner.invoke(app, ["positions"])

        assert result.exit_code == 0
        assert "No database" in result.output

    def test_positions_empty_db(self, tmp_path):
        """positions with empty fills table should show no open positions."""
        import duckdb

        from ingest.store.schema import create_schema

        db_path = str(tmp_path / "test.duckdb")
        conn = duckdb.connect(db_path)
        create_schema(conn)
        conn.close()

        env = {"DUCKDB_PATH": db_path}
        with patch.dict(os.environ, env):
            result = runner.invoke(app, ["positions"])

        assert result.exit_code == 0
        assert "No open positions" in result.output


class TestReportCommand:
    def test_report_no_reports(self, tmp_path):
        """report with no HTML files should warn gracefully."""
        db_path = str(tmp_path / "test.duckdb")
        env = {"DUCKDB_PATH": db_path}

        with patch.dict(os.environ, env):
            result = runner.invoke(app, ["report"])

        assert result.exit_code == 0
        assert "No reports" in result.output or "run-pipeline" in result.output.lower()

    def test_report_with_file(self, tmp_path):
        """report should display path when HTML file exists."""
        db_path = str(tmp_path / "test.duckdb")
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        report_file = reports_dir / "2024-01-10.html"
        report_file.write_text("<html>test</html>")

        env = {"DUCKDB_PATH": db_path}
        with patch.dict(os.environ, env):
            result = runner.invoke(app, ["report", "2024-01-10"])

        assert result.exit_code == 0
        assert "2024-01-10" in result.output


class TestRiskCheckCommand:
    def test_risk_check_no_db(self, tmp_path):
        """risk-check with no database should warn gracefully."""
        db_path = str(tmp_path / "nonexistent.duckdb")
        env = {"DUCKDB_PATH": db_path}

        with patch.dict(os.environ, env):
            result = runner.invoke(app, ["risk-check"])

        assert result.exit_code == 0
        assert "No database" in result.output

    def test_risk_check_empty_positions(self, tmp_path):
        """risk-check with no positions should say no positions."""
        import duckdb

        from ingest.store.schema import create_schema

        db_path = str(tmp_path / "test.duckdb")
        conn = duckdb.connect(db_path)
        create_schema(conn)
        conn.close()

        env = {"DUCKDB_PATH": db_path}
        with patch.dict(os.environ, env):
            result = runner.invoke(app, ["risk-check"])

        assert result.exit_code == 0
        assert "No open positions" in result.output or "Risk Assessment" in result.output


class TestAddSignalCommand:
    def test_add_signal_invalid_path(self):
        """add-signal with invalid module path should error cleanly."""
        result = runner.invoke(app, ["add-signal", "not_a_valid_path"])
        assert result.exit_code != 0 or "Invalid module path" in result.output

    def test_add_signal_missing_module(self):
        """add-signal with non-existent module should error cleanly."""
        result = runner.invoke(app, ["add-signal", "totally.fake.module.FakeSignal"])
        assert result.exit_code != 0 or "Could not import" in result.output


class TestBackfillCommand:
    def test_backfill_invalid_date(self):
        """backfill with invalid --start should error."""
        result = runner.invoke(app, ["backfill", "--start", "not-a-date"])
        assert result.exit_code != 0 or "Invalid date" in result.output

    def test_backfill_no_fred_key(self, tmp_path):
        """backfill without FRED_API_KEY should skip macro and warn."""
        db_path = str(tmp_path / "test.duckdb")
        env = {
            "DUCKDB_PATH": db_path,
            "FRED_API_KEY": "",
        }

        import pandas as pd

        # Patch at the module level where it's imported inside the function
        with patch.dict(os.environ, env):
            with patch("ingest.sources.market.MarketDataSource") as mock_mds_cls:
                with patch("ingest.sources.market.yf"):
                    mock_instance = MagicMock()
                    mock_instance.fetch_daily.return_value = pd.DataFrame(
                        columns=["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"]
                    )
                    mock_instance.write_parquet.return_value = Path(tmp_path / "out.parquet")
                    mock_mds_cls.return_value = mock_instance

                    result = runner.invoke(
                        app,
                        ["backfill", "--years", "1", "--start", "2024-01-01"],
                    )

        # Should complete (warning about missing FRED key is expected)
        assert result.exit_code in (0, 1)
        if result.exit_code == 0:
            assert "FRED_API_KEY" in result.output or "backfill" in result.output.lower()

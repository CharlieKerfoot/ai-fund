"""Tests for Phase 3: Backtest Engine."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from research.backtest.calendar import TradingCalendar
from research.backtest.engine import BacktestEngine, BacktestResult
from research.backtest.overfit import OverfitGuard, OverfitReport
from research.backtest.validation import WalkForwardResult, WalkForwardValidator
from research.signals.base import Signal, SignalMetadata
from research.signals.registry import SignalRegistry

# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

class MomentumSignal(Signal):
    """Concrete signal: 20-day price momentum (close[-1] / close[-20] - 1)."""

    @property
    def metadata(self) -> SignalMetadata:
        return SignalMetadata(
            name="momentum_20d",
            version="1.0.0",
            description="20-day price momentum",
            universe="test",
            frequency="weekly",
            lookback_days=30,
            features_required=["adj_close"],
        )

    def compute(self, features: pd.DataFrame, as_of: date) -> pd.Series:
        """Return momentum signal clipped to [-1, 1] indexed by symbol."""
        if features.empty:
            return pd.Series(dtype=float)

        df = features.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= pd.Timestamp(as_of)]

        result = {}
        for symbol, grp in df.groupby("symbol"):
            grp = grp.sort_values("date")
            if len(grp) < 2:
                result[symbol] = 0.0
                continue
            first = grp["adj_close"].iloc[0]
            last = grp["adj_close"].iloc[-1]
            if first == 0:
                result[symbol] = 0.0
            else:
                mom = (last / first) - 1.0
                result[symbol] = float(np.clip(mom, -1.0, 1.0))

        return pd.Series(result)

    def explain(self, symbol: str, features: pd.DataFrame, as_of: date) -> str:
        return f"Momentum signal for {symbol} as of {as_of}"


def _make_feature_store_with_prices(tmp_path, symbols: list[str], n_days: int = 400):
    """Create a FeatureStore populated with synthetic price data."""
    from ingest.store.feature_store import FeatureStore

    db_path = str(tmp_path / "bt_test.duckdb")
    parquet_dir = str(tmp_path / "parquet")
    store = FeatureStore(db_path=db_path, parquet_dir=parquet_dir)

    rows = []
    base_date = date(2022, 1, 3)  # Monday
    now = datetime.now(tz=UTC)

    rng = np.random.default_rng(42)
    for sym_i, sym in enumerate(symbols):
        price = 100.0 + sym_i * 20.0
        for i in range(n_days):
            d = base_date + timedelta(days=i)
            # skip weekends
            if d.weekday() >= 5:
                continue
            price *= 1.0 + rng.normal(0.0005, 0.01)
            rows.append({
                "symbol": sym,
                "date": d,
                "open": round(price * 0.999, 4),
                "high": round(price * 1.005, 4),
                "low": round(price * 0.995, 4),
                "close": round(price, 4),
                "adj_close": round(price, 4),
                "volume": 1_000_000,
                "ingested_at": now,
            })

    df = pd.DataFrame(rows)
    store.insert_prices(df, source="test")
    return store


# ---------------------------------------------------------------------------
# Signal tests
# ---------------------------------------------------------------------------

class TestSignalBase:
    def test_signal_compute_returns_series(self, tmp_path):
        """compute() must return a pd.Series indexed by symbol."""
        symbols = ["AAPL", "GOOG", "MSFT", "AMZN", "TSLA"]
        store = _make_feature_store_with_prices(tmp_path, symbols, n_days=100)

        price_df = store.query_prices(
            symbols=symbols,
            start=date(2022, 1, 3),
            end=date(2022, 6, 1),
            as_of=date(9999, 12, 31),
        )

        signal = MomentumSignal()
        result = signal.compute(price_df, as_of=date(2022, 6, 1))

        assert isinstance(result, pd.Series)
        assert set(result.index) == set(symbols)
        # Values should be in [-1, 1]
        assert (result >= -1.0).all()
        assert (result <= 1.0).all()
        store.close()

    def test_signal_metadata(self):
        """Signal metadata properties are accessible."""
        sig = MomentumSignal()
        assert sig.name == "momentum_20d"
        assert sig.version == "1.0.0"
        assert sig.metadata.frequency == "weekly"
        assert sig.metadata.lookback_days == 30


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestSignalRegistry:
    def test_register_and_get(self):
        registry = SignalRegistry()
        sig = MomentumSignal()
        registry.register(sig, weight=1.5)
        reg = registry.get("momentum_20d")
        assert reg is not None
        assert reg.active is True
        assert reg.weight == 1.5

    def test_deactivate_activate(self):
        registry = SignalRegistry()
        sig = MomentumSignal()
        registry.register(sig)
        registry.deactivate("momentum_20d")
        assert registry.get("momentum_20d").active is False
        assert len(registry.list_active()) == 0
        registry.activate("momentum_20d")
        assert len(registry.list_active()) == 1

    def test_weight_floor_enforcement(self):
        """Weight cannot drop below 50% of initial without floor being applied."""
        registry = SignalRegistry()
        registry.register(MomentumSignal(), weight=1.0)
        # Try to drop to 10% of initial (floor is 50%)
        registry.update_weight("momentum_20d", 0.1)
        reg = registry.get("momentum_20d")
        assert reg.weight == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# Backtest engine tests
# ---------------------------------------------------------------------------

class TestBacktestEngine:
    def test_backtest_engine_basic(self, tmp_path):
        """BacktestResult has all expected fields and sensible values."""
        symbols = ["AAPL", "GOOG", "MSFT", "AMZN", "TSLA"]
        store = _make_feature_store_with_prices(tmp_path, symbols, n_days=400)
        engine = BacktestEngine(feature_store=store, transaction_cost_bps=10.0)
        signal = MomentumSignal()

        result = engine.run(
            signal=signal,
            universe=symbols,
            start=date(2022, 6, 1),
            end=date(2023, 1, 31),
            rebalance_freq="weekly",
        )

        assert isinstance(result, BacktestResult)
        assert result.signal_name == "momentum_20d"
        assert result.signal_version == "1.0.0"
        assert result.start_date == date(2022, 6, 1)
        assert result.end_date == date(2023, 1, 31)
        assert isinstance(result.equity_curve, pd.Series)
        assert isinstance(result.returns, pd.Series)
        assert isinstance(result.positions, pd.DataFrame)
        assert len(result.equity_curve) > 0
        assert not math.isnan(result.sharpe)
        assert not math.isnan(result.max_drawdown)
        assert result.max_drawdown <= 0.0
        assert result.annual_volatility >= 0.0
        store.close()

    def test_backtest_transaction_costs(self, tmp_path):
        """Transaction costs reduce returns vs zero-cost backtest."""
        symbols = ["AAPL", "GOOG", "MSFT", "AMZN", "TSLA"]
        store = _make_feature_store_with_prices(tmp_path, symbols, n_days=400)
        signal = MomentumSignal()

        bt_start, bt_end = date(2022, 6, 1), date(2023, 1, 31)

        engine_with_tc = BacktestEngine(
            feature_store=store,
            transaction_cost_bps=50.0,
            slippage_bps=20.0,
        )
        engine_no_tc = BacktestEngine(
            feature_store=store,
            transaction_cost_bps=0.0,
            slippage_bps=0.0,
        )

        result_tc = engine_with_tc.run(signal, symbols, bt_start, bt_end)
        result_no_tc = engine_no_tc.run(signal, symbols, bt_start, bt_end)

        # With transaction costs, total return should be lower
        assert result_tc.equity_curve.iloc[-1] <= result_no_tc.equity_curve.iloc[-1]
        assert result_tc.transaction_costs_total > 0.0
        store.close()

    def test_weights_sum_to_one(self, tmp_path):
        """_compute_weights produces weights summing to 1."""
        symbols = ["AAPL", "GOOG", "MSFT", "AMZN", "TSLA", "META", "NVDA", "AMD", "NFLX", "ORCL"]
        store = _make_feature_store_with_prices(tmp_path, symbols, n_days=100)
        engine = BacktestEngine(feature_store=store)

        signal_values = pd.Series(
            {sym: float(i) for i, sym in enumerate(symbols)}
        )
        weights = engine._compute_weights(signal_values)

        assert abs(weights.sum() - 1.0) < 1e-6
        assert (weights >= 0.0).all()
        # Only top 20% should have non-zero weights (2 out of 10)
        assert (weights > 0).sum() == 2
        store.close()

    def test_weights_sum_to_one_small_universe(self, tmp_path):
        """_compute_weights works correctly with a small universe."""
        store = _make_feature_store_with_prices(tmp_path, ["A", "B", "C"], n_days=50)
        engine = BacktestEngine(feature_store=store)

        signal_values = pd.Series({"A": 0.8, "B": 0.5, "C": 0.2})
        weights = engine._compute_weights(signal_values)

        assert abs(weights.sum() - 1.0) < 1e-6
        assert (weights >= 0.0).all()
        store.close()


# ---------------------------------------------------------------------------
# Trading calendar tests
# ---------------------------------------------------------------------------

class TestTradingCalendar:
    def test_trading_calendar_is_trading_day(self):
        """NYSE holidays and weekends are excluded."""
        cal = TradingCalendar("NYSE")
        # NYSE closed on New Year's Day 2024 (observed Monday Jan 1)
        assert cal.is_trading_day(date(2024, 1, 1)) is False
        # Regular trading day
        assert cal.is_trading_day(date(2024, 1, 2)) is True
        # Saturday
        assert cal.is_trading_day(date(2024, 1, 6)) is False

    def test_trading_calendar_rebalance_dates(self):
        """Weekly rebalance gives approximately 52 dates per year."""
        cal = TradingCalendar("NYSE")
        dates = cal.get_rebalance_dates(date(2023, 1, 1), date(2023, 12, 31), freq="weekly")
        assert 50 <= len(dates) <= 54, f"Expected ~52 weekly rebalance dates, got {len(dates)}"

    def test_trading_calendar_monthly(self):
        """Monthly rebalance gives 12 dates per year."""
        cal = TradingCalendar("NYSE")
        dates = cal.get_rebalance_dates(date(2023, 1, 1), date(2023, 12, 31), freq="monthly")
        assert len(dates) == 12

    def test_previous_and_next_trading_day(self):
        """previous/next trading day skips weekends and holidays."""
        cal = TradingCalendar("NYSE")
        # Monday 2024-01-08 -> previous trading day = Friday 2024-01-05
        prev = cal.previous_trading_day(date(2024, 1, 8))
        assert prev == date(2024, 1, 5)
        # Friday 2024-01-05 -> next trading day = Monday 2024-01-08
        nxt = cal.next_trading_day(date(2024, 1, 5))
        assert nxt == date(2024, 1, 8)

    def test_get_trading_days(self):
        """get_trading_days returns correct number of trading days for a week."""
        cal = TradingCalendar("NYSE")
        # Week of 2024-01-02 (Tue-Fri, Mon is holiday)
        days = cal.get_trading_days(date(2024, 1, 2), date(2024, 1, 5))
        assert len(days) == 4


# ---------------------------------------------------------------------------
# Walk-forward validation tests
# ---------------------------------------------------------------------------

class TestWalkForwardValidator:
    def test_walk_forward_folds(self, tmp_path):
        """n_folds=3 produces 3 fold results with correct date ordering."""
        symbols = ["AAPL", "GOOG", "MSFT", "AMZN", "TSLA"]
        store = _make_feature_store_with_prices(tmp_path, symbols, n_days=700)
        engine = BacktestEngine(feature_store=store)
        validator = WalkForwardValidator(engine=engine, n_folds=3, min_train_days=30)
        signal = MomentumSignal()

        result = validator.validate(
            signal=signal,
            universe=symbols,
            start=date(2022, 6, 1),
            end=date(2023, 12, 31),
        )

        assert isinstance(result, WalkForwardResult)
        assert len(result.folds) == 3

        # Verify fold date ordering is non-overlapping
        for i, fold in enumerate(result.folds):
            assert fold["test_start"] < fold["test_end"]
            if i > 0:
                assert fold["test_start"] > result.folds[i - 1]["test_end"]

        store.close()


# ---------------------------------------------------------------------------
# Overfit guard tests
# ---------------------------------------------------------------------------

class TestOverfitGuard:
    def test_overfit_guard_reject(self):
        """High Sharpe with many trials -> REJECT."""
        guard = OverfitGuard()
        # 3.0 Sharpe but tested 100 parameter combinations over only 0.5 years
        report = guard.check(
            sharpe=3.0,
            n_trials=100,
            backtest_years=0.5,
            annual_vol=0.15,
            signal_name="overfit_signal",
        )
        assert isinstance(report, OverfitReport)
        assert report.recommendation == "REJECT"
        assert not report.sufficient_history

    def test_overfit_guard_pass(self):
        """Reasonable Sharpe with single trial and sufficient history -> PASS."""
        guard = OverfitGuard()
        report = guard.check(
            sharpe=1.5,
            n_trials=1,
            backtest_years=5.0,
            annual_vol=0.15,
            signal_name="clean_signal",
        )
        assert isinstance(report, OverfitReport)
        assert report.recommendation == "PASS"
        assert report.sufficient_history

    def test_overfit_minimum_history(self):
        """Insufficient backtest history -> REJECT regardless of Sharpe."""
        guard = OverfitGuard()
        # Only 0.25 years of history
        report = guard.check(
            sharpe=2.0,
            n_trials=10,
            backtest_years=0.25,
            annual_vol=0.20,
            signal_name="short_history",
        )
        assert report.recommendation == "REJECT"
        assert not report.sufficient_history

    def test_overfit_bonferroni_caution(self):
        """Moderate Sharpe with several trials -> CAUTION or REJECT."""
        guard = OverfitGuard()
        report = guard.check(
            sharpe=1.0,
            n_trials=20,
            backtest_years=3.0,
            annual_vol=0.15,
        )
        assert report.recommendation in ("CAUTION", "REJECT")

    def test_overfit_report_fields(self):
        """OverfitReport has all expected fields."""
        guard = OverfitGuard()
        report = guard.check(
            sharpe=1.2,
            n_trials=5,
            backtest_years=2.0,
            annual_vol=0.12,
            signal_name="test_signal",
        )
        assert report.signal_name == "test_signal"
        assert report.n_trials == 5
        assert isinstance(report.deflated_sharpe, float)
        assert isinstance(report.bonferroni_correction, float)
        assert isinstance(report.min_backtest_years, float)
        assert report.actual_backtest_years == pytest.approx(2.0)

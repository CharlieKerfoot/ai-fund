"""Tests for Phase 6: Execution layer."""

from __future__ import annotations

import os
from datetime import UTC, date, datetime

import duckdb
import pandas as pd
import pytest

from execution.orders.manager import OrderManager
from execution.orders.types import OrderSide
from execution.paper.simulator import Fill, PaperTradingSimulator
from execution.paper.slippage import SlippageModel
from execution.reporting.daily import DailyReporter
from execution.tracking.feedback import FeedbackLoop
from execution.tracking.pnl import DailyPnL, PnLTracker
from ingest.store.schema import create_schema

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def make_conn() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(":memory:")
    create_schema(conn)
    return conn


def make_slippage(base_bps: float = 5.0) -> SlippageModel:
    return SlippageModel(base_bps=base_bps)


def make_simulator(conn) -> PaperTradingSimulator:
    return PaperTradingSimulator(slippage_model=make_slippage(), conn=conn)


def make_order(symbol: str = "AAPL", side: str = "buy", quantity: float = 100.0) -> dict:
    return {
        "order_id": "test-order-001",
        "symbol": symbol,
        "quantity": quantity,
        "side": side,
        "date": date(2024, 1, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Slippage tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSlippageModel:
    def test_slippage_buy_worse_price(self):
        """Buy order gets higher fill price (slippage adds cost)."""
        model = SlippageModel(base_bps=5.0)
        market_price = 100.0
        fill_price = model.estimate(
            symbol="AAPL",
            quantity=100.0,  # positive = buy
            price=market_price,
            avg_daily_volume=1_000_000,
        )
        assert fill_price > market_price, (
            f"Buy fill price {fill_price} should be > market {market_price}"
        )

    def test_slippage_sell_better_price(self):
        """Sell order gets lower fill price (slippage reduces proceeds)."""
        model = SlippageModel(base_bps=5.0)
        market_price = 100.0
        fill_price = model.estimate(
            symbol="AAPL",
            quantity=-100.0,  # negative = sell
            price=market_price,
            avg_daily_volume=1_000_000,
        )
        assert fill_price < market_price, (
            f"Sell fill price {fill_price} should be < market {market_price}"
        )

    def test_slippage_large_order_higher_impact(self):
        """Larger order relative to ADV gets higher slippage."""
        model = SlippageModel(base_bps=5.0)
        adv = 1_000_000
        small_bps = model.estimate_cost_bps(quantity=100.0, price=100.0, avg_daily_volume=adv)
        large_bps = model.estimate_cost_bps(quantity=50_000.0, price=100.0, avg_daily_volume=adv)
        assert large_bps > small_bps, (
            f"Large order bps {large_bps:.4f} should exceed small order bps {small_bps:.4f}"
        )

    def test_slippage_base_bps_minimum(self):
        """Slippage should be at least base_bps even for tiny orders."""
        model = SlippageModel(base_bps=10.0)
        bps = model.estimate_cost_bps(quantity=1.0, price=100.0, avg_daily_volume=100_000_000)
        assert bps >= 10.0 - 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Simulator tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPaperTradingSimulator:
    def test_simulator_fill_writes_to_db(self):
        """Fill gets persisted to fills table."""
        conn = make_conn()
        sim = make_simulator(conn)
        order = make_order(symbol="AAPL", side="buy", quantity=100.0)

        fill = sim.simulate_fill(order, market_price=150.0, avg_daily_volume=5_000_000)

        assert isinstance(fill, Fill)
        assert fill.symbol == "AAPL"
        assert fill.quantity == 100.0  # positive (buy)

        # Verify persisted in DB
        row = conn.execute(
            "SELECT fill_id, symbol, quantity FROM fills WHERE fill_id = ?",
            [fill.fill_id],
        ).fetchone()
        assert row is not None
        assert row[1] == "AAPL"
        assert row[2] == 100.0

    def test_simulator_sell_fill_negative_quantity(self):
        """Sell fill has negative quantity in fills table."""
        conn = make_conn()
        sim = make_simulator(conn)
        order = make_order(symbol="MSFT", side="sell", quantity=50.0)

        fill = sim.simulate_fill(order, market_price=300.0, avg_daily_volume=3_000_000)

        assert fill.quantity == -50.0

    def test_positions_from_fills(self):
        """Multiple buys/sells compute correct net position."""
        conn = make_conn()
        sim = make_simulator(conn)

        # Buy 200 AAPL
        sim.simulate_fill(
            {"order_id": "o1", "symbol": "AAPL", "quantity": 200.0, "side": "buy", "date": date(2024, 1, 2)},
            market_price=150.0,
            avg_daily_volume=5_000_000,
        )
        # Sell 50 AAPL
        sim.simulate_fill(
            {"order_id": "o2", "symbol": "AAPL", "quantity": 50.0, "side": "sell", "date": date(2024, 1, 3)},
            market_price=152.0,
            avg_daily_volume=5_000_000,
        )
        # Buy 100 MSFT
        sim.simulate_fill(
            {"order_id": "o3", "symbol": "MSFT", "quantity": 100.0, "side": "buy", "date": date(2024, 1, 2)},
            market_price=300.0,
            avg_daily_volume=3_000_000,
        )

        positions = sim.get_positions(as_of=date(2024, 1, 5))
        assert not positions.empty

        pos_dict = {row["symbol"]: row["quantity"] for _, row in positions.iterrows()}
        assert pos_dict["AAPL"] == pytest.approx(150.0, abs=1e-6)
        assert pos_dict["MSFT"] == pytest.approx(100.0, abs=1e-6)

    def test_positions_fully_closed(self):
        """A position that nets to zero does not appear in get_positions."""
        conn = make_conn()
        sim = make_simulator(conn)

        sim.simulate_fill(
            {"order_id": "o1", "symbol": "AAPL", "quantity": 100.0, "side": "buy", "date": date(2024, 1, 2)},
            150.0, 5_000_000,
        )
        sim.simulate_fill(
            {"order_id": "o2", "symbol": "AAPL", "quantity": 100.0, "side": "sell", "date": date(2024, 1, 3)},
            152.0, 5_000_000,
        )

        positions = sim.get_positions(as_of=date(2024, 1, 5))
        if not positions.empty:
            pos_dict = {row["symbol"]: row["quantity"] for _, row in positions.iterrows()}
            assert "AAPL" not in pos_dict


# ─────────────────────────────────────────────────────────────────────────────
# Order manager tests
# ─────────────────────────────────────────────────────────────────────────────


class TestOrderManager:
    def test_order_manager_generates_buy(self):
        """target > current → buy order generated."""
        conn = make_conn()
        om = OrderManager(conn, min_order_value=100.0)

        target_weights = pd.Series({"AAPL": 0.50})
        current_positions = pd.DataFrame(columns=["symbol", "quantity", "avg_cost"])
        prices = pd.Series({"AAPL": 150.0})
        portfolio_value = 10_000.0

        orders = om.generate_orders(target_weights, current_positions, portfolio_value, prices)

        assert len(orders) == 1
        assert orders[0].symbol == "AAPL"
        assert orders[0].side == OrderSide.BUY

    def test_order_manager_generates_sell(self):
        """target < current → sell order generated."""
        conn = make_conn()
        om = OrderManager(conn, min_order_value=100.0)

        # Currently hold 100 shares, target is 0.1 weight
        target_weights = pd.Series({"AAPL": 0.10})
        current_positions = pd.DataFrame([
            {"symbol": "AAPL", "quantity": 100.0, "avg_cost": 150.0}
        ])
        prices = pd.Series({"AAPL": 150.0})
        portfolio_value = 20_000.0

        orders = om.generate_orders(target_weights, current_positions, portfolio_value, prices)

        assert len(orders) == 1
        assert orders[0].symbol == "AAPL"
        assert orders[0].side == OrderSide.SELL

    def test_order_manager_skips_small(self):
        """Small order (< min_order_value) is skipped."""
        conn = make_conn()
        om = OrderManager(conn, min_order_value=1000.0)

        # Target 0.001 weight: 0.001 * 10000 / 150 ≈ 0.067 shares → $10 → below $1000 threshold
        target_weights = pd.Series({"AAPL": 0.001})
        current_positions = pd.DataFrame(columns=["symbol", "quantity", "avg_cost"])
        prices = pd.Series({"AAPL": 150.0})
        portfolio_value = 10_000.0

        orders = om.generate_orders(target_weights, current_positions, portfolio_value, prices)
        assert len(orders) == 0

    def test_order_manager_persist_and_retrieve(self):
        """Persisted orders can be retrieved as pending."""
        conn = make_conn()
        om = OrderManager(conn, min_order_value=100.0)

        target_weights = pd.Series({"AAPL": 0.5, "MSFT": 0.5})
        current_positions = pd.DataFrame(columns=["symbol", "quantity", "avg_cost"])
        prices = pd.Series({"AAPL": 150.0, "MSFT": 300.0})
        portfolio_value = 100_000.0

        orders = om.generate_orders(target_weights, current_positions, portfolio_value, prices)
        om.persist_orders(orders)

        # Patch dates to today for retrieval
        for order in orders:
            pass  # order.date is already set to date.today() by Order.create()

        pending = om.get_pending_orders(orders[0].date)
        assert len(pending) == len(orders)


# ─────────────────────────────────────────────────────────────────────────────
# P&L tracker tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPnLTracker:
    def test_pnl_tracker_basic(self):
        """Insert fills, compute pnl, verify calculation."""
        conn = make_conn()
        make_simulator(conn)
        tracker = PnLTracker(conn, initial_capital=100_000.0)

        # Buy 100 AAPL at 150
        conn.execute(
            "INSERT INTO fills (fill_id, order_id, symbol, date, quantity, fill_price, slippage_bps, commission, filled_at) "
            "VALUES ('f1', 'o1', 'AAPL', '2024-01-02', 100.0, 150.0, 5.0, 0.5, '2024-01-02 09:30:00')"
        )

        prices = pd.Series({"AAPL": 155.0})
        pnl = tracker.compute_daily_pnl(date(2024, 1, 2), prices, regime="risk_on")

        assert isinstance(pnl, DailyPnL)
        assert pnl.portfolio_value > 0
        # Position value = 100 * 155 = 15500
        # Cash = 100000 - 100*150 = 85000
        # Total = 100500
        assert pnl.portfolio_value == pytest.approx(100_500.0, abs=1.0)
        assert pnl.regime == "risk_on"

    def test_pnl_equity_curve(self):
        """5 days of pnl tracked correctly."""
        conn = make_conn()
        tracker = PnLTracker(conn, initial_capital=100_000.0)

        date(2024, 1, 2)
        for i in range(5):
            d = date(2024, 1, 2 + i)
            portfolio_val = 100_000.0 + i * 1_000.0
            conn.execute(
                "INSERT OR REPLACE INTO daily_pnl (date, gross_pnl, net_pnl, transaction_costs, portfolio_value, cash, regime) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [d, 1000.0, 900.0, 100.0, portfolio_val, 50_000.0, "risk_on"],
            )

        curve = tracker.get_equity_curve(date(2024, 1, 2), date(2024, 1, 6))
        assert len(curve) == 5
        assert curve.iloc[0] == pytest.approx(100_000.0, abs=1e-6)
        assert curve.iloc[-1] == pytest.approx(104_000.0, abs=1e-6)

    def test_pnl_performance_metrics(self):
        """Performance metrics computed correctly."""
        conn = make_conn()
        tracker = PnLTracker(conn, initial_capital=100_000.0)

        # Insert 10 days of increasing portfolio values
        for i in range(10):
            d = date(2024, 1, 2 + i)
            conn.execute(
                "INSERT OR REPLACE INTO daily_pnl (date, gross_pnl, net_pnl, transaction_costs, portfolio_value, cash, regime) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [d, 500.0, 450.0, 50.0, 100_000.0 + i * 500.0, 50_000.0, "risk_on"],
            )

        metrics = tracker.get_performance_metrics()
        assert "sharpe" in metrics
        assert "sortino" in metrics
        assert "max_dd" in metrics
        assert "total_return" in metrics
        assert metrics["total_return"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# Feedback loop tests
# ─────────────────────────────────────────────────────────────────────────────


class _MockSignal:
    """Minimal signal stub for registry tests."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._version = "1.0.0"

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version


class TestFeedbackLoop:
    def _make_registry(self):
        from research.signals.registry import SignalRegistry
        return SignalRegistry()

    def test_feedback_loop_decay(self):
        """Signal with 40% hit rate (below 50%) gets weight below initial."""
        conn = make_conn()
        registry = self._make_registry()

        signal = _MockSignal("momentum")
        registry.register(signal, weight=1.0)

        # Insert 10 days of signals/fills where 4 match (buy) and 6 mismatch (sell)
        # All signals are positive (suggesting buy), but 6 fills are sells -> 40% hit rate
        # Use a run_date after all data so all 10 days are in the lookback window
        run_date = date(2024, 3, 15)  # far enough that all Jan data is in window
        for i in range(10):
            day = date(2024, 1, 2 + i)
            sig_value = 1.0  # all positive signals (suggesting buy)
            fill_qty = 100.0 if i < 4 else -100.0  # 4 buys match, 6 sells don't

            conn.execute(
                "INSERT OR REPLACE INTO signals (signal_id, symbol, date, value, confidence, signal_version, metadata) "
                "VALUES (?, 'AAPL', ?, ?, 1.0, '1.0.0', '{}')",
                ["momentum", day, sig_value],
            )
            conn.execute(
                "INSERT INTO fills (fill_id, order_id, symbol, date, quantity, fill_price, slippage_bps, commission, filled_at) "
                "VALUES (?, 'o1', 'AAPL', ?, ?, 150.0, 5.0, 0.5, ?)",
                [f"f{i}", day, fill_qty, datetime.now(tz=UTC)],
            )

        loop = FeedbackLoop(registry, conn)
        updated = loop.update(run_date, lookback_days=90)

        assert "momentum" in updated
        # With 40% hit rate: raw = 1.0 * (0.4 - 0.5) * 2 = -0.2
        # Smoothed: 0.1 * (-0.2) + 0.9 * 1.0 = -0.02 + 0.9 = 0.88
        # Floor = 0.5; 0.88 > 0.5 so no floor applied
        # Weight should be 0.88 < 1.0 (initial weight)
        assert updated["momentum"] < 1.0, (
            f"Expected weight < 1.0 (initial), got {updated['momentum']}"
        )

    def test_feedback_loop_floor(self):
        """Bad signal weight capped at 50% of initial."""
        conn = make_conn()
        registry = self._make_registry()

        signal = _MockSignal("reversal")
        registry.register(signal, weight=1.0)

        # No signals or fills -> default 0.5 hit rate -> apply anyway
        # Manually set current weight very low to test floor enforcement
        d = date(2024, 1, 10)

        # Force the registry weight down by calling update_weight below floor
        # The floor is enforced in update_weight itself
        registry.update_weight("reversal", 0.6)  # sets to 0.6 (above 0.5 floor)

        # Now insert signals that produce 0% hit rate (all mismatch)
        for i in range(10):
            day = date(2024, 1, 1 + i)
            conn.execute(
                "INSERT OR REPLACE INTO signals (signal_id, symbol, date, value, confidence, signal_version, metadata) "
                "VALUES (?, 'SPY', ?, 1.0, 1.0, '1.0.0', '{}')",
                ["reversal", day],
            )
            conn.execute(
                "INSERT INTO fills (fill_id, order_id, symbol, date, quantity, fill_price, slippage_bps, commission, filled_at) "
                "VALUES (?, 'o1', 'SPY', ?, -100.0, 400.0, 5.0, 0.5, ?)",
                [f"fr{i}", day, datetime.now(tz=UTC)],
            )

        loop = FeedbackLoop(registry, conn)
        updated = loop.update(d, lookback_days=63)

        # 0% hit rate: raw = 1.0 * (0.0 - 0.5) * 2 = -1.0
        # Floor = 0.5 * 1.0 = 0.5
        # Smoothed before floor: 0.1 * (-1.0) + 0.9 * 0.6 = -0.1 + 0.54 = 0.44 < 0.5 -> floor
        assert updated["reversal"] == pytest.approx(0.5, abs=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# Daily reporter tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDailyReporter:
    def test_daily_reporter_creates_file(self, tmp_path):
        """Generate report, verify HTML file created."""
        conn = make_conn()

        d = date(2024, 1, 10)
        conn.execute(
            "INSERT OR REPLACE INTO daily_pnl (date, gross_pnl, net_pnl, transaction_costs, portfolio_value, cash, regime) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [d, 500.0, 450.0, 50.0, 100_500.0, 85_000.0, "risk_on"],
        )

        reporter = DailyReporter(conn, reports_dir=str(tmp_path / "reports"))
        path = reporter.generate(d, regime="risk_on", degraded_mode=False)

        assert os.path.exists(path), f"Report file not found at {path}"
        content = open(path).read()
        assert "<html" in content.lower()
        assert "2024-01-10" in content
        assert "risk_on" in content

    def test_daily_reporter_degraded_mode(self, tmp_path):
        """Degraded mode flag appears in the report."""
        conn = make_conn()
        reporter = DailyReporter(conn, reports_dir=str(tmp_path / "reports"))
        d = date(2024, 1, 11)
        path = reporter.generate(d, regime="unknown", degraded_mode=True)

        content = open(path).read()
        assert "DEGRADED" in content

    def test_daily_reporter_multiple_dates(self, tmp_path):
        """Reports for multiple dates each get their own file."""
        conn = make_conn()
        reporter = DailyReporter(conn, reports_dir=str(tmp_path / "reports"))

        dates = [date(2024, 1, d) for d in range(2, 5)]
        for d in dates:
            path = reporter.generate(d)
            assert os.path.exists(path)

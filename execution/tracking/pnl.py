"""P&L tracking and performance metrics for the paper trading system."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DailyPnL:
    date: date
    gross_pnl: float
    net_pnl: float
    transaction_costs: float
    portfolio_value: float
    cash: float
    regime: str


class PnLTracker:
    def __init__(self, conn, initial_capital: float = 1_000_000) -> None:
        self._conn = conn
        self.initial_capital = initial_capital

    def compute_daily_pnl(
        self,
        as_of: date,
        prices: pd.Series,
        regime: str = "unknown",
    ) -> DailyPnL:
        """Compute and persist daily P&L.

        Steps:
        1. Get current positions (recomputed from fills)
        2. Mark positions to market
        3. gross_pnl = portfolio_value - yesterday's portfolio_value
        4. net_pnl = gross_pnl - today's transaction_costs (from fills)
        5. cash = initial_capital - sum(fill_price * quantity * sign for all fills)
        6. Write to daily_pnl table
        """
        # Step 1: Get current positions
        positions = self._get_positions(as_of)

        # Step 2: Mark to market
        position_value = 0.0
        for _, row in positions.iterrows():
            symbol = row["symbol"]
            quantity = float(row["quantity"])
            price = float(prices.get(symbol, 0.0))
            position_value += quantity * price

        # Step 3: Gross P&L vs yesterday
        yesterday = as_of - timedelta(days=1)
        prev_row = self._conn.execute(
            "SELECT portfolio_value FROM daily_pnl WHERE date = ?",
            [yesterday],
        ).fetchone()

        if prev_row is not None:
            prev_portfolio_value = float(prev_row[0])
        else:
            # First day: compare against initial capital deployed in positions
            prev_portfolio_value = self._compute_cost_basis(as_of) + self._compute_cash(as_of)

        # Step 4: Transaction costs from today's fills
        tc_row = self._conn.execute(
            """
            SELECT COALESCE(SUM(commission + ABS(quantity) * fill_price * slippage_bps / 10000.0), 0.0)
            FROM fills
            WHERE date = ?
            """,
            [as_of],
        ).fetchone()
        transaction_costs = float(tc_row[0]) if tc_row else 0.0

        # Step 5: Cash
        cash = self._compute_cash(as_of)

        portfolio_value = position_value + cash
        gross_pnl = portfolio_value - prev_portfolio_value
        net_pnl = gross_pnl - transaction_costs

        daily_pnl = DailyPnL(
            date=as_of,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            transaction_costs=transaction_costs,
            portfolio_value=portfolio_value,
            cash=cash,
            regime=regime,
        )

        # Step 6: Write to daily_pnl table
        self._conn.execute(
            """
            INSERT OR REPLACE INTO daily_pnl
                (date, gross_pnl, net_pnl, transaction_costs, portfolio_value, cash, regime)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                daily_pnl.date,
                daily_pnl.gross_pnl,
                daily_pnl.net_pnl,
                daily_pnl.transaction_costs,
                daily_pnl.portfolio_value,
                daily_pnl.cash,
                daily_pnl.regime,
            ],
        )

        return daily_pnl

    def _get_positions(self, as_of: date) -> pd.DataFrame:
        """Recompute positions from fills table."""
        return self._conn.execute(
            """
            SELECT
                symbol,
                SUM(quantity) AS quantity
            FROM fills
            WHERE date <= ?
            GROUP BY symbol
            HAVING SUM(quantity) != 0
            """,
            [as_of],
        ).df()

    def _compute_cash(self, as_of: date) -> float:
        """Cash = initial_capital - net cash spent on all fills.

        Cash flow = fill_price * quantity (positive quantity = buy = outflow,
        negative quantity = sell = inflow).
        """
        row = self._conn.execute(
            """
            SELECT COALESCE(SUM(fill_price * quantity), 0.0)
            FROM fills
            WHERE date <= ?
            """,
            [as_of],
        ).fetchone()
        cash_deployed = float(row[0]) if row else 0.0
        return float(self.initial_capital - cash_deployed)

    def _compute_cost_basis(self, as_of: date) -> float:
        """Compute current cost basis of all positions."""
        row = self._conn.execute(
            """
            SELECT COALESCE(SUM(fill_price * quantity), 0.0)
            FROM fills
            WHERE date <= ?
            """,
            [as_of],
        ).fetchone()
        return float(row[0]) if row else 0.0

    def get_equity_curve(self, start: date, end: date) -> pd.Series:
        """Returns portfolio_value indexed by date."""
        df = self._conn.execute(
            """
            SELECT date, portfolio_value
            FROM daily_pnl
            WHERE date >= ? AND date <= ?
            ORDER BY date
            """,
            [start, end],
        ).df()

        if df.empty:
            return pd.Series(dtype=float)

        df["date"] = pd.to_datetime(df["date"]).dt.date
        return pd.Series(df["portfolio_value"].values, index=df["date"].values)

    def get_performance_metrics(self) -> dict:
        """Returns Sharpe, Sortino, max drawdown, annual return, total return."""
        df = self._conn.execute(
            "SELECT date, portfolio_value FROM daily_pnl ORDER BY date"
        ).df()

        if len(df) < 2:
            return {
                "sharpe": float("nan"),
                "sortino": float("nan"),
                "max_dd": float("nan"),
                "annual_return": float("nan"),
                "total_return": float("nan"),
            }

        values = df["portfolio_value"].values.astype(float)
        daily_returns = np.diff(values) / values[:-1]

        total_return = (values[-1] - values[0]) / values[0]
        n_days = len(values)
        annual_return = (1.0 + total_return) ** (252.0 / n_days) - 1.0

        mean_ret = np.mean(daily_returns)
        std_ret = np.std(daily_returns, ddof=1)
        sharpe = float(mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else float("nan")

        downside = daily_returns[daily_returns < 0]
        sortino_std = np.std(downside, ddof=1) if len(downside) > 1 else 0.0
        sortino = float(mean_ret / sortino_std * np.sqrt(252)) if sortino_std > 0 else float("nan")

        # Max drawdown
        peak = values[0]
        max_dd = 0.0
        for v in values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd

        return {
            "sharpe": sharpe,
            "sortino": sortino,
            "max_dd": max_dd,
            "annual_return": annual_return,
            "total_return": total_return,
        }

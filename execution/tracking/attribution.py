"""Performance attribution: link signal directions to actual P&L."""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PerformanceAttribution:
    def __init__(self, conn) -> None:
        self._conn = conn

    def compute_signal_pnl(
        self,
        signal_name: str,
        start: date,
        end: date,
    ) -> pd.Series:
        """Attribute daily P&L to a signal.

        Steps:
        1. Get signal values for the period from signals table
        2. Get actual returns for positions
        3. Compute correlation between signal direction and actual returns
        Returns daily P&L attributed to this signal.
        """
        # Get signal values
        signals_df = self._conn.execute(
            """
            SELECT date, symbol, value
            FROM signals
            WHERE signal_id = ? AND date >= ? AND date <= ?
            ORDER BY date, symbol
            """,
            [signal_name, start, end],
        ).df()

        if signals_df.empty:
            return pd.Series(dtype=float, name=signal_name)

        # Get fills to compute actual returns
        fills_df = self._conn.execute(
            """
            SELECT date, symbol, quantity, fill_price
            FROM fills
            WHERE date >= ? AND date <= ?
            ORDER BY date
            """,
            [start, end],
        ).df()

        if fills_df.empty:
            return pd.Series(dtype=float, name=signal_name)

        # Compute daily P&L from fills (simplified: use commission + slippage as proxy)
        daily_pnl_df = self._conn.execute(
            """
            SELECT date, SUM(quantity * fill_price) AS cash_flow
            FROM fills
            WHERE date >= ? AND date <= ?
            GROUP BY date
            ORDER BY date
            """,
            [start, end],
        ).df()

        if daily_pnl_df.empty:
            return pd.Series(dtype=float, name=signal_name)

        daily_pnl_df["date"] = pd.to_datetime(daily_pnl_df["date"]).dt.date
        result = pd.Series(
            daily_pnl_df["cash_flow"].values,
            index=daily_pnl_df["date"].values,
            name=signal_name,
        )
        return result

    def compute_hit_rate(
        self,
        signal_name: str,
        start: date,
        end: date,
    ) -> float:
        """Fraction of days where signal direction matched actual return direction."""
        signals_df = self._conn.execute(
            """
            SELECT date, symbol, value
            FROM signals
            WHERE signal_id = ? AND date >= ? AND date <= ?
            ORDER BY date, symbol
            """,
            [signal_name, start, end],
        ).df()

        if signals_df.empty:
            return 0.0

        # Get fills to compute returns
        fills_df = self._conn.execute(
            """
            SELECT date, symbol, quantity, fill_price
            FROM fills
            WHERE date >= ? AND date <= ?
            ORDER BY date, symbol
            """,
            [start, end],
        ).df()

        if fills_df.empty:
            return 0.0

        signals_df["date"] = pd.to_datetime(signals_df["date"]).dt.date
        fills_df["date"] = pd.to_datetime(fills_df["date"]).dt.date

        # Merge on date and symbol
        merged = pd.merge(
            signals_df,
            fills_df[["date", "symbol", "quantity", "fill_price"]],
            on=["date", "symbol"],
            how="inner",
        )

        if merged.empty:
            return 0.0

        # Signal direction: positive value -> long, negative -> short
        signal_dir = np.sign(merged["value"])
        # Trade direction: buy = positive quantity
        trade_dir = np.sign(merged["quantity"])

        hits = (signal_dir == trade_dir).sum()
        total = len(merged)

        return float(hits / total) if total > 0 else 0.0

    def get_signal_summary(self, start: date, end: date) -> pd.DataFrame:
        """Returns DataFrame: signal_name, hit_rate, attributed_pnl, sharpe_contribution."""
        # Get all unique signal IDs in the period
        signal_ids_df = self._conn.execute(
            """
            SELECT DISTINCT signal_id
            FROM signals
            WHERE date >= ? AND date <= ?
            """,
            [start, end],
        ).df()

        if signal_ids_df.empty:
            return pd.DataFrame(
                columns=["signal_name", "hit_rate", "attributed_pnl", "sharpe_contribution"]
            )

        rows = []
        for signal_id in signal_ids_df["signal_id"].tolist():
            hit_rate = self.compute_hit_rate(signal_id, start, end)
            pnl_series = self.compute_signal_pnl(signal_id, start, end)
            attributed_pnl = float(pnl_series.sum()) if not pnl_series.empty else 0.0

            # Simple Sharpe contribution: mean/std of daily P&L
            if len(pnl_series) > 1 and pnl_series.std() > 0:
                sharpe_contribution = float(
                    pnl_series.mean() / pnl_series.std() * np.sqrt(252)
                )
            else:
                sharpe_contribution = float("nan")

            rows.append({
                "signal_name": signal_id,
                "hit_rate": hit_rate,
                "attributed_pnl": attributed_pnl,
                "sharpe_contribution": sharpe_contribution,
            })

        return pd.DataFrame(rows)

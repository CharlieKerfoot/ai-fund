"""Trading calendar utilities backed by pandas_market_calendars."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pandas_market_calendars as mcal


class TradingCalendar:
    """NYSE trading calendar wrapper providing holiday-aware date utilities."""

    def __init__(self, exchange: str = "NYSE") -> None:
        self._exchange = exchange
        self._calendar = mcal.get_calendar(exchange)

    def _schedule(self, start: date, end: date) -> pd.DatetimeIndex:
        """Return valid trading sessions between start and end (inclusive)."""
        schedule = self._calendar.schedule(
            start_date=start.isoformat(),
            end_date=end.isoformat(),
        )
        return schedule.index

    def is_trading_day(self, d: date) -> bool:
        """Return True if d is a valid NYSE trading day."""
        sessions = self._schedule(d, d)
        return len(sessions) > 0

    def get_trading_days(self, start: date, end: date) -> list[date]:
        """Return all trading days between start and end inclusive."""
        sessions = self._schedule(start, end)
        return [ts.date() for ts in sessions]

    def previous_trading_day(self, d: date) -> date:
        """Return the most recent trading day strictly before d."""
        # Look back up to 10 calendar days to find a trading day
        candidate = d - timedelta(days=1)
        for _ in range(10):
            if self.is_trading_day(candidate):
                return candidate
            candidate -= timedelta(days=1)
        raise ValueError(f"Could not find previous trading day before {d}")

    def next_trading_day(self, d: date) -> date:
        """Return the next trading day strictly after d."""
        candidate = d + timedelta(days=1)
        for _ in range(10):
            if self.is_trading_day(candidate):
                return candidate
            candidate += timedelta(days=1)
        raise ValueError(f"Could not find next trading day after {d}")

    def get_rebalance_dates(
        self,
        start: date,
        end: date,
        freq: str = "weekly",
    ) -> list[date]:
        """Return list of rebalance dates.

        weekly  -> every Friday (or last trading day of that week)
        monthly -> last trading day of each calendar month
        """
        trading_days = self.get_trading_days(start, end)
        if not trading_days:
            return []

        td_series = pd.Series(
            trading_days,
            index=pd.DatetimeIndex([pd.Timestamp(d) for d in trading_days]),
        )

        if freq == "weekly":
            # Group by ISO year-week, take the last trading day in each week
            rebalance = (
                td_series.groupby(pd.Grouper(freq="W-FRI"))
                .last()
                .dropna()
            )
        elif freq == "monthly":
            rebalance = (
                td_series.groupby(pd.Grouper(freq="ME"))
                .last()
                .dropna()
            )
        else:
            raise ValueError(f"Unsupported rebalance frequency: '{freq}'. Use 'weekly' or 'monthly'.")

        # Filter to [start, end] range
        result = [d for d in rebalance.values if start <= d <= end]
        return result

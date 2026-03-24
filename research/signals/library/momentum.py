"""Momentum signals: cross-sectional and time-series."""

from __future__ import annotations

from datetime import date

import pandas as pd

from research.signals.base import Signal, SignalMetadata


class CrossSectionalMomentum(Signal):
    """Rank symbols by past 12-1 month returns (skip last month to avoid short-term reversal)."""

    def __init__(self, lookback_days: int = 252, skip_days: int = 21) -> None:
        self._lookback_days = lookback_days
        self._skip_days = skip_days

    @property
    def metadata(self) -> SignalMetadata:
        return SignalMetadata(
            name="cross_sectional_momentum",
            version="1.0.0",
            description="Rank symbols by 12-1 month returns, cross-sectional rank normalized to [-1, 1].",
            universe="all",
            frequency="daily",
            lookback_days=self._lookback_days,
            features_required=["symbol", "date", "close"],
        )

    def compute(self, features: pd.DataFrame, as_of: date) -> pd.Series:
        as_of_ts = pd.Timestamp(as_of)
        df = features[features["date"] <= as_of_ts].copy()

        # For each symbol, get the close price at as_of - skip_days and as_of - lookback_days
        results = {}
        for symbol, grp in df.groupby("symbol"):
            grp = grp.sort_values("date")
            # Filter to valid dates
            start_mask = grp["date"] <= as_of_ts - pd.Timedelta(days=self._lookback_days)
            end_mask = grp["date"] <= as_of_ts - pd.Timedelta(days=self._skip_days)

            start_rows = grp[start_mask]
            end_rows = grp[end_mask]

            if len(grp) < 200 or start_rows.empty or end_rows.empty:
                continue

            price_start = start_rows.iloc[-1]["close"]
            price_end = end_rows.iloc[-1]["close"]

            if price_start <= 0:
                continue

            results[symbol] = price_end / price_start - 1.0

        if not results:
            return pd.Series(dtype=float)

        raw = pd.Series(results)
        # Cross-sectional rank percentile * 2 - 1 → [-1, 1]
        n = len(raw)
        if n == 1:
            return raw.rank(pct=True) * 2 - 1
        signal = raw.rank(pct=True) * 2 - 1
        return signal

    def explain(self, symbol: str, features: pd.DataFrame, as_of: date) -> str:
        signal = self.compute(features, as_of)
        if symbol not in signal.index:
            return f"{symbol}: insufficient data for cross-sectional momentum"
        val = signal[symbol]
        return (
            f"{symbol}: cross-sectional momentum signal = {val:.4f}. "
            f"Rank percentile in universe based on {self._lookback_days}-{self._skip_days} day return."
        )


class TimeSeriesMomentum(Signal):
    """Signal = sign(past 12m return) * |12m return| scaled to [-1, 1]."""

    def __init__(self, lookback_days: int = 252) -> None:
        self._lookback_days = lookback_days

    @property
    def metadata(self) -> SignalMetadata:
        return SignalMetadata(
            name="time_series_momentum",
            version="1.0.0",
            description="Sign of 12m return scaled by 95th percentile of absolute returns.",
            universe="all",
            frequency="daily",
            lookback_days=self._lookback_days,
            features_required=["symbol", "date", "close"],
        )

    def compute(self, features: pd.DataFrame, as_of: date) -> pd.Series:
        as_of_ts = pd.Timestamp(as_of)
        df = features[features["date"] <= as_of_ts].copy()

        results = {}
        for symbol, grp in df.groupby("symbol"):
            grp = grp.sort_values("date")
            start_mask = grp["date"] <= as_of_ts - pd.Timedelta(days=self._lookback_days)
            start_rows = grp[start_mask]

            if start_rows.empty or len(grp) < 2:
                continue

            price_start = start_rows.iloc[-1]["close"]
            price_end = grp.iloc[-1]["close"]

            if price_start <= 0:
                continue

            results[symbol] = price_end / price_start - 1.0

        if not results:
            return pd.Series(dtype=float)

        raw = pd.Series(results)
        abs_returns = raw.abs()
        p95 = abs_returns.quantile(0.95)

        if p95 == 0:
            return pd.Series(0.0, index=raw.index)

        signal = raw / p95
        # Clamp to [-1, 1]
        signal = signal.clip(-1.0, 1.0)
        return signal

    def explain(self, symbol: str, features: pd.DataFrame, as_of: date) -> str:
        signal = self.compute(features, as_of)
        if symbol not in signal.index:
            return f"{symbol}: insufficient data for time series momentum"
        val = signal[symbol]
        return (
            f"{symbol}: time-series momentum signal = {val:.4f}. "
            f"12m return scaled by 95th percentile of absolute returns in universe."
        )

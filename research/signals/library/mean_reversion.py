"""Mean reversion signals: RSI and Bollinger Bands."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from research.signals.base import Signal, SignalMetadata


class RSIMeanReversion(Signal):
    """14-day RSI mean reversion signal. RSI < 30 → strong buy, RSI > 70 → strong sell."""

    def __init__(self, rsi_period: int = 14) -> None:
        self._rsi_period = rsi_period

    @property
    def metadata(self) -> SignalMetadata:
        return SignalMetadata(
            name="rsi_mean_reversion",
            version="1.0.0",
            description="14-day RSI mean reversion: -(RSI - 50) / 50.",
            universe="all",
            frequency="daily",
            lookback_days=self._rsi_period + 10,
            features_required=["symbol", "date", "close"],
        )

    def _compute_rsi(self, closes: pd.Series) -> float:
        """Compute RSI for a series of closing prices."""
        delta = closes.diff().dropna()
        if len(delta) < self._rsi_period:
            return float("nan")

        gains = delta.clip(lower=0)
        losses = (-delta).clip(lower=0)

        avg_gain = gains.iloc[:self._rsi_period].mean()
        avg_loss = losses.iloc[:self._rsi_period].mean()

        # Wilder smoothing for remaining periods
        for i in range(self._rsi_period, len(gains)):
            avg_gain = (avg_gain * (self._rsi_period - 1) + gains.iloc[i]) / self._rsi_period
            avg_loss = (avg_loss * (self._rsi_period - 1) + losses.iloc[i]) / self._rsi_period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def compute(self, features: pd.DataFrame, as_of: date) -> pd.Series:
        as_of_ts = pd.Timestamp(as_of)
        df = features[features["date"] <= as_of_ts].copy()

        results = {}
        for symbol, grp in df.groupby("symbol"):
            grp = grp.sort_values("date")
            closes = grp["close"]

            if len(closes) < self._rsi_period + 1:
                continue

            rsi = self._compute_rsi(closes)
            if np.isnan(rsi):
                continue

            # signal = -(RSI - 50) / 50 → in [-1, 1] for RSI in [0, 100]
            results[symbol] = -(rsi - 50.0) / 50.0

        return pd.Series(results)

    def explain(self, symbol: str, features: pd.DataFrame, as_of: date) -> str:
        signal = self.compute(features, as_of)
        if symbol not in signal.index:
            return f"{symbol}: insufficient data for RSI mean reversion"
        val = signal[symbol]
        rsi = 50.0 - val * 50.0
        return (
            f"{symbol}: RSI mean reversion signal = {val:.4f} (RSI = {rsi:.1f}). "
            f"Negative signal = overbought (RSI > 50), positive = oversold (RSI < 50)."
        )


class BollingerMeanReversion(Signal):
    """Bollinger Band mean reversion: z-score of price vs 20-day band, negated and clamped."""

    def __init__(self, window: int = 20) -> None:
        self._window = window

    @property
    def metadata(self) -> SignalMetadata:
        return SignalMetadata(
            name="bollinger_mean_reversion",
            version="1.0.0",
            description="Bollinger Band z-score mean reversion: -z clamped to [-1, 1].",
            universe="all",
            frequency="daily",
            lookback_days=self._window + 5,
            features_required=["symbol", "date", "close"],
        )

    def compute(self, features: pd.DataFrame, as_of: date) -> pd.Series:
        as_of_ts = pd.Timestamp(as_of)
        df = features[features["date"] <= as_of_ts].copy()

        results = {}
        for symbol, grp in df.groupby("symbol"):
            grp = grp.sort_values("date")

            if len(grp) < self._window:
                continue

            closes = grp["close"]
            recent = closes.iloc[-self._window:]
            ma = recent.mean()
            std = recent.std(ddof=1)

            if std == 0 or np.isnan(std):
                continue

            current_price = closes.iloc[-1]
            z = (current_price - ma) / std

            # signal = -z clamped to [-2, 2] then divided by 2
            signal_val = -np.clip(z, -2.0, 2.0) / 2.0
            results[symbol] = signal_val

        return pd.Series(results)

    def explain(self, symbol: str, features: pd.DataFrame, as_of: date) -> str:
        signal = self.compute(features, as_of)
        if symbol not in signal.index:
            return f"{symbol}: insufficient data for Bollinger mean reversion"
        val = signal[symbol]
        return (
            f"{symbol}: Bollinger mean reversion signal = {val:.4f}. "
            f"Price above {self._window}-day band → negative (sell), below → positive (buy)."
        )

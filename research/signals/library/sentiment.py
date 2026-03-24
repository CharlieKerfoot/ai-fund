"""News/social sentiment signal."""

from __future__ import annotations

from datetime import date

import pandas as pd

from research.signals.base import Signal, SignalMetadata


class SentimentSignal(Signal):
    """5-day EWM of sentiment scores, cross-sectionally rank normalized to [-1, 1]."""

    def __init__(self, ewm_span: int = 5) -> None:
        self._ewm_span = ewm_span

    @property
    def metadata(self) -> SignalMetadata:
        return SignalMetadata(
            name="sentiment",
            version="1.0.0",
            description=(
                "5-day exponentially weighted mean of sentiment scores, "
                "cross-sectionally rank normalized to [-1, 1]."
            ),
            universe="all",
            frequency="daily",
            lookback_days=self._ewm_span * 3,
            features_required=["symbol", "date", "sentiment_score"],
        )

    def compute(self, features: pd.DataFrame, as_of: date) -> pd.Series:
        as_of_ts = pd.Timestamp(as_of)
        df = features[features["date"] <= as_of_ts].copy()

        if df.empty:
            return pd.Series(dtype=float)

        results = {}
        for symbol, grp in df.groupby("symbol"):
            grp = grp.sort_values("date")
            scores = grp["sentiment_score"].dropna()

            if scores.empty:
                continue

            ewm_val = scores.ewm(span=self._ewm_span, adjust=False).mean().iloc[-1]
            results[symbol] = ewm_val

        if not results:
            return pd.Series(dtype=float)

        raw = pd.Series(results)
        # Cross-sectional rank normalization to [-1, 1]
        signal = raw.rank(pct=True) * 2 - 1
        return signal

    def explain(self, symbol: str, features: pd.DataFrame, as_of: date) -> str:
        signal = self.compute(features, as_of)
        if symbol not in signal.index:
            return f"{symbol}: no sentiment data available"
        val = signal[symbol]
        return (
            f"{symbol}: sentiment signal = {val:.4f}. "
            f"{self._ewm_span}-day EWM of sentiment scores, rank normalized across universe."
        )

"""Post-earnings announcement drift (PEAD) signal."""

from __future__ import annotations

from datetime import date

import pandas as pd

from research.signals.base import Signal, SignalMetadata


class PostEarningsDrift(Signal):
    """Post-earnings announcement drift based on sentiment and guidance direction."""

    def __init__(self, holding_days: int = 5, lookback_days: int = 60) -> None:
        self._holding_days = holding_days
        self._lookback_days = lookback_days

    @property
    def metadata(self) -> SignalMetadata:
        return SignalMetadata(
            name="post_earnings_drift",
            version="1.0.0",
            description=(
                "PEAD: positive sentiment + up guidance → +0.8, "
                "negative sentiment + down/withdrawn → -0.8."
            ),
            universe="all",
            frequency="daily",
            lookback_days=self._lookback_days,
            features_required=["symbol", "date", "sentiment_score", "guidance_direction"],
        )

    def compute(self, features: pd.DataFrame, as_of: date) -> pd.Series:
        as_of_ts = pd.Timestamp(as_of)
        cutoff = as_of_ts - pd.Timedelta(days=self._lookback_days)

        df = features[
            (features["date"] <= as_of_ts) & (features["date"] >= cutoff)
        ].copy()

        if df.empty:
            return pd.Series(dtype=float)

        results = {}
        for symbol, grp in df.groupby("symbol"):
            grp = grp.sort_values("date")
            # Use only rows that have earnings data (non-null sentiment_score)
            earnings_rows = grp[grp["sentiment_score"].notna()]
            if earnings_rows.empty:
                results[symbol] = 0.0
                continue

            # Use the most recent earnings event
            latest = earnings_rows.iloc[-1]
            sentiment = latest["sentiment_score"]
            guidance = latest["guidance_direction"]

            if sentiment > 0.3 and guidance in ["up"]:
                results[symbol] = 0.8
            elif sentiment < -0.3 and guidance in ["down", "withdrawn"]:
                results[symbol] = -0.8
            else:
                results[symbol] = 0.0

        return pd.Series(results)

    def explain(self, symbol: str, features: pd.DataFrame, as_of: date) -> str:
        signal = self.compute(features, as_of)
        if symbol not in signal.index:
            return f"{symbol}: no recent earnings data within {self._lookback_days} days"
        val = signal[symbol]
        direction = "bullish" if val > 0 else ("bearish" if val < 0 else "neutral")
        return (
            f"{symbol}: post-earnings drift signal = {val:.4f} ({direction}). "
            f"Based on sentiment score and guidance direction from last earnings event."
        )

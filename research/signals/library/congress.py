"""Congressional trading signal."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from research.signals.base import Signal, SignalMetadata


class CongressionalTrading(Signal):
    """Congressional trading signal with recency weighting."""

    def __init__(self, lookback_days: int = 30) -> None:
        self._lookback_days = lookback_days

    @property
    def metadata(self) -> SignalMetadata:
        return SignalMetadata(
            name="congressional_trading",
            version="1.0.0",
            description=(
                "Congressional trading: +1 for large buys, -1 for large sells, "
                "with exponential recency decay."
            ),
            universe="all",
            frequency="daily",
            lookback_days=self._lookback_days,
            features_required=["symbol", "date", "transaction_type", "amount"],
        )

    def compute(self, features: pd.DataFrame, as_of: date) -> pd.Series:
        as_of_ts = pd.Timestamp(as_of)
        cutoff = as_of_ts - pd.Timedelta(days=self._lookback_days)

        df = features[
            (features["date"] <= as_of_ts) & (features["date"] >= cutoff)
        ].copy()

        if df.empty:
            return pd.Series(dtype=float)

        # Keep only rows with transaction data
        df = df[df["transaction_type"].notna() & df["amount"].notna()]
        if df.empty:
            return pd.Series(dtype=float)

        # Large transactions only (amount > 50000)
        large = df[df["amount"] > 50_000].copy()

        results = {}
        for symbol, grp in large.groupby("symbol"):
            grp = grp.sort_values("date")

            # Days since as_of for recency weight
            days_ago = (as_of_ts - grp["date"]).dt.days.astype(float)
            # Exponential decay: most recent = weight 1, older decays
            weights = np.exp(-days_ago / self._lookback_days)

            # Direction: +1 for buy, -1 for sell
            directions = grp["transaction_type"].map({"buy": 1.0, "sell": -1.0}).fillna(0.0)

            weighted_sum = (directions * weights).sum()
            weight_total = weights.sum()

            if weight_total == 0:
                continue

            signal_val = weighted_sum / weight_total
            results[symbol] = np.clip(signal_val, -1.0, 1.0)

        return pd.Series(results)

    def explain(self, symbol: str, features: pd.DataFrame, as_of: date) -> str:
        signal = self.compute(features, as_of)
        if symbol not in signal.index:
            return f"{symbol}: no large congressional transactions within {self._lookback_days} days"
        val = signal[symbol]
        direction = "buy" if val > 0 else ("sell" if val < 0 else "neutral")
        return (
            f"{symbol}: congressional trading signal = {val:.4f} ({direction}). "
            f"Recency-weighted large transactions over last {self._lookback_days} days."
        )

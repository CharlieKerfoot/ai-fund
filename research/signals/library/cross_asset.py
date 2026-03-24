"""Cross-asset macro signal (bonds → equities)."""

from __future__ import annotations

from datetime import date

import pandas as pd

from research.signals.base import Signal, SignalMetadata


class CrossAssetSignal(Signal):
    """Macro regime signal from rates, credit spreads, and VIX applied uniformly to all symbols."""

    def __init__(self, lookback_days: int = 252) -> None:
        self._lookback_days = lookback_days
        self._change_window = 21

    @property
    def metadata(self) -> SignalMetadata:
        return SignalMetadata(
            name="cross_asset",
            version="1.0.0",
            description=(
                "Macro regime from DGS10, BAA10Y, VIXCLS: "
                "rising rates/spreads/VIX>20 = negative; applied equally to all symbols."
            ),
            universe="all",
            frequency="daily",
            lookback_days=self._lookback_days,
            features_required=["symbol", "date", "DGS10", "BAA10Y", "VIXCLS"],
        )

    def _sign(self, val: float) -> float:
        if val > 0:
            return 1.0
        elif val < 0:
            return -1.0
        return 0.0

    def compute(self, features: pd.DataFrame, as_of: date) -> pd.Series:
        as_of_ts = pd.Timestamp(as_of)
        df = features[features["date"] <= as_of_ts].copy()

        if df.empty:
            return pd.Series(dtype=float)

        symbols = df["symbol"].unique().tolist()
        if not symbols:
            return pd.Series(dtype=float)

        # Use the first symbol's macro data (macro columns are same for all symbols)
        # Sort by date and get the most recent rows with macro data
        macro_cols = ["DGS10", "BAA10Y", "VIXCLS"]
        macro_df = df[df[macro_cols].notna().any(axis=1)].sort_values("date")

        if macro_df.empty:
            # Try with any non-null macro data
            return pd.Series(0.0, index=symbols)

        # Get current and lagged values (21 days ago)
        current_row = macro_df.iloc[-1]
        lag_mask = macro_df["date"] <= as_of_ts - pd.Timedelta(days=self._change_window)
        lag_rows = macro_df[lag_mask]

        if lag_rows.empty:
            return pd.Series(0.0, index=symbols)

        lag_row = lag_rows.iloc[-1]

        # Compute changes
        dgs10_change = current_row["DGS10"] - lag_row["DGS10"]
        baa10y_change = current_row["BAA10Y"] - lag_row["BAA10Y"]
        vix_level = current_row["VIXCLS"]

        # Signals
        yield_signal = -self._sign(dgs10_change)    # rising rates = negative
        spread_signal = -self._sign(baa10y_change)  # widening spreads = negative
        vix_signal = -self._sign(vix_level - 20.0)  # VIX > 20 = risk-off = negative

        combined = 0.3 * yield_signal + 0.4 * spread_signal + 0.3 * vix_signal

        # Apply same signal to all symbols
        return pd.Series(combined, index=symbols)

    def explain(self, symbol: str, features: pd.DataFrame, as_of: date) -> str:
        signal = self.compute(features, as_of)
        if symbol not in signal.index:
            return f"{symbol}: no macro data available"
        val = signal[symbol]
        regime = "risk-on" if val > 0 else ("risk-off" if val < 0 else "neutral")
        return (
            f"{symbol}: cross-asset macro signal = {val:.4f} ({regime}). "
            f"Driven by DGS10 change, BAA10Y spread change, and VIX level."
        )

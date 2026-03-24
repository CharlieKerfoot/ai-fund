"""SEC filing anomaly signal."""

from __future__ import annotations

from datetime import date

import pandas as pd

from research.signals.base import Signal, SignalMetadata


class FilingAnomaly(Signal):
    """SEC filing anomaly: going concern, risk factor changes, litigation exposure."""

    def __init__(self, lookback_days: int = 90) -> None:
        self._lookback_days = lookback_days

    @property
    def metadata(self) -> SignalMetadata:
        return SignalMetadata(
            name="filing_anomaly",
            version="1.0.0",
            description=(
                "SEC filing anomaly: going concern = -1.0, "
                "increased risk + high litigation = -0.7, decreased risk = +0.5."
            ),
            universe="all",
            frequency="daily",
            lookback_days=self._lookback_days,
            features_required=[
                "symbol", "date",
                "risk_factor_delta", "litigation_exposure", "going_concern",
            ],
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
            # Use only rows that have filing data
            filing_rows = grp[grp["risk_factor_delta"].notna() | grp["going_concern"].notna()]
            if filing_rows.empty:
                results[symbol] = 0.0
                continue

            # Use the most recent filing
            latest = filing_rows.iloc[-1]
            going_concern = latest.get("going_concern", False)
            risk_delta = latest.get("risk_factor_delta", None)
            litigation = latest.get("litigation_exposure", None)

            if bool(going_concern) is True:
                results[symbol] = -1.0
            elif risk_delta == "increased" and litigation in ["high"]:
                results[symbol] = -0.7
            elif risk_delta == "decreased":
                results[symbol] = 0.5
            else:
                results[symbol] = 0.0

        return pd.Series(results)

    def explain(self, symbol: str, features: pd.DataFrame, as_of: date) -> str:
        signal = self.compute(features, as_of)
        if symbol not in signal.index:
            return f"{symbol}: no recent filing data within {self._lookback_days} days"
        val = signal[symbol]
        return (
            f"{symbol}: filing anomaly signal = {val:.4f}. "
            f"Based on going concern, risk factor delta, and litigation exposure from SEC filings."
        )

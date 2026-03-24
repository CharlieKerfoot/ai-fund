"""Feedback loop: update signal weights based on rolling hit rates."""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

logger = logging.getLogger(__name__)


class FeedbackLoop:
    WEIGHT_FLOOR_PCT = 0.5  # signal weight floor = 50% of initial weight
    DECAY_ALPHA = 0.1       # exponential decay rate (smoothing weight for new value)

    def __init__(self, registry, conn) -> None:
        self._registry = registry
        self._conn = conn

    def update(
        self,
        as_of: date,
        lookback_days: int = 63,
    ) -> dict[str, float]:
        """Update signal weights based on rolling hit rates.

        For each active signal:
        1. Compute rolling hit rate over lookback_days
        2. Compute raw_weight = initial_weight * (hit_rate - 0.5) * 2
           (scale: 0 if 50% hit rate, 1x initial if 100% hit rate)
        3. Apply exponential smoothing: new_weight = alpha * raw + (1-alpha) * current
        4. Enforce floor: new_weight >= WEIGHT_FLOOR_PCT * initial_weight
        5. Alert if weight approaches floor (< 60% of initial)
        6. Update registry weights
        7. Return dict of {signal_name: new_weight}
        """
        start = as_of - timedelta(days=lookback_days)
        updated_weights: dict[str, float] = {}

        for reg in self._registry.list_active():
            signal_name = reg.signal.name
            initial_weight = reg.initial_weight
            current_weight = reg.weight

            hit_rate = self._compute_hit_rate(signal_name, start, as_of)

            # Scale: (hit_rate - 0.5) * 2 maps [0.5, 1.0] -> [0, 1]
            # and [0, 0.5] -> [-1, 0]; scale by initial weight
            raw_weight = initial_weight * (hit_rate - 0.5) * 2.0

            # Exponential smoothing: alpha * raw + (1-alpha) * current
            smoothed_weight = self.DECAY_ALPHA * raw_weight + (1.0 - self.DECAY_ALPHA) * current_weight

            # Enforce floor
            floor = self.WEIGHT_FLOOR_PCT * initial_weight
            if smoothed_weight < floor:
                smoothed_weight = floor

            # Alert if approaching floor
            if smoothed_weight < 0.6 * initial_weight:
                logger.warning(
                    "ALERT: Signal '%s' weight %.4f approaching floor "
                    "(< 60%% of initial %.4f). Consider review.",
                    signal_name,
                    smoothed_weight,
                    initial_weight,
                )

            # Update registry (update_weight enforces its own floor)
            self._registry.update_weight(signal_name, smoothed_weight)
            updated_weights[signal_name] = smoothed_weight

        return updated_weights

    def _compute_hit_rate(
        self,
        signal_name: str,
        start: date,
        end: date,
    ) -> float:
        """Compute fraction of days where signal direction matched fill direction."""
        signals_df = self._conn.execute(
            """
            SELECT date, symbol, value
            FROM signals
            WHERE signal_id = ? AND date >= ? AND date <= ?
            """,
            [signal_name, start, end],
        ).df()

        if signals_df.empty:
            return 0.5  # neutral hit rate if no data

        fills_df = self._conn.execute(
            """
            SELECT date, symbol, quantity
            FROM fills
            WHERE date >= ? AND date <= ?
            """,
            [start, end],
        ).df()

        if fills_df.empty:
            return 0.5

        signals_df["date"] = pd.to_datetime(signals_df["date"]).dt.date
        fills_df["date"] = pd.to_datetime(fills_df["date"]).dt.date

        import numpy as np

        merged = pd.merge(
            signals_df,
            fills_df[["date", "symbol", "quantity"]],
            on=["date", "symbol"],
            how="inner",
        )

        if merged.empty:
            return 0.5

        signal_dir = np.sign(merged["value"])
        trade_dir = np.sign(merged["quantity"])
        hits = (signal_dir == trade_dir).sum()
        return float(hits / len(merged))

    def get_weight_history(self) -> pd.DataFrame:
        """Returns DataFrame of weight evolution over time from registry performance logs."""
        rows = []
        for reg in self._registry.list_all():
            rows.append({
                "signal_name": reg.signal.name,
                "current_weight": reg.weight,
                "initial_weight": reg.initial_weight,
                "active": reg.active,
            })

        if not rows:
            return pd.DataFrame(
                columns=["signal_name", "current_weight", "initial_weight", "active"]
            )

        return pd.DataFrame(rows)

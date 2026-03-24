"""Square-root market impact slippage model for paper trading."""

from __future__ import annotations

import numpy as np


class SlippageModel:
    """Models market impact using a square-root model."""

    K = 50.0  # empirical constant for square-root impact

    def __init__(
        self,
        base_bps: float = 5.0,
        participation_rate: float = 0.01,
    ) -> None:
        self.base_bps = base_bps
        self.participation_rate = participation_rate

    def estimate_cost_bps(
        self,
        quantity: float,
        price: float,
        avg_daily_volume: float,
    ) -> float:
        """Returns slippage cost in basis points.

        impact_bps = base_bps + K * sqrt(order_size / avg_daily_volume)
        where order_size = abs(quantity) * price in dollars,
        and avg_daily_volume is in shares.
        """
        if avg_daily_volume <= 0 or price <= 0:
            return self.base_bps

        order_value = abs(quantity) * price
        adv_value = avg_daily_volume * price
        participation = order_value / adv_value

        impact_bps = self.base_bps + self.K * np.sqrt(participation)
        return float(impact_bps)

    def estimate(
        self,
        symbol: str,
        quantity: float,
        price: float,
        avg_daily_volume: float,
    ) -> float:
        """Compute fill price adjusted for slippage direction.

        Buys get a worse (higher) price; sells get a worse (lower) price.

        Returns: fill_price
        """
        cost_bps = self.estimate_cost_bps(quantity, price, avg_daily_volume)
        cost_fraction = cost_bps / 10_000.0

        if quantity > 0:
            # Buy: pay more
            fill_price = price * (1.0 + cost_fraction)
        else:
            # Sell: receive less
            fill_price = price * (1.0 - cost_fraction)

        return float(fill_price)

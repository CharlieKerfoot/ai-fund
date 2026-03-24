"""Paper trading simulator: fills orders with simulated slippage and commissions."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime

import pandas as pd
import structlog

from execution.paper.slippage import SlippageModel

log = structlog.get_logger()


@dataclass
class Fill:
    fill_id: str
    order_id: str
    symbol: str
    date: date
    quantity: float
    fill_price: float
    slippage_bps: float
    commission: float
    filled_at: datetime


class PaperTradingSimulator:
    COMMISSION_PER_SHARE = 0.005  # $0.005/share

    def __init__(self, slippage_model: SlippageModel, conn) -> None:
        self._slippage = slippage_model
        self._conn = conn

    def simulate_fill(
        self,
        order: dict,
        market_price: float,
        avg_daily_volume: float,
    ) -> Fill:
        """Simulate order fill with slippage and commission.

        Steps:
        1. Apply slippage model to get fill price
        2. Compute commission = COMMISSION_PER_SHARE * abs(quantity)
        3. Create Fill record
        4. Write to fills table
        5. Return Fill
        """
        symbol = order["symbol"]
        quantity = order["quantity"]
        # signed quantity: positive = buy, negative = sell
        side = order.get("side", "buy")
        if isinstance(side, str):
            signed_qty = quantity if side == "buy" else -quantity
        else:
            signed_qty = quantity if side.value == "buy" else -quantity

        fill_price = self._slippage.estimate(
            symbol=symbol,
            quantity=signed_qty,
            price=market_price,
            avg_daily_volume=avg_daily_volume,
        )

        slippage_bps = self._slippage.estimate_cost_bps(
            quantity=quantity,
            price=market_price,
            avg_daily_volume=avg_daily_volume,
        )

        commission = self.COMMISSION_PER_SHARE * abs(quantity)

        now = datetime.now(tz=UTC)
        fill = Fill(
            fill_id=str(uuid.uuid4()),
            order_id=order.get("order_id", str(uuid.uuid4())),
            symbol=symbol,
            date=order.get("date", date.today()),
            quantity=signed_qty,
            fill_price=fill_price,
            slippage_bps=slippage_bps,
            commission=commission,
            filled_at=now,
        )

        self._write_fill(fill)
        log.info(
            "Order filled",
            fill_id=fill.fill_id,
            symbol=fill.symbol,
            quantity=fill.quantity,
            fill_price=round(fill.fill_price, 4),
            slippage_bps=round(fill.slippage_bps, 2),
            commission=round(fill.commission, 4),
        )
        return fill

    def _write_fill(self, fill: Fill) -> None:
        """Persist fill to fills table."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO fills
                (fill_id, order_id, symbol, date, quantity,
                 fill_price, slippage_bps, commission, filled_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                fill.fill_id,
                fill.order_id,
                fill.symbol,
                fill.date,
                fill.quantity,
                fill.fill_price,
                fill.slippage_bps,
                fill.commission,
                fill.filled_at,
            ],
        )

    def get_positions(self, as_of: date) -> pd.DataFrame:
        """Recompute positions from fills table (not maintained incrementally).

        Returns DataFrame: symbol, quantity, avg_cost
        """
        result = self._conn.execute(
            """
            SELECT
                symbol,
                SUM(quantity) AS quantity,
                CASE
                    WHEN SUM(quantity) = 0 THEN 0.0
                    ELSE SUM(fill_price * ABS(quantity)) / SUM(ABS(quantity))
                END AS avg_cost
            FROM fills
            WHERE date <= ?
            GROUP BY symbol
            HAVING SUM(quantity) != 0
            """,
            [as_of],
        ).df()

        return result

    def get_portfolio_value(self, as_of: date, prices: pd.Series) -> float:
        """Mark positions to market using prices Series."""
        positions = self.get_positions(as_of)
        if positions.empty:
            return 0.0

        total = 0.0
        for _, row in positions.iterrows():
            symbol = row["symbol"]
            quantity = row["quantity"]
            price = prices.get(symbol, 0.0)
            total += quantity * price

        return float(total)

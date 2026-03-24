"""Order manager: generates, persists, and tracks orders."""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from execution.orders.types import Order, OrderSide, OrderStatus, OrderType

logger = logging.getLogger(__name__)


class OrderManager:
    def __init__(self, conn, min_order_value: float = 1000.0) -> None:
        self._conn = conn
        self.min_order_value = min_order_value

    def generate_orders(
        self,
        target_weights: pd.Series,
        current_positions: pd.DataFrame,
        portfolio_value: float,
        prices: pd.Series,
    ) -> list[Order]:
        """Diff target weights vs current positions and generate orders.

        Steps:
        1. target_shares = target_weight * portfolio_value / price
        2. current_shares = from positions DataFrame (0 if not held)
        3. delta_shares = target_shares - current_shares
        4. Skip orders < min_order_value
        5. Return list of Orders (BUY if delta > 0, SELL if delta < 0)
        """
        # Build current shares lookup from positions DataFrame
        current_shares: dict[str, float] = {}
        if not current_positions.empty and "symbol" in current_positions.columns:
            for _, row in current_positions.iterrows():
                current_shares[row["symbol"]] = float(row["quantity"])

        orders: list[Order] = []

        for symbol, target_weight in target_weights.items():
            price = prices.get(symbol)
            if price is None or price <= 0:
                logger.warning("No price for symbol %s, skipping order", symbol)
                continue

            target_shares = (target_weight * portfolio_value) / price
            current = current_shares.get(symbol, 0.0)
            delta = target_shares - current

            # Skip small orders
            order_value = abs(delta) * price
            if order_value < self.min_order_value:
                continue

            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            order = Order.create(
                symbol=symbol,
                side=side,
                quantity=abs(delta),
                order_type=OrderType.MARKET,
            )
            orders.append(order)

        return orders

    def persist_orders(self, orders: list[Order]) -> None:
        """Write orders to orders table."""
        for order in orders:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO orders
                    (order_id, symbol, date, side, quantity, order_type,
                     limit_price, status, created_at, filled_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    order.order_id,
                    order.symbol,
                    order.date,
                    order.side.value if isinstance(order.side, OrderSide) else order.side,
                    order.quantity,
                    order.order_type.value if isinstance(order.order_type, OrderType) else order.order_type,
                    order.limit_price,
                    order.status.value if isinstance(order.status, OrderStatus) else order.status,
                    order.created_at,
                    order.filled_at,
                ],
            )

    def cancel_order(self, order_id: str) -> None:
        """Cancel a pending order by setting status to cancelled."""
        self._conn.execute(
            """
            UPDATE orders SET status = ? WHERE order_id = ? AND status = ?
            """,
            [OrderStatus.CANCELLED.value, order_id, OrderStatus.PENDING.value],
        )

    def get_pending_orders(self, as_of: date) -> list[dict]:
        """Return pending orders for the given date."""
        result = self._conn.execute(
            """
            SELECT * FROM orders
            WHERE date = ? AND status = ?
            ORDER BY created_at
            """,
            [as_of, OrderStatus.PENDING.value],
        ).df()
        return result.to_dict(orient="records")

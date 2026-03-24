"""Order types and enumerations for the execution layer."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"
    TWAP = "twap"
    VWAP = "vwap"


class OrderStatus(StrEnum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"


@dataclass
class Order:
    order_id: str
    symbol: str
    date: date
    side: OrderSide
    quantity: float
    order_type: OrderType
    limit_price: float | None
    status: OrderStatus
    created_at: datetime
    filled_at: datetime | None = None

    @classmethod
    def create(
        cls,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
    ) -> Order:
        return cls(
            order_id=str(uuid.uuid4()),
            symbol=symbol,
            date=date.today(),
            side=side,
            quantity=abs(quantity),
            order_type=order_type,
            limit_price=limit_price,
            status=OrderStatus.PENDING,
            created_at=datetime.now(),
        )

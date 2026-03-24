"""Base classes for all trading signals."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass
class SignalMetadata:
    name: str
    version: str           # semver e.g. "1.0.0"
    description: str
    universe: str
    frequency: str         # "daily" | "weekly"
    lookback_days: int
    features_required: list[str]
    author: str = "system"
    created_at: str = ""   # ISO date string


class Signal(ABC):
    """Base class for all signals. Subclasses implement compute() and explain()."""

    @property
    @abstractmethod
    def metadata(self) -> SignalMetadata: ...

    @abstractmethod
    def compute(self, features: pd.DataFrame, as_of: date) -> pd.Series:
        """Compute signal values for all symbols as of given date.

        Args:
            features: DataFrame with MultiIndex (symbol, date) or columns [symbol, date, ...]
            as_of: Point-in-time date — only use data available as of this date

        Returns:
            pd.Series indexed by symbol, values in [-1, 1] range (negative=short, positive=long)
        """

    @abstractmethod
    def explain(self, symbol: str, features: pd.DataFrame, as_of: date) -> str:
        """Return human-readable explanation of signal value for a symbol."""

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def version(self) -> str:
        return self.metadata.version

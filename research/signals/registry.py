"""Signal registry for managing and weighting active signals."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

from research.signals.base import Signal

logger = logging.getLogger(__name__)


@dataclass
class SignalRegistration:
    signal: Signal
    active: bool
    registered_at: str
    weight: float = 1.0
    initial_weight: float = 1.0
    performance: dict = field(default_factory=dict)


class SignalRegistry:
    """Registry for managing signals, their weights, and performance metadata."""

    WEIGHT_FLOOR_RATIO = 0.5  # weight cannot drop below 50% of initial_weight without alert

    def __init__(self, conn=None) -> None:
        self._conn = conn
        self._registrations: dict[str, SignalRegistration] = {}

    def register(self, signal: Signal, weight: float = 1.0) -> None:
        """Register a signal with the given weight."""
        now = datetime.now(tz=UTC).date().isoformat()
        reg = SignalRegistration(
            signal=signal,
            active=True,
            registered_at=now,
            weight=weight,
            initial_weight=weight,
        )
        self._registrations[signal.name] = reg
        logger.info("Registered signal '%s' v%s (weight=%.2f)", signal.name, signal.version, weight)

    def deactivate(self, name: str) -> None:
        """Deactivate a registered signal by name."""
        reg = self._registrations.get(name)
        if reg is None:
            raise KeyError(f"Signal '{name}' not found in registry")
        reg.active = False
        logger.info("Deactivated signal '%s'", name)

    def activate(self, name: str) -> None:
        """Activate a previously deactivated signal by name."""
        reg = self._registrations.get(name)
        if reg is None:
            raise KeyError(f"Signal '{name}' not found in registry")
        reg.active = True
        logger.info("Activated signal '%s'", name)

    def get(self, name: str) -> SignalRegistration | None:
        """Return SignalRegistration for the given name, or None if not found."""
        return self._registrations.get(name)

    def list_active(self) -> list[SignalRegistration]:
        """Return all active signal registrations."""
        return [r for r in self._registrations.values() if r.active]

    def list_all(self) -> list[SignalRegistration]:
        """Return all signal registrations (active and inactive)."""
        return list(self._registrations.values())

    def update_weight(self, name: str, new_weight: float) -> None:
        """Update signal weight. Enforce floor: weight cannot drop below 50% of initial_weight
        without emitting a warning alert."""
        reg = self._registrations.get(name)
        if reg is None:
            raise KeyError(f"Signal '{name}' not found in registry")

        floor = reg.initial_weight * self.WEIGHT_FLOOR_RATIO
        if new_weight < floor:
            logger.warning(
                "ALERT: Signal '%s' weight %.4f is below floor %.4f (50%% of initial %.4f). "
                "Applying floor.",
                name,
                new_weight,
                floor,
                reg.initial_weight,
            )
            new_weight = floor

        old_weight = reg.weight
        reg.weight = new_weight
        logger.info("Updated weight for signal '%s': %.4f -> %.4f", name, old_weight, new_weight)

    def update_performance(self, name: str, metrics: dict) -> None:
        """Update rolling performance metrics for a signal."""
        reg = self._registrations.get(name)
        if reg is None:
            raise KeyError(f"Signal '{name}' not found in registry")
        reg.performance.update(metrics)

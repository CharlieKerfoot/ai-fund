"""Portfolio allocator: combines optimizer, constraints, and risk checks."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class PortfolioAllocator:
    """
    Orchestrates portfolio allocation:
    1. Filter universe to positive-signal assets
    2. Run HRP optimization
    3. Apply regime overlay
    4. Apply constraint engine
    5. Run risk limit check
    """

    def __init__(
        self,
        optimizer,
        constraint_engine,
        risk_checker,
        regime_detector=None,
    ) -> None:
        self.optimizer = optimizer
        self.constraint_engine = constraint_engine
        self.risk_checker = risk_checker
        self.regime_detector = regime_detector

    def allocate(
        self,
        signal_values: pd.Series,
        returns_history: pd.DataFrame,
        current_weights: pd.Series | None,
        regime,  # RegimeState | None
        constraints,  # PortfolioConstraints
    ) -> tuple[pd.Series, object]:
        """
        1. Filter universe: only symbols with positive signal (long-only)
        2. Run HRP optimizer on filtered universe
        3. Apply constraint engine (normalizes to sum=1.0)
        4. Regime overlay: reduce gross exposure in crisis/risk_off regime
           - risk_on: full allocation (sum=1.0)
           - risk_off: scale to 80% gross (sum=0.8, remainder in cash)
           - crisis: scale to 60% gross (sum=0.6, remainder in cash)
        5. Run risk limit check
        6. Return (final_weights, risk_result)
        """
        # Step 1: Filter to positive-signal assets
        positive_signals = signal_values[signal_values > 0]

        if positive_signals.empty:
            logger.warning("No positive-signal assets; returning empty weights")
            empty = pd.Series(dtype=float)
            from portfolio.risk.limits import RiskCheckResult
            risk_result = RiskCheckResult(
                passed=True,
                violations=[],
                constrained_weights=empty,
                alerts=["No positive-signal assets in universe"],
            )
            return empty, risk_result

        # Get returns for positive-signal assets only
        available_symbols = [
            s for s in positive_signals.index if s in returns_history.columns
        ]

        if not available_symbols:
            logger.warning("No return history for positive-signal assets")
            empty = pd.Series(dtype=float)
            from portfolio.risk.limits import RiskCheckResult
            risk_result = RiskCheckResult(
                passed=True,
                violations=[],
                constrained_weights=empty,
                alerts=["No return history for positive-signal assets"],
            )
            return empty, risk_result

        filtered_returns = returns_history[available_symbols].dropna(how="all")

        # Step 2: Run HRP optimizer
        weights = self.optimizer.optimize(filtered_returns)

        # Step 3: Apply constraint engine (normalizes to 1.0)
        weights = self.constraint_engine.apply(weights, current_weights, constraints)

        # Step 4: Regime overlay (applied after constraints to preserve gross exposure)
        weights = self._apply_regime_overlay(weights, regime)

        # Step 5: Run risk limit check
        risk_result = self.risk_checker.check(weights)

        # Use constrained weights from risk checker
        final_weights = risk_result.constrained_weights

        return final_weights, risk_result

    def _apply_regime_overlay(
        self, weights: pd.Series, regime
    ) -> pd.Series:
        """
        Scale portfolio gross exposure based on regime:
        - risk_on: 100% (full allocation)
        - risk_off: 80% (20% cash)
        - stagflation: 90%
        - deflation: 90%
        - crisis: 60% (40% cash)
        """
        if regime is None:
            return weights

        dominant = getattr(regime, "dominant_regime", "risk_on")

        scale_map = {
            "risk_on": 1.0,
            "risk_off": 0.8,
            "stagflation": 0.9,
            "deflation": 0.9,
            "crisis": 0.6,
        }

        scale = scale_map.get(dominant, 1.0)

        if scale < 1.0:
            # Scale down weights; remainder implicitly in cash (not included in weights)
            weights = weights * scale

        return weights

"""Risk limit checking and enforcement."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RiskCheckResult:
    passed: bool
    violations: list[str]
    constrained_weights: pd.Series  # weights after applying hard limits
    alerts: list[str]  # warnings that don't block execution


class RiskLimitChecker:
    """
    Checks portfolio weights and PnL against hard risk limits.
    Hard limits cause violations; soft limits generate alerts.
    """

    def __init__(self, limits: dict) -> None:
        self.limits = limits
        self.max_position_pct: float = limits.get("max_position_pct", 0.05)
        self.max_daily_loss_pct: float = limits.get("max_daily_loss_pct", 0.03)
        self.max_gross_leverage: float = limits.get("max_gross_leverage", 1.0)
        self.max_sector_pct: float = limits.get("max_sector_pct", 0.25)

    def check(
        self,
        weights: pd.Series,
        daily_pnl: float | None = None,
        portfolio_value: float | None = None,
    ) -> RiskCheckResult:
        """
        Hard limits:
        - max_position_pct: cap and renormalize
        - max_daily_loss_pct: halt execution (flag, not enforce)
        - max_gross_leverage: cap total weight

        Soft alerts:
        - Positions approaching limits (>80% of cap)
        """
        violations: list[str] = []
        alerts: list[str] = []

        if weights.empty:
            return RiskCheckResult(
                passed=True,
                violations=[],
                constrained_weights=weights.copy(),
                alerts=[],
            )

        w = weights.copy().astype(float)

        # Hard limit 1: max_position_pct
        over_limit = w[w > self.max_position_pct + 1e-6]
        if not over_limit.empty:
            for symbol, wt in over_limit.items():
                violations.append(
                    f"Position {symbol} = {wt:.2%} exceeds max {self.max_position_pct:.2%}"
                )
            # Cap and renormalize
            w = self._cap_and_renormalize(w, self.max_position_pct)

        # Hard limit 2: max_gross_leverage (sum of weights for long-only = 1, but check)
        gross_leverage = w.abs().sum()
        if gross_leverage > self.max_gross_leverage + 1e-6:
            violations.append(
                f"Gross leverage {gross_leverage:.3f} exceeds max {self.max_gross_leverage:.3f}"
            )
            # Scale down
            w = w / gross_leverage * self.max_gross_leverage

        # Hard limit 3: daily loss check (flag only — execution layer halts)
        if daily_pnl is not None and portfolio_value is not None and portfolio_value > 0:
            daily_loss_pct = -daily_pnl / portfolio_value
            if daily_loss_pct > self.max_daily_loss_pct:
                violations.append(
                    f"Daily loss {daily_loss_pct:.2%} exceeds limit {self.max_daily_loss_pct:.2%}; "
                    "execution should halt"
                )
        elif daily_pnl is not None and daily_pnl < 0 and portfolio_value is None:
            # Can still flag if pnl is negative without value context
            pass

        # Soft alerts: positions approaching limit (>80% of cap)
        approach_threshold = 0.80 * self.max_position_pct
        approaching = w[(w > approach_threshold) & (w <= self.max_position_pct + 1e-6)]
        for symbol, wt in approaching.items():
            alerts.append(
                f"Position {symbol} = {wt:.2%} approaching limit {self.max_position_pct:.2%} "
                f"(>{approach_threshold:.2%})"
            )

        passed = len(violations) == 0

        return RiskCheckResult(
            passed=passed,
            violations=violations,
            constrained_weights=w,
            alerts=alerts,
        )

    def _cap_and_renormalize(self, weights: pd.Series, max_pct: float) -> pd.Series:
        """Cap positions at max_pct and renormalize iteratively."""
        w = weights.copy()
        for _ in range(100):
            over = w > max_pct
            if not over.any():
                break
            excess = (w[over] - max_pct).sum()
            w[over] = max_pct
            under = ~over & (w > 0)
            if under.any():
                under_total = w[under].sum()
                if under_total > 0:
                    w[under] += excess * (w[under] / under_total)
            else:
                break
        return w

"""Stress testing via historical and hypothetical scenarios."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)

SCENARIOS: dict[str, dict] = {
    "2008_gfc": {
        "description": "2008 Global Financial Crisis",
        "start": "2008-09-01",
        "end": "2009-03-31",
        "equity_shock": -0.50,
        "vol_multiplier": 3.0,
        "credit_spread_bps": 500,
    },
    "2020_covid": {
        "description": "2020 COVID Crash",
        "start": "2020-02-19",
        "end": "2020-03-23",
        "equity_shock": -0.34,
        "vol_multiplier": 4.0,
        "credit_spread_bps": 300,
    },
    "2022_rates": {
        "description": "2022 Rate Shock",
        "start": "2022-01-01",
        "end": "2022-12-31",
        "equity_shock": -0.20,
        "vol_multiplier": 1.5,
        "credit_spread_bps": 150,
    },
}


@dataclass
class StressTestResult:
    scenario_name: str
    portfolio_loss_pct: float
    worst_position: str
    worst_position_loss: float
    max_drawdown_estimate: float
    breaches_daily_limit: bool


class StressTester:
    """
    Applies scenario shocks to estimate portfolio loss under stress conditions.
    """

    def __init__(self, daily_loss_limit: float = 0.03) -> None:
        self.daily_loss_limit = daily_loss_limit

    def run_scenario(
        self, weights: pd.Series, scenario_name: str
    ) -> StressTestResult:
        """Apply scenario shocks to estimate portfolio loss."""
        if scenario_name not in SCENARIOS:
            raise ValueError(
                f"Unknown scenario '{scenario_name}'. "
                f"Available: {list(SCENARIOS.keys())}"
            )

        scenario = SCENARIOS[scenario_name]
        equity_shock = scenario["equity_shock"]

        if weights.empty:
            return StressTestResult(
                scenario_name=scenario_name,
                portfolio_loss_pct=0.0,
                worst_position="",
                worst_position_loss=0.0,
                max_drawdown_estimate=0.0,
                breaches_daily_limit=False,
            )

        # Normalize weights
        total = weights.sum()
        if total <= 0:
            return StressTestResult(
                scenario_name=scenario_name,
                portfolio_loss_pct=0.0,
                worst_position="",
                worst_position_loss=0.0,
                max_drawdown_estimate=0.0,
                breaches_daily_limit=False,
            )

        w = weights / total

        # Apply equity shock uniformly to all positions (long-only portfolio)
        position_losses = w * equity_shock  # negative = loss

        portfolio_loss = float(position_losses.sum())  # negative

        # Find worst single-position loss
        worst_symbol = str(position_losses.idxmin())
        worst_loss = float(position_losses.min())  # negative

        # Estimate max drawdown: use equity shock as drawdown proxy
        # For more severe scenarios, assume some mean reversion
        vol_multiplier = scenario.get("vol_multiplier", 1.0)
        max_drawdown_estimate = abs(equity_shock) * (1.0 + 0.1 * (vol_multiplier - 1.0))

        # Portfolio loss is negative; check if abs(portfolio_loss) > daily_loss_limit
        # For scenario testing, we compare the full scenario loss to the daily limit
        # This is a flag-only check, not enforcement
        breaches_daily_limit = abs(portfolio_loss) > self.daily_loss_limit

        return StressTestResult(
            scenario_name=scenario_name,
            portfolio_loss_pct=portfolio_loss,  # negative value
            worst_position=worst_symbol,
            worst_position_loss=worst_loss,
            max_drawdown_estimate=max_drawdown_estimate,
            breaches_daily_limit=breaches_daily_limit,
        )

    def run_all(self, weights: pd.Series) -> list[StressTestResult]:
        """Run all defined scenarios."""
        results = []
        for scenario_name in SCENARIOS:
            try:
                result = self.run_scenario(weights, scenario_name)
                results.append(result)
            except Exception as e:
                logger.warning("Stress test '%s' failed: %s", scenario_name, e)
        return results

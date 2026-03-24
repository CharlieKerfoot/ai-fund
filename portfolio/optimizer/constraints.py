"""Portfolio constraint definitions and enforcement engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PortfolioConstraints:
    max_position_pct: float = 0.05
    max_sector_pct: float = 0.25
    max_turnover: float = 0.5  # max total weight change per rebalance
    max_gross_leverage: float = 1.0
    min_positions: int = 20
    sector_map: dict[str, str] = field(default_factory=dict)  # symbol -> sector


class ConstraintEngine:
    """Applies portfolio constraints iteratively and verifies compliance."""

    def apply(
        self,
        weights: pd.Series,
        current_weights: pd.Series | None,
        constraints: PortfolioConstraints,
    ) -> pd.Series:
        """
        Apply constraints iteratively:
        1. Normalize to sum to 1.0
        2. Iterate: cap positions + cap sectors until convergence
        3. Cap turnover vs current weights
        4. Re-normalize to sum to 1.0
        5. Verify min_positions (if fewer than min, add equal weight top names)
        """
        if weights.empty:
            return weights

        w = weights.copy().astype(float)

        # Initial normalize
        w = self._normalize(w)

        # Iteratively apply position and sector caps until convergence
        for _ in range(50):
            w_prev = w.copy()

            # Cap individual positions (redistributes excess to under-cap assets)
            w = self._cap_positions(w, constraints.max_position_pct)

            # Cap sectors (redistributes excess to non-sector assets)
            if constraints.sector_map:
                w = self._cap_sectors(w, constraints.sector_map, constraints.max_sector_pct)

            # Check convergence
            if (w - w_prev).abs().max() < 1e-8:
                break

        # Cap turnover
        if current_weights is not None and not current_weights.empty:
            w = self._cap_turnover(w, current_weights, constraints.max_turnover)

        # Final normalize
        w = self._normalize(w)

        # Ensure min_positions
        w = self._ensure_min_positions(w, constraints.min_positions)

        return w

    def check_violations(
        self,
        weights: pd.Series,
        constraints: PortfolioConstraints,
    ) -> list[str]:
        """Return list of constraint violation messages."""
        violations = []

        if weights.empty:
            return violations

        # Check individual position caps
        over_limit = weights[weights > constraints.max_position_pct + 1e-6]
        for symbol, w in over_limit.items():
            violations.append(
                f"Position {symbol} = {w:.2%} exceeds max {constraints.max_position_pct:.2%}"
            )

        # Check sector caps
        if constraints.sector_map:
            sector_weights = {}
            for symbol, w in weights.items():
                sector = constraints.sector_map.get(str(symbol))
                if sector:
                    sector_weights[sector] = sector_weights.get(sector, 0.0) + w
            for sector, sw in sector_weights.items():
                if sw > constraints.max_sector_pct + 1e-6:
                    violations.append(
                        f"Sector {sector} = {sw:.2%} exceeds max {constraints.max_sector_pct:.2%}"
                    )

        # Check min positions
        active = (weights > 1e-6).sum()
        if active < constraints.min_positions:
            violations.append(
                f"Only {active} positions, minimum is {constraints.min_positions}"
            )

        return violations

    def _cap_positions(self, weights: pd.Series, max_pct: float) -> pd.Series:
        """Cap individual positions and renormalize."""
        if max_pct >= 1.0:
            return weights

        # Iterative capping until all positions within limit
        w = weights.copy()
        for _ in range(100):
            over = w > max_pct
            if not over.any():
                break
            excess = (w[over] - max_pct).sum()
            w[over] = max_pct
            # Redistribute excess to under-cap positions
            under = ~over & (w > 0)
            if under.any():
                under_total = w[under].sum()
                if under_total > 0:
                    w[under] += excess * (w[under] / under_total)
            elif not over.any():
                break

        return w

    def _cap_sectors(
        self,
        weights: pd.Series,
        sector_map: dict[str, str],
        max_sector_pct: float,
    ) -> pd.Series:
        """
        Cap sector exposures. Excess weight is redistributed proportionally
        to assets outside the over-cap sector.
        """
        w = weights.copy()

        # Group by sector
        sectors: dict[str, list] = {}
        for symbol in w.index:
            sector = sector_map.get(str(symbol))
            if sector:
                sectors.setdefault(sector, []).append(symbol)

        for _ in range(50):
            any_over = False
            for sector, syms in sectors.items():
                sector_w = w[syms].sum()
                if sector_w > max_sector_pct + 1e-8:
                    any_over = True
                    # Scale down this sector
                    excess = sector_w - max_sector_pct
                    scale = max_sector_pct / sector_w
                    w[syms] *= scale
                    # Redistribute excess to assets NOT in this sector
                    outside_mask = ~w.index.isin(syms) & (w > 0)
                    outside_total = w[outside_mask].sum()
                    if outside_total > 0:
                        w[outside_mask] += excess * (w[outside_mask] / outside_total)
            if not any_over:
                break

        return w

    def _cap_turnover(
        self,
        weights: pd.Series,
        current_weights: pd.Series,
        max_turnover: float,
    ) -> pd.Series:
        """Limit turnover (total absolute weight change) to max_turnover."""
        # Align indices
        all_symbols = weights.index.union(current_weights.index)
        w = weights.reindex(all_symbols, fill_value=0.0)
        cw = current_weights.reindex(all_symbols, fill_value=0.0)

        # Only keep symbols that are in the proposed weights
        w = w[weights.index]
        cw = cw.reindex(weights.index, fill_value=0.0)

        turnover = (w - cw).abs().sum()

        if turnover <= max_turnover:
            return w

        # Scale changes proportionally to stay within turnover budget
        delta = w - cw
        scale = max_turnover / turnover
        w_capped = cw + delta * scale

        # Clip to non-negative and renormalize
        w_capped = w_capped.clip(lower=0)
        total = w_capped.sum()
        if total > 0:
            w_capped /= total

        return w_capped

    def _normalize(self, weights: pd.Series) -> pd.Series:
        """Normalize weights to sum to 1.0."""
        total = weights.sum()
        if total > 1e-10:
            return weights / total
        return weights

    def _ensure_min_positions(
        self, weights: pd.Series, min_positions: int
    ) -> pd.Series:
        """If fewer than min_positions active, add equal weight to top names."""
        active_count = (weights > 1e-6).sum()
        if active_count >= min_positions or len(weights) < min_positions:
            return weights

        w = weights.copy()
        # Sort by existing weight descending to find top names to activate
        w.index.tolist()
        zero_symbols = w[w <= 1e-6].index.tolist()

        needed = min_positions - active_count
        if not zero_symbols:
            return w

        # Add equal small weight to the needed symbols
        to_add = zero_symbols[:needed]
        # Use a small base allocation
        small_weight = 1e-4
        for sym in to_add:
            w[sym] = small_weight

        # Renormalize
        total = w.sum()
        if total > 0:
            w /= total

        return w

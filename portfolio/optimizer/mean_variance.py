"""Mean-Variance Optimizer via cvxpy."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MeanVarianceOptimizer:
    """
    Long-only Mean-Variance Optimization.
    Maximizes: mu'w - (lambda/2) * w'Sigma*w
    Subject to: sum(w) = 1, w >= 0, position constraints.
    """

    def __init__(self, risk_aversion: float = 1.0) -> None:
        self.risk_aversion = risk_aversion

    def optimize(
        self,
        expected_returns: pd.Series,
        covariance: pd.DataFrame,
        constraints: dict | None = None,
    ) -> pd.Series:
        """
        Long-only MVO via cvxpy.
        Objective: maximize mu'w - (lambda/2)*w'Sigma*w
        Subject to: sum(w) = 1, w >= 0, position constraints.
        Falls back to equal weight if problem infeasible.
        """
        try:
            import cvxpy as cp
        except ImportError:
            logger.error("cvxpy not installed; falling back to equal weight")
            return self._equal_weight(expected_returns.index)

        symbols = expected_returns.index.tolist()
        n = len(symbols)

        if n == 0:
            return pd.Series(dtype=float)

        mu = expected_returns.values.astype(float)
        sigma = covariance.loc[symbols, symbols].values.astype(float)

        # Ensure sigma is positive semidefinite
        sigma = (sigma + sigma.T) / 2
        min_eig = np.linalg.eigvalsh(sigma).min()
        if min_eig < 0:
            sigma += (-min_eig + 1e-8) * np.eye(n)

        w = cp.Variable(n)
        lam = self.risk_aversion

        objective = cp.Maximize(mu @ w - (lam / 2) * cp.quad_form(w, sigma))

        cvx_constraints = [
            cp.sum(w) == 1,
            w >= 0,
        ]

        # Position cap constraints
        if constraints:
            max_pos = constraints.get("max_position_pct", None)
            if max_pos is not None:
                cvx_constraints.append(w <= max_pos)

        problem = cp.Problem(objective, cvx_constraints)

        try:
            problem.solve(solver=cp.CLARABEL, warm_start=True)

            if problem.status in ("optimal", "optimal_inaccurate") and w.value is not None:
                raw_weights = np.array(w.value).flatten()
                raw_weights = np.clip(raw_weights, 0, None)
                total = raw_weights.sum()
                if total > 1e-10:
                    raw_weights /= total
                    return pd.Series(raw_weights, index=symbols)

            logger.warning(
                "MVO problem status '%s'; falling back to equal weight",
                problem.status,
            )
            return self._equal_weight(symbols)

        except Exception as e:
            logger.warning("MVO optimization failed (%s); falling back to equal weight", e)
            return self._equal_weight(symbols)

    def _equal_weight(self, symbols) -> pd.Series:
        """Return equal-weight portfolio."""
        n = len(symbols)
        if n == 0:
            return pd.Series(dtype=float)
        return pd.Series([1.0 / n] * n, index=symbols)

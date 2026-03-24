"""Risk Parity Optimizer: equal risk contribution via scipy."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class RiskParityOptimizer:
    """
    Equal Risk Contribution (ERC) optimization.
    Each asset contributes equally to total portfolio risk.
    """

    def optimize(
        self,
        covariance: pd.DataFrame,
        constraints: dict | None = None,
    ) -> pd.Series:
        """
        Equal risk contribution optimization via scipy.optimize.minimize.
        Objective: minimize sum of (w_i * (Sigma*w)_i - target_risk_i)^2
        where target_risk = 1/n for each asset.
        Falls back to inverse-vol weighting if optimization fails.
        """
        symbols = covariance.index.tolist()
        n = len(symbols)

        if n == 0:
            return pd.Series(dtype=float)

        if n == 1:
            return pd.Series([1.0], index=symbols)

        sigma = covariance.values.astype(float)

        # Ensure sigma is positive semidefinite
        sigma = (sigma + sigma.T) / 2
        min_eig = np.linalg.eigvalsh(sigma).min()
        if min_eig < 1e-8:
            sigma += (abs(min_eig) + 1e-8) * np.eye(n)

        target = 1.0 / n

        def objective(w: np.ndarray) -> float:
            """Sum of squared deviations from equal risk contribution."""
            portfolio_var = w @ sigma @ w
            if portfolio_var <= 0:
                return 1e10
            # Marginal risk contribution * weight = risk contribution
            mrc = sigma @ w
            rc = w * mrc
            total_rc = rc.sum()
            if total_rc <= 0:
                return 1e10
            rc_normalized = rc / total_rc
            return float(np.sum((rc_normalized - target) ** 2))

        def gradient(w: np.ndarray) -> np.ndarray:
            """Analytical gradient of the objective."""
            portfolio_var = w @ sigma @ w
            if portfolio_var <= 0:
                return np.zeros(n)
            mrc = sigma @ w
            rc = w * mrc
            total_rc = rc.sum()
            if total_rc <= 0:
                return np.zeros(n)
            rc_normalized = rc / total_rc
            rc_normalized - target

            # d(rc_i)/d(w_j) = delta_{ij} * mrc_i + w_i * (2*sigma_{ij}*w_j + ... )
            # Use numerical gradient for simplicity
            return None  # scipy will use numerical if jac=None

        # Initial guess: inverse-vol weights
        vols = np.sqrt(np.maximum(np.diag(sigma), 1e-10))
        inv_vol = 1.0 / vols
        w0 = inv_vol / inv_vol.sum()

        # Bounds: all weights non-negative
        bounds = [(1e-6, 1.0)] * n

        # Equality constraint: weights sum to 1
        eq_constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        # Position cap from constraints dict
        if constraints:
            max_pos = constraints.get("max_position_pct", None)
            if max_pos is not None:
                bounds = [(1e-6, max_pos)] * n

        try:
            result = minimize(
                objective,
                w0,
                method="SLSQP",
                bounds=bounds,
                constraints=eq_constraints,
                options={"ftol": 1e-9, "maxiter": 1000, "disp": False},
            )

            if result.success or result.fun < 1e-6:
                raw_weights = np.clip(result.x, 0, None)
                total = raw_weights.sum()
                if total > 1e-10:
                    raw_weights /= total
                    return pd.Series(raw_weights, index=symbols)

            logger.warning(
                "Risk parity optimization did not converge (fun=%.6f); "
                "falling back to inverse-vol",
                result.fun,
            )
            return self._inverse_vol_weights(sigma, symbols)

        except Exception as e:
            logger.warning(
                "Risk parity optimization failed (%s); falling back to inverse-vol", e
            )
            return self._inverse_vol_weights(sigma, symbols)

    def _inverse_vol_weights(
        self, sigma: np.ndarray, symbols: list
    ) -> pd.Series:
        """Inverse-volatility fallback weights."""
        vols = np.sqrt(np.maximum(np.diag(sigma), 1e-10))
        inv_vol = 1.0 / vols
        weights = inv_vol / inv_vol.sum()
        return pd.Series(weights, index=symbols)

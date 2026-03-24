"""Value at Risk (VaR) and Expected Shortfall calculations."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)


class VaRCalculator:
    """Computes parametric and historical VaR, plus Expected Shortfall."""

    def parametric_var(
        self,
        weights: pd.Series,
        returns: pd.DataFrame,
        confidence: float = 0.95,
        horizon_days: int = 1,
    ) -> float:
        """
        Parametric VaR assuming normal returns.
        Returns positive loss value (e.g., 0.02 means 2% loss).
        """
        # Align weights and returns
        common = weights.index.intersection(returns.columns)
        if len(common) == 0:
            return 0.0

        w = weights.reindex(common).fillna(0.0).values
        w = w / w.sum() if w.sum() > 0 else w

        ret = returns[common].dropna()
        if len(ret) < 2:
            return 0.0

        cov = ret.cov().values
        portfolio_variance = float(w @ cov @ w)
        portfolio_vol = np.sqrt(max(portfolio_variance, 0.0))

        # Scale for horizon
        portfolio_vol_scaled = portfolio_vol * np.sqrt(horizon_days)

        # VaR = z_alpha * sigma (positive loss)
        z = norm.ppf(confidence)
        var = z * portfolio_vol_scaled

        return float(max(var, 0.0))

    def historical_var(
        self,
        weights: pd.Series,
        returns: pd.DataFrame,
        confidence: float = 0.95,
        horizon_days: int = 1,
        lookback_days: int = 252,
    ) -> float:
        """
        Historical simulation VaR.
        Returns positive loss value at the given confidence level.
        """
        common = weights.index.intersection(returns.columns)
        if len(common) == 0:
            return 0.0

        w = weights.reindex(common).fillna(0.0).values
        w = w / w.sum() if w.sum() > 0 else w

        ret = returns[common].dropna()

        # Use lookback window
        if len(ret) > lookback_days:
            ret = ret.iloc[-lookback_days:]

        if len(ret) < 2:
            return 0.0

        # Portfolio daily returns
        port_returns = ret.values @ w

        # Scale for horizon (simple approximation)
        if horizon_days > 1:
            port_returns = port_returns * np.sqrt(horizon_days)

        # VaR = negative of the (1-confidence) quantile of returns
        var = -np.percentile(port_returns, (1 - confidence) * 100)

        return float(max(var, 0.0))

    def expected_shortfall(
        self,
        weights: pd.Series,
        returns: pd.DataFrame,
        confidence: float = 0.95,
    ) -> float:
        """
        CVaR / Expected Shortfall = mean of losses beyond VaR threshold.
        Always >= VaR.
        """
        common = weights.index.intersection(returns.columns)
        if len(common) == 0:
            return 0.0

        w = weights.reindex(common).fillna(0.0).values
        w = w / w.sum() if w.sum() > 0 else w

        ret = returns[common].dropna()
        if len(ret) < 2:
            return 0.0

        # Portfolio daily returns
        port_returns = ret.values @ w

        # Find the VaR threshold
        var_threshold = np.percentile(port_returns, (1 - confidence) * 100)

        # ES = mean of returns below VaR threshold (i.e., losses beyond VaR)
        tail_losses = port_returns[port_returns <= var_threshold]
        if len(tail_losses) == 0:
            return float(max(-var_threshold, 0.0))

        es = -tail_losses.mean()

        return float(max(es, 0.0))

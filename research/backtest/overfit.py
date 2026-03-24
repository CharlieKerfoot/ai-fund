"""Overfit protection using Deflated Sharpe Ratio and Bonferroni correction."""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import stats


@dataclass
class OverfitReport:
    signal_name: str
    n_trials: int                  # number of parameter combinations tested
    deflated_sharpe: float         # DSR (z-score)
    dsr_significant: bool          # True if DSR > 0 at 95% confidence (p-value < 0.05)
    bonferroni_correction: float   # adjusted p-value
    bh_significant: bool           # Benjamini-Hochberg significance
    min_backtest_years: float      # minimum backtest length (Bailey-Lopez de Prado)
    actual_backtest_years: float
    sufficient_history: bool
    recommendation: str            # "PASS" | "CAUTION" | "REJECT"


class OverfitGuard:
    """Implements overfit detection metrics from Bailey & Lopez de Prado (2014)."""

    def check(
        self,
        sharpe: float,
        n_trials: int,
        backtest_years: float,
        annual_vol: float,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
        signal_name: str = "unknown",
    ) -> OverfitReport:
        """Run overfit checks.

        1. Deflated Sharpe Ratio (Bailey-Lopez de Prado 2014)
        2. Bonferroni correction: p_adjusted = 1 - (1-p)^n_trials
        3. Minimum backtest length from Bailey-Lopez de Prado
        """
        n_trials = max(1, n_trials)

        # Convert annual Sharpe to per-observation Sharpe (daily)
        int(backtest_years * 252)  # approximate number of daily observations

        dsr = self._deflated_sharpe(
            sharpe=sharpe,
            n_trials=n_trials,
            backtest_years=backtest_years,
            annual_vol=annual_vol,
            skewness=skewness,
            kurtosis=kurtosis,
        )

        # DSR is a z-score: p-value from normal distribution
        p_dsr = float(stats.norm.sf(dsr))  # one-sided
        dsr_significant = p_dsr < 0.05

        # Bonferroni: adjusted p-value
        # P(at least one spurious | n_trials) = 1 - (1 - p_dsr)^n_trials
        bonferroni_p = 1.0 - (1.0 - p_dsr) ** n_trials
        bonferroni_significant = bonferroni_p < 0.05

        # Benjamini-Hochberg: for a single test with m comparisons,
        # significance threshold is 0.05 * rank/m; for rank=1 (worst case): 0.05/m
        bh_threshold = 0.05 / n_trials
        bh_significant = p_dsr < bh_threshold

        # Minimum backtest length (Bailey-Lopez de Prado formula)
        min_years = self._min_backtest_years(
            n_trials=n_trials,
            annual_vol=annual_vol,
            skewness=skewness,
            kurtosis=kurtosis,
        )
        sufficient_history = backtest_years >= min_years

        # Determine recommendation
        if not dsr_significant or not sufficient_history:
            recommendation = "REJECT"
        elif not bonferroni_significant or not bh_significant:
            recommendation = "CAUTION"
        else:
            recommendation = "PASS"

        return OverfitReport(
            signal_name=signal_name,
            n_trials=n_trials,
            deflated_sharpe=float(dsr),
            dsr_significant=dsr_significant,
            bonferroni_correction=float(bonferroni_p),
            bh_significant=bh_significant,
            min_backtest_years=float(min_years),
            actual_backtest_years=float(backtest_years),
            sufficient_history=sufficient_history,
            recommendation=recommendation,
        )

    def _deflated_sharpe(
        self,
        sharpe: float,
        n_trials: int,
        backtest_years: float,
        annual_vol: float,
        skewness: float,
        kurtosis: float,
    ) -> float:
        """Compute the Deflated Sharpe Ratio z-score.

        SR benchmark (expected max SR from n random strategies):
            SR* = (1 - euler_gamma) * Z^{-1}(1 - 1/n) + euler_gamma * Z^{-1}(1 - 1/(n*e))

        Variance of SR_hat (accounting for non-normality, T observations):
            V[SR_hat] = (1/T) * (1 + 0.5*SR^2 - skew*SR + ((kurtosis-3)/4)*SR^2)

        DSR = (SR_hat - SR*) / sqrt(V[SR_hat])
        """
        euler_gamma = 0.5772156649  # Euler-Mascheroni constant
        e = math.e

        # Convert annual Sharpe to per-observation (daily) Sharpe
        t_obs = backtest_years * 252
        sr_daily = sharpe / math.sqrt(252)

        # SR benchmark
        if n_trials == 1:
            sr_star_daily = 0.0
        else:
            z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
            z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * e))
            sr_star_annual = (1.0 - euler_gamma) * z1 + euler_gamma * z2
            sr_star_daily = sr_star_annual / math.sqrt(252)

        # Variance of SR_hat (per-observation)
        # V[SR_hat] = (1/T) * (1 + 0.5*SR^2 - skew*SR + ((kurtosis-3)/4)*SR^2)
        if t_obs <= 1:
            return 0.0

        var_sr = (1.0 / t_obs) * (
            1.0
            + 0.5 * sr_daily ** 2
            - skewness * sr_daily
            + ((kurtosis - 3.0) / 4.0) * sr_daily ** 2
        )
        if var_sr <= 0:
            return 0.0

        dsr = (sr_daily - sr_star_daily) / math.sqrt(var_sr)
        return float(dsr)

    def _min_backtest_years(
        self,
        n_trials: int,
        annual_vol: float,
        skewness: float,
        kurtosis: float,
        target_sharpe: float = 1.0,
    ) -> float:
        """Minimum backtest length from Bailey-Lopez de Prado.

        Derived from: T_min such that SR* < target_sharpe at the 5% significance level.

        Simplified: T_min = (Z_{1 - 1/n} / target_sr_daily)^2 * V_factor
        where V_factor accounts for non-normality.

        We solve: (target_sr_daily - sr_star_daily)^2 / V[SR_hat] = z_{0.95}^2
        """
        euler_gamma = 0.5772156649
        e = math.e
        z_95 = stats.norm.ppf(0.95)

        target_sr_daily = target_sharpe / math.sqrt(252)

        if n_trials <= 1:
            sr_star_daily = 0.0
        else:
            z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
            z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * e))
            sr_star_annual = (1.0 - euler_gamma) * z1 + euler_gamma * z2
            sr_star_daily = sr_star_annual / math.sqrt(252)

        # Variance factor (numerator of V, excluding 1/T)
        v_factor = (
            1.0
            + 0.5 * target_sr_daily ** 2
            - skewness * target_sr_daily
            + ((kurtosis - 3.0) / 4.0) * target_sr_daily ** 2
        )
        v_factor = max(v_factor, 1e-8)

        excess = target_sr_daily - sr_star_daily
        if excess <= 0:
            # Target SR is at or below benchmark — need infinite data
            return float("inf")

        # T_min such that DSR = z_95
        # z_95 = excess / sqrt(v_factor / T_min)
        # T_min = v_factor * (z_95 / excess)^2
        t_min_obs = v_factor * (z_95 / excess) ** 2
        return t_min_obs / 252.0  # convert to years

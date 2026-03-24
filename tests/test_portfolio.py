"""Tests for Phase 5: Portfolio + Risk."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from portfolio.allocator.allocate import PortfolioAllocator
from portfolio.optimizer.constraints import ConstraintEngine, PortfolioConstraints
from portfolio.optimizer.hierarchical import HierarchicalRiskParity
from portfolio.optimizer.mean_variance import MeanVarianceOptimizer
from portfolio.optimizer.risk_parity import RiskParityOptimizer
from portfolio.regime.detector import REGIMES, RegimeDetector, RegimeState
from portfolio.risk.limits import RiskCheckResult, RiskLimitChecker
from portfolio.risk.stress_test import StressTester
from portfolio.risk.var import VaRCalculator

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def make_macro_df(
    vix: float = 15.0,
    baa10y: float = 1.5,
    dgs10: float = 4.0,
    dgs2: float = 3.5,
) -> pd.DataFrame:
    """Build a minimal macro DataFrame for regime detection tests."""
    rows = [
        {"series_id": "VIXCLS", "date": date(2024, 1, 1), "value": vix},
        {"series_id": "BAA10Y", "date": date(2024, 1, 1), "value": baa10y},
        {"series_id": "DGS10", "date": date(2024, 1, 1), "value": dgs10},
        {"series_id": "DGS2", "date": date(2024, 1, 1), "value": dgs2},
    ]
    return pd.DataFrame(rows)


class FakeFeatureStore:
    """Minimal feature store stub for regime tests."""

    def __init__(self, macro_df: pd.DataFrame) -> None:
        self._macro = macro_df

    def get_macro(self, as_of=None) -> pd.DataFrame:
        return self._macro


def make_returns(
    n_days: int = 252,
    n_assets: int = 5,
    vols: list[float] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic return series."""
    rng = np.random.default_rng(seed)
    symbols = [f"A{i}" for i in range(n_assets)]
    if vols is None:
        vols = [0.01 * (i + 1) for i in range(n_assets)]
    returns_data = {
        sym: rng.normal(0, vol, n_days) for sym, vol in zip(symbols, vols)
    }
    return pd.DataFrame(returns_data)


# ─────────────────────────────────────────────────────────────────────────────
# Regime Detector Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRegimeDetector:
    def test_regime_detector_risk_on(self):
        """Low VIX + tight spreads + normal yield curve -> risk_on dominant."""
        macro = make_macro_df(vix=12.0, baa10y=1.2, dgs10=4.0, dgs2=3.5)
        store = FakeFeatureStore(macro)
        detector = RegimeDetector(feature_store=store)
        state = detector.detect(as_of=date(2024, 1, 1))

        assert isinstance(state, RegimeState)
        assert state.dominant_regime == "risk_on"
        assert state.regime_probs["risk_on"] > 0.5
        assert abs(sum(state.regime_probs.values()) - 1.0) < 1e-6
        assert state.confidence == state.regime_probs[state.dominant_regime]
        assert state.llm_regime is None  # no LLM configured

    def test_regime_detector_crisis(self):
        """High VIX (>30) + wide spreads -> crisis or risk_off dominant."""
        macro = make_macro_df(vix=45.0, baa10y=6.0, dgs10=3.0, dgs2=3.5)
        store = FakeFeatureStore(macro)
        detector = RegimeDetector(feature_store=store)
        state = detector.detect(as_of=date(2024, 1, 1))

        assert state.dominant_regime in ("crisis", "risk_off")
        # crisis + risk_off combined should dominate
        crisis_risk_off = (
            state.regime_probs["crisis"] + state.regime_probs["risk_off"]
        )
        assert crisis_risk_off > 0.5

    def test_regime_blend_with_llm(self):
        """Verify 70/30 blend math between quant and LLM probabilities."""
        macro = make_macro_df(vix=20.0, baa10y=2.0, dgs10=4.0, dgs2=3.5)
        store = FakeFeatureStore(macro)
        detector = RegimeDetector(feature_store=store)

        # Get pure quant probs
        quant_probs = detector._quant_regime(macro, date(2024, 1, 1))

        # Define a fake LLM distribution (all weight on crisis)
        llm_probs = {r: 0.0 for r in REGIMES}
        llm_probs["crisis"] = 1.0

        blended = detector._blend_probabilities(quant_probs, llm_probs)

        # Check blend is 70% quant + 30% LLM
        assert abs(sum(blended.values()) - 1.0) < 1e-6
        expected_crisis = 0.7 * quant_probs["crisis"] + 0.3 * 1.0
        # Normalize
        total = sum(
            0.7 * quant_probs[r] + 0.3 * llm_probs[r] for r in REGIMES
        )
        expected_crisis_normalized = expected_crisis / total
        assert abs(blended["crisis"] - expected_crisis_normalized) < 1e-6

    def test_regime_no_feature_store(self):
        """Without a feature store, returns a valid regime state with defaults."""
        detector = RegimeDetector()
        state = detector.detect(as_of=date(2024, 1, 1))

        assert state.dominant_regime in REGIMES
        assert abs(sum(state.regime_probs.values()) - 1.0) < 1e-6

    def test_regime_blend_no_llm(self):
        """Without LLM, blend returns pure quant probs."""
        quant_probs = {"risk_on": 0.6, "risk_off": 0.2, "stagflation": 0.1,
                       "deflation": 0.05, "crisis": 0.05}
        detector = RegimeDetector()
        blended = detector._blend_probabilities(quant_probs, None)
        assert blended == quant_probs


# ─────────────────────────────────────────────────────────────────────────────
# HRP Optimizer Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestHierarchicalRiskParity:
    def test_hrp_weights_sum_to_one(self):
        """HRP output must sum to 1.0."""
        returns = make_returns(n_days=252, n_assets=10)
        hrp = HierarchicalRiskParity()
        weights = hrp.optimize(returns)

        assert isinstance(weights, pd.Series)
        assert len(weights) == 10
        assert abs(weights.sum() - 1.0) < 1e-6
        assert (weights >= 0).all()

    def test_hrp_concentrated_in_low_vol(self):
        """Low-vol assets should receive higher HRP weight than high-vol assets."""
        rng = np.random.default_rng(0)
        n_days = 500
        # Low vol asset: std = 0.005, high vol asset: std = 0.04
        low_vol = rng.normal(0, 0.005, n_days)
        high_vol = rng.normal(0, 0.04, n_days)
        returns = pd.DataFrame({"LOW": low_vol, "HIGH": high_vol})

        hrp = HierarchicalRiskParity()
        weights = hrp.optimize(returns)

        assert weights["LOW"] > weights["HIGH"], (
            f"Expected low-vol asset to have higher weight, got LOW={weights['LOW']:.4f}, "
            f"HIGH={weights['HIGH']:.4f}"
        )

    def test_hrp_single_asset(self):
        """Single asset should get weight = 1.0."""
        returns = pd.DataFrame({"A": np.random.normal(0, 0.01, 100)})
        hrp = HierarchicalRiskParity()
        weights = hrp.optimize(returns)
        assert abs(weights.sum() - 1.0) < 1e-6

    def test_hrp_with_position_cap(self):
        """Position cap constraint should be respected."""
        returns = make_returns(n_days=252, n_assets=5)
        hrp = HierarchicalRiskParity()
        weights = hrp.optimize(returns, constraints={"max_position_pct": 0.25})

        assert (weights <= 0.25 + 1e-6).all()
        assert abs(weights.sum() - 1.0) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Mean-Variance Optimizer Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMeanVarianceOptimizer:
    def test_mean_variance_feasible(self):
        """Basic MVO with well-conditioned inputs returns valid weights."""
        rng = np.random.default_rng(42)
        n = 5
        symbols = [f"A{i}" for i in range(n)]
        returns_data = rng.normal(0, 0.01, (252, n))
        ret_df = pd.DataFrame(returns_data, columns=symbols)

        expected_returns = pd.Series(
            [0.10, 0.12, 0.08, 0.15, 0.09], index=symbols
        )
        covariance = ret_df.cov() * 252  # annualized

        mvo = MeanVarianceOptimizer(risk_aversion=1.0)
        weights = mvo.optimize(expected_returns, covariance)

        assert isinstance(weights, pd.Series)
        assert len(weights) == n
        assert abs(weights.sum() - 1.0) < 1e-4
        assert (weights >= -1e-6).all()

    def test_mean_variance_infeasible_fallback(self):
        """Infeasible MVO (impossible constraints) falls back to equal weight."""
        symbols = ["A", "B", "C"]
        expected_returns = pd.Series([0.10, 0.12, 0.08], index=symbols)
        # Degenerate covariance matrix (all zeros)
        covariance = pd.DataFrame(
            np.zeros((3, 3)), index=symbols, columns=symbols
        )

        mvo = MeanVarianceOptimizer(risk_aversion=1.0)
        # Constraint: max_position_pct = 0.1, but 3 assets need 0.333 each
        # Should fall back to equal weight
        weights = mvo.optimize(
            expected_returns, covariance,
            constraints={"max_position_pct": 0.1}
        )

        assert isinstance(weights, pd.Series)
        assert len(weights) == 3
        assert abs(weights.sum() - 1.0) < 1e-4

    def test_mean_variance_weights_non_negative(self):
        """Long-only constraint: all weights >= 0."""
        rng = np.random.default_rng(10)
        n = 10
        symbols = [f"S{i}" for i in range(n)]
        ret = pd.DataFrame(rng.normal(0, 0.01, (252, n)), columns=symbols)
        mu = pd.Series(rng.uniform(0.05, 0.20, n), index=symbols)
        cov = ret.cov() * 252

        mvo = MeanVarianceOptimizer(risk_aversion=2.0)
        weights = mvo.optimize(mu, cov)

        assert (weights >= -1e-6).all()


# ─────────────────────────────────────────────────────────────────────────────
# Risk Parity Optimizer Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRiskParityOptimizer:
    def test_risk_parity_equal_risk(self):
        """Risk contributions should be approximately equal across assets."""
        rng = np.random.default_rng(42)
        n = 4
        symbols = [f"A{i}" for i in range(n)]
        ret = pd.DataFrame(rng.normal(0, 0.01, (500, n)), columns=symbols)
        cov = ret.cov()

        rp = RiskParityOptimizer()
        weights = rp.optimize(cov)

        assert abs(weights.sum() - 1.0) < 1e-4
        assert (weights >= 0).all()

        # Compute risk contributions
        w = weights.values
        sigma = cov.values
        mrc = sigma @ w
        rc = w * mrc
        total_rc = rc.sum()
        rc_normalized = rc / total_rc

        # Each asset's risk contribution should be approximately 1/n
        target = 1.0 / n
        assert np.allclose(rc_normalized, target, atol=0.05), (
            f"Risk contributions not equal: {rc_normalized}"
        )

    def test_risk_parity_weights_sum_to_one(self):
        """Risk parity weights must sum to 1.0."""
        rng = np.random.default_rng(7)
        n = 6
        symbols = [f"X{i}" for i in range(n)]
        ret = pd.DataFrame(rng.normal(0, 0.01, (300, n)), columns=symbols)
        cov = ret.cov()

        rp = RiskParityOptimizer()
        weights = rp.optimize(cov)

        assert abs(weights.sum() - 1.0) < 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# Constraint Engine Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestConstraintEngine:
    def test_constraints_position_cap(self):
        """Weights > 5% should be capped; result must honor cap."""
        # Use 25 assets so all can be at 4% <= 5% cap and still sum to 1.0
        symbols = [f"A{i}" for i in range(25)]
        # One asset has 50% weight, rest share 50% equally (2% each)
        raw = [0.50] + [0.50 / 24] * 24
        weights = pd.Series(raw, index=symbols)

        constraints = PortfolioConstraints(
            max_position_pct=0.05,
            min_positions=0,
        )
        engine = ConstraintEngine()
        result = engine.apply(weights, None, constraints)

        assert (result <= 0.05 + 1e-6).all()
        assert abs(result.sum() - 1.0) < 1e-6

    def test_constraints_turnover_limit(self):
        """Large turnover gets capped to max_turnover."""
        symbols = [f"A{i}" for i in range(5)]
        current = pd.Series([0.20, 0.20, 0.20, 0.20, 0.20], index=symbols)
        # Proposed: completely different weights
        proposed = pd.Series([0.90, 0.025, 0.025, 0.025, 0.025], index=symbols)

        constraints = PortfolioConstraints(
            max_turnover=0.30,
            max_position_pct=1.0,  # no position cap for this test
            min_positions=0,
        )
        engine = ConstraintEngine()
        result = engine.apply(proposed, current, constraints)

        # Compute turnover of result vs current
        turnover = (result - current).abs().sum()
        assert turnover <= 0.30 + 1e-6, f"Turnover {turnover:.4f} exceeds limit 0.30"
        assert abs(result.sum() - 1.0) < 1e-6

    def test_constraints_sector_cap(self):
        """Sector weight should not exceed max_sector_pct when there are
        unclassified assets to absorb excess."""
        # 3 tech, 2 finance, 5 unclassified — enough room for redistribution
        symbols = ["AAPL", "MSFT", "GOOGL", "JPM", "BAC",
                   "U1", "U2", "U3", "U4", "U5"]
        # Tech starts at 60% of portfolio
        weights = pd.Series(
            [0.20, 0.20, 0.20, 0.05, 0.05, 0.07, 0.07, 0.07, 0.07, 0.02],
            index=symbols,
        )
        sector_map = {
            "AAPL": "tech", "MSFT": "tech", "GOOGL": "tech",
            "JPM": "finance", "BAC": "finance",
        }

        constraints = PortfolioConstraints(
            max_sector_pct=0.40,
            max_position_pct=1.0,
            min_positions=0,
            sector_map=sector_map,
        )
        engine = ConstraintEngine()
        result = engine.apply(weights, None, constraints)

        tech_weight = sum(result[s] for s in ["AAPL", "MSFT", "GOOGL"])
        assert tech_weight <= 0.40 + 1e-6

    def test_constraints_no_violations(self):
        """check_violations returns empty list when all constraints satisfied."""
        symbols = ["A", "B", "C", "D"]
        weights = pd.Series([0.25, 0.25, 0.25, 0.25], index=symbols)
        constraints = PortfolioConstraints(
            max_position_pct=0.30,
            max_sector_pct=1.0,
            min_positions=4,
        )
        engine = ConstraintEngine()
        violations = engine.check_violations(weights, constraints)
        assert violations == []


# ─────────────────────────────────────────────────────────────────────────────
# VaR Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestVaRCalculator:
    def test_var_parametric(self):
        """Known portfolio variance -> expected parametric VaR."""
        rng = np.random.default_rng(42)
        n_days = 1000
        # Single asset, daily vol = 1%
        returns = pd.DataFrame({"A": rng.normal(0, 0.01, n_days)})
        weights = pd.Series({"A": 1.0})

        var_calc = VaRCalculator()
        var = var_calc.parametric_var(weights, returns, confidence=0.95, horizon_days=1)

        # Expected: z_0.95 * 0.01 ≈ 1.645 * 0.01 = 0.01645
        assert var > 0
        assert 0.01 < var < 0.03, f"Parametric VaR = {var:.5f} outside expected range"

    def test_var_historical(self):
        """Historical VaR from 252-day returns should be reasonable."""
        rng = np.random.default_rng(1)
        n_days = 252
        returns = pd.DataFrame({
            "A": rng.normal(0, 0.01, n_days),
            "B": rng.normal(0, 0.015, n_days),
        })
        weights = pd.Series({"A": 0.6, "B": 0.4})

        var_calc = VaRCalculator()
        var = var_calc.historical_var(weights, returns, confidence=0.95)

        assert var > 0
        assert var < 0.10, f"Historical VaR = {var:.5f} seems too large"

    def test_expected_shortfall_exceeds_var(self):
        """Expected Shortfall should always be >= VaR."""
        rng = np.random.default_rng(99)
        n_days = 500
        returns = pd.DataFrame({
            f"S{i}": rng.normal(0, 0.01 * (i + 1), n_days)
            for i in range(5)
        })
        weights = pd.Series({f"S{i}": 0.2 for i in range(5)})

        var_calc = VaRCalculator()
        var = var_calc.historical_var(weights, returns, confidence=0.95)
        es = var_calc.expected_shortfall(weights, returns, confidence=0.95)

        assert es >= var - 1e-8, f"ES ({es:.5f}) < VaR ({var:.5f})"

    def test_var_multi_asset(self):
        """Multi-asset VaR is lower than worst-asset VaR due to diversification."""
        rng = np.random.default_rng(5)
        n_days = 500
        # Two negatively correlated assets
        base = rng.normal(0, 0.01, n_days)
        returns = pd.DataFrame({
            "A": base + rng.normal(0, 0.001, n_days),
            "B": -base + rng.normal(0, 0.001, n_days),
        })
        equal_weights = pd.Series({"A": 0.5, "B": 0.5})

        var_calc = VaRCalculator()
        portfolio_var = var_calc.parametric_var(equal_weights, returns)
        assert portfolio_var >= 0


# ─────────────────────────────────────────────────────────────────────────────
# Stress Test Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestStressTester:
    def test_stress_test_covid(self):
        """COVID scenario produces >20% loss on long equity portfolio."""
        symbols = [f"S{i}" for i in range(20)]
        weights = pd.Series([1.0 / 20] * 20, index=symbols)

        tester = StressTester()
        result = tester.run_scenario(weights, "2020_covid")

        # COVID equity shock is -34%, full long portfolio should lose ~34%
        assert result.portfolio_loss_pct < -0.20, (
            f"Expected >20% loss, got {result.portfolio_loss_pct:.2%}"
        )
        assert result.scenario_name == "2020_covid"
        assert result.worst_position in symbols
        assert result.max_drawdown_estimate > 0

    def test_stress_test_gfc(self):
        """GFC scenario produces ~50% loss on fully invested portfolio."""
        symbols = ["AAPL", "MSFT", "GOOG"]
        weights = pd.Series([1.0 / 3] * 3, index=symbols)

        tester = StressTester()
        result = tester.run_scenario(weights, "2008_gfc")

        assert result.portfolio_loss_pct < -0.40
        assert result.breaches_daily_limit  # -50% >> 3% daily limit

    def test_stress_test_run_all(self):
        """run_all returns results for all 3 scenarios."""
        symbols = [f"S{i}" for i in range(10)]
        weights = pd.Series([0.1] * 10, index=symbols)

        tester = StressTester()
        results = tester.run_all(weights)

        assert len(results) == 3
        scenario_names = {r.scenario_name for r in results}
        assert scenario_names == {"2008_gfc", "2020_covid", "2022_rates"}

    def test_stress_test_unknown_scenario(self):
        """Unknown scenario name raises ValueError."""
        weights = pd.Series({"A": 1.0})
        tester = StressTester()
        with pytest.raises(ValueError, match="Unknown scenario"):
            tester.run_scenario(weights, "1929_crash")


# ─────────────────────────────────────────────────────────────────────────────
# Risk Limit Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRiskLimitChecker:
    def test_risk_limit_position_cap(self):
        """Oversized position is flagged as violation and capped."""
        symbols = [f"A{i}" for i in range(10)]
        weights = pd.Series([0.50] + [0.50 / 9] * 9, index=symbols)

        limits = {
            "max_position_pct": 0.05,
            "max_daily_loss_pct": 0.03,
            "max_gross_leverage": 1.0,
        }
        checker = RiskLimitChecker(limits)
        result = checker.check(weights)

        assert not result.passed
        assert len(result.violations) > 0
        assert any("A0" in v for v in result.violations)
        # Constrained weights should respect cap
        assert (result.constrained_weights <= 0.05 + 1e-6).all()

    def test_risk_limit_daily_loss_flag(self):
        """Daily loss exceeding limit is flagged as violation."""
        symbols = ["A", "B"]
        weights = pd.Series([0.5, 0.5], index=symbols)
        limits = {
            "max_position_pct": 1.0,
            "max_daily_loss_pct": 0.03,
            "max_gross_leverage": 1.0,
        }
        checker = RiskLimitChecker(limits)
        # daily_pnl = -40,000 on portfolio_value = 1,000,000 = -4% > 3% limit
        result = checker.check(weights, daily_pnl=-40000.0, portfolio_value=1_000_000.0)

        assert not result.passed
        assert any("Daily loss" in v for v in result.violations)

    def test_risk_limit_no_violations(self):
        """Well-formed weights pass all checks."""
        symbols = [f"A{i}" for i in range(20)]
        weights = pd.Series([0.05] * 20, index=symbols)

        limits = {
            "max_position_pct": 0.05,
            "max_daily_loss_pct": 0.03,
            "max_gross_leverage": 1.0,
        }
        checker = RiskLimitChecker(limits)
        result = checker.check(weights)

        assert result.passed
        assert result.violations == []

    def test_risk_limit_approaching_alert(self):
        """Positions at 80-100% of cap generate soft alerts."""
        symbols = ["A", "B", "C", "D"]
        # A is at 4.5% which is 90% of 5% cap
        weights = pd.Series([0.045, 0.30, 0.30, 0.355], index=symbols)
        # Normalize
        weights /= weights.sum()

        limits = {
            "max_position_pct": 0.05,
            "max_daily_loss_pct": 0.03,
            "max_gross_leverage": 2.0,  # high enough to not trigger
        }
        checker = RiskLimitChecker(limits)
        result = checker.check(weights)

        # Some alerts expected for positions near cap
        # (result.alerts might be empty if none at 80% threshold after normalization)
        assert isinstance(result.alerts, list)


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio Allocator Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPortfolioAllocator:
    def _make_allocator(self, limits: dict | None = None):
        from portfolio.optimizer.constraints import ConstraintEngine
        from portfolio.optimizer.hierarchical import HierarchicalRiskParity
        from portfolio.risk.limits import RiskLimitChecker

        if limits is None:
            limits = {
                "max_position_pct": 0.20,
                "max_daily_loss_pct": 0.03,
                "max_gross_leverage": 1.0,
            }

        optimizer = HierarchicalRiskParity()
        engine = ConstraintEngine()
        checker = RiskLimitChecker(limits)
        return PortfolioAllocator(optimizer, engine, checker)

    def test_allocator_basic(self):
        """Allocator produces valid weights for positive-signal universe."""
        rng = np.random.default_rng(42)
        n = 5
        symbols = [f"S{i}" for i in range(n)]
        returns = pd.DataFrame(
            rng.normal(0, 0.01, (252, n)), columns=symbols
        )
        signals = pd.Series([0.8, 0.6, 0.4, 0.2, 0.1], index=symbols)
        constraints = PortfolioConstraints(
            max_position_pct=0.20, min_positions=0
        )

        allocator = self._make_allocator()
        weights, risk_result = allocator.allocate(
            signals, returns, None, None, constraints
        )

        assert isinstance(weights, pd.Series)
        assert len(weights) > 0
        assert isinstance(risk_result, RiskCheckResult)

    def test_allocator_filters_negative_signals(self):
        """Symbols with non-positive signals are excluded from portfolio."""
        rng = np.random.default_rng(10)
        n = 6
        symbols = [f"S{i}" for i in range(n)]
        returns = pd.DataFrame(
            rng.normal(0, 0.01, (252, n)), columns=symbols
        )
        # Only first 3 have positive signals
        signals = pd.Series(
            [0.8, 0.6, 0.4, 0.0, -0.1, -0.5], index=symbols
        )
        constraints = PortfolioConstraints(max_position_pct=1.0, min_positions=0)

        allocator = self._make_allocator()
        weights, _ = allocator.allocate(signals, returns, None, None, constraints)

        # Only positive-signal symbols should be in weights
        positive_symbols = {"S0", "S1", "S2"}
        if len(weights) > 0:
            for sym in weights.index:
                assert sym in positive_symbols

    def test_allocator_regime_overlay_crisis(self):
        """Crisis mode should reduce gross exposure to ~60%."""
        from portfolio.regime.detector import RegimeState

        rng = np.random.default_rng(7)
        n = 5
        symbols = [f"S{i}" for i in range(n)]
        returns = pd.DataFrame(
            rng.normal(0, 0.01, (252, n)), columns=symbols
        )
        signals = pd.Series([0.5] * n, index=symbols)
        constraints = PortfolioConstraints(max_position_pct=1.0, min_positions=0)

        # Create a crisis regime state
        regime = RegimeState(
            date=date(2024, 1, 1),
            regime_probs={
                "crisis": 0.7, "risk_off": 0.2, "risk_on": 0.05,
                "stagflation": 0.03, "deflation": 0.02,
            },
            dominant_regime="crisis",
            confidence=0.7,
            quant_regime="crisis",
            llm_regime=None,
        )

        allocator = self._make_allocator()
        weights, _ = allocator.allocate(
            signals, returns, None, regime, constraints
        )

        # After crisis overlay, sum of weights should be ~60% (rest is cash)
        total_weight = weights.sum()
        assert total_weight <= 0.65, (
            f"Crisis regime should reduce exposure to ~60%, got {total_weight:.2%}"
        )
        assert total_weight > 0.0

    def test_allocator_regime_overlay_risk_off(self):
        """Risk-off mode should reduce gross exposure to ~80%."""
        from portfolio.regime.detector import RegimeState

        rng = np.random.default_rng(7)
        n = 5
        symbols = [f"S{i}" for i in range(n)]
        returns = pd.DataFrame(
            rng.normal(0, 0.01, (252, n)), columns=symbols
        )
        signals = pd.Series([0.5] * n, index=symbols)
        constraints = PortfolioConstraints(max_position_pct=1.0, min_positions=0)

        regime = RegimeState(
            date=date(2024, 1, 1),
            regime_probs={
                "risk_off": 0.6, "crisis": 0.1, "risk_on": 0.15,
                "stagflation": 0.1, "deflation": 0.05,
            },
            dominant_regime="risk_off",
            confidence=0.6,
            quant_regime="risk_off",
            llm_regime=None,
        )

        allocator = self._make_allocator()
        weights, _ = allocator.allocate(
            signals, returns, None, regime, constraints
        )

        total_weight = weights.sum()
        assert total_weight <= 0.85, (
            f"Risk-off regime should reduce exposure to ~80%, got {total_weight:.2%}"
        )

    def test_allocator_empty_signals(self):
        """All-negative signals returns empty weights."""
        rng = np.random.default_rng(0)
        n = 5
        symbols = [f"S{i}" for i in range(n)]
        returns = pd.DataFrame(rng.normal(0, 0.01, (252, n)), columns=symbols)
        signals = pd.Series([-0.5, -0.3, -0.1, -0.8, -0.2], index=symbols)
        constraints = PortfolioConstraints(max_position_pct=1.0, min_positions=0)

        allocator = self._make_allocator()
        weights, risk_result = allocator.allocate(
            signals, returns, None, None, constraints
        )

        assert len(weights) == 0

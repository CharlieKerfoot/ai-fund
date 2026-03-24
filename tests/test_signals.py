"""Comprehensive tests for Phase 4: Signal Library."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from research.signals.combination import SignalCombiner
from research.signals.library.congress import CongressionalTrading
from research.signals.library.cross_asset import CrossAssetSignal
from research.signals.library.earnings_drift import PostEarningsDrift
from research.signals.library.filing_anomaly import FilingAnomaly
from research.signals.library.mean_reversion import BollingerMeanReversion, RSIMeanReversion
from research.signals.library.momentum import CrossSectionalMomentum, TimeSeriesMomentum
from research.signals.library.sentiment import SentimentSignal
from research.signals.registry import SignalRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_price_features(
    symbols: list[str] | None = None,
    days: int = 300,
    start_date: date | None = None,
) -> pd.DataFrame:
    """Return DataFrame with columns [symbol, date, close, open, high, low, volume]
    with realistic price data (random walk starting at 100).
    """
    if symbols is None:
        symbols = ["AAPL", "MSFT", "GOOG"]
    if start_date is None:
        start_date = date(2023, 1, 2)

    rng = np.random.default_rng(42)
    rows = []
    for sym in symbols:
        price = 100.0
        for i in range(days):
            d = start_date + timedelta(days=i)
            ret = rng.normal(0.0005, 0.015)
            price = price * (1 + ret)
            rows.append({
                "symbol": sym,
                "date": pd.Timestamp(d),
                "close": round(price, 4),
                "open": round(price * (1 - abs(rng.normal(0, 0.003))), 4),
                "high": round(price * (1 + abs(rng.normal(0, 0.005))), 4),
                "low": round(price * (1 - abs(rng.normal(0, 0.005))), 4),
                "volume": int(rng.integers(1_000_000, 10_000_000)),
            })
    return pd.DataFrame(rows)


def make_as_of(features: pd.DataFrame) -> date:
    """Return the latest date in features as a date object."""
    return features["date"].max().date()


# ---------------------------------------------------------------------------
# CrossSectionalMomentum
# ---------------------------------------------------------------------------

class TestCrossSectionalMomentum:
    def test_cross_sectional_momentum_shape(self):
        features = make_price_features(["AAPL", "MSFT", "GOOG"], days=300)
        as_of = make_as_of(features)
        signal = CrossSectionalMomentum()
        result = signal.compute(features, as_of)

        assert isinstance(result, pd.Series)
        # All symbols should be present (enough data)
        assert set(result.index) == {"AAPL", "MSFT", "GOOG"}
        assert result.between(-1, 1).all()

    def test_cross_sectional_momentum_direction(self):
        """Symbol with highest 12m return gets highest signal."""
        # Construct a features DataFrame where WINNER has the highest 12m return
        days = 300
        start = date(2023, 1, 2)
        rows = []

        # WINNER: strong uptrend
        price = 100.0
        for i in range(days):
            d = start + timedelta(days=i)
            price *= 1.002  # ~50% annual
            rows.append({"symbol": "WINNER", "date": pd.Timestamp(d), "close": price,
                         "open": price, "high": price, "low": price, "volume": 1_000_000})

        # LOSER: strong downtrend
        price = 100.0
        for i in range(days):
            d = start + timedelta(days=i)
            price *= 0.998  # ~-40% annual
            rows.append({"symbol": "LOSER", "date": pd.Timestamp(d), "close": price,
                         "open": price, "high": price, "low": price, "volume": 1_000_000})

        features = pd.DataFrame(rows)
        as_of = make_as_of(features)
        signal = CrossSectionalMomentum()
        result = signal.compute(features, as_of)

        assert result["WINNER"] > result["LOSER"]

    def test_cross_sectional_momentum_insufficient_data(self):
        """Symbols with < 200 days of data are excluded."""
        features = make_price_features(["AAPL"], days=150)
        as_of = make_as_of(features)
        signal = CrossSectionalMomentum()
        result = signal.compute(features, as_of)
        assert "AAPL" not in result.index


# ---------------------------------------------------------------------------
# TimeSeriesMomentum
# ---------------------------------------------------------------------------

class TestTimeSeriesMomentum:
    def test_time_series_momentum_positive(self):
        """Symbol up ~30% gets positive signal."""
        days = 280
        start = date(2023, 1, 2)
        rows = []
        price = 100.0
        for i in range(days):
            d = start + timedelta(days=i)
            price *= 1.001  # strong uptrend → well above start after 252 days
            rows.append({"symbol": "UP", "date": pd.Timestamp(d), "close": price,
                         "open": price, "high": price, "low": price, "volume": 1_000_000})

        features = pd.DataFrame(rows)
        as_of = make_as_of(features)
        signal = TimeSeriesMomentum()
        result = signal.compute(features, as_of)

        assert "UP" in result.index
        assert result["UP"] > 0

    def test_time_series_momentum_in_range(self):
        features = make_price_features(["AAPL", "MSFT"], days=300)
        as_of = make_as_of(features)
        signal = TimeSeriesMomentum()
        result = signal.compute(features, as_of)
        assert result.between(-1.0001, 1.0001).all()


# ---------------------------------------------------------------------------
# RSIMeanReversion
# ---------------------------------------------------------------------------

class TestRSIMeanReversion:
    def _make_consecutive_moves(self, symbol: str, n_up: int = 0, n_down: int = 0) -> pd.DataFrame:
        """Make a price series with consistent up or down days."""
        rows = []
        price = 100.0
        start = date(2024, 1, 1)
        # Warm-up period of 30 neutral days
        for i in range(30):
            d = start + timedelta(days=i)
            rows.append({"symbol": symbol, "date": pd.Timestamp(d), "close": price,
                         "open": price, "high": price, "low": price, "volume": 1_000_000})
        # Then consecutive up or down moves
        start2 = start + timedelta(days=30)
        n = max(n_up, n_down)
        for i in range(n):
            d = start2 + timedelta(days=i)
            if n_up > 0:
                price *= 1.02
            else:
                price *= 0.98
            rows.append({"symbol": symbol, "date": pd.Timestamp(d), "close": price,
                         "open": price, "high": price, "low": price, "volume": 1_000_000})
        return pd.DataFrame(rows)

    def test_rsi_overbought(self):
        """14 consecutive up days → negative RSI signal (overbought)."""
        features = self._make_consecutive_moves("OB", n_up=16)
        as_of = make_as_of(features)
        signal = RSIMeanReversion()
        result = signal.compute(features, as_of)

        assert "OB" in result.index
        assert result["OB"] < 0, f"Expected negative signal for overbought, got {result['OB']}"

    def test_rsi_oversold(self):
        """14 consecutive down days → positive RSI signal (oversold)."""
        features = self._make_consecutive_moves("OS", n_down=16)
        as_of = make_as_of(features)
        signal = RSIMeanReversion()
        result = signal.compute(features, as_of)

        assert "OS" in result.index
        assert result["OS"] > 0, f"Expected positive signal for oversold, got {result['OS']}"

    def test_rsi_in_range(self):
        features = make_price_features(["AAPL", "MSFT"], days=100)
        as_of = make_as_of(features)
        signal = RSIMeanReversion()
        result = signal.compute(features, as_of)
        assert result.between(-1.0001, 1.0001).all()


# ---------------------------------------------------------------------------
# BollingerMeanReversion
# ---------------------------------------------------------------------------

class TestBollingerMeanReversion:
    def test_bollinger_above_band(self):
        """Price well above upper band → negative signal."""
        rows = []
        start = date(2024, 1, 1)
        price = 100.0
        # Build stable base
        for i in range(25):
            d = start + timedelta(days=i)
            rows.append({"symbol": "HIGH", "date": pd.Timestamp(d), "close": price,
                         "open": price, "high": price, "low": price, "volume": 1_000_000})
        # Spike way above
        spike_price = price * 1.5
        d = start + timedelta(days=25)
        rows.append({"symbol": "HIGH", "date": pd.Timestamp(d), "close": spike_price,
                     "open": spike_price, "high": spike_price, "low": spike_price,
                     "volume": 1_000_000})

        features = pd.DataFrame(rows)
        as_of = make_as_of(features)
        signal = BollingerMeanReversion()
        result = signal.compute(features, as_of)

        assert "HIGH" in result.index
        assert result["HIGH"] < 0, f"Expected negative for above-band price, got {result['HIGH']}"

    def test_bollinger_below_band(self):
        """Price well below lower band → positive signal."""
        rows = []
        start = date(2024, 1, 1)
        price = 100.0
        for i in range(25):
            d = start + timedelta(days=i)
            rows.append({"symbol": "LOW", "date": pd.Timestamp(d), "close": price,
                         "open": price, "high": price, "low": price, "volume": 1_000_000})
        # Drop way below
        drop_price = price * 0.5
        d = start + timedelta(days=25)
        rows.append({"symbol": "LOW", "date": pd.Timestamp(d), "close": drop_price,
                     "open": drop_price, "high": drop_price, "low": drop_price,
                     "volume": 1_000_000})

        features = pd.DataFrame(rows)
        as_of = make_as_of(features)
        signal = BollingerMeanReversion()
        result = signal.compute(features, as_of)

        assert "LOW" in result.index
        assert result["LOW"] > 0


# ---------------------------------------------------------------------------
# PostEarningsDrift
# ---------------------------------------------------------------------------

class TestPostEarningsDrift:
    def _make_earnings_features(
        self,
        symbol: str,
        sentiment: float,
        guidance: str,
        days_ago: int = 5,
    ) -> pd.DataFrame:
        as_of = date(2024, 6, 1)
        earnings_date = as_of - timedelta(days=days_ago)
        return pd.DataFrame([{
            "symbol": symbol,
            "date": pd.Timestamp(earnings_date),
            "sentiment_score": sentiment,
            "guidance_direction": guidance,
        }])

    def test_post_earnings_drift_bullish(self):
        """Positive sentiment + up guidance → +0.8."""
        features = self._make_earnings_features("AAPL", sentiment=0.6, guidance="up")
        as_of = date(2024, 6, 1)
        signal = PostEarningsDrift()
        result = signal.compute(features, as_of)

        assert "AAPL" in result.index
        assert result["AAPL"] == pytest.approx(0.8)

    def test_post_earnings_drift_bearish(self):
        """Negative sentiment + down guidance → -0.8."""
        features = self._make_earnings_features("MSFT", sentiment=-0.5, guidance="down")
        as_of = date(2024, 6, 1)
        signal = PostEarningsDrift()
        result = signal.compute(features, as_of)

        assert "MSFT" in result.index
        assert result["MSFT"] == pytest.approx(-0.8)

    def test_post_earnings_drift_no_recent(self):
        """No recent earnings (> 60 days ago) → 0 signal."""
        features = self._make_earnings_features("GOOG", sentiment=0.6, guidance="up", days_ago=65)
        as_of = date(2024, 6, 1)
        signal = PostEarningsDrift()
        result = signal.compute(features, as_of)

        # Outside lookback window → no rows → empty result
        assert "GOOG" not in result.index or result.get("GOOG", 0.0) == 0.0

    def test_post_earnings_drift_neutral(self):
        """Moderate sentiment → 0.0 signal."""
        features = self._make_earnings_features("TSLA", sentiment=0.1, guidance="neutral")
        as_of = date(2024, 6, 1)
        signal = PostEarningsDrift()
        result = signal.compute(features, as_of)

        assert "TSLA" in result.index
        assert result["TSLA"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# FilingAnomaly
# ---------------------------------------------------------------------------

class TestFilingAnomaly:
    def _make_filing_features(
        self,
        symbol: str,
        going_concern: bool = False,
        risk_delta: str | None = None,
        litigation: str | None = None,
        days_ago: int = 10,
    ) -> pd.DataFrame:
        as_of = date(2024, 6, 1)
        filing_date = as_of - timedelta(days=days_ago)
        return pd.DataFrame([{
            "symbol": symbol,
            "date": pd.Timestamp(filing_date),
            "going_concern": going_concern,
            "risk_factor_delta": risk_delta,
            "litigation_exposure": litigation,
        }])

    def test_filing_anomaly_going_concern(self):
        """Going concern = True → -1.0."""
        features = self._make_filing_features("RISK", going_concern=True)
        as_of = date(2024, 6, 1)
        signal = FilingAnomaly()
        result = signal.compute(features, as_of)

        assert "RISK" in result.index
        assert result["RISK"] == pytest.approx(-1.0)

    def test_filing_anomaly_increased_risk_high_litigation(self):
        """increased risk + high litigation → -0.7."""
        features = self._make_filing_features(
            "BAD", risk_delta="increased", litigation="high"
        )
        as_of = date(2024, 6, 1)
        signal = FilingAnomaly()
        result = signal.compute(features, as_of)

        assert "BAD" in result.index
        assert result["BAD"] == pytest.approx(-0.7)

    def test_filing_anomaly_decreased_risk(self):
        """Decreased risk factor → +0.5."""
        features = self._make_filing_features("GOOD", risk_delta="decreased")
        as_of = date(2024, 6, 1)
        signal = FilingAnomaly()
        result = signal.compute(features, as_of)

        assert "GOOD" in result.index
        assert result["GOOD"] == pytest.approx(0.5)

    def test_filing_anomaly_no_filing(self):
        """No recent filing (outside 90 days) → empty result / 0.0."""
        features = self._make_filing_features("NOFILING", risk_delta="increased", days_ago=95)
        as_of = date(2024, 6, 1)
        signal = FilingAnomaly()
        result = signal.compute(features, as_of)

        assert "NOFILING" not in result.index or result.get("NOFILING", 0.0) == 0.0


# ---------------------------------------------------------------------------
# SentimentSignal
# ---------------------------------------------------------------------------

class TestSentimentSignal:
    def test_sentiment_combines_ewm(self):
        """Recent positive sentiment → higher rank → positive signal."""
        as_of = date(2024, 6, 1)
        rows = []
        # BULL: consistently positive sentiment
        for i in range(10):
            d = as_of - timedelta(days=9 - i)
            rows.append({"symbol": "BULL", "date": pd.Timestamp(d), "sentiment_score": 0.8})
        # NEUTRAL: flat sentiment
        for i in range(10):
            d = as_of - timedelta(days=9 - i)
            rows.append({"symbol": "NEUTRAL", "date": pd.Timestamp(d), "sentiment_score": 0.0})
        # BEAR: consistently negative sentiment
        for i in range(10):
            d = as_of - timedelta(days=9 - i)
            rows.append({"symbol": "BEAR", "date": pd.Timestamp(d), "sentiment_score": -0.8})

        features = pd.DataFrame(rows)
        signal = SentimentSignal()
        result = signal.compute(features, as_of)

        assert "BULL" in result.index
        assert "BEAR" in result.index
        assert result["BULL"] > result["NEUTRAL"]
        assert result["NEUTRAL"] > result["BEAR"]
        assert result["BULL"] > 0
        assert result["BEAR"] < 0

    def test_sentiment_in_range(self):
        as_of = date(2024, 6, 1)
        rows = []
        for sym in ["A", "B", "C"]:
            for i in range(15):
                d = as_of - timedelta(days=14 - i)
                score = {"A": 0.5, "B": 0.0, "C": -0.5}[sym]
                rows.append({"symbol": sym, "date": pd.Timestamp(d), "sentiment_score": score})
        features = pd.DataFrame(rows)
        signal = SentimentSignal()
        result = signal.compute(features, as_of)
        assert result.between(-1.0001, 1.0001).all()


# ---------------------------------------------------------------------------
# CongressionalTrading
# ---------------------------------------------------------------------------

class TestCongressionalTrading:
    def test_congressional_buy_signal(self):
        """Recent large buy → positive signal near +1.0."""
        as_of = date(2024, 6, 1)
        features = pd.DataFrame([{
            "symbol": "AAPL",
            "date": pd.Timestamp(as_of - timedelta(days=2)),
            "transaction_type": "buy",
            "amount": 100_000.0,
        }])
        signal = CongressionalTrading()
        result = signal.compute(features, as_of)

        assert "AAPL" in result.index
        assert result["AAPL"] > 0.5

    def test_congressional_sell_signal(self):
        """Recent large sell → negative signal near -1.0."""
        as_of = date(2024, 6, 1)
        features = pd.DataFrame([{
            "symbol": "MSFT",
            "date": pd.Timestamp(as_of - timedelta(days=3)),
            "transaction_type": "sell",
            "amount": 200_000.0,
        }])
        signal = CongressionalTrading()
        result = signal.compute(features, as_of)

        assert "MSFT" in result.index
        assert result["MSFT"] < -0.5

    def test_congressional_small_transaction_excluded(self):
        """Small transaction (< $50k) → excluded from signal."""
        as_of = date(2024, 6, 1)
        features = pd.DataFrame([{
            "symbol": "GOOG",
            "date": pd.Timestamp(as_of - timedelta(days=2)),
            "transaction_type": "buy",
            "amount": 10_000.0,
        }])
        signal = CongressionalTrading()
        result = signal.compute(features, as_of)

        assert "GOOG" not in result.index

    def test_congressional_recency_weighting(self):
        """More recent transaction gets higher weight than older one."""
        as_of = date(2024, 6, 1)
        features = pd.DataFrame([
            # Recent buy (strong signal)
            {
                "symbol": "TSLA",
                "date": pd.Timestamp(as_of - timedelta(days=1)),
                "transaction_type": "buy",
                "amount": 100_000.0,
            },
            # Old sell (weaker signal due to decay)
            {
                "symbol": "TSLA",
                "date": pd.Timestamp(as_of - timedelta(days=28)),
                "transaction_type": "sell",
                "amount": 100_000.0,
            },
        ])
        signal = CongressionalTrading()
        result = signal.compute(features, as_of)

        # Recent buy dominates → net positive
        assert "TSLA" in result.index
        assert result["TSLA"] > 0


# ---------------------------------------------------------------------------
# CrossAssetSignal
# ---------------------------------------------------------------------------

class TestCrossAssetSignal:
    def _make_macro_features(
        self,
        symbols: list[str],
        dgs10_change: float,
        baa10y_change: float,
        vix_level: float,
    ) -> pd.DataFrame:
        """Build a features DataFrame with macro columns for given symbols."""
        as_of = date(2024, 6, 1)
        rows = []
        # Current macro values
        for sym in symbols:
            rows.append({
                "symbol": sym,
                "date": pd.Timestamp(as_of),
                "DGS10": 4.5 + dgs10_change,
                "BAA10Y": 1.5 + baa10y_change,
                "VIXCLS": vix_level,
            })
        # Lagged macro values (22 days ago for 21-day change)
        for sym in symbols:
            rows.append({
                "symbol": sym,
                "date": pd.Timestamp(as_of - timedelta(days=22)),
                "DGS10": 4.5,
                "BAA10Y": 1.5,
                "VIXCLS": vix_level,
            })
        return pd.DataFrame(rows)

    def test_cross_asset_risk_off(self):
        """Rising rates + widening spreads + VIX=30 → negative signal."""
        symbols = ["AAPL", "MSFT", "GOOG"]
        features = self._make_macro_features(
            symbols,
            dgs10_change=0.3,   # rates rising
            baa10y_change=0.2,  # spreads widening
            vix_level=30.0,     # VIX > 20
        )
        as_of = date(2024, 6, 1)
        signal = CrossAssetSignal()
        result = signal.compute(features, as_of)

        assert len(result) > 0
        # All symbols should get same negative signal
        assert (result < 0).all(), f"Expected all negative, got {result}"
        # Check all equal (macro applied uniformly)
        assert result.nunique() == 1

    def test_cross_asset_risk_on(self):
        """Falling rates + tightening spreads + VIX=15 → positive signal."""
        symbols = ["AAPL", "MSFT"]
        features = self._make_macro_features(
            symbols,
            dgs10_change=-0.3,  # rates falling
            baa10y_change=-0.2, # spreads tightening
            vix_level=15.0,     # VIX < 20
        )
        as_of = date(2024, 6, 1)
        signal = CrossAssetSignal()
        result = signal.compute(features, as_of)

        assert len(result) > 0
        assert (result > 0).all(), f"Expected all positive, got {result}"

    def test_cross_asset_same_signal_all_symbols(self):
        """All symbols receive the identical macro signal."""
        symbols = ["AAPL", "MSFT", "GOOG", "AMZN"]
        features = self._make_macro_features(
            symbols, dgs10_change=0.2, baa10y_change=0.1, vix_level=25.0
        )
        as_of = date(2024, 6, 1)
        signal = CrossAssetSignal()
        result = signal.compute(features, as_of)

        assert len(result) == len(symbols)
        assert result.nunique() == 1


# ---------------------------------------------------------------------------
# SignalCombiner
# ---------------------------------------------------------------------------

class TestSignalCombiner:
    @pytest.fixture
    def registry(self):
        return SignalRegistry()

    @pytest.fixture
    def combiner(self, registry):
        return SignalCombiner(registry)

    def test_combiner_equal_weight(self, combiner):
        """Equal weight of two opposite signals → ~0."""
        s1 = pd.Series({"AAPL": 1.0, "MSFT": 0.5})
        s2 = pd.Series({"AAPL": -1.0, "MSFT": -0.5})
        result = combiner.equal_weight({"sig1": s1, "sig2": s2})

        assert abs(result["AAPL"]) < 1e-9
        assert abs(result["MSFT"]) < 1e-9

    def test_combiner_weighted_combine(self, combiner):
        """Weighted combine with explicit weights."""
        s1 = pd.Series({"AAPL": 1.0, "MSFT": 0.0})
        s2 = pd.Series({"AAPL": 0.0, "MSFT": 1.0})
        result = combiner.combine([s1, s2], [0.8, 0.2])

        assert result["AAPL"] == pytest.approx(0.8)
        assert result["MSFT"] == pytest.approx(0.2)

    def test_combiner_output_in_range(self, combiner):
        """Combined output stays in [-1, 1]."""
        s1 = pd.Series({"AAPL": 0.9, "MSFT": -0.9})
        s2 = pd.Series({"AAPL": 0.8, "MSFT": -0.8})
        result = combiner.combine([s1, s2], [0.5, 0.5])
        assert result.between(-1.0001, 1.0001).all()

    def test_combiner_correlation(self, combiner):
        """Two identical signals → correlation = 1.0."""
        dates = pd.date_range("2024-01-01", periods=50)
        vals = np.random.default_rng(1).normal(0, 0.3, 50)
        s1 = pd.Series(vals, index=dates)
        s2 = pd.Series(vals, index=dates)

        corr = combiner.compute_correlation_matrix({"sig1": s1, "sig2": s2})
        assert corr.loc["sig1", "sig2"] == pytest.approx(1.0, abs=1e-9)

    def test_combiner_correlation_independent(self, combiner):
        """Two uncorrelated signals → correlation near 0."""
        rng = np.random.default_rng(99)
        dates = pd.date_range("2024-01-01", periods=500)
        s1 = pd.Series(rng.normal(0, 1, 500), index=dates)
        s2 = pd.Series(rng.normal(0, 1, 500), index=dates)

        corr = combiner.compute_correlation_matrix({"sig1": s1, "sig2": s2})
        assert abs(corr.loc["sig1", "sig2"]) < 0.15

    def test_combiner_diversification(self, combiner):
        """Orthogonal signals → high diversification score (> 0.5)."""
        rng = np.random.default_rng(77)
        dates = pd.date_range("2024-01-01", periods=500)
        s1 = pd.Series(rng.normal(0, 1, 500), index=dates)
        s2 = pd.Series(rng.normal(0, 1, 500), index=dates)
        s3 = pd.Series(rng.normal(0, 1, 500), index=dates)

        metrics = combiner.analyze_diversification({"s1": s1, "s2": s2, "s3": s3})

        assert "mean_pairwise_corr" in metrics
        assert "max_corr" in metrics
        assert "min_corr" in metrics
        assert "diversification_score" in metrics
        assert metrics["diversification_score"] > 0.5

    def test_combiner_diversification_correlated(self, combiner):
        """Highly correlated signals → low diversification score."""
        rng = np.random.default_rng(55)
        dates = pd.date_range("2024-01-01", periods=200)
        base = rng.normal(0, 1, 200)
        s1 = pd.Series(base, index=dates)
        s2 = pd.Series(base + rng.normal(0, 0.01, 200), index=dates)

        metrics = combiner.analyze_diversification({"s1": s1, "s2": s2})
        assert metrics["diversification_score"] < 0.1

    def test_optimal_weights_sharpe(self, combiner):
        """Higher Sharpe signal gets higher weight."""
        rng = np.random.default_rng(11)
        dates = pd.date_range("2024-01-01", periods=200)

        # Strong positive returns
        strong = pd.Series(rng.normal(0.05, 0.1, 200), index=dates)
        # Weak positive returns
        weak = pd.Series(rng.normal(0.005, 0.1, 200), index=dates)

        weights = combiner.estimate_optimal_weights({"strong": strong, "weak": weak})

        assert "strong" in weights
        assert "weak" in weights
        assert weights["strong"] > weights["weak"]
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_optimal_weights_negative_sharpe_excluded(self, combiner):
        """Signals with negative Sharpe → weight = 0 (or equal fallback)."""
        rng = np.random.default_rng(22)
        dates = pd.date_range("2024-01-01", periods=200)

        good = pd.Series(rng.normal(0.05, 0.1, 200), index=dates)
        bad = pd.Series(rng.normal(-0.05, 0.1, 200), index=dates)

        weights = combiner.estimate_optimal_weights({"good": good, "bad": bad})

        assert weights["bad"] == pytest.approx(0.0, abs=1e-9)
        assert weights["good"] == pytest.approx(1.0, abs=1e-9)

    def test_combiner_missing_symbols_handled(self, combiner):
        """Symbols present in only one signal are handled gracefully."""
        s1 = pd.Series({"AAPL": 0.5, "MSFT": 0.3})
        s2 = pd.Series({"AAPL": 0.2, "GOOG": 0.4})  # GOOG only in s2, MSFT only in s1
        result = combiner.equal_weight({"sig1": s1, "sig2": s2})

        assert "AAPL" in result.index
        assert "MSFT" in result.index
        assert "GOOG" in result.index
        # MSFT and GOOG only in one signal → renormalized to that signal's value
        assert result["MSFT"] == pytest.approx(0.3)
        assert result["GOOG"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Metadata validation
# ---------------------------------------------------------------------------

class TestMetadata:
    """All signals must expose correct metadata."""

    @pytest.mark.parametrize("signal_cls,kwargs", [
        (CrossSectionalMomentum, {}),
        (TimeSeriesMomentum, {}),
        (RSIMeanReversion, {}),
        (BollingerMeanReversion, {}),
        (PostEarningsDrift, {}),
        (FilingAnomaly, {}),
        (SentimentSignal, {}),
        (CongressionalTrading, {}),
        (CrossAssetSignal, {}),
    ])
    def test_metadata_name_non_empty(self, signal_cls, kwargs):
        sig = signal_cls(**kwargs)
        assert sig.metadata.name != ""
        assert sig.metadata.version != ""
        assert sig.metadata.lookback_days > 0
        assert isinstance(sig.metadata.features_required, list)
        assert len(sig.metadata.features_required) > 0

    def test_signal_name_property(self):
        sig = CrossSectionalMomentum()
        assert sig.name == sig.metadata.name
        assert sig.version == sig.metadata.version

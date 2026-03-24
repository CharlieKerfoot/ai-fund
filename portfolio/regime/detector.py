"""Regime detector: blends quantitative macro signals with optional LLM assessment."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
import structlog

log = structlog.get_logger()

REGIMES = ["risk_on", "risk_off", "stagflation", "deflation", "crisis"]


@dataclass
class RegimeState:
    date: date
    regime_probs: dict[str, float]  # sum to 1.0
    dominant_regime: str
    confidence: float  # max probability
    quant_regime: str  # from quantitative features alone
    llm_regime: str | None  # from LLM assessment (None if degraded)
    blend_weight_quant: float = 0.7  # 70/30 quant/LLM blend


class RegimeDetector:
    """Detects market regime from macro features, optionally blending with LLM."""

    def __init__(self, feature_store=None, llm_regime=None) -> None:
        self.feature_store = feature_store
        self.llm_regime = llm_regime

    def detect(self, as_of: date) -> RegimeState:
        """
        Quantitative regime from macro features:
        - VIX > 25 -> crisis/risk_off weight
        - BAA10Y spread > 3.0 -> risk_off
        - Yield curve inverted (DGS10 < DGS2) -> stagflation/deflation weight
        - All signals benign -> risk_on

        Blend with LLM if available (70% quant, 30% LLM).
        """
        macro = self._get_macro(as_of)

        quant_probs = self._quant_regime(macro, as_of)
        quant_dominant = max(quant_probs, key=quant_probs.__getitem__)

        llm_probs = None
        llm_regime_name = None

        if self.llm_regime is not None:
            try:
                macro_summary = self._build_macro_summary(macro, as_of)
                llm_probs = self.llm_regime.assess(macro_summary, [], as_of)
                if llm_probs is not None:
                    llm_regime_name = max(llm_probs, key=llm_probs.__getitem__)
            except Exception as exc:
                log.warning("LLM regime assessment failed; using quant only", error=str(exc))
                llm_probs = None

        blended = self._blend_probabilities(quant_probs, llm_probs)
        dominant = max(blended, key=blended.__getitem__)
        confidence = blended[dominant]

        log.info(
            "Regime detected",
            date=str(as_of),
            regime=dominant,
            confidence=round(confidence, 4),
            quant_regime=quant_dominant,
            llm_regime=llm_regime_name,
        )

        return RegimeState(
            date=as_of,
            regime_probs=blended,
            dominant_regime=dominant,
            confidence=confidence,
            quant_regime=quant_dominant,
            llm_regime=llm_regime_name,
        )

    def _get_macro(self, as_of: date) -> pd.DataFrame:
        """Retrieve macro data from feature_store or return empty DataFrame."""
        if self.feature_store is None:
            return pd.DataFrame()

        try:
            macro = self.feature_store.get_macro(as_of=as_of)
            return macro
        except Exception:
            return pd.DataFrame()

    def _build_macro_summary(self, macro: pd.DataFrame, as_of: date) -> str:
        """Build a text summary of current macro conditions."""
        if macro.empty:
            return f"Macro data unavailable as of {as_of}"

        lines = [f"Macro indicators as of {as_of}:"]
        for _, row in macro.iterrows():
            lines.append(f"  {row.get('series_id', 'unknown')}: {row.get('value', 'N/A')}")
        return "\n".join(lines)

    def _get_series_value(self, macro: pd.DataFrame, series_id: str) -> float | None:
        """Extract the most recent value for a given series_id."""
        if macro.empty:
            return None
        if "series_id" not in macro.columns:
            return None
        subset = macro[macro["series_id"] == series_id]
        if subset.empty:
            return None
        if "date" in subset.columns:
            subset = subset.sort_values("date")
        return float(subset.iloc[-1]["value"])

    def _quant_regime(self, macro: pd.DataFrame, as_of: date) -> dict[str, float]:
        """Returns probability dict over REGIMES using macro indicators."""
        vix = self._get_series_value(macro, "VIXCLS")
        baa10y = self._get_series_value(macro, "BAA10Y")
        dgs10 = self._get_series_value(macro, "DGS10")
        dgs2 = self._get_series_value(macro, "DGS2")

        # Initialize scores
        scores = {regime: 0.0 for regime in REGIMES}

        # VIX signals
        if vix is not None:
            if vix > 40:
                scores["crisis"] += 3.0
                scores["risk_off"] += 1.0
            elif vix > 30:
                scores["crisis"] += 1.5
                scores["risk_off"] += 1.5
            elif vix > 25:
                scores["risk_off"] += 2.0
                scores["crisis"] += 0.5
            elif vix < 15:
                scores["risk_on"] += 2.0
            else:
                scores["risk_on"] += 1.0

        # Credit spread signals
        if baa10y is not None:
            if baa10y > 5.0:
                scores["crisis"] += 2.0
                scores["risk_off"] += 1.0
            elif baa10y > 3.0:
                scores["risk_off"] += 2.0
            elif baa10y < 1.5:
                scores["risk_on"] += 1.5
            else:
                scores["risk_on"] += 0.5

        # Yield curve signals
        if dgs10 is not None and dgs2 is not None:
            spread = dgs10 - dgs2
            if spread < -0.5:
                scores["stagflation"] += 1.5
                scores["deflation"] += 0.5
                scores["risk_off"] += 0.5
            elif spread < 0:
                scores["stagflation"] += 0.5
                scores["deflation"] += 0.5
            else:
                scores["risk_on"] += 0.5

        # Default benign case: if no signals fired at all
        total_score = sum(scores.values())
        if total_score == 0:
            scores["risk_on"] = 1.0
            total_score = 1.0

        # Normalize to probabilities
        probs = {k: v / total_score for k, v in scores.items()}
        return probs

    def _blend_probabilities(
        self, quant_probs: dict, llm_probs: dict | None
    ) -> dict[str, float]:
        """70% quant + 30% LLM if LLM available, else 100% quant."""
        if llm_probs is None:
            return dict(quant_probs)

        w_q = 0.7
        w_l = 0.3

        blended = {}
        for regime in REGIMES:
            q = quant_probs.get(regime, 0.0)
            llm_p = llm_probs.get(regime, 0.0)
            blended[regime] = w_q * q + w_l * llm_p

        # Normalize in case of floating point drift
        total = sum(blended.values())
        if total > 0:
            blended = {k: v / total for k, v in blended.items()}
        return blended

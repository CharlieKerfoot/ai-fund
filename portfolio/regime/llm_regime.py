"""LLM-based regime assessor using Claude API."""

from __future__ import annotations

import json
import logging
from datetime import date

logger = logging.getLogger(__name__)

REGIMES = ["risk_on", "risk_off", "stagflation", "deflation", "crisis"]


class LLMRegimeAssessor:
    """Calls Claude to assess market regime probabilities from macro context."""

    def __init__(self, api_key: str, model: str = "claude-opus-4-6") -> None:
        self.api_key = api_key
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def assess(
        self,
        macro_summary: str,
        news_headlines: list[str],
        as_of: date,
    ) -> dict[str, float] | None:
        """
        Call Claude API with macro summary + news to get regime probabilities.
        Returns None on any API failure (degraded mode).

        Prompt asks Claude to return JSON: {"risk_on": 0.x, "risk_off": 0.x, ...}
        Validates probabilities sum to 1.0 +/- 0.01.
        """
        try:
            client = self._get_client()
            prompt = self._build_prompt(macro_summary, news_headlines, as_of)

            message = client.messages.create(
                model=self.model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )

            response_text = message.content[0].text
            return self._parse_response(response_text)

        except Exception as e:
            logger.warning("LLM regime assessment failed (degraded mode): %s", e)
            return None

    def _build_prompt(
        self, macro_summary: str, news_headlines: list[str], as_of: date
    ) -> str:
        headlines_text = ""
        if news_headlines:
            headlines_text = "\n\nRecent news headlines:\n" + "\n".join(
                f"- {h}" for h in news_headlines[:10]
            )

        return f"""You are a quantitative macro analyst. Based on the following market data,
assess the probability of each market regime as of {as_of}.

{macro_summary}{headlines_text}

The five regimes are:
- risk_on: growth accelerating, low volatility, tight credit spreads
- risk_off: growth slowing, elevated volatility, widening spreads
- stagflation: high inflation with slowing growth, inverted yield curve
- deflation: falling prices, deflationary pressures, very low rates
- crisis: extreme stress, VIX > 30, credit markets seizing

Return ONLY a JSON object with probability estimates that sum to 1.0, like:
{{"risk_on": 0.4, "risk_off": 0.3, "stagflation": 0.1, "deflation": 0.1, "crisis": 0.1}}

No explanation, just the JSON object."""

    def _parse_response(self, response_text: str) -> dict[str, float] | None:
        """Parse and validate regime probabilities from LLM response."""
        try:
            # Try to extract JSON from response
            text = response_text.strip()

            # Find JSON object in the response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start == -1 or end == 0:
                logger.warning("No JSON found in LLM regime response")
                return None

            json_str = text[start:end]
            probs = json.loads(json_str)

            # Validate all required regimes present
            for regime in REGIMES:
                if regime not in probs:
                    logger.warning("Missing regime '%s' in LLM response", regime)
                    return None

            # Validate probabilities are numeric and non-negative
            float_probs = {}
            for regime in REGIMES:
                val = float(probs[regime])
                if val < 0:
                    logger.warning("Negative probability for regime '%s'", regime)
                    return None
                float_probs[regime] = val

            # Validate sum to 1.0 ± 0.01
            total = sum(float_probs.values())
            if abs(total - 1.0) > 0.01:
                logger.warning(
                    "LLM regime probs sum to %.4f, not 1.0; normalizing", total
                )
                if total <= 0:
                    return None
                float_probs = {k: v / total for k, v in float_probs.items()}

            return float_probs

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning("Failed to parse LLM regime response: %s", e)
            return None

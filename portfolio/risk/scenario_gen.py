"""LLM-based tail-risk scenario generator."""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


class LLMScenarioGenerator:
    """Generates portfolio-specific tail-risk scenarios using Claude."""

    def __init__(self, api_key: str, model: str = "claude-opus-4-6") -> None:
        self.api_key = api_key
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def generate_scenarios(
        self,
        portfolio_description: str,
        current_regime: str,
        top_positions: list[str],
    ) -> list[dict] | None:
        """
        Ask Claude to generate 3 tail-risk scenarios specific to current portfolio.
        Returns list of dicts: [{name, description, trigger, equity_shock, probability}]
        Returns None on API failure (degraded mode).
        """
        try:
            client = self._get_client()
            prompt = self._build_prompt(
                portfolio_description, current_regime, top_positions
            )

            message = client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )

            response_text = message.content[0].text
            return self._parse_response(response_text)

        except Exception as e:
            logger.warning(
                "LLM scenario generation failed (degraded mode): %s", e
            )
            return None

    def _build_prompt(
        self,
        portfolio_description: str,
        current_regime: str,
        top_positions: list[str],
    ) -> str:
        positions_str = ", ".join(top_positions[:10]) if top_positions else "unknown"

        return f"""You are a quantitative risk manager. Generate exactly 3 specific tail-risk
scenarios for the following portfolio.

Portfolio description: {portfolio_description}
Current market regime: {current_regime}
Top positions: {positions_str}

For each scenario, provide a JSON object with these fields:
- name: short scenario name (string)
- description: 1-2 sentence description (string)
- trigger: what event triggers this scenario (string)
- equity_shock: estimated portfolio loss as a decimal (e.g., -0.25 for -25%)
- probability: estimated 12-month probability (decimal between 0 and 1)

Return ONLY a JSON array of exactly 3 scenario objects, like:
[
  {{"name": "...", "description": "...", "trigger": "...", "equity_shock": -0.20, "probability": 0.10}},
  {{"name": "...", "description": "...", "trigger": "...", "equity_shock": -0.30, "probability": 0.05}},
  {{"name": "...", "description": "...", "trigger": "...", "equity_shock": -0.15, "probability": 0.15}}
]

No explanation outside the JSON array."""

    def _parse_response(self, response_text: str) -> list[dict] | None:
        """Parse and validate scenario list from LLM response."""
        try:
            text = response_text.strip()

            # Find JSON array in the response
            start = text.find("[")
            end = text.rfind("]") + 1
            if start == -1 or end == 0:
                logger.warning("No JSON array found in LLM scenario response")
                return None

            json_str = text[start:end]
            scenarios = json.loads(json_str)

            if not isinstance(scenarios, list):
                logger.warning("Expected list of scenarios, got %s", type(scenarios))
                return None

            required_fields = {"name", "description", "trigger", "equity_shock", "probability"}
            validated = []
            for i, scenario in enumerate(scenarios):
                if not isinstance(scenario, dict):
                    logger.warning("Scenario %d is not a dict", i)
                    continue
                missing = required_fields - set(scenario.keys())
                if missing:
                    logger.warning("Scenario %d missing fields: %s", i, missing)
                    continue

                # Validate numeric fields
                try:
                    eq_shock = float(scenario["equity_shock"])
                    prob = float(scenario["probability"])
                    if not (-1.0 <= eq_shock <= 0.0):
                        eq_shock = max(-1.0, min(0.0, eq_shock))
                    if not (0.0 <= prob <= 1.0):
                        prob = max(0.0, min(1.0, prob))
                    scenario["equity_shock"] = eq_shock
                    scenario["probability"] = prob
                    validated.append(scenario)
                except (ValueError, TypeError) as e:
                    logger.warning("Scenario %d has invalid numeric fields: %s", i, e)
                    continue

            if not validated:
                logger.warning("No valid scenarios parsed from LLM response")
                return None

            return validated

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Failed to parse LLM scenario response: %s", e)
            return None

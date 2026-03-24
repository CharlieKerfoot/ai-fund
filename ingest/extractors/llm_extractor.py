"""LLM-based feature extraction using the Anthropic Claude API."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

import structlog

log = structlog.get_logger()

# Pricing for claude-opus-4-6 in USD per million tokens
_INPUT_COST_PER_M = 15.0
_OUTPUT_COST_PER_M = 75.0


@dataclass
class EarningsFeatures:
    sentiment_score: float          # -1.0 to 1.0
    guidance_direction: str         # "up", "down", "flat", "withdrawn", "none"
    management_confidence: float    # 0.0 to 1.0
    key_risks: list[str] = field(default_factory=list)   # up to 5 risk phrases
    hedging_language_count: int = 0
    forward_looking_positive: int = 0
    forward_looking_negative: int = 0


@dataclass
class FilingFeatures:
    risk_factor_delta: str          # "increased", "decreased", "unchanged"
    new_risk_factors: list[str] = field(default_factory=list)
    accounting_changes: list[str] = field(default_factory=list)
    litigation_exposure: str = "none"  # "high", "medium", "low", "none"
    going_concern: bool = False
    revenue_trend: str = "stable"   # "growing", "declining", "stable", "mixed"


class LLMExtractor:
    """Extract structured features from text using Claude."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-opus-4-6",
        daily_budget_usd: float = 5.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._daily_budget_usd = daily_budget_usd
        self._cost_today: float = 0.0
        self._budget_date: date = datetime.now(tz=UTC).date()
        self._client = None  # lazy init

    def _get_client(self):
        """Lazily initialize Anthropic client."""
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def _reset_budget_if_new_day(self) -> None:
        today = datetime.now(tz=UTC).date()
        if today != self._budget_date:
            self._cost_today = 0.0
            self._budget_date = today

    def check_budget(self) -> bool:
        """Returns True if daily budget has not been exceeded."""
        self._reset_budget_if_new_day()
        return self._cost_today < self._daily_budget_usd

    def get_cost_today(self) -> float:
        """Returns estimated USD cost of today's API calls."""
        self._reset_budget_if_new_day()
        return self._cost_today

    def _track_usage(self, input_tokens: int, output_tokens: int) -> None:
        cost = (input_tokens / 1_000_000) * _INPUT_COST_PER_M
        cost += (output_tokens / 1_000_000) * _OUTPUT_COST_PER_M
        self._cost_today += cost

    def _chunk_text(self, text: str, max_chars: int = 80000) -> list[str]:
        """Split text into chunks, preserving paragraph boundaries."""
        if len(text) <= max_chars:
            return [text]

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + max_chars
            if end >= len(text):
                chunks.append(text[start:])
                break
            # Try to break at a paragraph boundary
            boundary = text.rfind("\n\n", start, end)
            if boundary == -1 or boundary <= start:
                # Fall back to newline
                boundary = text.rfind("\n", start, end)
            if boundary == -1 or boundary <= start:
                # Fall back to space
                boundary = text.rfind(" ", start, end)
            if boundary == -1 or boundary <= start:
                boundary = end
            chunks.append(text[start:boundary])
            start = boundary + 1

        return [c for c in chunks if c.strip()]

    def _call_claude(self, prompt: str) -> str | None:
        """Call Claude API. Returns None on any API error (degraded mode)."""
        if not self.check_budget():
            log.warning(
                "Daily budget exceeded; skipping LLM extraction",
                budget_usd=self._daily_budget_usd,
                cost_today=round(self._cost_today, 4),
            )
            return None
        try:
            client = self._get_client()
            log.info("Calling Claude API", model=self._model, cost_today=round(self._cost_today, 4))
            response = client.messages.create(
                model=self._model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            self._track_usage(input_tokens, output_tokens)
            log.info(
                "Claude API call complete",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_today=round(self._cost_today, 4),
            )
            return response.content[0].text
        except Exception as exc:
            log.warning("Claude API call failed (degraded mode)", error=str(exc))
            return None

    def extract_earnings(
        self,
        text: str,
        ticker: str,
        quarter: str,
    ) -> EarningsFeatures | None:
        """Extract structured features from an earnings call transcript.

        Returns None if API unavailable (degraded mode).
        """
        log.info("Starting earnings extraction", ticker=ticker, quarter=quarter)
        chunks = self._chunk_text(text)
        all_results: list[dict] = []

        for chunk in chunks:
            prompt = f"""You are a financial analyst extracting structured data from an earnings call transcript.

Ticker: {ticker}
Quarter: {quarter}

Transcript excerpt:
{chunk}

Extract the following features and return them as a single JSON object with exactly these keys:
- "sentiment_score": float from -1.0 (very negative) to 1.0 (very positive)
- "guidance_direction": one of "up", "down", "flat", "withdrawn", "none"
- "management_confidence": float from 0.0 (very uncertain) to 1.0 (very confident)
- "key_risks": list of up to 5 short risk phrases mentioned
- "hedging_language_count": integer count of hedging phrases (e.g., "may", "could", "might", "uncertain")
- "forward_looking_positive": integer count of positive forward-looking statements
- "forward_looking_negative": integer count of negative forward-looking statements

Respond with ONLY the JSON object, no other text."""

            raw = self._call_claude(prompt)
            if raw is None:
                return None
            try:
                parsed = json.loads(raw)
                all_results.append(parsed)
            except (json.JSONDecodeError, ValueError) as exc:
                log.warning("Failed to parse earnings JSON from Claude", error=str(exc))
                continue

        if not all_results:
            return None

        return self._aggregate_earnings(all_results)

    def _aggregate_earnings(self, results: list[dict]) -> EarningsFeatures:
        """Aggregate results from multiple chunks."""
        len(results)

        def avg_float(key: str, default: float = 0.0) -> float:
            vals = [r.get(key, default) for r in results if isinstance(r.get(key), (int, float))]
            return sum(vals) / len(vals) if vals else default

        def union_list(key: str) -> list[str]:
            seen: set[str] = set()
            out: list[str] = []
            for r in results:
                for item in r.get(key, []):
                    if isinstance(item, str) and item not in seen:
                        seen.add(item)
                        out.append(item)
            return out[:5]

        def most_common_str(key: str, choices: list[str], default: str) -> str:
            counts: dict[str, int] = {}
            for r in results:
                v = r.get(key, default)
                if v in choices:
                    counts[v] = counts.get(v, 0) + 1
            return max(counts, key=lambda k: counts[k]) if counts else default

        def sum_int(key: str) -> int:
            return sum(int(r.get(key, 0)) for r in results if isinstance(r.get(key), (int, float)))

        guidance_choices = ["up", "down", "flat", "withdrawn", "none"]
        return EarningsFeatures(
            sentiment_score=round(avg_float("sentiment_score"), 4),
            guidance_direction=most_common_str("guidance_direction", guidance_choices, "none"),
            management_confidence=round(avg_float("management_confidence"), 4),
            key_risks=union_list("key_risks"),
            hedging_language_count=sum_int("hedging_language_count"),
            forward_looking_positive=sum_int("forward_looking_positive"),
            forward_looking_negative=sum_int("forward_looking_negative"),
        )

    def extract_filing(
        self,
        text: str,
        ticker: str,
        filing_type: str,
    ) -> FilingFeatures | None:
        """Extract structured features from an SEC filing.

        Returns None if API unavailable.
        """
        log.info("Starting filing extraction", ticker=ticker, filing_type=filing_type)
        chunks = self._chunk_text(text)
        all_results: list[dict] = []

        for chunk in chunks:
            prompt = f"""You are a financial analyst extracting structured data from an SEC filing.

Ticker: {ticker}
Filing type: {filing_type}

Filing excerpt:
{chunk}

Extract the following features and return them as a single JSON object with exactly these keys:
- "risk_factor_delta": one of "increased", "decreased", "unchanged"
- "new_risk_factors": list of new risk factors mentioned (strings)
- "accounting_changes": list of accounting policy changes mentioned (strings)
- "litigation_exposure": one of "high", "medium", "low", "none"
- "going_concern": boolean, true if going concern language is detected
- "revenue_trend": one of "growing", "declining", "stable", "mixed"

Respond with ONLY the JSON object, no other text."""

            raw = self._call_claude(prompt)
            if raw is None:
                return None
            try:
                parsed = json.loads(raw)
                all_results.append(parsed)
            except (json.JSONDecodeError, ValueError) as exc:
                log.warning("Failed to parse filing JSON from Claude", error=str(exc))
                continue

        if not all_results:
            return None

        return self._aggregate_filing(all_results)

    def _aggregate_filing(self, results: list[dict]) -> FilingFeatures:
        """Aggregate filing results from multiple chunks."""

        def union_list(key: str) -> list[str]:
            seen: set[str] = set()
            out: list[str] = []
            for r in results:
                for item in r.get(key, []):
                    if isinstance(item, str) and item not in seen:
                        seen.add(item)
                        out.append(item)
            return out

        def most_common_str(key: str, choices: list[str], default: str) -> str:
            counts: dict[str, int] = {}
            for r in results:
                v = r.get(key, default)
                if v in choices:
                    counts[v] = counts.get(v, 0) + 1
            return max(counts, key=lambda k: counts[k]) if counts else default

        def any_bool(key: str) -> bool:
            return any(bool(r.get(key, False)) for r in results)

        delta_choices = ["increased", "decreased", "unchanged"]
        litigation_choices = ["high", "medium", "low", "none"]
        trend_choices = ["growing", "declining", "stable", "mixed"]

        return FilingFeatures(
            risk_factor_delta=most_common_str("risk_factor_delta", delta_choices, "unchanged"),
            new_risk_factors=union_list("new_risk_factors"),
            accounting_changes=union_list("accounting_changes"),
            litigation_exposure=most_common_str("litigation_exposure", litigation_choices, "none"),
            going_concern=any_bool("going_concern"),
            revenue_trend=most_common_str("revenue_trend", trend_choices, "stable"),
        )

"""
engagement_scorer.py — EngagementScorer

Four-dimension scoring model for YouTube content keywords:

  1. title_score       — How clickable/compelling is this as a video title?
  2. trend_score       — How trendy/in-demand is this topic right now?
  3. competition_score — How winnable is this niche (inverse of saturation)?
  4. monetization_score — Advertiser value and CPM potential?

Each dimension is 0–100. A weighted composite score is computed.
Claude API is used to evaluate dimensions 1 and 4; trend data feeds dimension 2;
YouTube search volume proxies dimension 3.

Usage:
    scorer = EngagementScorer()
    result = scorer.score("best budget microphone 2024")
    print(result.composite_score, result.rationale)
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import anthropic
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ScoreResult:
    """Engagement scores for a single keyword."""

    keyword: str
    title_score: float = 0.0            # 0–100 clickability / title appeal
    trend_score: float = 0.0            # 0–100 trend momentum
    competition_score: float = 0.0      # 0–100 (high = low competition)
    monetization_score: float = 0.0     # 0–100 advertiser value
    composite_score: float = 0.0        # weighted average
    rationale: str = ""
    scored_at: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.scored_at:
            self.scored_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "keyword": self.keyword,
            "title_score": round(self.title_score, 2),
            "trend_score": round(self.trend_score, 2),
            "competition_score": round(self.competition_score, 2),
            "monetization_score": round(self.monetization_score, 2),
            "composite_score": round(self.composite_score, 2),
            "rationale": self.rationale,
            "scored_at": self.scored_at,
        }


# ---------------------------------------------------------------------------
# Weights for composite score
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: dict[str, float] = {
    "title_score": 0.30,
    "trend_score": 0.30,
    "competition_score": 0.20,
    "monetization_score": 0.20,
}


# ---------------------------------------------------------------------------
# Claude-powered scorer
# ---------------------------------------------------------------------------

_CLAUDE_MODEL = os.getenv("CLAUDE_SCORER_MODEL", "claude-haiku-4-5-20251001")

_SCORING_SYSTEM_PROMPT = """You are an expert YouTube content strategist and SEO analyst.
You will evaluate keywords/topics for their potential as YouTube video subjects.

For each keyword, provide scores (0–100) for:

1. title_score: How compelling and clickable would a video title based on this be?
   Consider: emotional triggers, curiosity gap, specificity, search intent match.

2. monetization_score: How valuable is this topic to advertisers?
   Consider: CPM estimates, buyer intent, product/service overlap,
   demographic quality (high-income audiences score higher).

Return ONLY a JSON object in this exact format (no markdown, no explanation):
{
  "title_score": 75,
  "monetization_score": 60,
  "rationale": "one or two sentences explaining the scores"
}
"""


class ClaudeScorer:
    """Uses Claude API to score title appeal and monetization potential."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = _CLAUDE_MODEL,
        max_tokens: int = 200,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("ANTHROPIC_API_KEY", "")
        self.model = model
        self.max_tokens = max_tokens
        self._client: anthropic.Anthropic | None = None

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            if not self.api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def score(self, keyword: str) -> tuple[float, float, str]:
        """
        Score a keyword for title appeal and monetization potential.

        Returns:
            (title_score, monetization_score, rationale)
        """
        prompt = (
            f'Evaluate this YouTube content keyword:\n"{keyword}"\n\n'
            "Provide title_score (clickability 0-100) and monetization_score "
            "(advertiser value 0-100)."
        )

        try:
            client = self._get_client()
            message = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=_SCORING_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text = message.content[0].text.strip()

            # Strip markdown code fences if present
            if raw_text.startswith("```"):
                raw_text = raw_text.strip("`").strip()
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:].strip()

            data = json.loads(raw_text)
            title_score = float(data.get("title_score", 50.0))
            monetization_score = float(data.get("monetization_score", 50.0))
            rationale = str(data.get("rationale", ""))

            # Clamp to valid range
            title_score = max(0.0, min(100.0, title_score))
            monetization_score = max(0.0, min(100.0, monetization_score))

            logger.debug(
                "Claude scored '%s': title=%.1f, monetization=%.1f",
                keyword, title_score, monetization_score,
            )
            return title_score, monetization_score, rationale

        except Exception as exc:
            logger.error("Claude scoring failed for '%s': %s", keyword, exc)
            return 50.0, 50.0, f"Scoring unavailable: {exc}"


# ---------------------------------------------------------------------------
# Heuristic scorers (no API calls)
# ---------------------------------------------------------------------------

def _score_trend(interest_score: float) -> float:
    """
    Convert a raw trend interest score (0–100) into a normalised trend_score.

    Uses a mild log boost so that moderate trends aren't penalised too heavily.
    """
    import math
    if interest_score <= 0:
        return 0.0
    # interest_score is already 0–100 from pytrends/YouTube
    # Apply a mild sigmoid-like scaling: keep high scores high, lift low scores
    normalised = min(100.0, interest_score * 1.1)
    return round(normalised, 2)


def _score_competition(
    keyword: str,
    youtube_result_count: int | None = None,
) -> float:
    """
    Estimate competition score (0–100, high = low competition = easier to rank).

    Heuristics:
      - Short generic keywords → low score (high competition)
      - Long-tail (4+ words) → higher score
      - Very high YouTube result counts → lower score
      - Question-form keywords → slight boost (lower competition niche)
    """
    import math

    score = 50.0  # baseline

    words = keyword.strip().split()
    word_count = len(words)

    # Long-tail boost
    if word_count >= 5:
        score += 20.0
    elif word_count >= 4:
        score += 12.0
    elif word_count >= 3:
        score += 6.0
    elif word_count <= 1:
        score -= 15.0

    # Question-form boost
    question_starters = {"how", "what", "why", "when", "where", "which", "who", "can", "does"}
    if words[0].lower() in question_starters:
        score += 8.0

    # Year/version specificity boost
    import re
    if re.search(r"\b(202[3-9]|203\d)\b", keyword):
        score += 10.0

    # YouTube result-count penalty (if data available)
    if youtube_result_count is not None and youtube_result_count > 0:
        # Very saturated (>1M results) lowers competition score
        saturation_penalty = math.log10(youtube_result_count) * 3
        score -= saturation_penalty

    return round(max(0.0, min(100.0, score)), 2)


# ---------------------------------------------------------------------------
# Main EngagementScorer
# ---------------------------------------------------------------------------

class EngagementScorer:
    """
    Four-dimension engagement scorer for YouTube content keywords.

    Scoring pipeline:
      - Dimensions 1 (title) and 4 (monetization): Claude API
      - Dimension 2 (trend): pass-through from TrendSignal.interest_score
      - Dimension 3 (competition): heuristic model

    Args:
        use_claude: If False, uses fixed fallback scores for Claude dimensions.
                    Useful for testing without API calls.
        weights: Custom dimension weights (must sum to 1.0).
        claude_api_key: Override for the Anthropic API key.
    """

    def __init__(
        self,
        use_claude: bool = True,
        weights: dict[str, float] | None = None,
        claude_api_key: str | None = None,
    ) -> None:
        self.use_claude = use_claude
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self._validate_weights()
        self._claude = ClaudeScorer(api_key=claude_api_key) if use_claude else None
        logger.info(
            "EngagementScorer initialised (use_claude=%s, weights=%s)",
            use_claude, self.weights,
        )

    def _validate_weights(self) -> None:
        """Ensure weights are valid and sum to approximately 1.0."""
        required = {"title_score", "trend_score", "competition_score", "monetization_score"}
        if set(self.weights.keys()) != required:
            raise ValueError(f"weights must have keys: {required}")
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"weights must sum to 1.0, got {total:.4f}")

    def score(
        self,
        keyword: str,
        trend_interest: float = 0.0,
        youtube_result_count: int | None = None,
    ) -> ScoreResult:
        """
        Score a single keyword across all four dimensions.

        Args:
            keyword: The keyword/topic string to score.
            trend_interest: Pre-fetched interest score from TrendScrapingEngine (0–100).
            youtube_result_count: Number of YouTube search results (for competition model).

        Returns:
            ScoreResult with all four dimension scores and composite.
        """
        logger.debug("Scoring keyword: '%s'", keyword)

        # Dimension 2: trend (heuristic)
        trend_score = _score_trend(trend_interest)

        # Dimension 3: competition (heuristic)
        competition_score = _score_competition(keyword, youtube_result_count)

        # Dimensions 1 and 4: Claude API (or fallback)
        if self.use_claude and self._claude:
            title_score, monetization_score, rationale = self._claude.score(keyword)
        else:
            # Fallback heuristics when Claude is unavailable
            title_score = self._heuristic_title_score(keyword)
            monetization_score = 50.0
            rationale = "Scored using heuristics (Claude API unavailable)"

        # Composite
        composite = (
            title_score * self.weights["title_score"]
            + trend_score * self.weights["trend_score"]
            + competition_score * self.weights["competition_score"]
            + monetization_score * self.weights["monetization_score"]
        )

        result = ScoreResult(
            keyword=keyword,
            title_score=round(title_score, 2),
            trend_score=round(trend_score, 2),
            competition_score=round(competition_score, 2),
            monetization_score=round(monetization_score, 2),
            composite_score=round(composite, 2),
            rationale=rationale,
            raw={
                "trend_interest_input": trend_interest,
                "youtube_result_count": youtube_result_count,
            },
        )

        logger.info(
            "Scored '%s': composite=%.1f (title=%.1f, trend=%.1f, comp=%.1f, mono=%.1f)",
            keyword, composite, title_score, trend_score, competition_score, monetization_score,
        )
        return result

    def score_batch(
        self,
        keywords: list[str],
        trend_map: dict[str, float] | None = None,
        result_count_map: dict[str, int] | None = None,
    ) -> list[ScoreResult]:
        """
        Score a list of keywords.

        Args:
            keywords: List of keywords.
            trend_map: Optional {keyword: interest_score} from TrendScrapingEngine.
            result_count_map: Optional {keyword: youtube_result_count}.

        Returns:
            List of ScoreResult, sorted by composite_score descending.
        """
        trend_map = trend_map or {}
        result_count_map = result_count_map or {}
        results: list[ScoreResult] = []

        for kw in keywords:
            result = self.score(
                keyword=kw,
                trend_interest=trend_map.get(kw, 0.0),
                youtube_result_count=result_count_map.get(kw),
            )
            results.append(result)

        results.sort(key=lambda r: r.composite_score, reverse=True)
        return results

    @staticmethod
    def _heuristic_title_score(keyword: str) -> float:
        """
        Simple heuristic title score when Claude is not available.

        Rewards: question words, numbers, emotional words, "how to", etc.
        """
        score = 40.0
        lower = keyword.lower()
        words = lower.split()

        emotional_words = {
            "best", "worst", "secret", "amazing", "shocking", "truth",
            "ultimate", "complete", "proven", "easy", "simple", "fast",
            "free", "new", "top", "must", "never", "always", "every",
        }
        for word in words:
            if word in emotional_words:
                score += 8.0
                break  # only count once

        import re
        if re.search(r"\b(how\s+to|step[- ]by[- ]step|beginners?|tutorial)\b", lower):
            score += 12.0

        if re.search(r"\b\d+\b", keyword):
            score += 8.0  # listicle/numbered boost

        if re.search(r"\b(202[3-9]|203\d)\b", keyword):
            score += 5.0  # year specificity

        word_count = len(words)
        if 4 <= word_count <= 8:
            score += 5.0
        elif word_count > 10:
            score -= 5.0

        return round(max(0.0, min(100.0, score)), 2)

"""
hook_generator.py — HookGenerator

Generates 5 hook variants for a YouTube Shorts topic using Claude API,
scores each on curiosity and clarity (0–10), and returns the best one.

Usage:
    gen = HookGenerator()
    result = gen.generate(topic="stoic quotes", score=82.5, emotion="curiosity")
    print(result.text, result.curiosity_score, result.clarity_score)
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

_MODEL = "claude-sonnet-4-5"

_SYSTEM_PROMPT = """You are an expert YouTube Shorts scriptwriter specialising in
viral hooks. Your hooks must be punchy, under 15 words, and immediately grab attention.

Given a topic, engagement score, and dominant emotion, generate exactly 5 hook variants.
Each hook must:
- Be 10–15 words long
- Match the dominant emotion
- Create immediate curiosity or shock
- Work as the opening line of a 15-second video

For each hook also provide:
- curiosity_score: 0–10 (how much it makes viewer want to keep watching)
- clarity_score: 0–10 (how clearly it signals what the video is about)

Respond ONLY with a JSON array of exactly 5 objects, no markdown:
[
  {"text": "hook text here", "curiosity_score": 8, "clarity_score": 7},
  ...
]"""


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class HookVariant:
    """A single generated hook with its quality scores."""

    text: str
    curiosity_score: float       # 0–10
    clarity_score: float         # 0–10

    @property
    def combined_score(self) -> float:
        """Equal-weight combination of curiosity and clarity."""
        return (self.curiosity_score + self.clarity_score) / 2.0


@dataclass
class HookResult:
    """Output of HookGenerator.generate() — best hook plus all variants."""

    topic: str
    emotion: str
    input_score: float           # composite engagement score from scorer
    best: HookVariant            # highest combined_score
    variants: list[HookVariant]  # all 5 generated variants
    generated_at: str = ""
    raw_response: str = ""

    def __post_init__(self) -> None:
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "emotion": self.emotion,
            "input_score": self.input_score,
            "best_hook": self.best.text,
            "best_curiosity": self.best.curiosity_score,
            "best_clarity": self.best.clarity_score,
            "best_combined": round(self.best.combined_score, 2),
            "all_variants": [
                {
                    "text": v.text,
                    "curiosity_score": v.curiosity_score,
                    "clarity_score": v.clarity_score,
                    "combined_score": round(v.combined_score, 2),
                }
                for v in self.variants
            ],
            "generated_at": self.generated_at,
        }


# ---------------------------------------------------------------------------
# HookGenerator
# ---------------------------------------------------------------------------

class HookGenerator:
    """
    Generates 5 hook variants via Claude API and selects the best one.

    Args:
        api_key: Anthropic API key. If None, reads ANTHROPIC_API_KEY from env.
        model: Claude model ID to use.
        max_tokens: Max tokens for the API response.
    """

    NUM_VARIANTS = 5

    def __init__(
        self,
        api_key: str | None = None,
        model: str = _MODEL,
        max_tokens: int = 600,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("ANTHROPIC_API_KEY", "")
        self.model = model
        self.max_tokens = max_tokens
        self._client: anthropic.Anthropic | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        topic: str,
        score: float,
        emotion: str,
    ) -> HookResult:
        """
        Generate 5 hook variants and return the best one.

        Args:
            topic: The video topic or keyword (e.g. "stoic quotes").
            score: Composite engagement score from EngagementScorer (0–100).
            emotion: Dominant emotion to target (e.g. "curiosity", "shock",
                     "inspiration", "fear", "humour").

        Returns:
            HookResult containing all variants and the selected best hook.

        Raises:
            ValueError: If ANTHROPIC_API_KEY is not configured.
        """
        client = self._get_client()
        prompt = self._build_prompt(topic, score, emotion)

        logger.info(
            "Generating hooks for topic='%s' emotion='%s' score=%.1f",
            topic, emotion, score,
        )

        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
        logger.debug("Raw hook response: %s", raw)

        variants = self._parse_variants(raw)
        best = self._select_best(variants)

        logger.info(
            "Selected hook (combined=%.1f): %s",
            best.combined_score, best.text,
        )
        return HookResult(
            topic=topic,
            emotion=emotion,
            input_score=score,
            best=best,
            variants=variants,
            raw_response=raw,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            if not self.api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def _build_prompt(self, topic: str, score: float, emotion: str) -> str:
        return (
            f"Topic: {topic}\n"
            f"Engagement score: {score:.1f}/100\n"
            f"Dominant emotion: {emotion}\n\n"
            f"Generate {self.NUM_VARIANTS} hook variants for a YouTube Shorts video."
        )

    def _parse_variants(self, raw: str) -> list[HookVariant]:
        """
        Parse the JSON array from Claude's response into HookVariant objects.

        Falls back to a safe default list if parsing fails, so the pipeline
        never hard-crashes on a malformed API response.
        """
        text = raw.strip()

        # Strip markdown fences if present
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            ).strip()

        try:
            data = json.loads(text)
            if not isinstance(data, list):
                raise ValueError("Expected JSON array")

            variants: list[HookVariant] = []
            for item in data[:self.NUM_VARIANTS]:
                variants.append(
                    HookVariant(
                        text=str(item.get("text", "")).strip(),
                        curiosity_score=float(
                            max(0.0, min(10.0, item.get("curiosity_score", 5.0)))
                        ),
                        clarity_score=float(
                            max(0.0, min(10.0, item.get("clarity_score", 5.0)))
                        ),
                    )
                )

            if not variants:
                raise ValueError("No variants parsed")

            # Pad to NUM_VARIANTS if Claude returned fewer
            while len(variants) < self.NUM_VARIANTS:
                variants.append(HookVariant(text="Hook unavailable", curiosity_score=0.0, clarity_score=0.0))

            return variants

        except Exception as exc:
            logger.error("Failed to parse hook variants: %s | raw=%s", exc, raw[:200])
            # Return safe fallback variants so downstream never crashes
            return [
                HookVariant(text="Hook generation failed", curiosity_score=0.0, clarity_score=0.0)
                for _ in range(self.NUM_VARIANTS)
            ]

    @staticmethod
    def _select_best(variants: list[HookVariant]) -> HookVariant:
        """Return the variant with the highest combined curiosity + clarity score."""
        return max(variants, key=lambda v: v.combined_score)

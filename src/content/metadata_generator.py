"""
metadata_generator.py — MetadataGenerator

Generates SEO-optimised YouTube metadata from a topic and script:
  - Title: max 60 characters
  - Description: max 200 characters, ends with "Comment below 👇"
  - Hashtags: exactly 15, always includes #Shorts

Usage:
    gen = MetadataGenerator()
    result = gen.generate(topic="stoic quotes", script="Most people ignore...")
    print(result.title)
    print(result.description)
    print(result.hashtags)
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import anthropic
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-5"

_SYSTEM_PROMPT = """You are a YouTube SEO specialist who writes viral metadata for Shorts targeting high-CPM English-speaking audiences in the US, UK, Canada, and Australia.

Given a topic and a short script, generate:
1. title: SEO-optimised YouTube title
   - Max 60 characters (HARD LIMIT)
   - No clickbait that misrepresents content
   - Include the main keyword near the front
   - No hashtags in the title

2. description: Video description
   - Max 200 characters (HARD LIMIT — count carefully)
   - Summarise the video value in 1–2 sentences
   - Must end EXACTLY with: Comment below 👇
   - No URLs

3. hashtags: Array of exactly 15 hashtag strings
   - Each starts with #
   - No spaces within a hashtag (use CamelCase for multi-word)
   - Must include #Shorts
   - Use location-neutral English finance/self-improvement terms
   - REQUIRED tags (must include all of these): #personalfinance, #wealthbuilding,
     #financialfreedom, #moneyadvice, #moneytips, #financetok, #Shorts
   - Additional tags to choose from: #investing, #financialtips,
     #moneymindset, #buildwealth, #investingforbeginners
   - DO NOT include any regional or country-specific tags
     (e.g. NO #Nigeria, #Africa, #Naira, #Lagos or similar)
   - Mix: broad (e.g. #Motivation) and specific (e.g. #PersonalFinance)
   - All lowercase except for CamelCase and #Shorts

4. cta_product: The exact name of the free product being offered in the CTA
   (copied verbatim from the input). If no product is provided, return an
   empty string.

Respond ONLY with a JSON object, no markdown:
{
  "title": "...",
  "description": "...",
  "hashtags": ["#Shorts", "#tag2", ...],
  "cta_product": "..."
}"""

# ---------------------------------------------------------------------------
# Validation constants
# ---------------------------------------------------------------------------

TITLE_MAX_CHARS      = 60
DESCRIPTION_MAX_CHARS = 200
REQUIRED_HASHTAG_COUNT = 15
REQUIRED_HASHTAG      = "#Shorts"
DESCRIPTION_SUFFIX    = "Comment below 👇"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class MetadataResult:
    """
    SEO metadata for a YouTube Shorts video.
    """

    topic: str
    title: str
    description: str
    hashtags: list[str]          # exactly 15 strings starting with #
    is_valid: bool
    cta_product: str = ""        # free product name referenced in the CTA
    validation_errors: list[str] = field(default_factory=list)
    generated_at: str = ""
    raw_response: str = ""

    def __post_init__(self) -> None:
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "title": self.title,
            "description": self.description,
            "hashtags": self.hashtags,
            "hashtags_string": " ".join(self.hashtags),
            "cta_product": self.cta_product,
            "is_valid": self.is_valid,
            "validation_errors": self.validation_errors,
            "generated_at": self.generated_at,
        }


# ---------------------------------------------------------------------------
# MetadataGenerator
# ---------------------------------------------------------------------------

class MetadataGenerator:
    """
    Generates SEO-optimised YouTube metadata via Claude API.

    Args:
        api_key: Anthropic API key. If None, reads ANTHROPIC_API_KEY from env.
        model: Claude model ID.
        max_tokens: Max tokens for the API response.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = _MODEL,
        max_tokens: int = 500,
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
        script: str,
        cta_product: str = "",
        category: str = "",
        video_number: int = 0,
    ) -> MetadataResult:
        """
        Generate title, description, hashtags, and cta_product for a YouTube Shorts video.

        Args:
            topic: The video topic or keyword.
            script: The full script text (used for SEO context).
            cta_product: Name of the free product offered in the CTA.
            category: Channel category slug (e.g. "money"). When provided,
                      the description is augmented with a Gumroad link block
                      and required hashtags are force-inserted.

        Returns:
            MetadataResult with title, description, hashtags, cta_product,
            and validation status.

        Raises:
            ValueError: If ANTHROPIC_API_KEY is not configured.
        """
        client = self._get_client()
        prompt = self._build_prompt(topic, script, cta_product)

        logger.info("Generating metadata for topic='%s'", topic)

        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
        logger.debug("Raw metadata response: %s", raw)

        parsed = self._parse_response(raw)
        result = self._build_result(topic, parsed, raw)

        # Post-process with product info when category is known
        if category:
            result = self._apply_product_overrides(result, category, video_number)

        logger.info(
            "Metadata generated: title=%d chars, desc=%d chars, tags=%d, valid=%s",
            len(result.title),
            len(result.description),
            len(result.hashtags),
            result.is_valid,
        )
        if result.validation_errors:
            logger.warning("Validation issues: %s", result.validation_errors)

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            if not self.api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def _build_prompt(self, topic: str, script: str, cta_product: str = "") -> str:
        # Truncate script to first 300 chars to stay within context budget
        script_preview = script[:300].strip()
        lines = [
            f"Topic: {topic}",
            f"\nScript:\n{script_preview}",
        ]
        if cta_product:
            lines.append(f"\nFree product for CTA: {cta_product}")
        lines.append("\nGenerate the YouTube metadata now.")
        return "\n".join(lines)

    def _parse_response(self, raw: str) -> dict[str, Any]:
        """Parse Claude JSON response; return safe empty dict on failure."""
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
            return {
                "title":       str(data.get("title", "")).strip(),
                "description": str(data.get("description", "")).strip(),
                "hashtags":    [
                    str(h).strip()
                    for h in data.get("hashtags", [])
                    if str(h).strip()
                ],
                "cta_product": str(data.get("cta_product", "")).strip(),
            }
        except Exception as exc:
            logger.error(
                "Failed to parse metadata JSON: %s | raw=%s", exc, raw[:200]
            )
            return {"title": "", "description": "", "hashtags": [], "cta_product": ""}

    def _build_result(
        self, topic: str, parsed: dict[str, Any], raw: str
    ) -> MetadataResult:
        """Assemble MetadataResult and run all validation checks."""
        title       = parsed.get("title", "")
        description = parsed.get("description", "")
        hashtags    = parsed.get("hashtags", [])
        cta_product = parsed.get("cta_product", "")

        errors = self._validate(title, description, hashtags)

        return MetadataResult(
            topic=topic,
            title=title,
            description=description,
            hashtags=hashtags,
            cta_product=cta_product,
            is_valid=len(errors) == 0,
            validation_errors=errors,
            raw_response=raw,
        )

    # Required tags forced into every video when a category is known
    _REQUIRED_TAGS: list[str] = [
        "#Shorts", "#personalfinance", "#financialfreedom",
        "#wealthbuilding", "#moneyadvice", "#moneytips", "#financetok",
    ]

    _UPSELL_LINE: str = "Want the whole system built for you? Comment SYSTEM on any video."

    def _apply_product_overrides(
        self,
        result: "MetadataResult",
        category: str,
        video_number: int = 0,
    ) -> "MetadataResult":
        """
        Augment description with Gumroad block and ensure required hashtags are
        present.  Only called when a channel category is provided.

        On every 5th video (video_number > 0 and video_number % 5 == 0) a soft
        upsell line is appended before the follow line.
        """
        from config.constants import PRODUCTS
        product = PRODUCTS.get(category, {})

        # --- Description: append Gumroad block ---
        if product:
            short_name  = product.get("short_name", "")
            gumroad_url = product.get("gumroad_url", "")
            desc = result.description
            # Strip trailing suffix so we can re-append after the Gumroad block
            if desc.endswith(DESCRIPTION_SUFFIX):
                desc = desc[: -len(DESCRIPTION_SUFFIX)].rstrip()

            upsell_block = ""
            if video_number > 0 and video_number % 5 == 0:
                upsell_block = f"{self._UPSELL_LINE}\n\n"
                logger.info("Soft upsell injected for video_number=%d", video_number)

            result.description = (
                f"{desc}\n\n"
                f"FREE RESOURCE: {short_name}\n"
                f"Download here: {gumroad_url}\n\n"
                f"{upsell_block}"
                f"Follow @moneyheresy for weekly wealth truths. {DESCRIPTION_SUFFIX}"
            )
            result.cta_product = short_name

        # --- Hashtags: force-insert required tags, keep total at 15 ---
        required_lower = {t.lower() for t in self._REQUIRED_TAGS}
        existing = list(result.hashtags)
        non_required = [t for t in existing if t.lower() not in required_lower]
        merged = list(self._REQUIRED_TAGS) + non_required
        # Deduplicate while preserving order (required tags first)
        seen: set[str] = set()
        deduped: list[str] = []
        for tag in merged:
            key = tag.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(tag)
        # Trim or pad to exactly 15
        result.hashtags = deduped[:15]
        if len(result.hashtags) < 15:
            # Pad with generic tags if Claude was stingy
            for extra in ("#motivation", "#success", "#mindset",
                          "#goals", "#hustle", "#money", "#wealth",
                          "#finance"):
                if len(result.hashtags) >= 15:
                    break
                if extra.lower() not in {t.lower() for t in result.hashtags}:
                    result.hashtags.append(extra)

        return result

    @staticmethod
    def _validate(
        title: str,
        description: str,
        hashtags: list[str],
    ) -> list[str]:
        """
        Validate all three metadata fields.
        Returns a list of error strings (empty = valid).
        """
        errors: list[str] = []

        # --- Title ---
        if not title.strip():
            errors.append("title is empty")
        elif len(title) > TITLE_MAX_CHARS:
            errors.append(
                f"title is {len(title)} chars, exceeds limit of {TITLE_MAX_CHARS}"
            )

        # --- Description ---
        if not description.strip():
            errors.append("description is empty")
        else:
            if len(description) > DESCRIPTION_MAX_CHARS:
                errors.append(
                    f"description is {len(description)} chars, "
                    f"exceeds limit of {DESCRIPTION_MAX_CHARS}"
                )
            if not description.rstrip().endswith(DESCRIPTION_SUFFIX):
                errors.append(
                    f"description must end with \"{DESCRIPTION_SUFFIX}\""
                )

        # --- Hashtags ---
        if len(hashtags) != REQUIRED_HASHTAG_COUNT:
            errors.append(
                f"expected {REQUIRED_HASHTAG_COUNT} hashtags, got {len(hashtags)}"
            )

        # All hashtags must start with #
        bad_format = [h for h in hashtags if not h.startswith("#")]
        if bad_format:
            errors.append(f"hashtags missing # prefix: {bad_format[:3]}")

        # Must include #Shorts
        if REQUIRED_HASHTAG not in hashtags:
            errors.append(f"{REQUIRED_HASHTAG} not in hashtags")

        # No spaces inside hashtags
        spaced = [h for h in hashtags if " " in h]
        if spaced:
            errors.append(f"hashtags contain spaces: {spaced[:3]}")

        return errors

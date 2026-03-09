"""
script_generator.py — ScriptGenerator

Generates a 15-second YouTube Shorts script with exactly 4 timed parts,
then validates word count and ending punctuation.

Script structure:
  Hook       (2s)  — the attention-grabbing opener (from HookGenerator)
  Statement  (4s)  — the core claim or insight
  Twist      (4s)  — the unexpected angle or subversion
  Question   (3s)  — the closing question that drives comments

Total: ~13s of speech + 2s buffer = 15s.
Word count must stay under 75 words.
Script must end with a question mark.

Usage:
    gen = ScriptGenerator()
    result = gen.generate(topic="stoic quotes", hook="Most people ignore this...")
    print(result.full_script)
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

_SYSTEM_PROMPT = """You are a YouTube Shorts scriptwriter who specialises in viral
15-second videos. You write tight, conversational scripts — no filler, no fluff.

Given a topic, an opening hook, and an exact CTA line, generate a 4-part script:
  1. hook      (~2s, 8–12 words)  — use the provided hook verbatim
  2. statement (~4s, 12–16 words) — bold claim or surprising insight
  3. twist     (~4s, 12–16 words) — the unexpected flip or deeper truth
  4. question  (~3s)              — use the EXACT CTA text provided, word for word.
                                    Do NOT rephrase or improvise the CTA.

Rules:
- hook + statement + twist MUST total 40 words or fewer
- Total word count across all 4 parts MUST be under 75 words
- Write in second person ("you", "your") where natural
- No hashtags, no emojis, no stage directions

Respond ONLY with a JSON object, no markdown:
{
  "hook": "...",
  "statement": "...",
  "twist": "...",
  "question": "..."
}"""


# ---------------------------------------------------------------------------
# Validation constants
# ---------------------------------------------------------------------------

MAX_WORDS = 75
REQUIRED_PARTS = ("hook", "statement", "twist", "question")

PART_WORD_LIMITS = {
    "hook":      (8, 15),
    "statement": (12, 25),
    "twist":     (12, 25),
    "question":  (6, 15),
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ScriptPart:
    """A single timed segment of the script."""

    name: str
    text: str
    duration_seconds: int

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass
class ScriptResult:
    """
    The validated 4-part script returned by ScriptGenerator.generate().
    """

    topic: str
    hook: str
    statement: str
    twist: str
    question: str
    full_script: str             # newline-joined concatenation of all 4 parts
    word_count: int
    is_valid: bool               # True only if all validation rules pass
    validation_errors: list[str] = field(default_factory=list)
    generated_at: str = ""
    raw_response: str = ""

    def __post_init__(self) -> None:
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "hook": self.hook,
            "statement": self.statement,
            "twist": self.twist,
            "question": self.question,
            "full_script": self.full_script,
            "word_count": self.word_count,
            "is_valid": self.is_valid,
            "validation_errors": self.validation_errors,
            "generated_at": self.generated_at,
        }

    @property
    def parts(self) -> list[ScriptPart]:
        """Return all 4 parts as ordered ScriptPart objects."""
        return [
            ScriptPart("hook",      self.hook,      2),
            ScriptPart("statement", self.statement, 4),
            ScriptPart("twist",     self.twist,     4),
            ScriptPart("question",  self.question,  3),
        ]


# ---------------------------------------------------------------------------
# ScriptGenerator
# ---------------------------------------------------------------------------

class ScriptGenerator:
    """
    Generates and validates a 15-second YouTube Shorts script.

    Args:
        api_key: Anthropic API key. If None, reads ANTHROPIC_API_KEY from env.
        model: Claude model ID.
        max_tokens: Max tokens for the API response.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = _MODEL,
        max_tokens: int = 400,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("ANTHROPIC_API_KEY", "")
        self.model = model
        self.max_tokens = max_tokens
        self._client: anthropic.Anthropic | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, topic: str, hook: str, cta_script: str = "") -> ScriptResult:
        """
        Generate a 4-part 15-second script for the given topic and hook.

        Args:
            topic: The video topic or keyword.
            hook: The opening hook line (from HookGenerator or custom).
            cta_script: Exact CTA sentence(s) to use verbatim as the question
                        part (e.g. from PRODUCTS[category]["cta_script"]).
                        If empty, Claude improvises the question freely.

        Returns:
            ScriptResult with all parts, full_script, word_count, and
            validation status.

        Raises:
            ValueError: If ANTHROPIC_API_KEY is not configured.
        """
        client = self._get_client()
        prompt = self._build_prompt(topic, hook, cta_script)

        logger.info("Generating script for topic='%s'", topic)

        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
        logger.debug("Raw script response: %s", raw)

        parts = self._parse_parts(raw)
        result = self._build_result(topic, parts, raw)

        logger.info(
            "Script generated: %d words, valid=%s",
            result.word_count, result.is_valid,
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

    def _build_prompt(self, topic: str, hook: str, cta_script: str = "") -> str:
        lines = [
            f"Topic: {topic}",
            f"Hook (use verbatim): {hook}",
        ]
        if cta_script:
            lines.append(
                f"Exact CTA for question (copy word-for-word into the question field): {cta_script}"
            )
        lines.append("\nWrite the 4-part script now.")
        return "\n".join(lines)

    def _parse_parts(self, raw: str) -> dict[str, str]:
        """
        Parse Claude's JSON response into a {part_name: text} dict.

        Returns a safe fallback dict on any parse failure.
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
            return {part: str(data.get(part, "")).strip() for part in REQUIRED_PARTS}
        except Exception as exc:
            logger.error("Failed to parse script JSON: %s | raw=%s", exc, raw[:200])
            return {part: "" for part in REQUIRED_PARTS}

    def _build_result(
        self, topic: str, parts: dict[str, str], raw: str
    ) -> ScriptResult:
        """Assemble ScriptResult and run all validation checks."""
        hook      = parts.get("hook", "")
        statement = parts.get("statement", "")
        twist     = parts.get("twist", "")
        question  = parts.get("question", "")

        full_script = "\n".join(filter(None, [hook, statement, twist, question]))
        word_count  = len(full_script.split()) if full_script.strip() else 0

        errors = self._validate(hook, statement, twist, question, word_count)

        return ScriptResult(
            topic=topic,
            hook=hook,
            statement=statement,
            twist=twist,
            question=question,
            full_script=full_script,
            word_count=word_count,
            is_valid=len(errors) == 0,
            validation_errors=errors,
            raw_response=raw,
        )

    @staticmethod
    def _validate(
        hook: str,
        statement: str,
        twist: str,
        question: str,
        word_count: int,
    ) -> list[str]:
        """
        Run all validation rules.  Returns a list of error strings;
        empty list means the script is fully valid.
        """
        errors: list[str] = []

        # All parts must be non-empty
        for name, text in [
            ("hook", hook),
            ("statement", statement),
            ("twist", twist),
            ("question", question),
        ]:
            if not text.strip():
                errors.append(f"{name} is empty")

        # Total word count
        if word_count >= MAX_WORDS:
            errors.append(
                f"word_count={word_count} exceeds limit of {MAX_WORDS - 1}"
            )

        # Script must end with a question mark
        full = "\n".join(filter(None, [hook, statement, twist, question]))
        if full and not full.rstrip().endswith("?"):
            errors.append("script does not end with a question mark")

        # question part specifically must end with '?'
        if question.strip() and not question.rstrip().endswith("?"):
            errors.append("question part does not end with a question mark")

        return errors

"""
script_generator.py — ScriptGenerator

Generates a 45-50 second YouTube Shorts script with 5 timed parts,
following the Palki Sharma narrative formula:
  Hook       — Provocative opening
  Statement  — Analogy/micro-story + fact/data point
  Twist      — Escalation + reframe
  Landing    — Pause beat / emotional punchline
  Question   — Direct question to viewer + CTA

Target: 80–100 words for natural 45–50 second delivery.
Script must contain at least one question mark.

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

_SYSTEM_PROMPT = """You are a YouTube Shorts scriptwriter in the style of Palki Sharma.
You write punchy, conversational 45-50 second scripts for faceless finance channels.
No fluff, no filler. Every sentence earns its place.

TARGET AUDIENCE: United States (primary), United Kingdom, Canada, Australia (secondary).

NEVER reference: India, rupees, Indian statistics, Asian or African markets,
chai, cricket, Bollywood.

CURRENCY RULES — critical for voiceover naturalness:
- Never use symbols: $, £, €, ₹
- Never use abbreviations: USD, GBP, Rs, INR
- Always write amounts in full words: "one hundred US dollars" not "$100",
  "sixty thousand US dollars per year" not "$60k"

STATISTICS RULES:
- Only use US, UK, Canadian or Australian data
- Cite: Federal Reserve, US Bureau of Labor Statistics, Bank of England
- "Average salary" = approximately 60,000 US dollars per year (US context)

CULTURAL CONTEXT:
- Use Western references: 401k, mortgage, Wall Street, Silicon Valley, pension, NHS
- Relatable scenarios: "working a 9-to-5 in New York", "paying rent in London",
  "saving for a house deposit in Sydney"

EXAMPLE:
BAD:  "In India, 78% of salaried professionals live paycheck to paycheck.
       Every rupee you earn..."
GOOD: "In America, 78 percent of full-time workers live paycheck to paycheck.
       Every US dollar you earn gets taxed first..."

Given a topic, an opening hook, and an exact CTA line, write a 5-part script that
follows this narrative arc:

PART 1 — hook (10–15 words)
Drop the viewer into tension immediately. State the uncomfortable truth bluntly.
Use the provided hook verbatim.

PART 2 — statement (22–30 words)
Pack in beats 2 and 3:
  Beat 2: A specific, relatable micro-scenario the viewer can picture themselves in.
           Not a statistic — a situation. (2 sentences)
  Beat 3: One sharp, credible number or fact that validates the tension. No fluff. (1 sentence)

PART 3 — twist (18–25 words)
Pack in beats 4 and 5:
  Beat 4: Deepen the problem. Show it is worse than the viewer realised. (1–2 sentences)
  Beat 5: The reframe — the insight moment that makes them feel they just learned something. (1 sentence)

PART 4 — landing (8–12 words)
A short pause beat that lands the emotional point. This is the punchline moment —
a sharp, declarative sentence that lets the insight sink in before the CTA.
Not a question. Not filler. A gut-punch that makes the viewer nod.
Example: "That's not an accident. It's by design."

PART 5 — question (15–20 words)
  One personal, direct question that pulls the viewer in. (1 sentence, ends with ?)
  Then the exact CTA text provided, word for word. Do NOT rephrase or improvise.
  The CTA is the final sentence of the entire script. Nothing after it.

WRITING RULES — follow these strictly:
- Total word count across all 5 parts: 80–95 words. HARD MAXIMUM: 100 words total.
- BEFORE responding, count every word in your script. If it exceeds 100, cut until it does not.
- Never use em dashes (—) in the script body
- Never use "it is worth noting", "in conclusion", "furthermore", "moreover"
- Write as if talking to one specific person
- Short sentences. Punchy. Direct.
- No sentence longer than 20 words
- Use "..." for a natural pause between thoughts
- Use "!" for genuine emphasis only — maximum once per script
- For shocked/surprised beats, write a very short fragment: "Wait. What?"
- For tension build, use progressively shorter sentences
- No hashtags, no emojis, no stage directions

Respond ONLY with a JSON object, no markdown:
{
  "hook": "...",
  "statement": "...",
  "twist": "...",
  "landing": "...",
  "question": "..."
}"""


# ---------------------------------------------------------------------------
# Validation constants
# ---------------------------------------------------------------------------

MAX_WORDS = 106   # hard ceiling; scripts of ≥106 words fail validation
RETRY_WORD_LIMIT = 100  # trigger one brevity retry if word_count > this
REQUIRED_PARTS = ("hook", "statement", "twist", "landing", "question")

PART_WORD_LIMITS = {
    "hook":      (10, 15),
    "statement": (22, 30),
    "twist":     (18, 25),
    "landing":   (8, 12),
    "question":  (15, 20),
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
    The validated 5-part script returned by ScriptGenerator.generate().
    """

    topic: str
    hook: str
    statement: str
    twist: str
    landing: str
    question: str
    full_script: str             # newline-joined concatenation of all 5 parts
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
            "landing": self.landing,
            "question": self.question,
            "full_script": self.full_script,
            "word_count": self.word_count,
            "is_valid": self.is_valid,
            "validation_errors": self.validation_errors,
            "generated_at": self.generated_at,
        }

    @property
    def parts(self) -> list[ScriptPart]:
        """Return all 5 parts as ordered ScriptPart objects."""
        return [
            ScriptPart("hook",      self.hook,      2),
            ScriptPart("statement", self.statement, 4),
            ScriptPart("twist",     self.twist,     3),
            ScriptPart("landing",   self.landing,   2),
            ScriptPart("question",  self.question,  2),
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
        max_tokens: int = 800,
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

        # Word count retry: if too long, retry once with a tighter prompt
        if self._parts_too_long(parts):
            parts, raw = self._retry_for_brevity(topic, hook, parts, raw, client)

        # CTA enforcement: ensure question field contains the verbatim CTA text
        if cta_script:
            parts, cta_path = self._enforce_cta(parts, cta_script, topic, hook, client)
            logger.info("CTA enforcement: %s", cta_path)

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
        lines.append("\nWrite the 5-part script now.")
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

    @staticmethod
    def _cta_matches(question: str, cta_script: str) -> bool:
        """Return True if the question field contains the expected CTA.

        Two CTA formats are valid:
          1. Subscribe CTA — question must contain 'subscribe' and 'daily'
          2. Lead-magnet CTA — question must contain the trigger keyword
             (SYSTEM, AUTOMATE, or BLUEPRINT)

        Falls back to verbatim check for unrecognised CTA formats.
        """
        q_lower = question.strip().lower()
        cta_lower = cta_script.strip().lower()

        if not q_lower:
            return False

        # Subscribe-only CTA: just needs 'subscribe' and 'daily'
        if "subscribe" in cta_lower and "comment" not in cta_lower:
            return "subscribe" in q_lower and ("bank" in q_lower or "hear" in q_lower)

        # Lead-magnet CTA: must contain the trigger keyword
        trigger_keywords = ["system", "automate", "blueprint"]
        for kw in trigger_keywords:
            if kw in cta_lower:
                return kw in q_lower

        # Fallback: verbatim check for unrecognised CTA formats
        return cta_lower in q_lower

    def _enforce_cta(
        self,
        parts: dict[str, str],
        cta_script: str,
        topic: str,
        hook: str,
        client: "anthropic.Anthropic",
    ) -> tuple[dict[str, str], str]:
        """
        Ensure the question field contains the verbatim CTA text.

        Three outcomes (logged by caller):
          "CTA matched verbatim -- OK"
          "CTA drift detected -- regenerated"
          "CTA drift -- force replaced"

        Returns (parts, path_label).
        """
        if self._cta_matches(parts.get("question", ""), cta_script):
            return parts, "CTA matched verbatim -- OK"

        logger.warning(
            "CTA drift detected -- regenerating script. "
            "Expected CTA: %r | Got question: %r",
            cta_script, parts.get("question", ""),
        )

        # Retry with a stronger correction prompt
        correction_prompt = (
            f"Topic: {topic}\n"
            f"Hook (use verbatim): {hook}\n"
            f"Your previous response did not use the exact CTA provided. "
            f"You MUST copy this text word for word into the question field: {cta_script}\n"
            f"\nWrite the 5-part script now."
        )
        try:
            message = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": correction_prompt}],
            )
            raw2 = message.content[0].text.strip()
            parts2 = self._parse_parts(raw2)

            if self._cta_matches(parts2.get("question", ""), cta_script):
                return parts2, "CTA drift detected -- regenerated"

            logger.warning(
                "CTA drift persists after retry -- force replacing question field. "
                "Got: %r", parts2.get("question", ""),
            )
        except Exception as exc:
            logger.error("CTA retry API call failed: %s -- force replacing", exc)

        # Force replace: keep hook/statement/twist from original, inject CTA verbatim
        parts["question"] = cta_script
        return parts, "CTA drift -- force replaced"

    def _parts_too_long(self, parts: dict[str, str]) -> bool:
        """Return True if combined word count of all parts exceeds RETRY_WORD_LIMIT."""
        all_text = " ".join(parts.get(p, "") for p in REQUIRED_PARTS)
        word_count = len(all_text.split()) if all_text.strip() else 0
        has_content = all(parts.get(p, "").strip() for p in REQUIRED_PARTS)
        return has_content and word_count > RETRY_WORD_LIMIT

    def _retry_for_brevity(
        self,
        topic: str,
        hook: str,
        parts: dict[str, str],
        raw: str,
        client: "anthropic.Anthropic",
    ) -> tuple[dict[str, str], str]:
        """
        Retry the script generation with a tighter word-count instruction.

        Called once when the initial response exceeds RETRY_WORD_LIMIT words.
        Returns the retry parts + raw on success, or the original on failure.
        """
        word_count = len(
            " ".join(parts.get(p, "") for p in REQUIRED_PARTS).split()
        )
        logger.warning(
            "Script too long (%d words) — retrying with 95-word cap", word_count
        )
        retry_prompt = (
            f"Topic: {topic}\n"
            f"Hook (use verbatim): {hook}\n"
            f"Your previous script was {word_count} words — way over the 100-word hard max. "
            f"Rewrite it in EXACTLY 85 words or fewer. Cut filler, shorten sentences, "
            f"remove adjectives. Every word must earn its place.\n"
            f"\nWrite the 5-part script now."
        )
        try:
            message = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": retry_prompt}],
            )
            raw2 = message.content[0].text.strip()
            parts2 = self._parse_parts(raw2)
            has_content = all(parts2.get(p, "").strip() for p in REQUIRED_PARTS)
            if has_content:
                new_count = len(
                    " ".join(parts2.get(p, "") for p in REQUIRED_PARTS).split()
                )
                logger.info(
                    "Word count retry: %d → %d words", word_count, new_count
                )
                # Only accept retry if it actually reduced below MAX_WORDS
                if new_count < MAX_WORDS:
                    return parts2, raw2
                logger.warning(
                    "Retry still over limit (%d words) — keeping original",
                    new_count,
                )
        except Exception as exc:
            logger.error("Word count retry API call failed: %s", exc)
        return parts, raw

    def _build_result(
        self, topic: str, parts: dict[str, str], raw: str
    ) -> ScriptResult:
        """Assemble ScriptResult and run all validation checks."""
        hook      = parts.get("hook", "")
        statement = parts.get("statement", "")
        twist     = parts.get("twist", "")
        landing   = parts.get("landing", "")
        question  = parts.get("question", "")

        full_script = "\n".join(filter(None, [hook, statement, twist, landing, question]))
        word_count  = len(full_script.split()) if full_script.strip() else 0

        errors = self._validate(hook, statement, twist, landing, question, word_count)

        return ScriptResult(
            topic=topic,
            hook=hook,
            statement=statement,
            twist=twist,
            landing=landing,
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
        landing: str,
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
            ("landing", landing),
            ("question", question),
        ]:
            if not text.strip():
                errors.append(f"{name} is empty")

        # Total word count
        if word_count >= MAX_WORDS:
            errors.append(
                f"word_count={word_count} exceeds limit of {MAX_WORDS - 1}"
            )

        # Script must contain at least one question mark
        full = "\n".join(filter(None, [hook, statement, twist, landing, question]))
        if full and "?" not in full:
            errors.append("script does not contain a question mark")

        # question part must contain a question mark
        if question.strip() and "?" not in question:
            errors.append("question part does not contain a question mark")

        return errors

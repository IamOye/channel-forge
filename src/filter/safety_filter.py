"""
safety_filter.py — SafetyFilter

Two-stage keyword safety classification:
  Stage 1: rule-based blocklist (fast, free, offline)
  Stage 2: Claude API classifier (nuanced, used only if stage 1 passes)

Usage:
    filt = SafetyFilter()
    result = filt.check("make money fast")
    print(result.is_safe, result.reason)
"""

import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone

import anthropic
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SafetyResult:
    """Result of a safety classification."""

    keyword: str
    is_safe: bool
    reason: str | None          # why it was blocked, or None if safe
    method: str                 # 'blocklist' | 'claude_api' | 'allowed'
    confidence: float = 1.0
    checked_at: str = ""

    def __post_init__(self) -> None:
        if not self.checked_at:
            self.checked_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "keyword": self.keyword,
            "is_safe": int(self.is_safe),
            "block_reason": self.reason,
            "method": self.method,
            "confidence": self.confidence,
            "checked_at": self.checked_at,
        }


# ---------------------------------------------------------------------------
# Blocklist
# ---------------------------------------------------------------------------

# Rule-based blocklist: exact tokens and regex patterns.
# Keep this conservative — only clear-cut content-policy violations.
_BLOCKED_TOKENS: frozenset[str] = frozenset({
    # Violence / harm
    "kill", "murder", "assault", "torture", "suicide", "self-harm",
    "bomb", "explosive", "terrorism", "terrorist",
    # Adult / explicit
    "porn", "pornography", "xxx", "nude", "naked", "sex tape",
    "onlyfans leak", "hentai",
    # Hate speech
    "nazi", "white supremacy", "racial slur",
    # Drugs (black-market context)
    "buy cocaine", "buy heroin", "buy meth", "drug trafficking",
    # Illegal
    "child abuse", "csam", "cp ", "piracy site", "crack software",
    "hack account", "steal identity",
    # Scams
    "ponzi scheme", "pyramid scheme",
})

# Regex patterns (case-insensitive) for more flexible matching
_BLOCKED_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(how\s+to\s+(kill|murder|harm|hurt)\s+\w+)\b", re.IGNORECASE),
    re.compile(r"\b(free\s+(porn|sex|xxx|nude))\b", re.IGNORECASE),
    re.compile(r"\b(buy\s+(drugs?|weed|cocaine|heroin|meth))\b", re.IGNORECASE),
    re.compile(r"\b(hack\s+(into|someone|password|account))\b", re.IGNORECASE),
    re.compile(r"\b(crack\s+(serial|keygen|license))\b", re.IGNORECASE),
    re.compile(r"\b(child\s+(porn|sex|abuse|exploitation))\b", re.IGNORECASE),
]


def _blocklist_check(keyword: str) -> tuple[bool, str | None]:
    """
    Stage-1: token and regex blocklist check.

    Returns:
        (is_blocked, reason) — is_blocked=True means the keyword is NOT safe.
    """
    lower = keyword.lower()

    # Token check
    for token in _BLOCKED_TOKENS:
        if token in lower:
            return True, f"blocked token: '{token}'"

    # Regex check
    for pattern in _BLOCKED_PATTERNS:
        match = pattern.search(lower)
        if match:
            return True, f"blocked pattern: '{match.group(0)}'"

    return False, None


# ---------------------------------------------------------------------------
# Claude API classifier
# ---------------------------------------------------------------------------

_CLAUDE_MODEL = os.getenv("CLAUDE_SAFETY_MODEL", "claude-haiku-4-5-20251001")

_SYSTEM_PROMPT = """You are a YouTube content safety classifier for ChannelForge.
Your job is to decide whether a keyword or topic is safe for use in a
family-friendly, advertiser-friendly YouTube channel.

Classify each keyword as SAFE or UNSAFE.

Respond ONLY with a JSON object in this exact format:
{"safe": true, "confidence": 0.95, "reason": "brief reason if unsafe"}

Guidelines:
- SAFE: educational, informational, how-to, entertainment, pop-culture, business,
  technology, health (general), finance (legal), lifestyle topics.
- UNSAFE: violence, explicit sexual content, hate speech, illegal activities,
  dangerous/harmful instructions, scams, misinformation promotion.
- When in doubt about borderline topics, lean SAFE if the content could be
  presented responsibly on a mainstream YouTube channel.
"""


class ClaudeSafetyClassifier:
    """Uses Claude API to classify borderline keywords."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = _CLAUDE_MODEL,
        max_tokens: int = 100,
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

    def classify(self, keyword: str) -> SafetyResult:
        """
        Call Claude to classify a keyword.

        Returns:
            SafetyResult with method='claude_api'.
        """
        import json

        prompt = f'Classify this YouTube content keyword: "{keyword}"'
        try:
            client = self._get_client()
            message = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text = message.content[0].text.strip()

            # Strip markdown code fences if present
            if raw_text.startswith("```"):
                raw_text = raw_text.strip("`").strip()
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:].strip()

            data = json.loads(raw_text)
            is_safe = bool(data.get("safe", True))
            confidence = float(data.get("confidence", 0.8))
            reason = data.get("reason") if not is_safe else None

            logger.debug(
                "Claude classified '%s' as %s (confidence=%.2f)",
                keyword, "SAFE" if is_safe else "UNSAFE", confidence,
            )
            return SafetyResult(
                keyword=keyword,
                is_safe=is_safe,
                reason=reason,
                method="claude_api",
                confidence=confidence,
            )

        except Exception as exc:
            logger.error("Claude safety classification failed for '%s': %s", keyword, exc)
            # Fail open (assume safe) on API errors to not block the pipeline
            return SafetyResult(
                keyword=keyword,
                is_safe=True,
                reason=None,
                method="claude_api",
                confidence=0.5,
            )


# ---------------------------------------------------------------------------
# Main SafetyFilter
# ---------------------------------------------------------------------------

class SafetyFilter:
    """
    Two-stage safety filter for YouTube content keywords.

    Stage 1 — Blocklist: instant, no API calls, catches obvious violations.
    Stage 2 — Claude API: nuanced classification for anything that passes stage 1.

    Args:
        use_claude: If True, run Claude API on keywords that pass the blocklist.
                    Set False in tests or when ANTHROPIC_API_KEY is not available.
        claude_api_key: Override for the Anthropic API key.
        extra_blocked_tokens: Additional tokens to add to the blocklist.
    """

    def __init__(
        self,
        use_claude: bool = True,
        claude_api_key: str | None = None,
        extra_blocked_tokens: list[str] | None = None,
    ) -> None:
        self.use_claude = use_claude
        self._extra_tokens: frozenset[str] = frozenset(
            t.lower() for t in (extra_blocked_tokens or [])
        )
        self._claude = ClaudeSafetyClassifier(api_key=claude_api_key) if use_claude else None

    def check(self, keyword: str) -> SafetyResult:
        """
        Run the full two-stage safety check on a single keyword.

        Args:
            keyword: The keyword/topic string to check.

        Returns:
            SafetyResult with is_safe, reason, method, confidence.
        """
        if not keyword or not keyword.strip():
            return SafetyResult(
                keyword=keyword,
                is_safe=False,
                reason="empty keyword",
                method="blocklist",
                confidence=1.0,
            )

        # Stage 1: blocklist
        lower = keyword.lower()
        for token in self._extra_tokens:
            if token in lower:
                logger.info("Blocked by extra token '%s': %s", token, keyword)
                return SafetyResult(
                    keyword=keyword,
                    is_safe=False,
                    reason=f"blocked token: '{token}'",
                    method="blocklist",
                    confidence=1.0,
                )

        is_blocked, reason = _blocklist_check(keyword)
        if is_blocked:
            logger.info("Blocked by blocklist: '%s' — %s", keyword, reason)
            return SafetyResult(
                keyword=keyword,
                is_safe=False,
                reason=reason,
                method="blocklist",
                confidence=1.0,
            )

        # Stage 2: Claude API (optional)
        if self.use_claude and self._claude:
            logger.debug("Sending to Claude classifier: '%s'", keyword)
            return self._claude.classify(keyword)

        # Passed all checks
        return SafetyResult(
            keyword=keyword,
            is_safe=True,
            reason=None,
            method="allowed",
            confidence=1.0,
        )

    def check_batch(self, keywords: list[str]) -> list[SafetyResult]:
        """
        Check a batch of keywords.

        Args:
            keywords: List of keyword strings.

        Returns:
            List of SafetyResult, one per keyword, in the same order.
        """
        results: list[SafetyResult] = []
        for kw in keywords:
            results.append(self.check(kw))
        return results

    def filter_safe(self, keywords: list[str]) -> list[str]:
        """
        Return only the keywords that pass the safety filter.

        Args:
            keywords: List of keyword strings.

        Returns:
            Filtered list containing only safe keywords.
        """
        results = self.check_batch(keywords)
        safe = [r.keyword for r in results if r.is_safe]
        blocked = len(keywords) - len(safe)
        if blocked:
            logger.info("SafetyFilter blocked %d/%d keywords", blocked, len(keywords))
        return safe

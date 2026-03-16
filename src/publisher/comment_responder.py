"""
comment_responder.py — CommentResponder

Generates personalised YouTube comment replies via Claude API.

Each reply is tailored to the specific comment, commenter name,
video category, and trigger type.  Replies are capped at 500 characters
(YouTube's reply limit) and truncated at the last complete sentence if
Claude returns something longer.

Trigger types
-------------
  YES          — commenter typed the CTA keyword (YES, SALARY, FREE, etc.)
  COMPLIMENT   — positive comment or personal story
  NEGATIVE     — critical, dismissive, or hostile comment
  QUESTION     — genuine question about money / investing
  HOT_LEAD     — mentions SYSTEM, done-for-you, or wants it built for them
  GENERAL      — anything else

Usage
-----
    responder = CommentResponder()
    reply = responder.generate_reply(
        comment_text="Yes",
        commenter_name="Jordan",
        category="money",
        trigger_type="YES",
    )
    print(reply.text)
"""

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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_REPLY_CHARS: int = 500

TRIGGER_YES        = "YES"
TRIGGER_COMPLIMENT = "COMPLIMENT"
TRIGGER_NEGATIVE   = "NEGATIVE"
TRIGGER_QUESTION   = "QUESTION"
TRIGGER_HOT_LEAD   = "HOT_LEAD"
TRIGGER_GENERAL    = "GENERAL"

VALID_TRIGGERS = {
    TRIGGER_YES,
    TRIGGER_COMPLIMENT,
    TRIGGER_NEGATIVE,
    TRIGGER_QUESTION,
    TRIGGER_HOT_LEAD,
    TRIGGER_GENERAL,
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You write YouTube comment replies for the channel Money Heresy, a finance \
channel that exposes uncomfortable truths about money and wealth.

Your replies must feel like they came from a real, warm, intelligent person \
who genuinely read the comment, not a bot.

TONE RULES:
- Warm, genuine, and conversational
- Never robotic or corporate
- Short, 2 to 4 sentences maximum
- Sound like a knowledgeable friend, not a financial institution
- Never preachy or condescending

FORMATTING RULES:
- Never use em dashes
- Never use bullet points in replies
- Never use hashtags in replies
- Write in plain conversational sentences
- Never start with "Great comment!" or "Thanks for sharing!" as these are \
generic and feel fake

CONTENT RULES:

If the comment is a YES trigger:
- Acknowledge what they commented briefly
- Send the relevant Gumroad link naturally
- Add one warm sentence of encouragement
- Keep it under 4 sentences total

If the comment is a compliment or positive story:
- Read the actual comment carefully
- Respond to the specific detail they shared
- Accept the compliment genuinely without being sycophantic
- Add one sentence that connects their experience to a broader truth
- Never say "you are so right" or "absolutely" as these feel hollow

If the comment is negative, critical or dismissive:
- Never respond defensively
- Never match their energy
- Smile through it and respond with warmth
- Find something genuine to acknowledge even in a negative comment
- If completely baseless, simply thank them for watching and wish them well
- Never delete or dismiss their view

If the comment is a question:
- Answer it directly and specifically
- If the answer is in one of the guides, mention the guide naturally
- Keep it conversational

If the comment mentions SYSTEM or done-for-you:
- Respond warmly and personally
- Express genuine interest in their situation
- Ask one specific open question such as "What kind of income stream are you \
trying to build?" or "What is the main thing holding you back right now?"
- Never mention price or package tiers
- Feel like the beginning of a real conversation

PERSONALIZATION:
- Use the commenter's name if available
- Reference specific words or details from their comment to show you read it
- Every reply should feel like it could only have been written for that \
specific comment

EXAMPLES:

EXAMPLE 1 — Compliment with personal story:
Comment: "When my husband got a raise he saved more money instead of spending \
more money. We lived below our means so we could buy a nice little house while \
our children were very young."
Reply: "That is the move right there. Most people treat a raise as permission \
to spend more and the paycheck-to-paycheck cycle just resets at a higher \
number. You and your husband broke that pattern early and bought an asset \
while others were upgrading their lifestyle. Your kids benefited from a \
financial decision you made before they were old enough to understand it. \
That is real generational thinking."

EXAMPLE 2 — YES trigger:
Comment: "Yes"
Reply: "Here you go! Your free Wealth Systems Blueprint is ready at \
shextroll.gumroad.com/l/wealth-systems-blueprint No sign-up needed, just \
instant access. Hope it changes how you see your money."

EXAMPLE 3 — Negative or dismissive comment:
Comment: "This is all obvious stuff anyone already knows"
Reply: "Fair point and you are probably further ahead than most people who \
find this channel. If any of it was useful even as a reminder, glad it showed \
up. Appreciate you watching."

EXAMPLE 4 — Question:
Comment: "How do I actually start investing with only 50 dollars a month?"
Reply: "50 US dollars a month is genuinely enough to start. Open a free \
account with Fidelity or Schwab, find their total market index fund, and \
set up an automatic monthly transfer. The habit matters more than the amount \
at this stage. The Wealth Systems Blueprint goes deeper on this if you want \
the full picture."
"""

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ReplyResult:
    """Result of a single comment reply generation."""

    comment_text: str
    commenter_name: str
    category: str
    trigger_type: str
    text: str               # the final reply text (truncated to ≤500 chars)
    char_count: int = 0
    was_truncated: bool = False
    is_valid: bool = True
    validation_errors: list[str] = field(default_factory=list)
    generated_at: str = ""
    raw_response: str = ""

    def __post_init__(self) -> None:
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()
        self.char_count = len(self.text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "comment_text":   self.comment_text,
            "commenter_name": self.commenter_name,
            "category":       self.category,
            "trigger_type":   self.trigger_type,
            "text":           self.text,
            "char_count":     self.char_count,
            "was_truncated":  self.was_truncated,
            "is_valid":       self.is_valid,
            "validation_errors": self.validation_errors,
            "generated_at":   self.generated_at,
        }


# ---------------------------------------------------------------------------
# CommentResponder
# ---------------------------------------------------------------------------


class CommentResponder:
    """
    Generates personalised YouTube comment replies via Claude API.

    Args:
        api_key: Anthropic API key. If None, reads ANTHROPIC_API_KEY from env.
        model: Claude model ID.
        max_tokens: Max tokens for API response.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = _MODEL,
        max_tokens: int = 300,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("ANTHROPIC_API_KEY", "")
        self.model = model
        self.max_tokens = max_tokens
        self._client: anthropic.Anthropic | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_reply(
        self,
        comment_text: str,
        commenter_name: str = "",
        category: str = "money",
        trigger_type: str = TRIGGER_GENERAL,
        video_title: str = "",
        video_url: str = "",
    ) -> ReplyResult:
        """
        Generate a personalised reply to a YouTube comment.

        Args:
            comment_text: The raw comment text from the viewer.
            commenter_name: Display name of the commenter (empty if unknown).
            category: Channel category slug ("money", "career", "success").
            trigger_type: One of TRIGGER_* constants.
            video_title: Title of the video being commented on (used for HOT_LEAD alerts).
            video_url: URL of the video (used for HOT_LEAD alerts).

        Returns:
            ReplyResult with the final reply text (≤500 chars).

        Raises:
            ValueError: If ANTHROPIC_API_KEY is not configured.
        """
        client = self._get_client()

        gumroad_url = self._get_gumroad_url(category)
        prompt = self._build_prompt(comment_text, commenter_name, category, trigger_type, gumroad_url)

        logger.info(
            "Generating reply: trigger=%s category=%s commenter=%r",
            trigger_type, category, commenter_name,
        )

        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
        logger.debug("Raw reply response: %s", raw)

        text, was_truncated = self._enforce_length(raw)

        errors = self._validate(text, trigger_type)

        result = ReplyResult(
            comment_text=comment_text,
            commenter_name=commenter_name,
            category=category,
            trigger_type=trigger_type,
            text=text,
            was_truncated=was_truncated,
            is_valid=len(errors) == 0,
            validation_errors=errors,
            raw_response=raw,
        )

        logger.info(
            "Reply generated: %d chars, truncated=%s, valid=%s",
            result.char_count, was_truncated, result.is_valid,
        )
        if errors:
            logger.warning("Reply validation issues: %s", errors)

        # Notification 9 — hot lead alert
        if trigger_type == TRIGGER_HOT_LEAD:
            try:
                from src.notifications.telegram_notifier import TelegramNotifier
                TelegramNotifier().notify_hot_lead(
                    commenter_name=commenter_name or "unknown",
                    video_title=video_title or "(unknown video)",
                    comment_text=comment_text,
                    video_url=video_url or "",
                )
            except Exception:
                pass

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

    @staticmethod
    def _get_gumroad_url(category: str) -> str:
        """Return the Gumroad URL for the given category from PRODUCTS config."""
        from config.constants import PRODUCTS
        return PRODUCTS.get(category, {}).get("gumroad_url", "")

    @staticmethod
    def _build_prompt(
        comment_text: str,
        commenter_name: str,
        category: str,
        trigger_type: str,
        gumroad_url: str,
    ) -> str:
        name_line = f"Commenter name: {commenter_name}" if commenter_name else "Commenter name: (unknown)"
        url_line  = f"Relevant Gumroad URL if needed: {gumroad_url}" if gumroad_url else ""
        parts = [
            name_line,
            f"Their comment: {comment_text}",
            f"Video category: {category}",
            f"Trigger type: {trigger_type}",
        ]
        if url_line:
            parts.append(url_line)
        parts.append("\nWrite a reply to this comment.")
        return "\n".join(parts)

    @staticmethod
    def _enforce_length(text: str, max_chars: int = MAX_REPLY_CHARS) -> tuple[str, bool]:
        """
        Ensure text is within max_chars.

        If it exceeds the limit, truncate at the last complete sentence
        boundary (period, exclamation, or question mark) before the limit.
        Falls back to hard truncation if no sentence boundary is found.

        Returns (final_text, was_truncated).
        """
        if len(text) <= max_chars:
            return text, False

        # Find last sentence-ending punctuation before the limit
        window = text[:max_chars]
        match = None
        for m in re.finditer(r"[.!?]", window):
            match = m
        if match:
            truncated = text[: match.end()].strip()
        else:
            truncated = window.rstrip()

        return truncated, True

    @staticmethod
    def _validate(text: str, trigger_type: str) -> list[str]:
        """Return a list of validation error strings (empty = valid)."""
        errors: list[str] = []

        if not text.strip():
            errors.append("reply text is empty")

        if len(text) > MAX_REPLY_CHARS:
            errors.append(f"reply is {len(text)} chars, exceeds limit of {MAX_REPLY_CHARS}")

        if "\u2014" in text or "\u2013" in text:
            errors.append("reply contains em/en dash — remove per formatting rules")

        if trigger_type not in VALID_TRIGGERS:
            errors.append(f"unknown trigger_type={trigger_type!r}")

        return errors

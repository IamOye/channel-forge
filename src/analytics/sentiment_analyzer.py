"""
sentiment_analyzer.py — SentimentAnalyzer

Fetches up to 50 YouTube video comments and uses Claude API (claude-sonnet-4-5)
to analyze audience sentiment. Optionally injects a followup topic into the
production queue when debate_intensity == "viral".

Usage:
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze("YT_abc123", "Stoic Wisdom")
    print(result.sentiment_score, result.debate_intensity)
"""

import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-5"
DB_PATH = Path("data/processed/channel_forge.db")
CREDENTIALS_DIR = Path(".credentials")

_SYSTEM_PROMPT = """You are an expert YouTube audience analyst.
Given a list of YouTube comments on a video, analyze the overall audience sentiment
and reaction.

Return ONLY a JSON object in this exact format (no markdown):
{
  "sentiment_score": 0.72,
  "dominant_reaction": "inspired",
  "debate_intensity": "low",
  "summary": "brief explanation"
}

Rules:
- sentiment_score: -1.0 (very negative) to +1.0 (very positive)
- dominant_reaction: one of: inspired, curious, amused, angry, sad, skeptical, excited, bored
- debate_intensity: one of: low, medium, high, viral
  - viral = the comments show extreme polarization, heated arguments, or mass sharing intent
- summary: 1-2 sentences describing the audience mood"""

_SCORED_TOPICS_DDL = """
CREATE TABLE IF NOT EXISTS scored_topics (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    keyword     TEXT    NOT NULL,
    category    TEXT    NOT NULL DEFAULT 'success',
    score       REAL    NOT NULL DEFAULT 0,
    source      TEXT    NOT NULL DEFAULT 'manual',
    created_at  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_scored_keyword ON scored_topics (keyword);
CREATE INDEX IF NOT EXISTS idx_scored_score   ON scored_topics (score DESC);
"""

_VALID_INTENSITIES = {"low", "medium", "high", "viral"}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SentimentResult:
    """Sentiment analysis result for a YouTube video's comments."""

    video_id: str
    video_title: str
    comment_count: int
    sentiment_score: float          # -1.0 to +1.0
    dominant_reaction: str          # e.g. "inspired"
    debate_intensity: str           # low/medium/high/viral
    summary: str = ""
    followup_injected: bool = False
    analyzed_at: str = ""
    is_valid: bool = True
    error: str = ""

    def __post_init__(self) -> None:
        if not self.analyzed_at:
            self.analyzed_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id":          self.video_id,
            "video_title":       self.video_title,
            "comment_count":     self.comment_count,
            "sentiment_score":   round(self.sentiment_score, 4),
            "dominant_reaction": self.dominant_reaction,
            "debate_intensity":  self.debate_intensity,
            "summary":           self.summary,
            "followup_injected": self.followup_injected,
            "analyzed_at":       self.analyzed_at,
            "is_valid":          self.is_valid,
            "error":             self.error,
        }


# ---------------------------------------------------------------------------
# SentimentAnalyzer
# ---------------------------------------------------------------------------

class SentimentAnalyzer:
    """
    Analyzes YouTube video comment sentiment using Claude API.

    Args:
        api_key:         Anthropic API key. If None, reads ANTHROPIC_API_KEY.
        db_path:         SQLite path for injecting followup topics.
        credentials_dir: Directory with OAuth tokens for YouTube API.
        max_comments:    Max comments to fetch per video (default 50).
    """

    VIRAL_FOLLOWUP_SCORE = 92.0

    def __init__(
        self,
        api_key: str | None = None,
        db_path: str | Path = DB_PATH,
        credentials_dir: str | Path = CREDENTIALS_DIR,
        max_comments: int = 50,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("ANTHROPIC_API_KEY", "")
        self.db_path = Path(db_path)
        self.credentials_dir = Path(credentials_dir)
        self.max_comments = max_comments
        self._client: anthropic.Anthropic | None = None
        self._ensure_table()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        video_id: str,
        video_title: str = "",
        channel_key: str = "default",
    ) -> SentimentResult:
        """
        Fetch comments for video_id and run Claude sentiment analysis.

        If debate_intensity == "viral", a suggested followup topic is injected
        into the scored_topics table with score=92.

        Args:
            video_id:    YouTube video ID.
            video_title: Title of the video (used for context in analysis).
            channel_key: OAuth credential key for the YouTube API.

        Returns:
            SentimentResult with sentiment_score, dominant_reaction, debate_intensity.
        """
        logger.info("Analyzing sentiment for video: %s", video_id)
        try:
            comments = self._fetch_comments(video_id, channel_key)
            result = self._analyze_sentiment(video_id, video_title, comments)
            if result.debate_intensity == "viral":
                self._inject_followup_topic(video_title)
                result.followup_injected = True
                logger.info(
                    "Viral debate detected — followup topic injected for: %s", video_title
                )
            return result
        except Exception as exc:
            logger.error("Sentiment analysis failed for %s: %s", video_id, exc)
            return SentimentResult(
                video_id=video_id,
                video_title=video_title,
                comment_count=0,
                sentiment_score=0.0,
                dominant_reaction="unknown",
                debate_intensity="low",
                is_valid=False,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Private: YouTube comments API (lazy Google imports)
    # ------------------------------------------------------------------

    def _load_credentials(self, channel_key: str):
        """Load OAuth credentials from .credentials/{channel_key}_token.json."""
        from google.oauth2.credentials import Credentials  # lazy
        token_path = self.credentials_dir / f"{channel_key}_token.json"
        if not token_path.exists():
            raise FileNotFoundError(f"Credentials not found: {token_path}")
        data = json.loads(token_path.read_text())
        return Credentials(
            token=data.get("token"),
            refresh_token=data.get("refresh_token"),
            token_uri=data.get("token_uri", "https://oauth2.googleapis.com/token"),
            client_id=data.get("client_id"),
            client_secret=data.get("client_secret"),
            scopes=data.get("scopes"),
        )

    def _build_youtube_service(self, credentials):
        """Build YouTube Data API v3 service."""
        from googleapiclient.discovery import build  # lazy
        return build("youtube", "v3", credentials=credentials)

    def _fetch_comments(self, video_id: str, channel_key: str) -> list[str]:
        """
        Fetch up to max_comments top-level comments from the video.

        Returns a list of comment text strings.
        """
        credentials = self._load_credentials(channel_key)
        service = self._build_youtube_service(credentials)

        response = service.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=self.max_comments,
            textFormat="plainText",
            order="relevance",
        ).execute()

        comments: list[str] = []
        for item in response.get("items", []):
            text = (
                item
                .get("snippet", {})
                .get("topLevelComment", {})
                .get("snippet", {})
                .get("textDisplay", "")
            ).strip()
            if text:
                comments.append(text)
        return comments

    # ------------------------------------------------------------------
    # Private: Claude sentiment analysis
    # ------------------------------------------------------------------

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            if not self.api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def _analyze_sentiment(
        self, video_id: str, video_title: str, comments: list[str]
    ) -> SentimentResult:
        """Call Claude API to analyze sentiment of a comment list."""
        if not comments:
            return SentimentResult(
                video_id=video_id,
                video_title=video_title,
                comment_count=0,
                sentiment_score=0.0,
                dominant_reaction="unknown",
                debate_intensity="low",
                summary="No comments available to analyze.",
            )

        prompt = (
            f'Video title: "{video_title}"\n\n'
            f"Top {len(comments)} comments:\n"
            + "\n".join(f"- {c}" for c in comments[:50])
        )

        client = self._get_client()
        message = client.messages.create(
            model=_MODEL,
            max_tokens=300,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
        logger.debug("Raw sentiment response: %s", raw)

        return self._parse_result(video_id, video_title, len(comments), raw)

    def _parse_result(
        self,
        video_id: str,
        video_title: str,
        comment_count: int,
        raw: str,
    ) -> SentimentResult:
        """Parse Claude's JSON response into a SentimentResult."""
        text = raw
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                line for line in lines if not line.strip().startswith("```")
            ).strip()

        try:
            data = json.loads(text)
            sentiment_score   = float(max(-1.0, min(1.0, data.get("sentiment_score", 0.0))))
            dominant_reaction = str(data.get("dominant_reaction", "unknown"))
            debate_intensity  = str(data.get("debate_intensity", "low"))
            summary           = str(data.get("summary", ""))

            if debate_intensity not in _VALID_INTENSITIES:
                debate_intensity = "low"

            return SentimentResult(
                video_id=video_id,
                video_title=video_title,
                comment_count=comment_count,
                sentiment_score=sentiment_score,
                dominant_reaction=dominant_reaction,
                debate_intensity=debate_intensity,
                summary=summary,
            )
        except Exception as exc:
            logger.error(
                "Failed to parse sentiment response: %s | raw=%s", exc, raw[:200]
            )
            return SentimentResult(
                video_id=video_id,
                video_title=video_title,
                comment_count=comment_count,
                sentiment_score=0.0,
                dominant_reaction="unknown",
                debate_intensity="low",
                is_valid=False,
                error=f"Parse error: {exc}",
            )

    # ------------------------------------------------------------------
    # Private: followup topic injection
    # ------------------------------------------------------------------

    def _inject_followup_topic(self, video_title: str) -> None:
        """Insert a suggested followup topic into scored_topics with score=92."""
        keyword = f"followup: {video_title}"
        now = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO scored_topics (keyword, category, score, source, created_at)
                VALUES (?, 'success', ?, 'sentiment_viral', ?)
                """,
                (keyword, self.VIRAL_FOLLOWUP_SCORE, now),
            )
            conn.commit()
            logger.info("Injected viral followup topic: %s (score=92)", keyword)
        finally:
            conn.close()

    def _ensure_table(self) -> None:
        """Create the scored_topics table if it does not exist."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path)
            try:
                conn.executescript(_SCORED_TOPICS_DDL)
                conn.commit()
            finally:
                conn.close()
        except Exception as exc:
            logger.warning("Could not initialise scored_topics table: %s", exc)

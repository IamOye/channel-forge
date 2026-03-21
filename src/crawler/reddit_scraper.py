"""
reddit_scraper.py — RedditScraper

Scrapes Reddit's public JSON endpoints (no API key required) to extract
high-signal financial, career, and success topics for YouTube Shorts.

Subreddits:
    money:   personalfinance, financialindependence, povertyfinance,
             investing, wallstreetbets, Fire, Frugal
    career:  antiwork, careerguidance, cscareerquestions, jobs, salary
    success: getmotivated, selfimprovement, productivity, Entrepreneur

Usage:
    scraper = RedditScraper()
    topics = scraper.scrape_finance_subreddits(category="money")
    # or all categories:
    topics = scraper.scrape_finance_subreddits()
"""

import difflib
import json
import logging
import os
import socket
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUBREDDITS: dict[str, list[str]] = {
    "money": [
        "personalfinance",
        "financialindependence",
        "povertyfinance",
        "investing",
        "wallstreetbets",
        "Fire",
        "Frugal",
    ],
    "career": [
        "antiwork",
        "careerguidance",
        "cscareerquestions",
        "jobs",
        "salary",
    ],
    "success": [
        "getmotivated",
        "selfimprovement",
        "productivity",
        "Entrepreneur",
    ],
}

CATEGORY_MAP: dict[str, str] = {
    "personalfinance": "money",
    "financialindependence": "money",
    "povertyfinance": "money",
    "investing": "money",
    "wallstreetbets": "money",
    "Fire": "money",
    "Frugal": "money",
    "antiwork": "career",
    "careerguidance": "career",
    "cscareerquestions": "career",
    "jobs": "career",
    "salary": "career",
    "getmotivated": "success",
    "selfimprovement": "success",
    "productivity": "success",
    "Entrepreneur": "success",
}

_REDDIT_BASE = "https://www.reddit.com/r"
_HEADERS = {
    "User-Agent": "ChannelForge/1.0 topic-research",
    "Accept": "application/json",
}
_REQUEST_TIMEOUT = 10.0      # seconds per request
_RATE_LIMIT_DELAY = 2.0      # seconds between requests
_MIN_SCORE = 100             # minimum upvotes
_MIN_COMMENTS = 20           # minimum comment count
_BATCH_SIZE = 10             # posts per Claude extraction call
_DEDUP_THRESHOLD = 0.70      # SequenceMatcher similarity cutoff

_MOD_PREFIXES = (
    "[META]", "[MOD]", "[Weekly]", "[Daily]",
    "[Monthly]", "[Megathread]",
)
_MIN_TITLE_LEN = 20
_MAX_TITLE_LEN = 200

_DEFAULT_DB = Path(os.getenv("DB_PATH", "data/processed/channel_forge.db"))

_DATACENTER_HOSTNAMES = (
    "railway", "render", "heroku", "fly",
    "aws", "gcp", "azure", "digitalocean", "linode",
)


def _is_datacenter() -> bool:
    """Return True when running on a known cloud/datacenter host."""
    try:
        hostname = socket.gethostname().lower()
        return any(x in hostname for x in _DATACENTER_HOSTNAMES)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RedditTopic:
    """A single topic extracted from a Reddit post."""

    keyword: str
    category: str
    score: float
    subreddit: str
    upvotes: int
    pain_level: int
    source: str = "reddit"
    used: int = 0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RedditScraper:
    """
    Scrapes Reddit public JSON endpoints to source YouTube Shorts topics.

    No API key required — uses public .json endpoints with a 2-second rate
    limit delay between requests.  Claude API (optional) is used for batch
    topic extraction; falls back to raw title trimming when unavailable.
    """

    def __init__(
        self,
        anthropic_api_key: str | None = None,
        db_path: Path | None = None,
    ) -> None:
        self.anthropic_api_key = (
            anthropic_api_key
            if anthropic_api_key is not None
            else os.getenv("ANTHROPIC_API_KEY", "")
        )
        self.db_path = db_path or _DEFAULT_DB

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scrape_finance_subreddits(
        self, category: str | None = None
    ) -> list[RedditTopic]:
        """
        Scrape Reddit for financial/career/success topics.

        Args:
            category: One of 'money', 'career', 'success'. If None, scrapes all.

        Returns:
            List of RedditTopic objects (already persisted to DB).
        """
        if _is_datacenter():
            logger.info(
                "[reddit] Datacenter IP detected — Reddit scraper disabled on cloud servers. "
                "Run locally for Reddit topics."
            )
            return []

        categories_to_scrape = (
            {category: SUBREDDITS[category]}
            if category and category in SUBREDDITS
            else SUBREDDITS
        )

        # Load existing topics for deduplication
        existing_topics = self._load_existing_topics()

        all_topics: list[RedditTopic] = []
        first_request = True

        for cat, subs in categories_to_scrape.items():
            for sub in subs:
                for kind in ("hot", "top"):
                    if not first_request:
                        time.sleep(_RATE_LIMIT_DELAY)
                    first_request = False

                    posts = self._fetch_subreddit(sub, kind)
                    qualified = self._filter_posts(posts)
                    if not qualified:
                        continue

                    extracted = self._extract_topics_batch(qualified, sub)

                    for item in extracted:
                        topic_text = item.get("topic", "").strip()
                        pain_level = int(item.get("pain_level", 5))
                        post_idx = item.get("post_index", 1) - 1

                        if not topic_text:
                            continue

                        # Get upvotes from original post
                        if 0 <= post_idx < len(qualified):
                            upvotes = int(qualified[post_idx].get("score", 100))
                        else:
                            upvotes = 100

                        if self._is_duplicate(topic_text, existing_topics):
                            logger.debug(
                                "[reddit] Skipping duplicate topic: %s", topic_text
                            )
                            continue

                        score = self._calculate_score(upvotes, pain_level)
                        reddit_topic = RedditTopic(
                            keyword=topic_text,
                            category=cat,
                            score=score,
                            subreddit=sub,
                            upvotes=upvotes,
                            pain_level=pain_level,
                            source=f"reddit/{sub}",
                        )
                        all_topics.append(reddit_topic)
                        existing_topics.append(topic_text)  # prevent intra-run dups

        saved_count = self._save_to_db(all_topics)
        logger.info(
            "[reddit] scrape_finance_subreddits complete: %d topics saved (category=%s)",
            saved_count, category or "all",
        )
        return all_topics

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------

    def _fetch_subreddit(self, subreddit: str, kind: str = "hot") -> list[dict]:
        """
        Fetch posts from a subreddit's hot or top-week JSON endpoint.

        Args:
            subreddit: Subreddit name (e.g. 'personalfinance').
            kind: 'hot' or 'top'.

        Returns:
            List of post data dicts. Empty list on any error.
        """
        if kind == "top":
            url = f"{_REDDIT_BASE}/{subreddit}/top.json?t=week&limit=25"
        else:
            url = f"{_REDDIT_BASE}/{subreddit}/hot.json?limit=25"

        try:
            resp = httpx.get(url, headers=_HEADERS, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            children = data.get("data", {}).get("children", [])
            return [child.get("data", {}) for child in children]
        except Exception as exc:
            logger.warning(
                "[reddit] Failed to fetch r/%s/%s: %s", subreddit, kind, exc
            )
            return []

    # ------------------------------------------------------------------
    # Filter
    # ------------------------------------------------------------------

    def _filter_posts(self, posts: list[dict]) -> list[dict]:
        """
        Return only posts that meet all quality criteria.

        Criteria:
        - score >= 100 upvotes
        - num_comments >= 20
        - stickied is False
        - selftext is not '[removed]' or '[deleted]'
        - title does not start with mod prefixes
        - title length in [20, 200] characters
        """
        qualified: list[dict] = []
        for post in posts:
            if post.get("stickied", False):
                continue
            if int(post.get("score", 0)) < _MIN_SCORE:
                continue
            if int(post.get("num_comments", 0)) < _MIN_COMMENTS:
                continue
            selftext = post.get("selftext", "")
            if selftext in ("[removed]", "[deleted]"):
                continue
            title = (post.get("title") or "")
            if any(title.startswith(prefix) for prefix in _MOD_PREFIXES):
                continue
            if not (_MIN_TITLE_LEN <= len(title) <= _MAX_TITLE_LEN):
                continue
            qualified.append(post)
        return qualified

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------

    def _calculate_score(self, upvotes: int, pain_level: int) -> float:
        """
        Compute composite score from upvote tier and pain_level (1–10).

        Base scores:
            >= 10 000 → 98
            >= 5 000  → 95
            >= 1 000  → 92
            >= 500    → 88
            >= 100    → 75

        Final = base * (pain_level / 10) * 1.2, capped at 99.
        """
        if upvotes >= 10000:
            base = 98.0
        elif upvotes >= 5000:
            base = 95.0
        elif upvotes >= 1000:
            base = 92.0
        elif upvotes >= 500:
            base = 88.0
        else:
            base = 75.0

        final = base * (pain_level / 10) * 1.2
        return min(99.0, round(final, 1))

    # ------------------------------------------------------------------
    # Claude extraction
    # ------------------------------------------------------------------

    def _extract_topics_batch(
        self, posts: list[dict], subreddit: str
    ) -> list[dict[str, Any]]:
        """
        Extract YouTube Short topics from Reddit posts using Claude.

        Posts are processed in batches of _BATCH_SIZE to reduce API calls.
        Falls back to raw title trimming on any API failure.

        Returns:
            List of dicts with keys: post_index, topic, pain_level.
        """
        results: list[dict[str, Any]] = []
        for batch_start in range(0, len(posts), _BATCH_SIZE):
            batch = posts[batch_start: batch_start + _BATCH_SIZE]
            batch_results = self._extract_batch(batch, offset=batch_start)
            results.extend(batch_results)
        return results

    def _extract_batch(
        self, posts: list[dict], offset: int = 0
    ) -> list[dict[str, Any]]:
        """
        Call Claude to extract a single batch of post titles into topics.

        Falls back to `_fallback_extraction` if no API key or on any error.
        """
        if not self.anthropic_api_key:
            return self._fallback_extraction(posts, offset)

        titles_text = "\n".join(
            f"{i + 1}. {post.get('title', '')}"
            for i, post in enumerate(posts)
        )
        prompt = (
            "These are Reddit post titles from finance subreddits. "
            "For each one extract the core financial pain point or insight "
            "as a YouTube Short topic for a US audience.\n\n"
            "Rules:\n"
            "- 8-12 words maximum\n"
            "- Must be a statement or revelation not a question\n"
            "- Western market context only\n"
            "- No rupees, no Indian references\n"
            "- Sound like a Shorts hook not a Reddit title\n\n"
            f"Posts:\n{titles_text}\n\n"
            "Return JSON array of objects:\n"
            '[{"post_index": 1, "topic": "extracted topic string", "pain_level": 1-10}]'
        )

        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.anthropic_api_key)
            message = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = "\n".join(
                    line
                    for line in raw.splitlines()
                    if not line.strip().startswith("```")
                ).strip()
            items = json.loads(raw)
            if not isinstance(items, list):
                raise ValueError("Expected a JSON array")
            # Adjust indices for batch offset
            for item in items:
                if "post_index" in item:
                    item["post_index"] = int(item["post_index"]) + offset
            return items
        except Exception as exc:
            logger.warning(
                "[reddit] Claude batch extraction failed (%s) — using fallback", exc
            )
            return self._fallback_extraction(posts, offset)

    def _fallback_extraction(
        self, posts: list[dict], offset: int = 0
    ) -> list[dict[str, Any]]:
        """Return raw post title (trimmed to 12 words) as topic fallback."""
        results: list[dict[str, Any]] = []
        for i, post in enumerate(posts):
            title = (post.get("title") or "").strip()
            words = title.split()[:12]
            topic = " ".join(words)
            if topic:
                results.append({
                    "post_index": offset + i + 1,
                    "topic": topic,
                    "pain_level": 5,
                })
        return results

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _is_duplicate(self, topic: str, existing: list[str]) -> bool:
        """
        Return True if topic similarity > 70% with any existing topic.

        Uses difflib.SequenceMatcher on lower-cased strings.
        """
        topic_lower = topic.lower()
        for existing_topic in existing:
            ratio = difflib.SequenceMatcher(
                None, topic_lower, existing_topic.lower()
            ).ratio()
            if ratio > _DEDUP_THRESHOLD:
                return True
        return False

    def _load_existing_topics(self) -> list[str]:
        """Load all keywords from scored_topics for dedup checking."""
        try:
            if not self.db_path.exists():
                return []
            conn = sqlite3.connect(self.db_path)
            try:
                rows = conn.execute(
                    "SELECT keyword FROM scored_topics"
                ).fetchall()
                return [r[0] for r in rows]
            finally:
                conn.close()
        except Exception as exc:
            logger.warning("[reddit] Could not load existing topics: %s", exc)
            return []

    # ------------------------------------------------------------------
    # DB storage
    # ------------------------------------------------------------------

    def _save_to_db(self, topics: list[RedditTopic]) -> int:
        """
        Persist RedditTopic objects to the scored_topics table.

        Returns:
            Number of rows inserted (0 on empty input or error).
        """
        if not topics:
            return 0
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS scored_topics (
                        id         INTEGER PRIMARY KEY AUTOINCREMENT,
                        keyword    TEXT    NOT NULL,
                        category   TEXT    NOT NULL DEFAULT 'success',
                        score      REAL    NOT NULL DEFAULT 0,
                        source     TEXT    NOT NULL DEFAULT 'manual',
                        used       INTEGER NOT NULL DEFAULT 0,
                        created_at TEXT    NOT NULL DEFAULT (datetime('now'))
                    )
                """)
                conn.executemany(
                    """
                    INSERT INTO scored_topics
                        (keyword, category, score, source, used, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            t.keyword, t.category, t.score,
                            t.source, t.used, t.created_at,
                        )
                        for t in topics
                    ],
                )
                conn.commit()
                count = len(topics)
                logger.info("[reddit] Saved %d topics to scored_topics", count)
                return count
            finally:
                conn.close()
        except Exception as exc:
            logger.error("[reddit] Failed to save topics to DB: %s", exc)
            return 0

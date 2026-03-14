"""
competitor_scraper.py — CompetitorScraper

Scrapes competitor YouTube channels and trending finance content to build
a continuous supply of high-signal video topics.

Three discovery methods:
  1. scrape_competitor_topics(category)  — latest high-view videos from rival channels
  2. scrape_trending_finance_topics()    — YouTube trending + Shorts + keyword search
  3. mine_comment_topics(video_ids)      — extract question patterns from own video comments

Topics are stored in the competitor_topics table and returned as plain strings
ready for injection into the scored_topics table via TopicQueue.

Usage:
    scraper = CompetitorScraper()
    topics = scraper.scrape_competitor_topics("money")
    trending = scraper.scrape_trending_finance_topics()
"""

import logging
import os
import re
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from config.constants import (
    AUTOCOMPLETE_SEED_KEYWORDS,
    COMPETITOR_CHANNELS,
    COMPETITOR_HIGH_SIGNAL_MIN_VIEWS,
    FINANCE_SEARCH_KEYWORDS,
    TRENDING_SEARCH_KEYWORDS,
)

load_dotenv()

logger = logging.getLogger(__name__)

_YOUTUBE_BASE = "https://www.googleapis.com/youtube/v3"
_AUTOCOMPLETE_URL = "http://suggestqueries.google.com/complete/search"
_DEFAULT_DB = Path("data/processed/channel_forge.db")

# Regex patterns that signal a viewer question
_QUESTION_PATTERNS = re.compile(
    r"\b(why|how|what|when|where|who|can you|do you|is it|should i|"
    r"could you|help me|explain|anyone know)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class CompetitorTopic:
    """A single topic extracted from a competitor or trending video."""

    __slots__ = (
        "channel_name", "original_title", "extracted_topic",
        "view_count", "category", "source", "scraped_at",
    )

    def __init__(
        self,
        channel_name: str,
        original_title: str,
        extracted_topic: str,
        view_count: int = 0,
        category: str = "money",
        source: str = "competitor",
    ) -> None:
        self.channel_name    = channel_name
        self.original_title  = original_title
        self.extracted_topic = extracted_topic
        self.view_count      = view_count
        self.category        = category
        self.source          = source
        self.scraped_at      = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel_name":    self.channel_name,
            "original_title":  self.original_title,
            "extracted_topic": self.extracted_topic,
            "view_count":      self.view_count,
            "category":        self.category,
            "source":          self.source,
            "scraped_at":      self.scraped_at,
        }


# ---------------------------------------------------------------------------
# CompetitorScraper
# ---------------------------------------------------------------------------


class CompetitorScraper:
    """
    Scrapes YouTube competitor channels and trending finance content.

    Args:
        api_key: YouTube Data API v3 key. If None, reads YOUTUBE_API_KEY from env.
        anthropic_api_key: Anthropic key for topic extraction. If None, reads env.
        db_path: Path to SQLite DB for storing results.
        timeout: HTTP timeout in seconds.
        max_retries: Number of HTTP retries per request.
    """

    def __init__(
        self,
        api_key: str | None = None,
        anthropic_api_key: str | None = None,
        db_path: str | Path = _DEFAULT_DB,
        timeout: float = 20.0,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("YOUTUBE_API_KEY", "")
        self.anthropic_api_key = (
            anthropic_api_key if anthropic_api_key is not None
            else os.getenv("ANTHROPIC_API_KEY", "")
        )
        self.db_path = Path(db_path)
        self.max_retries = max_retries
        self._client = httpx.Client(timeout=timeout)
        self._anthropic: Any = None  # lazy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scrape_competitor_topics(self, category: str) -> list[str]:
        """
        Fetch recent high-performing videos from configured competitor channels.

        For each channel in COMPETITOR_CHANNELS[category], pulls the last 30 days
        of videos ordered by viewCount.  Videos with >= COMPETITOR_HIGH_SIGNAL_MIN_VIEWS
        have their titles processed by Claude to extract a clean topic phrase.

        Returns:
            List of extracted topic strings (may be empty if API key absent).
        """
        if not self.api_key:
            logger.warning("YOUTUBE_API_KEY not set — skipping competitor scraping")
            return []

        channels = COMPETITOR_CHANNELS.get(category, [])
        if not channels:
            logger.warning("No competitor channels configured for category '%s'", category)
            return []

        results: list[CompetitorTopic] = []
        published_after = (datetime.now(timezone.utc) - timedelta(days=30)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        for channel in channels:
            channel_name = channel["name"]
            channel_id   = channel["id"]
            logger.info("Scraping competitor channel: %s", channel_name)

            videos = self._fetch_channel_videos(channel_id, published_after)
            for video in videos:
                view_count = int(video.get("view_count", 0))
                if view_count >= COMPETITOR_HIGH_SIGNAL_MIN_VIEWS:
                    topic = self._extract_topic(video["title"], category)
                    if topic:
                        ct = CompetitorTopic(
                            channel_name=channel_name,
                            original_title=video["title"],
                            extracted_topic=topic,
                            view_count=view_count,
                            category=category,
                            source="COMPETITOR_HIGH_SIGNAL",
                        )
                        results.append(ct)

            time.sleep(0.5)

        self._save_to_db(results)
        return [ct.extracted_topic for ct in results]

    def scrape_trending_finance_topics(self) -> list[str]:
        """
        Discover trending finance topics via three YouTube approaches.

        Approach 1: YouTube most-popular videos (category 25 — News & Politics)
        Approach 2: Search for recent finance Shorts (videoDuration=short, last 7 days)
        Approach 3: Keyword-based view-count search across FINANCE_SEARCH_KEYWORDS

        Returns:
            Combined deduplicated list of extracted topic strings.
        """
        if not self.api_key:
            logger.warning("YOUTUBE_API_KEY not set — skipping trending scrape")
            return []

        results: list[CompetitorTopic] = []

        # Approach 1: trending by category
        results.extend(self._scrape_trending_by_category())

        # Approach 2: recent finance Shorts
        results.extend(self._scrape_finance_shorts())

        # Approach 3: keyword search
        results.extend(self._scrape_keyword_search())

        # Dedup by extracted_topic
        seen: set[str] = set()
        unique: list[CompetitorTopic] = []
        for ct in results:
            key = ct.extracted_topic.lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(ct)

        self._save_to_db(unique)
        return [ct.extracted_topic for ct in unique]

    def mine_comment_topics(self, video_ids: list[str]) -> list[str]:
        """
        Extract question-pattern topics from comments on own uploaded videos.

        Fetches the top 20 comments per video, filters for viewer questions
        (containing "why", "how", "what", etc.), and extracts the core topic
        via Claude.  These are tagged as VIEWER_REQUESTED priority.

        Args:
            video_ids: YouTube video IDs from uploaded_videos table.

        Returns:
            List of extracted topic strings.
        """
        if not self.api_key:
            logger.warning("YOUTUBE_API_KEY not set — skipping comment mining")
            return []

        results: list[CompetitorTopic] = []

        for video_id in video_ids:
            comments = self._fetch_comments(video_id)
            for comment in comments:
                if _QUESTION_PATTERNS.search(comment):
                    topic = self._extract_topic_from_comment(comment)
                    if topic:
                        results.append(CompetitorTopic(
                            channel_name="own_video",
                            original_title=comment[:120],
                            extracted_topic=topic,
                            view_count=0,
                            category="money",      # refined downstream
                            source="VIEWER_REQUESTED",
                        ))
            time.sleep(0.3)

        self._save_to_db(results)
        return [ct.extracted_topic for ct in results]

    def scrape_search_autocomplete(self, category: str) -> list[str]:
        """
        Fetch YouTube search autocomplete suggestions for seed keywords.

        Uses the YouTube Suggest API to discover what people are actively
        typing right now — no API key required.

        Args:
            category: Channel category ("money", "career", "success").

        Returns:
            Deduplicated list of suggestion strings stored as AUTOCOMPLETE
            topics (priority 85).
        """
        seeds = AUTOCOMPLETE_SEED_KEYWORDS.get(category, [])
        if not seeds:
            logger.warning("No autocomplete seeds for category '%s'", category)
            return []

        results: list[CompetitorTopic] = []
        seen: set[str] = set()

        for seed in seeds:
            suggestions = self._fetch_autocomplete(seed)
            for suggestion in suggestions:
                key = suggestion.lower().strip()
                if key and key not in seen:
                    seen.add(key)
                    results.append(CompetitorTopic(
                        channel_name="youtube_autocomplete",
                        original_title=seed,
                        extracted_topic=suggestion,
                        view_count=0,
                        category=category,
                        source="AUTOCOMPLETE",
                    ))
            time.sleep(0.3)

        self._save_to_db(results)
        logger.info(
            "Autocomplete: %d topics scraped for category '%s'",
            len(results), category,
        )
        return [ct.extracted_topic for ct in results]

    def scrape_trending_search_topics(self, category: str = "money") -> list[str]:
        """
        Search YouTube for recent high-view Shorts, extract fresh topic angles.

        Queries videos published in the last 7 days, ordered by viewCount,
        with videoDuration=short. Claude extracts a concise topic angle from
        each title. Stored as TRENDING_SEARCH priority (score 80).

        Args:
            category: Channel category for storing results.

        Returns:
            List of extracted topic strings (empty if no API key).
        """
        if not self.api_key:
            logger.warning("YOUTUBE_API_KEY not set — skipping trending search")
            return []

        published_after = (
            datetime.now(timezone.utc) - timedelta(days=7)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        results: list[CompetitorTopic] = []
        seen: set[str] = set()

        for keyword in TRENDING_SEARCH_KEYWORDS:
            params = {
                "part":           "snippet",
                "q":              f"{keyword} money 2026",
                "type":           "video",
                "order":          "viewCount",
                "publishedAfter": published_after,
                "videoDuration":  "short",
                "regionCode":     "US",
                "maxResults":     25,
                "key":            self.api_key,
            }
            items = self._get(_YOUTUBE_BASE + "/search", params)
            video_ids = [
                i["id"]["videoId"] for i in items if "videoId" in i.get("id", {})
            ]
            stats_map = self._fetch_video_stats(video_ids) if video_ids else {}

            for item in items[:5]:
                title  = item.get("snippet", {}).get("title", "").strip()
                vid_id = item.get("id", {}).get("videoId", "")
                views  = stats_map.get(vid_id, 0)
                title_key = title.lower()
                if title and title_key not in seen:
                    seen.add(title_key)
                    topic = self._extract_trending_topic(title, views)
                    if topic:
                        results.append(CompetitorTopic(
                            channel_name="trending_search",
                            original_title=title,
                            extracted_topic=topic,
                            view_count=views,
                            category=category,
                            source="TRENDING_SEARCH",
                        ))
            time.sleep(0.5)

        self._save_to_db(results)
        logger.info("Trending search: %d topics scraped", len(results))
        return [ct.extracted_topic for ct in results]

    # ------------------------------------------------------------------
    # Private: YouTube API helpers
    # ------------------------------------------------------------------

    def _fetch_channel_videos(
        self, channel_id: str, published_after: str, max_results: int = 20
    ) -> list[dict[str, Any]]:
        """Search a channel's recent videos, returning [{title, view_count}]."""
        params = {
            "part":           "snippet",
            "channelId":      channel_id,
            "order":          "viewCount",
            "maxResults":     max_results,
            "type":           "video",
            "publishedAfter": published_after,
            "key":            self.api_key,
        }
        items = self._get(_YOUTUBE_BASE + "/search", params)
        if not items:
            return []

        video_ids = [i["id"]["videoId"] for i in items if "videoId" in i.get("id", {})]
        if not video_ids:
            return []

        # Fetch statistics in one batch call
        stats_map = self._fetch_video_stats(video_ids)

        videos: list[dict[str, Any]] = []
        for item in items:
            vid_id = item.get("id", {}).get("videoId", "")
            title  = item.get("snippet", {}).get("title", "").strip()
            if title and vid_id:
                videos.append({
                    "title":      title,
                    "view_count": stats_map.get(vid_id, 0),
                })
        return videos

    def _fetch_video_stats(self, video_ids: list[str]) -> dict[str, int]:
        """Return {video_id: view_count} for a batch of video IDs."""
        params = {
            "part": "statistics",
            "id":   ",".join(video_ids),
            "key":  self.api_key,
        }
        items = self._get(_YOUTUBE_BASE + "/videos", params)
        result: dict[str, int] = {}
        for item in items:
            vid_id  = item.get("id", "")
            views   = int(item.get("statistics", {}).get("viewCount", 0))
            result[vid_id] = views
        return result

    def _fetch_comments(self, video_id: str, max_results: int = 20) -> list[str]:
        """Fetch top-level comment text for a video."""
        params = {
            "part":       "snippet",
            "videoId":    video_id,
            "order":      "relevance",
            "maxResults": max_results,
            "key":        self.api_key,
        }
        items = self._get(_YOUTUBE_BASE + "/commentThreads", params)
        texts: list[str] = []
        for item in items:
            snippet = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
            text = snippet.get("textOriginal", "").strip()
            if text:
                texts.append(text)
        return texts

    def _scrape_trending_by_category(self) -> list[CompetitorTopic]:
        """Approach 1: YouTube most-popular videos in News & Politics (cat 25)."""
        params = {
            "part":              "snippet,statistics",
            "chart":             "mostPopular",
            "regionCode":        "US",
            "videoCategoryId":   "25",
            "maxResults":        50,
            "key":               self.api_key,
        }
        items = self._get(_YOUTUBE_BASE + "/videos", params)
        results: list[CompetitorTopic] = []
        for item in items:
            title      = item.get("snippet", {}).get("title", "").strip()
            view_count = int(item.get("statistics", {}).get("viewCount", 0))
            if title and view_count > 50_000:
                topic = self._extract_topic(title, "money")
                if topic:
                    results.append(CompetitorTopic(
                        channel_name="youtube_trending",
                        original_title=title,
                        extracted_topic=topic,
                        view_count=view_count,
                        category="money",
                        source="YOUTUBE_TRENDING",
                    ))
        return results

    def _scrape_finance_shorts(self) -> list[CompetitorTopic]:
        """Approach 2: Recent finance Shorts from last 7 days."""
        published_after = (datetime.now(timezone.utc) - timedelta(days=7)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        results: list[CompetitorTopic] = []
        # Cycle through a subset of keywords to stay within quota
        for keyword in FINANCE_SEARCH_KEYWORDS[:5]:
            params = {
                "part":           "snippet",
                "q":              f"{keyword} #shorts",
                "type":           "video",
                "order":          "viewCount",
                "publishedAfter": published_after,
                "videoDuration":  "short",
                "regionCode":     "US",
                "maxResults":     20,
                "key":            self.api_key,
            }
            items = self._get(_YOUTUBE_BASE + "/search", params)
            video_ids = [i["id"]["videoId"] for i in items if "videoId" in i.get("id", {})]
            stats_map = self._fetch_video_stats(video_ids) if video_ids else {}

            for item in items:
                vid_id = item.get("id", {}).get("videoId", "")
                title  = item.get("snippet", {}).get("title", "").strip()
                views  = stats_map.get(vid_id, 0)
                if title and views > 10_000:
                    topic = self._extract_topic(title, "money")
                    if topic:
                        results.append(CompetitorTopic(
                            channel_name="youtube_shorts",
                            original_title=title,
                            extracted_topic=topic,
                            view_count=views,
                            category="money",
                            source="YOUTUBE_TRENDING",
                        ))
            time.sleep(0.5)
        return results

    def _scrape_keyword_search(self) -> list[CompetitorTopic]:
        """Approach 3: View-count search across FINANCE_SEARCH_KEYWORDS."""
        results: list[CompetitorTopic] = []
        for keyword in FINANCE_SEARCH_KEYWORDS:
            params = {
                "part":       "snippet",
                "q":          keyword,
                "type":       "video",
                "order":      "viewCount",
                "regionCode": "US",
                "maxResults": 10,
                "key":        self.api_key,
            }
            items = self._get(_YOUTUBE_BASE + "/search", params)
            for item in items[:3]:   # only top 3 per keyword
                title = item.get("snippet", {}).get("title", "").strip()
                if title:
                    topic = self._extract_topic(title, "money")
                    if topic:
                        results.append(CompetitorTopic(
                            channel_name="keyword_search",
                            original_title=title,
                            extracted_topic=topic,
                            view_count=0,
                            category="money",
                            source="YOUTUBE_KEYWORD",
                        ))
            time.sleep(0.3)
        return results

    def _fetch_autocomplete(self, query: str) -> list[str]:
        """
        Call the YouTube Suggest API and return up to 10 suggestion strings.

        The endpoint returns JSONP: window.google.ac.h(["query",[...],...])
        Handles both JSONP and plain JSON response formats.
        """
        import json as _json

        params = {"client": "youtube", "ds": "yt", "q": query}
        try:
            resp = self._client.get(_AUTOCOMPLETE_URL, params=params)
            resp.raise_for_status()
            text = resp.text.strip()

            # Strip JSONP wrapper if present
            if text.startswith("window.google.ac.h("):
                text = text[len("window.google.ac.h("):]
                if text.endswith(")"):
                    text = text[:-1]

            data = _json.loads(text)
            suggestions_raw = data[1] if len(data) > 1 else []

            suggestions: list[str] = []
            for item in suggestions_raw:
                if isinstance(item, list) and item:
                    s = str(item[0]).strip()
                    if s:
                        suggestions.append(s)
                elif isinstance(item, str) and item.strip():
                    suggestions.append(item.strip())
            return suggestions[:10]
        except Exception as exc:
            logger.warning("Autocomplete fetch failed for '%s': %s", query, exc)
            return []

    def _extract_trending_topic(self, title: str, view_count: int) -> str:
        """
        Extract a fresh topic angle from a trending video title.

        Uses Claude with view-count context. Falls back to heuristic when
        Claude API is unavailable.
        """
        if not self.anthropic_api_key:
            return self._heuristic_extract(title)

        views_str = f"{view_count:,}" if view_count else "many"
        prompt = (
            f"This YouTube Short got {views_str} views in 5 days. "
            f"Extract the core financial topic as a fresh angle for a new video in under 10 words. "
            f'Title: "{title}"\n'
            f"Return only the new topic angle, plain lowercase, no hashtags."
        )
        try:
            client = self._get_anthropic_client()
            message = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=60,
                messages=[{"role": "user", "content": prompt}],
            )
            topic = message.content[0].text.strip().strip('"').strip("'")
            word_count = len(topic.split())
            if 3 <= word_count <= 15 and "http" not in topic:
                return topic
            return ""
        except Exception as exc:
            logger.warning("Trending topic extraction failed: %s", exc)
            return self._heuristic_extract(title)

    # ------------------------------------------------------------------
    # Private: HTTP helper
    # ------------------------------------------------------------------

    def _get(self, url: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        """GET request with retries; returns items list or []."""
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._client.get(url, params=params)
                resp.raise_for_status()
                return resp.json().get("items", [])
            except Exception as exc:
                logger.warning("HTTP attempt %d/%d failed: %s", attempt, self.max_retries, exc)
                if attempt < self.max_retries:
                    time.sleep(2.0 * attempt)
        logger.error("HTTP request failed after %d retries: %s", self.max_retries, url)
        return []

    # ------------------------------------------------------------------
    # Private: Claude topic extraction
    # ------------------------------------------------------------------

    def _get_anthropic_client(self) -> Any:
        if self._anthropic is None:
            import anthropic as _ant
            if not self.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            self._anthropic = _ant.Anthropic(api_key=self.anthropic_api_key)
        return self._anthropic

    def _extract_topic(self, title: str, category: str) -> str:
        """
        Use Claude to extract a clean 5-10 word topic from a video title.

        Falls back to a simple heuristic if Claude API is unavailable.

        Returns empty string on failure.
        """
        if not self.anthropic_api_key:
            return self._heuristic_extract(title)

        prompt = (
            f"Extract the core financial topic from this YouTube title as a "
            f"5 to 10 word topic suitable for a new video. "
            f"Write it in plain lowercase English, no hashtags, no punctuation at the end. "
            f'Title: "{title}"\n'
            f"Return only the topic, no explanation."
        )
        try:
            client = self._get_anthropic_client()
            message = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=60,
                messages=[{"role": "user", "content": prompt}],
            )
            topic = message.content[0].text.strip().strip('"').strip("'")
            # Sanity check: 3–20 words, no URLs
            word_count = len(topic.split())
            if 3 <= word_count <= 20 and "http" not in topic:
                return topic
            logger.debug("Extracted topic out of range (%d words): %r", word_count, topic)
            return ""
        except Exception as exc:
            logger.warning("Claude topic extraction failed: %s", exc)
            return self._heuristic_extract(title)

    def _extract_topic_from_comment(self, comment: str) -> str:
        """Extract a video topic idea from a viewer comment using Claude."""
        if not self.anthropic_api_key:
            return ""

        prompt = (
            f"A viewer left this comment on a finance YouTube video: \"{comment[:300]}\"\n\n"
            f"Extract the financial question or topic the viewer wants answered, "
            f"as a 5 to 10 word YouTube video topic in plain lowercase English. "
            f"Return only the topic, no explanation."
        )
        try:
            client = self._get_anthropic_client()
            message = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=60,
                messages=[{"role": "user", "content": prompt}],
            )
            topic = message.content[0].text.strip().strip('"').strip("'")
            word_count = len(topic.split())
            if 3 <= word_count <= 20 and "http" not in topic:
                return topic
            return ""
        except Exception as exc:
            logger.warning("Comment topic extraction failed: %s", exc)
            return ""

    @staticmethod
    def _heuristic_extract(title: str) -> str:
        """
        Simple heuristic: lowercase, strip hashtags, truncate to 10 words.

        Used as a fallback when Claude is unavailable.
        """
        cleaned = re.sub(r"#\w+", "", title).strip()
        cleaned = re.sub(r"[^\w\s]", " ", cleaned).strip()
        words = cleaned.lower().split()
        return " ".join(words[:10]) if words else ""

    # ------------------------------------------------------------------
    # Private: DB storage
    # ------------------------------------------------------------------

    def _save_to_db(self, topics: list[CompetitorTopic]) -> None:
        """Persist CompetitorTopic objects to the competitor_topics table."""
        if not topics:
            return
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS competitor_topics (
                        id              INTEGER PRIMARY KEY AUTOINCREMENT,
                        channel_name    TEXT    NOT NULL,
                        original_title  TEXT    NOT NULL,
                        extracted_topic TEXT    NOT NULL,
                        view_count      INTEGER NOT NULL DEFAULT 0,
                        category        TEXT    NOT NULL DEFAULT 'money',
                        source          TEXT    NOT NULL DEFAULT 'competitor',
                        used            INTEGER NOT NULL DEFAULT 0,
                        scraped_at      TEXT    NOT NULL DEFAULT (datetime('now'))
                    )
                """)
                conn.executemany(
                    """
                    INSERT INTO competitor_topics
                        (channel_name, original_title, extracted_topic,
                         view_count, category, source, scraped_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (ct.channel_name, ct.original_title, ct.extracted_topic,
                         ct.view_count, ct.category, ct.source, ct.scraped_at)
                        for ct in topics
                    ],
                )
                conn.commit()
                logger.info("Saved %d competitor topics to DB", len(topics))
            finally:
                conn.close()
        except Exception as exc:
            logger.error("Failed to save competitor topics to DB: %s", exc)

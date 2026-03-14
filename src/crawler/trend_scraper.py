"""
trend_scraper.py — TrendScrapingEngine

Fetches trending keywords/topics from:
  - Google Trends (via pytrends)
  - YouTube Trending (via YouTube Data API v3)

Usage:
    engine = TrendScrapingEngine()
    results = engine.fetch_all(keywords=["AI tools", "passive income"])
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# urllib3 v2.0 compatibility shim for pytrends
# urllib3 v2.0 renamed Retry's `method_whitelist` parameter to `allowed_methods`.
# pytrends still passes `method_whitelist`, so we patch Retry.__init__ to
# silently map the old name to the new one before importing pytrends.
# ---------------------------------------------------------------------------
try:
    import urllib3.util.retry as _retry_mod

    _orig_retry_init = _retry_mod.Retry.__init__

    def _compat_retry_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        if "method_whitelist" in kwargs:
            kwargs.setdefault("allowed_methods", kwargs.pop("method_whitelist"))
        _orig_retry_init(self, *args, **kwargs)

    _retry_mod.Retry.__init__ = _compat_retry_init  # type: ignore[method-assign]
except Exception:
    pass  # if urllib3 is not present or already compatible, do nothing

from pytrends.request import TrendReq

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class TrendSignal:
    """A single trend data point from any source."""

    keyword: str
    source: str                         # 'google' | 'youtube'
    region: str = "US"
    interest_score: float = 0.0         # 0–100 normalised
    related_queries: list[str] = field(default_factory=list)
    fetched_at: str = ""
    raw_json: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.fetched_at:
            self.fetched_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dict (for DB insertion)."""
        return {
            "keyword": self.keyword,
            "source": self.source,
            "region": self.region,
            "interest_score": self.interest_score,
            "related_query": json.dumps(self.related_queries),
            "fetched_at": self.fetched_at,
            "raw_json": json.dumps(self.raw_json),
        }


# ---------------------------------------------------------------------------
# Google Trends scraper
# ---------------------------------------------------------------------------

class GoogleTrendsScraper:
    """Wraps pytrends to pull interest-over-time and related queries."""

    def __init__(
        self,
        hl: str = "en-US",
        tz: int = 360,
        region: str = "US",
        retries: int = 3,
        backoff: float = 2.0,
    ) -> None:
        self.hl = hl
        self.tz = tz
        self.region = region
        self.retries = retries
        self.backoff = backoff
        self._pytrends = TrendReq(hl=hl, tz=tz, timeout=(10, 30), retries=retries, backoff_factor=backoff)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(self, keywords: list[str], timeframe: str = "today 3-m") -> list[TrendSignal]:
        """
        Fetch Google Trends data for a list of keywords.

        Args:
            keywords: Up to 5 keywords per request (pytrends limit).
            timeframe: pytrends timeframe string, e.g. 'today 3-m'.

        Returns:
            List of TrendSignal objects.
        """
        signals: list[TrendSignal] = []

        # pytrends allows max 5 keywords per build_payload call
        for chunk in self._chunk(keywords, size=5):
            chunk_signals = self._fetch_chunk(chunk, timeframe)
            signals.extend(chunk_signals)
            # polite delay between requests
            time.sleep(1.0)

        return signals

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_chunk(self, keywords: list[str], timeframe: str) -> list[TrendSignal]:
        """Fetch a single chunk (≤5 keywords) from Google Trends."""
        for attempt in range(1, self.retries + 1):
            try:
                self._pytrends.build_payload(
                    keywords,
                    cat=0,
                    timeframe=timeframe,
                    geo=self.region,
                    gprop="",
                )
                iot = self._pytrends.interest_over_time()
                related = self._pytrends.related_queries()
                return self._parse_results(keywords, iot, related)
            except Exception as exc:
                logger.warning(
                    "Google Trends attempt %d/%d failed for %s: %s",
                    attempt, self.retries, keywords, exc,
                )
                if attempt < self.retries:
                    time.sleep(self.backoff * attempt)
                else:
                    logger.error("Giving up on Google Trends for keywords: %s", keywords)
                    return []
        return []

    def _parse_results(
        self,
        keywords: list[str],
        iot: Any,
        related: dict[str, Any],
    ) -> list[TrendSignal]:
        """Convert raw pytrends DataFrames into TrendSignal objects."""
        signals: list[TrendSignal] = []

        for kw in keywords:
            interest_score = 0.0
            raw: dict[str, Any] = {}

            if iot is not None and not iot.empty and kw in iot.columns:
                series = iot[kw].dropna()
                if len(series) > 0:
                    interest_score = float(series.mean())
                raw["interest_over_time"] = series.tolist()

            related_list: list[str] = []
            if kw in related:
                top_df = related[kw].get("top")
                if top_df is not None and not top_df.empty:
                    related_list = top_df["query"].head(10).tolist()
                    raw["related_top"] = top_df.to_dict(orient="records")

            signals.append(
                TrendSignal(
                    keyword=kw,
                    source="google",
                    region=self.region,
                    interest_score=round(interest_score, 2),
                    related_queries=related_list,
                    raw_json=raw,
                )
            )

        return signals

    @staticmethod
    def _chunk(lst: list[str], size: int) -> list[list[str]]:
        """Split list into chunks of at most `size`."""
        return [lst[i : i + size] for i in range(0, len(lst), size)]


# ---------------------------------------------------------------------------
# YouTube Trends scraper
# ---------------------------------------------------------------------------

class YouTubeTrendsScraper:
    """
    Fetches YouTube trending videos using the YouTube Data API v3.

    Requires env var: YOUTUBE_API_KEY
    """

    BASE_URL = "https://www.googleapis.com/youtube/v3"

    def __init__(
        self,
        api_key: str | None = None,
        region: str = "US",
        max_results: int = 50,
        retries: int = 3,
        timeout: float = 20.0,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("YOUTUBE_API_KEY", "")
        self.region = region
        self.max_results = max_results
        self.retries = retries
        self._client = httpx.Client(timeout=timeout)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_trending(self, category_id: str = "0") -> list[TrendSignal]:
        """
        Fetch YouTube trending videos and extract keywords from their titles.

        Args:
            category_id: YouTube video category ID ('0' = all categories).

        Returns:
            List of TrendSignal objects derived from video titles.
        """
        if not self.api_key:
            logger.warning("YOUTUBE_API_KEY not set — skipping YouTube trends")
            return []

        videos = self._fetch_videos(category_id)
        return self._videos_to_signals(videos)

    def fetch_for_keywords(self, keywords: list[str]) -> list[TrendSignal]:
        """
        Search YouTube for each keyword and return view-count-based signals.

        Args:
            keywords: List of search terms.

        Returns:
            List of TrendSignal objects, one per keyword.
        """
        if not self.api_key:
            logger.warning("YOUTUBE_API_KEY not set — skipping YouTube keyword search")
            return []

        signals: list[TrendSignal] = []
        for kw in keywords:
            signal = self._search_keyword(kw)
            if signal:
                signals.append(signal)
            time.sleep(0.5)
        return signals

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_videos(self, category_id: str) -> list[dict[str, Any]]:
        """Call the YouTube videos.list endpoint for trending content."""
        params = {
            "part": "snippet,statistics",
            "chart": "mostPopular",
            "regionCode": self.region,
            "videoCategoryId": category_id,
            "maxResults": self.max_results,
            "key": self.api_key,
        }
        for attempt in range(1, self.retries + 1):
            try:
                resp = self._client.get(f"{self.BASE_URL}/videos", params=params)
                resp.raise_for_status()
                return resp.json().get("items", [])
            except httpx.HTTPStatusError as exc:
                logger.warning(
                    "YouTube API HTTP error attempt %d/%d: %s",
                    attempt, self.retries, exc,
                )
            except Exception as exc:
                logger.warning(
                    "YouTube API error attempt %d/%d: %s",
                    attempt, self.retries, exc,
                )
            if attempt < self.retries:
                time.sleep(2.0 * attempt)
        logger.error("YouTube trending fetch failed after %d retries", self.retries)
        return []

    def _search_keyword(self, keyword: str) -> TrendSignal | None:
        """Search YouTube for a keyword and derive an interest score."""
        params = {
            "part": "snippet",
            "q": keyword,
            "type": "video",
            "order": "viewCount",
            "maxResults": 10,
            "regionCode": self.region,
            "key": self.api_key,
        }
        for attempt in range(1, self.retries + 1):
            try:
                resp = self._client.get(f"{self.BASE_URL}/search", params=params)
                resp.raise_for_status()
                data = resp.json()
                items = data.get("items", [])
                total = data.get("pageInfo", {}).get("totalResults", 0)

                # Normalise result count to a 0–100 score (log scale)
                import math
                score = min(100.0, math.log1p(total) * 10) if total > 0 else 0.0

                related = [
                    item["snippet"]["title"]
                    for item in items
                    if "snippet" in item
                ]

                return TrendSignal(
                    keyword=keyword,
                    source="youtube",
                    region=self.region,
                    interest_score=round(score, 2),
                    related_queries=related[:10],
                    raw_json={"total_results": total, "top_titles": related},
                )
            except Exception as exc:
                logger.warning(
                    "YouTube search attempt %d/%d for '%s': %s",
                    attempt, self.retries, keyword, exc,
                )
                if attempt < self.retries:
                    time.sleep(2.0 * attempt)
        return None

    def _videos_to_signals(self, videos: list[dict[str, Any]]) -> list[TrendSignal]:
        """Convert trending video items into TrendSignal objects."""
        signals: list[TrendSignal] = []
        for video in videos:
            snippet = video.get("snippet", {})
            stats = video.get("statistics", {})
            title = snippet.get("title", "").strip()
            if not title:
                continue

            view_count = int(stats.get("viewCount", 0))
            import math
            score = min(100.0, math.log1p(view_count) / math.log1p(1_000_000) * 100)

            signals.append(
                TrendSignal(
                    keyword=title,
                    source="youtube",
                    region=self.region,
                    interest_score=round(score, 2),
                    related_queries=[snippet.get("channelTitle", "")],
                    raw_json={
                        "video_id": video.get("id", ""),
                        "view_count": view_count,
                        "like_count": int(stats.get("likeCount", 0)),
                        "category_id": snippet.get("categoryId", ""),
                    },
                )
            )
        return signals


# ---------------------------------------------------------------------------
# Unified engine
# ---------------------------------------------------------------------------

class TrendScrapingEngine:
    """
    Orchestrates all trend scrapers and returns a unified list of TrendSignals.

    Usage:
        engine = TrendScrapingEngine()
        signals = engine.fetch_all(keywords=["AI tools", "Python tutorial"])
    """

    def __init__(
        self,
        region: str = "US",
        google_enabled: bool = True,
        youtube_enabled: bool = True,
    ) -> None:
        self.region = region
        self.google: GoogleTrendsScraper | None = (
            GoogleTrendsScraper(region=region) if google_enabled else None
        )
        self.youtube: YouTubeTrendsScraper | None = (
            YouTubeTrendsScraper(region=region) if youtube_enabled else None
        )
        logger.info(
            "TrendScrapingEngine initialised (google=%s, youtube=%s)",
            google_enabled, youtube_enabled,
        )

    def fetch_all(
        self,
        keywords: list[str],
        include_youtube_trending: bool = False,
    ) -> list[TrendSignal]:
        """
        Fetch trends from all enabled sources for the given keywords.

        Args:
            keywords: Seed keywords to look up.
            include_youtube_trending: If True, also fetch global YouTube trending.

        Returns:
            Combined, deduplicated list of TrendSignal objects.
        """
        all_signals: list[TrendSignal] = []

        if self.google:
            logger.info("Fetching Google Trends for %d keywords…", len(keywords))
            try:
                g_signals = self.google.fetch(keywords)
                all_signals.extend(g_signals)
                logger.info("Google Trends: %d signals returned", len(g_signals))
            except Exception as exc:
                logger.error("Google Trends fetch failed: %s", exc)

        if self.youtube:
            logger.info("Fetching YouTube Trends for %d keywords…", len(keywords))
            try:
                yt_signals = self.youtube.fetch_for_keywords(keywords)
                all_signals.extend(yt_signals)
                logger.info("YouTube keyword search: %d signals returned", len(yt_signals))

                if include_youtube_trending:
                    trending = self.youtube.fetch_trending()
                    all_signals.extend(trending)
                    logger.info("YouTube trending: %d signals returned", len(trending))
            except Exception as exc:
                logger.error("YouTube Trends fetch failed: %s", exc)

        logger.info("TrendScrapingEngine total signals: %d", len(all_signals))
        return all_signals

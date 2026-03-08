"""
pixabay_fetcher.py — PixabayFetcher

Downloads royalty-free stock videos from the Pixabay API.

Usage:
    fetcher = PixabayFetcher()
    result = fetcher.fetch(topic_id="stoic_001", category="success")
    print(result.video_path)
"""

import logging
import os
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

_PIXABAY_API_URL = "https://pixabay.com/api/videos/"

# Maps topic category → list of search keywords (tried in order until one returns results)
KEYWORD_MAP: dict[str, list[str]] = {
    "money":      ["finance coins money", "wealth abundance"],
    "career":     ["office professional business", "workplace success"],
    "success":    ["trophy achievement winner", "success celebration"],
    "motivation": ["sunrise energy running", "motivation inspiration"],
    "stoicism":   ["mountains nature calm", "peaceful landscape"],
    "mindset":    ["meditation focus mind", "brain thinking"],
    "health":     ["fitness exercise healthy", "yoga wellness"],
    "default":    ["nature landscape", "cityscape abstract"],
}

MIN_VIDEO_DURATION_SECONDS = 10
OUTPUT_DIR = Path("data/raw")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class PixabayVideo:
    """Metadata for a single Pixabay video result."""

    pixabay_id: int
    duration: int            # seconds
    width: int
    height: int
    download_url: str
    page_url: str
    tags: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pixabay_id":   self.pixabay_id,
            "duration":     self.duration,
            "width":        self.width,
            "height":       self.height,
            "download_url": self.download_url,
            "page_url":     self.page_url,
            "tags":         self.tags,
        }


@dataclass
class FetchResult:
    """Result of a Pixabay video fetch + download operation."""

    topic_id: str
    video_path: str
    video_meta: PixabayVideo | None
    is_valid: bool
    validation_errors: list[str] = field(default_factory=list)
    fetched_at: str = ""

    def __post_init__(self) -> None:
        if not self.fetched_at:
            self.fetched_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic_id":          self.topic_id,
            "video_path":        self.video_path,
            "video_meta":        self.video_meta.to_dict() if self.video_meta else None,
            "is_valid":          self.is_valid,
            "validation_errors": self.validation_errors,
            "fetched_at":        self.fetched_at,
        }


# ---------------------------------------------------------------------------
# PixabayFetcher
# ---------------------------------------------------------------------------

class PixabayFetcher:
    """
    Fetches and downloads portrait-oriented stock video from Pixabay.

    Args:
        api_key: Pixabay API key. If None, reads PIXABAY_API_KEY from env.
        output_dir: Directory to save downloaded videos.
        min_duration: Minimum video duration in seconds.
    """

    def __init__(
        self,
        api_key: str | None = None,
        output_dir: str | Path = OUTPUT_DIR,
        min_duration: int = MIN_VIDEO_DURATION_SECONDS,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("PIXABAY_API_KEY", "")
        self.output_dir = Path(output_dir)
        self.min_duration = min_duration

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(
        self,
        topic_id: str,
        category: str = "default",
    ) -> FetchResult:
        """
        Search Pixabay for a suitable portrait video and download it.

        Args:
            topic_id: Unique identifier used in output filename.
            category: Topic category key used to select search keywords.

        Returns:
            FetchResult with video_path and validation status.

        Raises:
            ValueError: If PIXABAY_API_KEY is not configured.
        """
        if not self.api_key:
            raise ValueError("PIXABAY_API_KEY not set")

        keywords = KEYWORD_MAP.get(category.lower(), KEYWORD_MAP["default"])
        video = self._search_videos(keywords)

        if video is None:
            return FetchResult(
                topic_id=topic_id,
                video_path="",
                video_meta=None,
                is_valid=False,
                validation_errors=["no suitable video found on Pixabay"],
            )

        output_path = self.output_dir / f"{topic_id}_stock.mp4"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._download_video(video.download_url, output_path)
        logger.info(
            "Downloaded stock video %s → %s (%ds)",
            video.pixabay_id, output_path, video.duration,
        )

        return FetchResult(
            topic_id=topic_id,
            video_path=str(output_path),
            video_meta=video,
            is_valid=True,
        )

    def fetch_multiple(
        self,
        topic_id: str,
        keywords_list: list[str],
        count: int = 4,
    ) -> list[str]:
        """
        Fetch multiple stock videos — one per keyword phrase — and return local paths.

        Args:
            topic_id: Unique identifier used in output filenames
                      (e.g. "stoic_001" → "stoic_001_stock_0.mp4", …).
            keywords_list: Keyword phrases to search; one video per phrase.
            count: Maximum number of clips to fetch (uses first `count` phrases).

        Returns:
            List of local file paths. May be shorter than count if some
            keyword phrases return no results or duplicate video IDs.

        Raises:
            ValueError: If PIXABAY_API_KEY is not configured.
        """
        if not self.api_key:
            raise ValueError("PIXABAY_API_KEY not set")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        seen_ids: set[int] = set()

        for i, phrase in enumerate(keywords_list[:count]):
            video = self._query_api(phrase)
            if video is None:
                logger.warning("No video found for phrase=%r (clip %d)", phrase, i)
                continue
            if video.pixabay_id in seen_ids:
                logger.debug(
                    "Duplicate pixabay_id=%d skipped for phrase=%r",
                    video.pixabay_id, phrase,
                )
                continue

            seen_ids.add(video.pixabay_id)
            output_path = self.output_dir / f"{topic_id}_stock_{i}.mp4"
            self._download_video(video.download_url, output_path)
            logger.info(
                "Fetched clip %d: pixabay_id=%d → %s",
                len(paths), video.pixabay_id, output_path,
            )
            paths.append(str(output_path))

        return paths

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _search_videos(self, keywords: list[str]) -> PixabayVideo | None:
        """Try each keyword phrase in order; return first suitable video or None."""
        for phrase in keywords:
            video = self._query_api(phrase)
            if video is not None:
                return video
        return None

    def _query_api(self, query: str) -> PixabayVideo | None:
        """Query Pixabay API for portrait videos matching query."""
        params = {
            "key":         self.api_key,
            "q":           query,
            "video_type":  "film",
            "orientation": "vertical",
            "per_page":    50,
            "min_width":   1080,
        }
        try:
            resp = httpx.get(_PIXABAY_API_URL, params=params, timeout=15.0)
            resp.raise_for_status()
            hits = resp.json().get("hits", [])
        except Exception as exc:
            logger.error("Pixabay API call failed for query=%r: %s", query, exc)
            return None

        for hit in hits:
            duration = int(hit.get("duration", 0))
            if duration < self.min_duration:
                continue

            videos_dict = hit.get("videos", {})
            url, width, height = self._best_url(videos_dict)
            if not url:
                continue

            # Skip videos that don't meet the minimum width after size selection
            if width < 1080:
                continue

            return PixabayVideo(
                pixabay_id=hit.get("id", 0),
                duration=duration,
                width=width,
                height=height,
                download_url=url,
                page_url=hit.get("pageURL", ""),
                tags=hit.get("tags", ""),
            )
        return None

    @staticmethod
    def _best_url(videos_dict: dict) -> tuple[str, int, int]:
        """Pick the largest available download URL from Pixabay videos dict.

        Returns:
            (url, width, height) — all empty/zero if nothing found.
        """
        for quality in ("large", "medium", "small", "tiny"):
            entry = videos_dict.get(quality, {})
            url = entry.get("url", "")
            if url:
                return url, int(entry.get("width", 0)), int(entry.get("height", 0))
        return "", 0, 0

    @staticmethod
    def _download_video(url: str, output_path: Path) -> None:
        """Stream-download a video file to disk."""
        with httpx.stream("GET", url, timeout=60.0, follow_redirects=True) as resp:
            resp.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=65536):
                    f.write(chunk)

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
import shutil
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

# Quality filters applied to every candidate before selection
MIN_VIDEO_DURATION_SECONDS = 5          # reject very short clips
MAX_VIDEO_DURATION_SECONDS = 30         # reject unnecessarily large files
MIN_VIDEO_WIDTH = 1080                  # Full HD minimum width
MIN_VIDEO_HEIGHT = 1080                 # Accept portrait AND landscape; scoring prefers portrait
MAX_FILE_SIZE_BYTES = 40 * 1024 * 1024  # 40 MB — skip large downloads
MIN_FILE_SIZE_BYTES = 100 * 1024        # 100 KB — anything smaller is corrupt

# Composition scoring weights
_SCORE_PORTRAIT_BONUS    = 2   # width < height
_SCORE_IDEAL_RATIO_BONUS = 2   # ratio 0.5–0.6 (close to 9:16)
_SCORE_GOOD_DURATION     = 1   # duration 8–20 s
_SCORE_LANDSCAPE_PENALTY = -2  # width/height ratio > 1.5

REQUEST_TIMEOUT = 30.0                  # seconds for API calls
DOWNLOAD_TIMEOUT = 30.0                 # seconds for file downloads

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

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"{topic_id}_stock.mp4"

        keywords = KEYWORD_MAP.get(category.lower(), KEYWORD_MAP["default"])
        video = self._search_and_download(keywords, output_path)

        if video is None:
            return FetchResult(
                topic_id=topic_id,
                video_path="",
                video_meta=None,
                is_valid=False,
                validation_errors=["no suitable video found on Pixabay"],
            )

        logger.info(
            "Downloaded stock video %s -> %s (%ds)",
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

        For each phrase, all qualifying API candidates are tried in order until one
        downloads successfully (passes size and corruption checks).  If every candidate
        for a phrase fails, the last successfully downloaded clip is copied as a
        fallback so the video always has enough b-roll.

        Args:
            topic_id: Unique identifier used in output filenames
                      (e.g. "stoic_001" → "stoic_001_stock_0.mp4", …).
            keywords_list: Keyword phrases to search; one video per phrase.
            count: Maximum number of clips to fetch (uses first `count` phrases).

        Returns:
            List of local file paths. Falls back to clip duplication so the
            returned list length equals the number of phrases attempted (after
            the first successful download).

        Raises:
            ValueError: If PIXABAY_API_KEY is not configured.
        """
        if not self.api_key:
            raise ValueError("PIXABAY_API_KEY not set")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        seen_ids: set[int] = set()
        last_good_path: str | None = None

        for i, phrase in enumerate(keywords_list[:count]):
            output_path = self.output_dir / f"{topic_id}_stock_{i}.mp4"
            candidates = self._query_api(phrase)

            downloaded = False
            for video in candidates:
                if video.pixabay_id in seen_ids:
                    logger.debug(
                        "Duplicate pixabay_id=%d skipped for phrase=%r",
                        video.pixabay_id, phrase,
                    )
                    continue

                ok = self._download_verified(video.download_url, output_path)
                if ok:
                    seen_ids.add(video.pixabay_id)
                    paths.append(str(output_path))
                    last_good_path = str(output_path)
                    logger.info(
                        "Fetched clip %d: pixabay_id=%d phrase=%r -> %s",
                        len(paths), video.pixabay_id, phrase, output_path,
                    )
                    downloaded = True
                    break

            if not downloaded:
                if last_good_path is not None:
                    shutil.copy2(last_good_path, output_path)
                    paths.append(str(output_path))
                    logger.warning(
                        "No clip for phrase=%r — duplicated %s as fallback clip %d",
                        phrase, last_good_path, i,
                    )
                else:
                    logger.warning(
                        "No clip for phrase=%r and no fallback clip available yet (clip %d skipped)",
                        phrase, i,
                    )

        return paths

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _search_and_download(
        self, keywords: list[str], output_path: Path
    ) -> PixabayVideo | None:
        """Try each keyword phrase; download the first verified candidate."""
        for phrase in keywords:
            for video in self._query_api(phrase):
                if self._download_verified(video.download_url, output_path):
                    return video
        return None

    def _query_api(self, query: str) -> list[PixabayVideo]:
        """Query Pixabay API and return all qualifying portrait video candidates.

        Filters applied server-side:
          - video_type=film (real footage only, no animation)
          - orientation=vertical (portrait mode for Shorts)
          - min_width=1080
          - order=popular (highest quality score first)

        Filters applied client-side:
          - width  >= MIN_VIDEO_WIDTH  (1080)
          - height >= MIN_VIDEO_HEIGHT (1920)
          - duration in [MIN_VIDEO_DURATION_SECONDS, MAX_VIDEO_DURATION_SECONDS]

        Returns an empty list on API error.
        """
        params = {
            "key":         self.api_key,
            "q":           query,
            "video_type":  "film",
            "orientation": "vertical",
            "per_page":    50,
            "min_width":   MIN_VIDEO_WIDTH,
            "order":       "popular",
        }
        try:
            resp = httpx.get(_PIXABAY_API_URL, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            hits = resp.json().get("hits", [])
        except Exception as exc:
            logger.error("Pixabay API call failed for query=%r: %s", query, exc)
            return []

        results: list[PixabayVideo] = []
        for hit in hits:
            duration = int(hit.get("duration", 0))
            if duration < self.min_duration or duration > MAX_VIDEO_DURATION_SECONDS:
                continue

            videos_dict = hit.get("videos", {})
            url, width, height = self._best_url(videos_dict)
            if not url:
                continue

            if width < MIN_VIDEO_WIDTH or height < MIN_VIDEO_HEIGHT:
                continue

            results.append(PixabayVideo(
                pixabay_id=hit.get("id", 0),
                duration=duration,
                width=width,
                height=height,
                download_url=url,
                page_url=hit.get("pageURL", ""),
                tags=hit.get("tags", ""),
            ))

        # Sort by composition score — portrait and 9:16 clips float to the top
        results.sort(key=self._score_clip, reverse=True)
        logger.debug("Query %r → %d qualifying candidates", query, len(results))
        return results

    @staticmethod
    def _score_clip(video: "PixabayVideo") -> int:
        """Compute a composition score for a candidate clip.

        Higher score = better suited for 9:16 portrait frame.
        Landscape clips are penalised but not excluded; _fit_clip will
        centre-crop them correctly during video assembly.
        """
        score = 0
        w, h, d = video.width, video.height, video.duration

        if h > 0:
            ratio = w / h
            if w < h:                           # portrait orientation
                score += _SCORE_PORTRAIT_BONUS
            if 0.5 <= ratio <= 0.6:             # close to 9:16
                score += _SCORE_IDEAL_RATIO_BONUS
            if ratio > 1.5:                     # landscape
                score += _SCORE_LANDSCAPE_PENALTY

        if 8 <= d <= 20:
            score += _SCORE_GOOD_DURATION

        return score

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
    def _download_verified(url: str, output_path: Path) -> bool:
        """Download a video, checking size limits and file integrity.

        Checks performed:
          - Content-Length header < MAX_FILE_SIZE_BYTES (40 MB) before downloading
          - Downloaded file size > MIN_FILE_SIZE_BYTES (100 KB) after download
          - Downloaded file size < MAX_FILE_SIZE_BYTES (40 MB) after download

        Returns:
            True if the file was downloaded and passes all checks, False otherwise.
            Partial or corrupt files are deleted before returning False.
        """
        try:
            with httpx.stream(
                "GET", url, timeout=DOWNLOAD_TIMEOUT, follow_redirects=True
            ) as resp:
                resp.raise_for_status()

                # Pre-flight size check via Content-Length header
                content_length = int(resp.headers.get("content-length", 0))
                if content_length > MAX_FILE_SIZE_BYTES:
                    logger.debug(
                        "Skipping %s — Content-Length %dMB exceeds 40MB limit",
                        url, content_length // (1024 * 1024),
                    )
                    return False

                with open(output_path, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=65536):
                        f.write(chunk)

        except Exception as exc:
            logger.warning("Download failed for %s: %s", url, exc)
            output_path.unlink(missing_ok=True)
            return False

        # Post-download integrity checks
        actual_size = output_path.stat().st_size

        if actual_size < MIN_FILE_SIZE_BYTES:
            logger.warning(
                "Corrupt download (%d bytes < 100KB) — discarding: %s",
                actual_size, output_path,
            )
            output_path.unlink(missing_ok=True)
            return False

        if actual_size > MAX_FILE_SIZE_BYTES:
            logger.warning(
                "Oversized download (%dMB > 40MB) — discarding: %s",
                actual_size // (1024 * 1024), output_path,
            )
            output_path.unlink(missing_ok=True)
            return False

        return True

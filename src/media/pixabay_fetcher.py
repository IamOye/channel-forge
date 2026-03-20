"""
pixabay_fetcher.py — PixabayFetcher

Downloads royalty-free stock videos from the Pixabay API.

Usage:
    fetcher = PixabayFetcher()
    result = fetcher.fetch(topic_id="stoic_001", category="success")
    print(result.video_path)
"""

import json
import logging
import os
import random
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

# Portrait ratio constraints — clips outside this range stretch badly in 9:16 frame
MIN_PORTRAIT_RATIO = 0.50   # below this: too narrow (very tall, unusual)
MAX_PORTRAIT_RATIO = 0.62   # above this: too wide → stretched when cropped to 1080×1920

# Finance-specific fallback search queries (tried in order when primary search fails)
FALLBACK_QUERIES: list[str] = [
    "business meeting office professional",
    "money cash currency finance",
    "smartphone app technology person",
    "city skyline urban success",
    "laptop computer working desk",
    "investment chart graph growth",
    "handshake deal agreement business",
    "confident professional walking",
]

# Max seconds a single clip should contribute to the output video
MAX_CLIP_OUTPUT_SECONDS = 12

_PIXABAY_PHOTO_API_URL = "https://pixabay.com/api/"
MAX_PHOTO_PORTRAIT_RATIO = 0.65  # width/height must be below this for portrait photos

# Minimum relevance score to accept a clip (Claude scores 1–10)
# 6 for primary video clips (finance stock is limited); photos/illustrations use 7
_MIN_RELEVANCE_SCORE = 6
# Trigger additional human-focused search if fewer than this many clips pass scoring
_MIN_CLIPS_AFTER_SCORING = 4


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
        anthropic_api_key: str | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("PIXABAY_API_KEY", "")
        self.output_dir = Path(output_dir)
        self.min_duration = min_duration
        self.anthropic_api_key = (
            anthropic_api_key if anthropic_api_key is not None
            else os.getenv("ANTHROPIC_API_KEY", "")
        )

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
        topic: str = "",
        script_preview: str = "",
    ) -> list[str]:
        """
        Fetch multiple stock video clips and return local paths.

        Flow:
          1. Collect all qualifying candidates from every keyword phrase (deduped).
          2. If ``topic`` and ``anthropic_api_key`` are set, score candidates for
             relevance via Claude (FIX 2); reject below score 6.  When fewer than
             ``_MIN_CLIPS_AFTER_SCORING`` pass, one additional human-focused search
             is run.
          3. Download up to ``count`` clips.
          4. If fewer than 2 clips were downloaded, fill remaining slots from the
             finance-specific ``FALLBACK_QUERIES`` (FIX 3).

        Args:
            topic_id: Unique identifier used in output filenames.
            keywords_list: Keyword phrases to search.
            count: Maximum number of clips to return.
            topic: Video topic string used for relevance scoring (optional).
            script_preview: First ~40 words of script, used for scoring (optional).

        Returns:
            List of local file paths (may be shorter than ``count`` if fallback
            also fails).

        Raises:
            ValueError: If PIXABAY_API_KEY is not configured.
        """
        if not self.api_key:
            raise ValueError("PIXABAY_API_KEY not set")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        seen_ids: set[int] = set()

        # 1. Collect all candidates from every keyword phrase (deduped)
        all_candidates: list[PixabayVideo] = []
        for phrase in keywords_list[:count]:
            for video in self._query_api(phrase):
                if video.pixabay_id not in seen_ids:
                    all_candidates.append(video)
                    seen_ids.add(video.pixabay_id)

        # 2. Relevance scoring (FIX 2) — only if topic + Anthropic key are available
        if topic and all_candidates and self.anthropic_api_key:
            all_candidates = self.score_clip_relevance(all_candidates, topic, script_preview)
            if len(all_candidates) < _MIN_CLIPS_AFTER_SCORING:
                extra_query = f"human financial person {topic}"
                logger.info(
                    "[pixabay] Only %d clips after scoring — running extra search: %r",
                    len(all_candidates), extra_query,
                )
                for video in self._query_api(extra_query):
                    if video.pixabay_id not in seen_ids:
                        all_candidates.append(video)
                        seen_ids.add(video.pixabay_id)

        # 3. Download up to `count` clips
        paths: list[str] = []
        for video in all_candidates:
            if len(paths) >= count:
                break
            output_path = self.output_dir / f"{topic_id}_stock_{len(paths)}.mp4"
            ok = self._download_verified(video.download_url, output_path)
            if ok:
                paths.append(str(output_path))
                logger.info(
                    "Fetched clip %d: pixabay_id=%d -> %s",
                    len(paths), video.pixabay_id, output_path,
                )

        # 4. Fallback from curated library if fewer than 2 clips found (FIX 3)
        if len(paths) < 2:
            logger.warning(
                "[pixabay] Only %d clip(s) after main search — filling from fallback library",
                len(paths),
            )
            paths = self._fill_from_fallback(topic_id, paths, seen_ids, count)

        return paths

    # ------------------------------------------------------------------
    # Relevance scoring (FIX 2)
    # ------------------------------------------------------------------

    def score_clip_relevance(
        self,
        clips: list[PixabayVideo],
        topic: str,
        script_preview: str,
    ) -> list[PixabayVideo]:
        """Score clip relevance via Claude API; return only clips scoring >= 6.

        Falls back to returning all clips unchanged if the API call fails or
        ``anthropic_api_key`` is not set.

        Args:
            clips: Candidate clips to evaluate.
            topic: Video topic (e.g. "money habits").
            script_preview: First ~40 words of the script for context.

        Returns:
            Filtered list of clips with relevance score >= 6.
        """
        if not self.anthropic_api_key or not clips:
            return clips

        try:
            import anthropic  # lazy import

            client = anthropic.Anthropic(api_key=self.anthropic_api_key)
            clip_descriptions = [
                {"clip_id": v.pixabay_id, "tags": v.tags, "duration": v.duration}
                for v in clips
            ]
            prompt = (
                "Rate each stock video clip's relevance to this YouTube Short "
                "about finance/money.\n"
                "Score 1-10:\n"
                "- 1-3: Completely irrelevant (animals, nature, sports, food, children)\n"
                "- 4-5: Abstract scenes, empty rooms, generic landscapes\n"
                "- 6-7: Business/office/professional content that could plausibly "
                "accompany finance content even if not directly about money\n"
                "- 8-10: Explicitly financial content (money, charts, banks, investing, "
                "wealth, budgets, salary)\n\n"
                f"Topic: {topic}\n"
                f"Script (first 40 words): {script_preview[:200]}\n\n"
                f"Clips to rate:\n{json.dumps(clip_descriptions, indent=2)}\n\n"
                'Return JSON array only: [{"clip_id": x, "score": y, "reason": "z"}]'
            )
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            scores_data = json.loads(raw)
            score_map = {item["clip_id"]: item["score"] for item in scores_data}

            filtered = [v for v in clips if score_map.get(v.pixabay_id, 0) >= _MIN_RELEVANCE_SCORE]
            logger.info(
                "[pixabay] Relevance scoring: %d clips → %d passed (score >= %d)",
                len(clips), len(filtered), _MIN_RELEVANCE_SCORE,
            )
            return filtered

        except Exception as exc:
            logger.warning(
                "[pixabay] Relevance scoring failed (returning all clips): %s", exc
            )
            return clips

    # ------------------------------------------------------------------
    # Fallback clip library (FIX 3)
    # ------------------------------------------------------------------

    def _fill_from_fallback(
        self,
        topic_id: str,
        existing_paths: list[str],
        seen_ids: set[int],
        target_count: int,
    ) -> list[str]:
        """Fill remaining clip slots using finance-specific fallback queries.

        Searches FALLBACK_QUERIES in order until target_count is reached.
        Skips clips already downloaded (by pixabay_id).  Failures per query
        are caught and logged; the method never raises.

        Args:
            topic_id: Used to name output files.
            existing_paths: Already-downloaded clip paths.
            seen_ids: Pixabay IDs already downloaded (mutated in-place).
            target_count: Desired total number of clips.

        Returns:
            Updated paths list (may still be short if all fallback queries fail).
        """
        paths = list(existing_paths)
        need = max(2, target_count) - len(paths)
        if need <= 0:
            return paths

        for query in FALLBACK_QUERIES:
            if need <= 0:
                break
            try:
                candidates = self._query_api(query)
                for video in candidates:
                    if need <= 0:
                        break
                    if video.pixabay_id in seen_ids:
                        continue
                    output_path = self.output_dir / f"{topic_id}_stock_{len(paths)}.mp4"
                    if self._download_verified(video.download_url, output_path):
                        seen_ids.add(video.pixabay_id)
                        paths.append(str(output_path))
                        need -= 1
                        logger.info(
                            "[pixabay] Filled fallback slot %d from query=%r (clip_id=%d)",
                            len(paths) - 1, query, video.pixabay_id,
                        )
            except Exception as exc:
                logger.warning("[pixabay] Fallback query %r failed: %s", query, exc)

        return paths

    def fetch_photos(
        self,
        topic_id: str,
        phrase: str,
        count: int = 2,
    ) -> list[dict]:
        """Fetch portrait stock photos from Pixabay image API.

        Filters: portrait only (width/height < MAX_PHOTO_PORTRAIT_RATIO = 0.65).
        Downloads to output_dir/{topic_id}_photo_{n}.jpg.

        Returns:
            List of dicts with keys: id, local_path, width, height, tags.
            Returns [] on API error (never raises).

        Raises:
            ValueError: If PIXABAY_API_KEY is not configured.
        """
        if not self.api_key:
            raise ValueError("PIXABAY_API_KEY not set")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        params = {
            "key":         self.api_key,
            "q":           phrase,
            "image_type":  "photo",
            "orientation": "vertical",
            "min_width":   1080,
            "per_page":    20,
            "safesearch":  "true",
            "order":       "popular",
        }
        try:
            resp = httpx.get(_PIXABAY_PHOTO_API_URL, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            hits = resp.json().get("hits", [])
        except Exception as exc:
            logger.error("[pixabay] Photo API call failed for query=%r: %s", phrase, exc)
            return []

        results: list[dict] = []
        for hit in hits:
            if len(results) >= count:
                break
            width  = int(hit.get("imageWidth", 0))
            height = int(hit.get("imageHeight", 0))
            if width < 1080 or height == 0:
                continue
            ratio = width / height
            if ratio >= MAX_PHOTO_PORTRAIT_RATIO:
                logger.debug(
                    "[pixabay] Rejected photo %d — ratio %.2f not portrait",
                    hit.get("id", 0), ratio,
                )
                continue

            url = (
                hit.get("fullHDURL")
                or hit.get("largeImageURL")
                or hit.get("webformatURL", "")
            )
            if not url:
                continue

            idx         = len(results)
            output_path = self.output_dir / f"{topic_id}_photo_{idx}.jpg"
            if not self._download_photo(url, output_path):
                continue

            results.append({
                "id":         hit.get("id", 0),
                "local_path": str(output_path),
                "width":      width,
                "height":     height,
                "tags":       hit.get("tags", ""),
            })
            logger.info(
                "[pixabay] Downloaded photo %d: %dx%d -> %s",
                hit.get("id", 0), width, height, output_path,
            )

        return results

    def _download_photo(self, url: str, output_path: Path) -> bool:
        """Download a photo URL to output_path with retry on 429. Returns True on success."""
        import time as _time

        max_retries = 3
        for attempt in range(max_retries):
            try:
                with httpx.stream(
                    "GET", url, timeout=DOWNLOAD_TIMEOUT, follow_redirects=True
                ) as resp:
                    if resp.status_code == 429:
                        wait = 2 ** attempt
                        logger.info("[pixabay] 429 rate limit — retrying in %ds (attempt %d)", wait, attempt + 1)
                        _time.sleep(wait)
                        continue
                    resp.raise_for_status()
                    with open(output_path, "wb") as f:
                        for chunk in resp.iter_bytes(chunk_size=65536):
                            f.write(chunk)
                return output_path.stat().st_size > 1024  # at least 1 KB
            except Exception as exc:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.debug("[pixabay] Photo download attempt %d failed: %s — retrying in %ds", attempt + 1, exc, wait)
                    _time.sleep(wait)
                else:
                    logger.warning("[pixabay] Photo download failed for %s: %s", url, exc)

        output_path.unlink(missing_ok=True)
        return False

    def fetch_illustrations(
        self,
        topic_id: str,
        phrase: str,
        count: int = 2,
    ) -> list[dict]:
        """Fetch illustrations/vectors from Pixabay image API.

        Searches with image_type=illustration for financial infographics,
        charts, money symbols, and business concept art. Downloads to
        output_dir/{topic_id}_illust_{n}.jpg.

        Returns:
            List of dicts with keys: id, local_path, width, height, tags.
            Returns [] on API error (never raises).
        """
        if not self.api_key:
            raise ValueError("PIXABAY_API_KEY not set")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        params = {
            "key":         self.api_key,
            "q":           phrase,
            "image_type":  "illustration",
            "orientation": "vertical",
            "min_width":   720,
            "per_page":    20,
            "safesearch":  "true",
            "order":       "popular",
        }
        try:
            resp = httpx.get(_PIXABAY_PHOTO_API_URL, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            hits = resp.json().get("hits", [])
        except Exception as exc:
            logger.error("[pixabay] Illustration API failed for query=%r: %s", phrase, exc)
            return []

        results: list[dict] = []
        for hit in hits:
            if len(results) >= count:
                break
            width = int(hit.get("imageWidth", 0))
            height = int(hit.get("imageHeight", 0))
            if width < 720 or height == 0:
                continue

            url = (
                hit.get("fullHDURL")
                or hit.get("largeImageURL")
                or hit.get("webformatURL", "")
            )
            if not url:
                continue

            idx = len(results)
            output_path = self.output_dir / f"{topic_id}_illust_{idx}.jpg"
            if not self._download_photo(url, output_path):
                continue

            results.append({
                "id":         hit.get("id", 0),
                "local_path": str(output_path),
                "width":      width,
                "height":     height,
                "tags":       hit.get("tags", ""),
            })
            logger.info(
                "[pixabay] Downloaded illustration %d: %dx%d -> %s",
                hit.get("id", 0), width, height, output_path,
            )

        return results

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

            # FIX 1: Reject clips whose aspect ratio falls outside portrait range
            if height > 0:
                ratio = width / height
                if ratio < MIN_PORTRAIT_RATIO or ratio > MAX_PORTRAIT_RATIO:
                    logger.debug(
                        "[pixabay] Rejected clip %d — ratio %.2f too wide for portrait",
                        hit.get("id", 0), ratio,
                    )
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

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
MIN_VIDEO_WIDTH = 1920                  # Full HD portrait minimum width
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

# Aerial-first fallback search queries (tried in order when primary search fails)
FALLBACK_QUERIES: list[str] = [
    # Aerial/drone
    "aerial city skyline drone",
    "drone highway traffic moving",
    "aerial green landscape nature",
    "luxury neighbourhood aerial view",
    "city buildings skyscrapers aerial",
    "aerial coastline beach drone",
    "drone forest trees aerial",
    "aerial mountains landscape",
    "drone river city aerial",
    "aerial desert landscape drone",
    # Finance/business
    "money cash currency finance",
    "business meeting office professional",
    "stock market trading finance",
    "person walking city street",
    "coffee shop work laptop",
    "city traffic rush hour",
    "shopping mall retail people",
    "construction building crane",
    "sunset cityscape golden hour",
    "people walking downtown city",
]

# Max seconds a single clip should contribute to the output video
MAX_CLIP_OUTPUT_SECONDS = 12

# Illustration queries by category — rotated for variety across videos
ILLUSTRATION_QUERIES: dict[str, list[str]] = {
    "money": [
        "money growth chart infographic",
        "financial freedom wealth illustration",
        "investment returns graph colorful",
        "savings piggy bank infographic",
        "debt vs wealth comparison chart",
        "rich poor wealth gap infographic",
    ],
    "career": [
        "career growth ladder illustration",
        "salary negotiation infographic",
        "job market skills chart",
        "workplace productivity infographic",
        "career path roadmap illustration",
        "remote work future infographic",
    ],
    "success": [
        "success mindset infographic",
        "habits of successful people chart",
        "goal setting roadmap illustration",
        "morning routine infographic",
        "discipline vs motivation chart",
        "wealth mindset vs poverty mindset",
    ],
}

_PIXABAY_PHOTO_API_URL = "https://pixabay.com/api/"
MAX_PHOTO_PORTRAIT_RATIO = 0.65  # width/height must be below this for portrait photos

# Minimum relevance score to accept a clip (Claude scores 1–10)
# 6 for primary video clips (finance stock is limited); photos/illustrations use 7
_MIN_RELEVANCE_SCORE = 6
# Trigger additional human-focused search if fewer than this many clips pass scoring
_MIN_CLIPS_AFTER_SCORING = 4


# ---------------------------------------------------------------------------
# Clip history helpers (used by pixabay, pexels, unsplash fetchers)
# ---------------------------------------------------------------------------

_CLIP_HISTORY_DDL = """
CREATE TABLE IF NOT EXISTS clip_history (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    clip_id  TEXT NOT NULL,
    source   TEXT NOT NULL,
    query    TEXT NOT NULL DEFAULT '',
    topic_id TEXT NOT NULL DEFAULT '',
    used_at  TEXT NOT NULL DEFAULT (datetime('now'))
)
"""


def _ensure_clip_history(conn: "sqlite3.Connection") -> None:
    """Create clip_history table if it doesn't exist."""
    conn.execute(_CLIP_HISTORY_DDL)


def _clip_already_used(db_path: "Path | str | None", source: str, clip_id: str) -> bool:
    """Return True if clip_id/source already exists in clip_history."""
    if db_path is None:
        return False
    db = Path(db_path)
    if not db.exists():
        return False
    try:
        import sqlite3 as _sq
        conn = _sq.connect(db)
        _ensure_clip_history(conn)
        row = conn.execute(
            "SELECT 1 FROM clip_history WHERE source = ? AND clip_id = ? LIMIT 1",
            (source, clip_id),
        ).fetchone()
        conn.close()
        return row is not None
    except Exception:
        return False


def _clip_history_record(
    db_path: "Path | str | None",
    source: str,
    clip_id: str,
    query: str,
    topic_id: str,
) -> None:
    """Insert a row into clip_history. Swallows all errors."""
    if db_path is None:
        logger.debug("[clip_history] db_path is None — skipping record")
        return
    db = Path(db_path)
    if not db.exists():
        logger.warning("[clip_history] DB does not exist at %s — skipping", db)
        return
    try:
        import sqlite3 as _sq
        conn = _sq.connect(db)
        _ensure_clip_history(conn)
        conn.execute(
            "INSERT INTO clip_history (clip_id, source, query, topic_id) VALUES (?, ?, ?, ?)",
            (clip_id, source, query, topic_id),
        )
        conn.commit()
        conn.close()
        logger.info("[clip_history] Recorded %s clip %s → db: %s", source, clip_id, db)
    except Exception as exc:
        logger.warning("[pixabay] clip_history record failed: %s", exc)


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
        db_path: "Path | str | None" = None,
    ) -> list[str]:
        """
        Fetch multiple stock video clips and return local paths.

        Flow:
          0. Try Pexels first if PEXELS_API_KEY is set; use those clips if >= count.
          1. Collect all qualifying Pixabay candidates (deduped + clip_history filtered).
          2. If ``topic`` and ``anthropic_api_key`` are set, score candidates for
             relevance via Claude; reject below score 6.
          3. Download up to ``count`` clips; record each in clip_history.
          4. If fewer than 2 clips were downloaded, fill remaining slots from the
             finance-specific ``FALLBACK_QUERIES``.

        Args:
            topic_id: Unique identifier used in output filenames.
            keywords_list: Keyword phrases to search.
            count: Maximum number of clips to return.
            topic: Video topic string used for relevance scoring (optional).
            script_preview: First ~40 words of script, used for scoring (optional).
            db_path: SQLite DB path for clip_history deduplication (optional).

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

        # 0. Try Pexels first if PEXELS_API_KEY is configured
        pexels_paths: list[str] = []
        pexels_key = os.getenv("PEXELS_API_KEY", "")
        if pexels_key:
            try:
                from src.media.pexels_fetcher import PexelsFetcher  # lazy
                pf = PexelsFetcher(
                    api_key=pexels_key,
                    output_dir=self.output_dir,
                    anthropic_api_key=self.anthropic_api_key,
                )
                for phrase in keywords_list[:count]:
                    if len(pexels_paths) >= count:
                        break
                    clips = pf.fetch_clips(
                        query=phrase,
                        min_width=MIN_VIDEO_WIDTH,
                        max_clips=count - len(pexels_paths),
                        topic_id=topic_id,
                        db_path=db_path,
                    )
                    pexels_paths.extend(clips)
            except Exception as exc:
                logger.warning(
                    "[pixabay] Pexels fetch failed — falling through to Pixabay: %s", exc
                )

        if len(pexels_paths) >= count:
            logger.info("[pixabay] Returning %d Pexels clips (Pixabay not needed)", count)
            return pexels_paths[:count]

        # 1. Collect all Pixabay candidates (deduped + clip_history filtered)
        all_candidates: list[PixabayVideo] = []
        phrase_map: dict[int, str] = {}  # pixabay_id -> phrase for history recording
        for phrase in keywords_list[:count]:
            for video in self._query_api(phrase):
                if video.pixabay_id not in seen_ids:
                    if _clip_already_used(db_path, "pixabay", str(video.pixabay_id)):
                        logger.debug(
                            "[pixabay] Skipping already-used clip_id=%d", video.pixabay_id
                        )
                        continue
                    all_candidates.append(video)
                    seen_ids.add(video.pixabay_id)
                    phrase_map[video.pixabay_id] = phrase

        # 2. Relevance scoring — only if topic + Anthropic key are available
        if topic and all_candidates and self.anthropic_api_key:
            all_candidates = self.score_clip_relevance(all_candidates, topic, script_preview)
            random.shuffle(all_candidates)  # randomize among equally-scored clips
            if len(all_candidates) < _MIN_CLIPS_AFTER_SCORING:
                extra_query = f"human financial person {topic}"
                logger.info(
                    "[pixabay] Only %d clips after scoring — running extra search: %r",
                    len(all_candidates), extra_query,
                )
                for video in self._query_api(extra_query):
                    if video.pixabay_id not in seen_ids:
                        if not _clip_already_used(db_path, "pixabay", str(video.pixabay_id)):
                            all_candidates.append(video)
                            seen_ids.add(video.pixabay_id)
                            phrase_map[video.pixabay_id] = extra_query

        # 3. Download up to `count` clips (supplement any Pexels clips already gathered)
        paths: list[str] = list(pexels_paths)
        pixabay_count = 0
        for video in all_candidates:
            if len(paths) >= count:
                break
            output_path = self.output_dir / f"{topic_id}_stock_{pixabay_count}.mp4"
            ok = self._download_verified(video.download_url, output_path)
            if ok:
                paths.append(str(output_path))
                pixabay_count += 1
                _clip_history_record(
                    db_path, "pixabay", str(video.pixabay_id),
                    phrase_map.get(video.pixabay_id, ""), topic_id,
                )
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
            paths = self._fill_from_fallback(topic_id, paths, seen_ids, count, db_path=db_path)

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
                "Score each Pixabay clip for use in a YouTube Shorts finance video.\n"
                f"Topic: {topic}\n\n"
                "IMPORTANT: Aerial drone footage of cities, highways, coastlines, forests, "
                "neighbourhoods and skylines always scores 8-10 regardless of the specific "
                "financial topic — these are pre-approved visual metaphors for wealth, "
                "ambition and scale. Only score below 6 for: food, animals, cartoons, "
                "abstract art, or clearly unrelated content.\n\n"
                "Scoring guide:\n"
                "8-10: Cinematic aerial/drone footage of cities, highways, luxury areas, "
                "coastlines, landscapes. Any high quality aerial shot.\n"
                "8-10: Crisp financial imagery (money, charts, banks, professional settings).\n"
                "6-7:  Office workers, business people, professional settings.\n"
                "4-5:  Generic street level footage, people walking, non-specific scenes.\n"
                "1-3:  Animals, food, sports, nature closeups, children, anything unrelated "
                "to aspirational/financial context.\n\n"
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
        db_path: "Path | str | None" = None,
    ) -> list[str]:
        """Fill remaining clip slots using finance-specific fallback queries.

        Searches FALLBACK_QUERIES in order until target_count is reached.
        Skips clips already downloaded (by pixabay_id) or in clip_history.
        Failures per query are caught and logged; the method never raises.

        Args:
            topic_id: Used to name output files.
            existing_paths: Already-downloaded clip paths.
            seen_ids: Pixabay IDs already downloaded (mutated in-place).
            target_count: Desired total number of clips.
            db_path: SQLite DB path for clip_history deduplication (optional).

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
                    if _clip_already_used(db_path, "pixabay", str(video.pixabay_id)):
                        continue
                    output_path = self.output_dir / f"{topic_id}_stock_{len(paths)}.mp4"
                    if self._download_verified(video.download_url, output_path):
                        seen_ids.add(video.pixabay_id)
                        paths.append(str(output_path))
                        need -= 1
                        _clip_history_record(
                            db_path, "pixabay", str(video.pixabay_id), query, topic_id
                        )
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
        db_path: "Path | str | None" = None,
    ) -> list[dict]:
        """Fetch portrait stock photos — tries Unsplash first, then Pixabay.

        Filters: portrait only (width/height < MAX_PHOTO_PORTRAIT_RATIO = 0.65).
        Downloads to output_dir/{topic_id}_photo_{n}.jpg (Pixabay) or
        output_dir/{topic_id}_unsplash_{n}.jpg (Unsplash).

        Returns:
            List of dicts with keys: id, local_path, width, height, tags.
            Returns [] on API error (never raises).

        Raises:
            ValueError: If PIXABAY_API_KEY is not configured.
        """
        # Try Unsplash first if UNSPLASH_ACCESS_KEY is configured
        unsplash_key = os.getenv("UNSPLASH_ACCESS_KEY", "")
        if unsplash_key:
            try:
                from src.media.unsplash_fetcher import UnsplashFetcher  # lazy
                uf = UnsplashFetcher(access_key=unsplash_key, output_dir=self.output_dir)
                photos = uf.fetch_photos(
                    query=phrase, min_width=1080, topic_id=topic_id,
                    db_path=db_path, count=count,
                )
                if photos:
                    logger.info(
                        "[pixabay] Using %d Unsplash photo(s) for topic=%s", len(photos), topic_id
                    )
                    return photos
            except Exception as exc:
                logger.warning(
                    "[pixabay] Unsplash fetch failed — falling through to Pixabay: %s", exc
                )

        if not self.api_key:
            raise ValueError("PIXABAY_API_KEY not set")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        params = {
            "key":         self.api_key,
            "q":           phrase,
            "image_type":  "photo",
            "orientation": "vertical",
            "min_width":   1920,
            "per_page":    20,
            "safesearch":  "true",
            "order":       "popular",
            "page":        random.randint(1, 4),
        }
        try:
            resp = httpx.get(_PIXABAY_PHOTO_API_URL, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            hits = resp.json().get("hits", [])
        except Exception as exc:
            logger.error("[pixabay] Photo API call failed for query=%r: %s", phrase, exc)
            return []

        random.shuffle(hits)  # randomize selection among qualifying photos
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
            "image_type":  "vector",
            "orientation": "vertical",
            "min_width":   1280,
            "per_page":    50,
            "safesearch":  "true",
            "order":       "popular",
            "page":        random.randint(1, 4),
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
        # Boost quality for aerial/drone queries
        query_lower = query.lower()
        is_aerial = any(w in query_lower for w in ("aerial", "drone", "skyline"))

        params = {
            "key":         self.api_key,
            "q":           query,
            "video_type":  "film",
            "orientation": "vertical",
            "per_page":    50,
            "min_width":   MIN_VIDEO_WIDTH,
            "order":       "popular",
            "page":        random.randint(1, 4),
        }
        if is_aerial:
            params["category"] = "travel"
            params["editors_choice"] = "true"
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

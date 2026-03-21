"""
pexels_fetcher.py — PexelsFetcher

Downloads royalty-free portrait stock video and photos from the Pexels API.
Used as the primary b-roll source; pixabay_fetcher falls back to Pixabay if
Pexels returns fewer clips than required.

API docs: https://www.pexels.com/api/documentation/
"""

import json
import logging
import os
import random
import sqlite3
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_PEXELS_VIDEO_API_URL = "https://api.pexels.com/videos/search"
_PEXELS_PHOTO_API_URL = "https://api.pexels.com/v1/search"

OUTPUT_DIR = Path("data/raw")
MIN_VIDEO_WIDTH = 1080
REQUEST_TIMEOUT = 30.0
DOWNLOAD_TIMEOUT = 30.0
MAX_FILE_SIZE_BYTES = 40 * 1024 * 1024
MIN_FILE_SIZE_BYTES = 100 * 1024
_MIN_RELEVANCE_SCORE = 6
MAX_PHOTO_PORTRAIT_RATIO = 0.65  # width/height — reject landscape/square photos


# ---------------------------------------------------------------------------
# Clip history helpers
# ---------------------------------------------------------------------------

def _clip_already_used(db_path: "Path | str | None", source: str, clip_id: str) -> bool:
    """Return True if clip_id/source already exists in clip_history."""
    if db_path is None:
        return False
    db = Path(db_path)
    if not db.exists():
        return False
    try:
        conn = sqlite3.connect(db)
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
        return
    db = Path(db_path)
    if not db.exists():
        return
    try:
        conn = sqlite3.connect(db)
        conn.execute(
            "INSERT INTO clip_history (clip_id, source, query, topic_id) VALUES (?, ?, ?, ?)",
            (clip_id, source, query, topic_id),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.warning("[pexels] clip_history record failed: %s", exc)


# ---------------------------------------------------------------------------
# PexelsFetcher
# ---------------------------------------------------------------------------

class PexelsFetcher:
    """
    Fetches portrait-oriented stock video and photos from the Pexels API.

    Args:
        api_key: Pexels API key. If None, reads PEXELS_API_KEY from env.
        output_dir: Directory to save downloaded files.
        anthropic_api_key: Used for Claude relevance scoring. If None, reads from env.
    """

    def __init__(
        self,
        api_key: str | None = None,
        output_dir: str | Path = OUTPUT_DIR,
        anthropic_api_key: str | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("PEXELS_API_KEY", "")
        self.output_dir = Path(output_dir)
        self.anthropic_api_key = (
            anthropic_api_key if anthropic_api_key is not None
            else os.getenv("ANTHROPIC_API_KEY", "")
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_clips(
        self,
        query: str,
        min_width: int = MIN_VIDEO_WIDTH,
        max_clips: int = 4,
        topic_id: str = "pexels",
        db_path: "Path | str | None" = None,
    ) -> list[str]:
        """
        Search Pexels for portrait video clips and download them.

        Flow:
          1. Search with portrait + large filters.
          2. Filter for vertical video files (height > width, width >= min_width).
          3. Skip clip_ids already in clip_history (source='pexels').
          4. Score relevance via Claude (if anthropic_api_key is set).
          5. Download passing clips to data/raw/{topic_id}_pexels_{N}.mp4.
          6. Record each downloaded clip_id in clip_history.

        Returns:
            List of local file paths. Returns [] if no API key or no results.
        """
        if not self.api_key:
            logger.debug("[pexels] No PEXELS_API_KEY — skipping video fetch")
            return []

        self.output_dir.mkdir(parents=True, exist_ok=True)

        page = random.randint(1, 4)
        try:
            resp = httpx.get(
                _PEXELS_VIDEO_API_URL,
                headers={"Authorization": self.api_key},
                params={
                    "query":       query,
                    "orientation": "portrait",
                    "size":        "large",
                    "per_page":    40,
                    "page":        page,
                },
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            videos = resp.json().get("videos", [])
        except Exception as exc:
            logger.warning("[pexels] Video search failed for query=%r: %s", query, exc)
            return []

        # Build candidate list
        candidates: list[dict] = []
        for vid in videos:
            clip_id = str(vid.get("id", 0))
            if _clip_already_used(db_path, "pexels", clip_id):
                logger.debug("[pexels] Skipping already-used clip_id=%s", clip_id)
                continue
            video_file = self._best_video_file(vid.get("video_files", []), min_width)
            if video_file is None:
                continue
            candidates.append({
                "clip_id":  clip_id,
                "url":      video_file["link"],
                "width":    video_file["width"],
                "height":   video_file["height"],
                "duration": vid.get("duration", 0),
                "tags":     " ".join(t.get("title", "") for t in vid.get("tags", [])),
            })

        if not candidates:
            logger.info("[pexels] No portrait candidates for query=%r", query)
            return []

        # Relevance scoring via Claude (optional)
        if self.anthropic_api_key:
            candidates = self._score_relevance(candidates, query)

        random.shuffle(candidates)  # randomize among equally-qualified clips

        # Download
        paths: list[str] = []
        for cand in candidates:
            if len(paths) >= max_clips:
                break
            out = self.output_dir / f"{topic_id}_pexels_{len(paths)}.mp4"
            if self._download_file(cand["url"], out):
                _clip_history_record(db_path, "pexels", cand["clip_id"], query, topic_id)
                paths.append(str(out))
                logger.info("[pexels] Downloaded clip %s -> %s", cand["clip_id"], out)

        return paths

    def fetch_photos(
        self,
        query: str,
        min_width: int = 1080,
        topic_id: str = "pexels",
        db_path: "Path | str | None" = None,
        count: int = 2,
    ) -> list[dict]:
        """
        Search Pexels for portrait photos and download them.

        Returns:
            List of dicts with keys: id, local_path, width, height, tags.
            Returns [] if no API key or on error.
        """
        if not self.api_key:
            logger.debug("[pexels] No PEXELS_API_KEY — skipping photo fetch")
            return []

        self.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            resp = httpx.get(
                _PEXELS_PHOTO_API_URL,
                headers={"Authorization": self.api_key},
                params={
                    "query":       query,
                    "orientation": "portrait",
                    "per_page":    20,
                },
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            photos = resp.json().get("photos", [])
        except Exception as exc:
            logger.warning("[pexels] Photo search failed for query=%r: %s", query, exc)
            return []

        results: list[dict] = []
        for photo in photos:
            if len(results) >= count:
                break
            clip_id = str(photo.get("id", 0))
            if _clip_already_used(db_path, "pexels_photo", clip_id):
                continue
            w = photo.get("width", 0)
            h = photo.get("height", 0)
            if w < min_width or h == 0 or (w / h) >= MAX_PHOTO_PORTRAIT_RATIO:
                continue
            url = (
                photo.get("src", {}).get("original", "")
                or photo.get("src", {}).get("large2x", "")
            )
            if not url:
                continue
            out = self.output_dir / f"{topic_id}_pexels_photo_{len(results)}.jpg"
            if not self._download_file(url, out):
                continue
            _clip_history_record(db_path, "pexels_photo", clip_id, query, topic_id)
            results.append({
                "id":         int(clip_id),
                "local_path": str(out),
                "width":      w,
                "height":     h,
                "tags":       photo.get("alt", ""),
            })
            logger.info("[pexels] Downloaded photo %s -> %s", clip_id, out)

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _best_video_file(video_files: list[dict], min_width: int) -> dict | None:
        """Pick the best portrait video file (height > width, width >= min_width).

        Returns the highest-resolution qualifying file, or None if none qualify.
        """
        portrait = [
            f for f in video_files
            if f.get("height", 0) > f.get("width", 0)
            and f.get("width", 0) >= min_width
        ]
        if not portrait:
            return None
        return max(portrait, key=lambda f: f.get("width", 0))

    def _score_relevance(self, candidates: list[dict], query: str) -> list[dict]:
        """Score candidates with Claude; return only those >= _MIN_RELEVANCE_SCORE.

        Falls back to returning all candidates unchanged on error.
        """
        if not candidates or not self.anthropic_api_key:
            return candidates
        try:
            import anthropic  # lazy
            client = anthropic.Anthropic(api_key=self.anthropic_api_key)
            clip_descriptions = [
                {"clip_id": c["clip_id"], "tags": c["tags"], "duration": c["duration"]}
                for c in candidates
            ]
            prompt = (
                "Score each Pexels clip for use in a YouTube Shorts finance video.\n"
                f"Topic: {query}\n\n"
                "IMPORTANT: Aerial drone footage of cities, highways, coastlines, forests, "
                "neighbourhoods and skylines always scores 8-10. Only score below 6 for: "
                "food, animals, cartoons, abstract art, or clearly unrelated content.\n\n"
                "Scoring guide:\n"
                "8-10: Cinematic aerial/drone footage of cities, highways, luxury areas, "
                "coastlines, landscapes.\n"
                "8-10: Crisp financial imagery (money, charts, banks, professional settings).\n"
                "6-7:  Office workers, business people, professional settings.\n"
                "4-5:  Generic street level footage, non-specific scenes.\n"
                "1-3:  Animals, food, sports, children, anything unrelated.\n\n"
                f"Clips to rate:\n{json.dumps(clip_descriptions, indent=2)}\n\n"
                'Return JSON array only: [{"clip_id": "x", "score": y, "reason": "z"}]'
            )
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
            raw = raw.strip()
            scores_data = json.loads(raw)
            score_map = {str(item["clip_id"]): item["score"] for item in scores_data}
            filtered = [
                c for c in candidates
                if score_map.get(c["clip_id"], 0) >= _MIN_RELEVANCE_SCORE
            ]
            logger.info(
                "[pexels] Relevance scoring: %d clips → %d passed",
                len(candidates), len(filtered),
            )
            return filtered
        except json.JSONDecodeError as e:
            logger.warning("[pexels] Relevance scoring JSON parse failed: %s | raw: %.200s", e, raw)
            return candidates
        except Exception as exc:
            logger.warning("[pexels] Relevance scoring failed (returning all): %s", exc)
            return candidates

    @staticmethod
    def _download_file(url: str, output_path: Path) -> bool:
        """Download url to output_path. Returns True on success."""
        try:
            with httpx.stream(
                "GET", url, timeout=DOWNLOAD_TIMEOUT, follow_redirects=True
            ) as resp:
                resp.raise_for_status()
                content_length = int(resp.headers.get("content-length", 0))
                if content_length > MAX_FILE_SIZE_BYTES:
                    return False
                with open(output_path, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=65536):
                        f.write(chunk)
            size = output_path.stat().st_size
            if size < MIN_FILE_SIZE_BYTES or size > MAX_FILE_SIZE_BYTES:
                output_path.unlink(missing_ok=True)
                return False
            return True
        except Exception as exc:
            logger.warning("[pexels] Download failed for %s: %s", url, exc)
            output_path.unlink(missing_ok=True)
            return False

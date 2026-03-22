"""
unsplash_fetcher.py — UnsplashFetcher

Downloads royalty-free portrait images from the Unsplash API.
Used as the primary still-image source; pixabay_fetcher falls back to
Pixabay photos if Unsplash returns nothing or UNSPLASH_ACCESS_KEY is unset.

API docs: https://unsplash.com/documentation
Guideline: call photo.links.download_location for every image downloaded
           (required by Unsplash API terms — tracks stats, no file needed).
"""

import logging
import os
import sqlite3
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_UNSPLASH_SEARCH_URL = "https://api.unsplash.com/search/photos"

OUTPUT_DIR = Path("data/raw")
REQUEST_TIMEOUT = 30.0
DOWNLOAD_TIMEOUT = 30.0
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024
MIN_FILE_SIZE_BYTES = 50 * 1024


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
        logger.info("[clip_history] Recorded %s clip %s → db: %s", source, clip_id, db)
    except Exception as exc:
        logger.warning("[unsplash] clip_history record failed: %s", exc)


# ---------------------------------------------------------------------------
# UnsplashFetcher
# ---------------------------------------------------------------------------

class UnsplashFetcher:
    """
    Fetches portrait photos from the Unsplash API.

    Args:
        access_key: Unsplash access key. If None, reads UNSPLASH_ACCESS_KEY from env.
        output_dir: Directory to save downloaded files.
    """

    def __init__(
        self,
        access_key: str | None = None,
        output_dir: str | Path = OUTPUT_DIR,
    ) -> None:
        self.access_key = (
            access_key if access_key is not None
            else os.getenv("UNSPLASH_ACCESS_KEY", "")
        )
        self.output_dir = Path(output_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_photos(
        self,
        query: str,
        min_width: int = 1080,
        topic_id: str = "unsplash",
        db_path: "Path | str | None" = None,
        count: int = 2,
    ) -> list[dict]:
        """
        Search Unsplash for portrait photos and download them.

        Flow:
          1. Search with orientation=portrait.
          2. Skip landscape/square results (width >= height).
          3. Skip clip_ids already in clip_history (source='unsplash').
          4. Build portrait-cropped URL (?w=1080&h=1920&fit=crop&q=85).
          5. Fire-and-forget GET to photo.links.download_location (API requirement).
          6. Download to data/raw/{topic_id}_unsplash_{N}.jpg.
          7. Record in clip_history.

        Returns:
            List of dicts with keys: id, local_path, width, height, tags.
            Returns [] if no access key or on error.
        """
        if not self.access_key:
            logger.debug("[unsplash] No UNSPLASH_ACCESS_KEY — skipping")
            return []

        self.output_dir.mkdir(parents=True, exist_ok=True)
        headers = {"Authorization": f"Client-ID {self.access_key}"}

        try:
            resp = httpx.get(
                _UNSPLASH_SEARCH_URL,
                headers=headers,
                params={
                    "query":       query,
                    "orientation": "portrait",
                    "per_page":    20,
                },
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            photos = resp.json().get("results", [])
        except Exception as exc:
            logger.warning("[unsplash] Photo search failed for query=%r: %s", query, exc)
            return []

        results: list[dict] = []
        for photo in photos:
            if len(results) >= count:
                break

            clip_id = str(photo.get("id", ""))
            if not clip_id:
                continue

            if _clip_already_used(db_path, "unsplash", clip_id):
                logger.debug("[unsplash] Skipping already-used clip_id=%s", clip_id)
                continue

            w = photo.get("width", 0)
            h = photo.get("height", 0)
            if w < min_width or h == 0 or w >= h:
                continue  # reject landscape or square

            raw_url = photo.get("urls", {}).get("raw", "")
            if not raw_url:
                continue
            url = f"{raw_url}&w=1080&h=1920&fit=crop&q=85"

            # Fire-and-forget download tracking (required by Unsplash API guidelines)
            dl_location = photo.get("links", {}).get("download_location", "")
            if dl_location:
                try:
                    httpx.get(dl_location, headers=headers, timeout=5.0)
                except Exception:
                    pass  # stats tracking — failure is not fatal

            out = self.output_dir / f"{topic_id}_unsplash_{len(results)}.jpg"
            if not self._download_file(url, out):
                continue

            _clip_history_record(db_path, "unsplash", clip_id, query, topic_id)
            results.append({
                "id":         clip_id,
                "local_path": str(out),
                "width":      1080,
                "height":     1920,
                "tags":       (
                    photo.get("alt_description", "")
                    or photo.get("description", "")
                    or ""
                ),
            })
            logger.info("[unsplash] Downloaded photo %s -> %s", clip_id, out)

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _download_file(url: str, output_path: Path) -> bool:
        """Download url to output_path. Returns True on success."""
        try:
            with httpx.stream(
                "GET", url, timeout=DOWNLOAD_TIMEOUT, follow_redirects=True
            ) as resp:
                resp.raise_for_status()
                with open(output_path, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=65536):
                        f.write(chunk)
            size = output_path.stat().st_size
            if size < MIN_FILE_SIZE_BYTES or size > MAX_FILE_SIZE_BYTES:
                output_path.unlink(missing_ok=True)
                return False
            return True
        except Exception as exc:
            logger.warning("[unsplash] Download failed for %s: %s", url, exc)
            output_path.unlink(missing_ok=True)
            return False

"""
youtube_uploader.py — YouTubeUploader + QuotaTracker

Uploads MP4 videos to YouTube using the Data API v3 with resumable uploads.
OAuth credentials are loaded from .credentials/{channel_key}_token.json

Quota tracking:
  - Video upload:     1600 units
  - Thumbnail upload:   50 units
  - Metadata update:    50 units
  Daily limit (default 10 000) read from YOUTUBE_DAILY_QUOTA_LIMIT env var.
  At 80 % → WARNING log.
  At 95 % → CRITICAL log.
  At 100 % → upload is skipped; payload queued to data/output/quota_queue/.

Usage:
    uploader = YouTubeUploader(channel_key="main")
    result = uploader.upload(
        topic_id="stoic_001",
        video_path="data/output/stoic_001_final.mp4",
        metadata={
            "title": "Stoic Secret Most People Ignore",
            "description": "Learn the ancient stoic method. Comment below 👇",
            "tags": ["#Shorts", "#Stoicism"],
            "category_id": "22",
        },
        publish_at="2025-01-01T08:00:00+01:00",  # optional
    )
    print(result.youtube_video_id)
"""

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CREDENTIALS_DIR = Path(".credentials")
DEFAULT_CATEGORY_ID = "22"          # People & Blogs
CHUNK_SIZE = 5 * 1024 * 1024        # 5 MB resumable upload chunks
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5
YOUTUBE_UPLOAD_SCOPE = "https://www.googleapis.com/auth/youtube.upload"

# HTTP status codes that warrant a retry
_RETRYABLE_STATUSES = (500, 502, 503, 504)

# Quota costs per operation (YouTube Data API v3 units)
QUOTA_UNITS: dict[str, int] = {
    "video_upload":     1600,
    "thumbnail_upload":   50,
    "metadata_update":    50,
}

DEFAULT_DAILY_QUOTA  = 10_000
_WARN_THRESHOLD      = 0.80   # 80 %
_CRIT_THRESHOLD      = 0.95   # 95 %
_DEFAULT_DB_PATH     = Path("data/processed/channel_forge.db")
_QUOTA_QUEUE_DIR     = Path("data/output/quota_queue")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class UploadResult:
    """Result of a YouTubeUploader.upload() call."""

    topic_id: str
    youtube_video_id: str
    youtube_url: str
    title: str
    is_valid: bool
    validation_errors: list[str] = field(default_factory=list)
    uploaded_at: str = ""
    publish_at: str = ""    # ISO 8601 scheduled publish time, or "" if immediate

    def __post_init__(self) -> None:
        if not self.uploaded_at:
            self.uploaded_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic_id":          self.topic_id,
            "youtube_video_id":  self.youtube_video_id,
            "youtube_url":       self.youtube_url,
            "title":             self.title,
            "is_valid":          self.is_valid,
            "validation_errors": self.validation_errors,
            "uploaded_at":       self.uploaded_at,
            "publish_at":        self.publish_at,
        }


# ---------------------------------------------------------------------------
# QuotaTracker
# ---------------------------------------------------------------------------

class QuotaTracker:
    """
    Tracks YouTube Data API v3 quota usage in the channel_forge.db database.

    Args:
        db_path:     Path to channel_forge.db. Defaults to data/processed/channel_forge.db.
        daily_limit: Max units per day. Defaults to YOUTUBE_DAILY_QUOTA_LIMIT env var
                     or 10 000.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        daily_limit: int | None = None,
    ) -> None:
        self.db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self.daily_limit = (
            daily_limit
            if daily_limit is not None
            else int(os.getenv("YOUTUBE_DAILY_QUOTA_LIMIT", str(DEFAULT_DAILY_QUOTA)))
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_daily_usage(self, date: str | None = None) -> int:
        """Return total units used today (UTC). Returns 0 if DB/table absent."""
        date = date or self._today()
        if not self.db_path.exists():
            return 0
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                row = conn.execute(
                    "SELECT COALESCE(SUM(units_used), 0) FROM youtube_quota_usage WHERE date = ?",
                    (date,),
                ).fetchone()
                return int(row[0]) if row else 0
            finally:
                conn.close()
        except sqlite3.OperationalError:
            return 0  # table doesn't exist yet

    def can_upload(self) -> bool:
        """Return True if daily quota has not been reached."""
        return self.get_daily_usage() < self.daily_limit

    def units_remaining(self) -> int:
        """Return units available for the rest of today."""
        return max(0, self.daily_limit - self.get_daily_usage())

    def record(self, operation: str, units: int) -> int:
        """
        Insert a quota usage row and return the new cumulative daily total.

        Emits WARNING at ≥ 80 % and CRITICAL at ≥ 95 % of the daily limit.
        DB failures are logged and swallowed — never propagate.
        """
        date = self._today()
        current  = self.get_daily_usage(date)
        new_cumulative = current + units

        if self.db_path.exists():
            try:
                conn = sqlite3.connect(self.db_path)
                try:
                    conn.execute(
                        """INSERT INTO youtube_quota_usage
                               (date, operation, units_used, cumulative_daily)
                           VALUES (?, ?, ?, ?)""",
                        (date, operation, units, new_cumulative),
                    )
                    conn.commit()
                finally:
                    conn.close()
            except Exception as exc:
                logger.warning("QuotaTracker.record failed: %s", exc)
        else:
            logger.warning(
                "QuotaTracker: DB not found at %s — quota row not saved", self.db_path
            )

        # Threshold alerts
        if self.daily_limit > 0:
            pct = new_cumulative / self.daily_limit
            remaining = max(0, self.daily_limit - new_cumulative)
            if pct >= _CRIT_THRESHOLD:
                logger.critical(
                    "YouTube quota critical — pausing uploads until reset at "
                    "midnight Pacific / 08:00 WAT"
                )
            elif pct >= _WARN_THRESHOLD:
                logger.warning(
                    "YouTube quota at 80%% — %d units remaining today", remaining
                )

        return new_cumulative

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# YouTubeUploader
# ---------------------------------------------------------------------------

class YouTubeUploader:
    """
    Uploads MP4 files to YouTube using the Data API v3.

    Google API client libraries are lazy-imported so the module loads cleanly
    even when google-api-python-client is not installed.

    Args:
        channel_key:      Selects which credential file to load.
                          File path: {credentials_dir}/{channel_key}_token.json
        credentials_dir:  Directory containing OAuth token JSON files.
        quota_tracker:    QuotaTracker instance. A default one is created if None.
    """

    def __init__(
        self,
        channel_key: str = "default",
        credentials_dir: str | Path = CREDENTIALS_DIR,
        quota_tracker: QuotaTracker | None = None,
    ) -> None:
        self.channel_key = channel_key
        self.credentials_dir = Path(credentials_dir)
        self.quota_tracker = quota_tracker if quota_tracker is not None else QuotaTracker()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upload(
        self,
        topic_id: str,
        video_path: str | Path,
        metadata: dict[str, Any],
        publish_at: str | None = None,
        thumbnail_path: str | Path = "",
    ) -> UploadResult:
        """
        Upload a video file to YouTube.

        If the daily quota is exhausted the video is NOT uploaded — the
        payload is saved to data/output/quota_queue/ for next-day retry and
        an UploadResult with is_valid=False is returned.

        Args:
            topic_id:        Unique topic identifier (used for logging).
            video_path:      Path to the MP4 file to upload.
            metadata:        Dict with keys: title, description, tags, category_id.
            publish_at:      Optional ISO 8601 datetime string for scheduled publish.
                             When set, video is uploaded as private and scheduled.
            thumbnail_path:  Optional path to a JPEG thumbnail.

        Returns:
            UploadResult with youtube_video_id on success, is_valid=False on error.
        """
        video_path = Path(video_path)

        # --- Quota gate ---
        if not self.quota_tracker.can_upload():
            self._queue_for_next_day(topic_id, video_path, metadata, publish_at, thumbnail_path)
            return UploadResult(
                topic_id=topic_id,
                youtube_video_id="",
                youtube_url="",
                title=metadata.get("title", ""),
                is_valid=False,
                validation_errors=["quota exceeded: video queued for next day"],
                publish_at=publish_at or "",
            )

        errors = self._validate_inputs(video_path, metadata)
        if errors:
            return UploadResult(
                topic_id=topic_id,
                youtube_video_id="",
                youtube_url="",
                title=metadata.get("title", ""),
                is_valid=False,
                validation_errors=errors,
                publish_at=publish_at or "",
            )

        logger.info("Uploading video for topic_id=%s title='%s'", topic_id, metadata.get("title", ""))

        try:
            credentials = self._load_credentials()
            service = self._build_service(credentials)
            body = self._build_body(metadata, publish_at)
            video_id = self._execute_upload(service, body, video_path)
            self.quota_tracker.record("video_upload", QUOTA_UNITS["video_upload"])

            url = f"https://www.youtube.com/watch?v={video_id}"

            # Attempt thumbnail upload if provided
            if thumbnail_path:
                self._upload_thumbnail(service, video_id, Path(thumbnail_path))
                self.quota_tracker.record("thumbnail_upload", QUOTA_UNITS["thumbnail_upload"])

            logger.info("Upload complete: topic_id=%s -> %s", topic_id, url)
            return UploadResult(
                topic_id=topic_id,
                youtube_video_id=video_id,
                youtube_url=url,
                title=metadata.get("title", ""),
                is_valid=True,
                publish_at=publish_at or "",
            )

        except Exception as exc:
            logger.error("Upload failed for topic_id=%s: %s", topic_id, exc)
            return UploadResult(
                topic_id=topic_id,
                youtube_video_id="",
                youtube_url="",
                title=metadata.get("title", ""),
                is_valid=False,
                validation_errors=[str(exc)],
                publish_at=publish_at or "",
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_credentials(self):
        """Load OAuth2 credentials from the token file for this channel.

        Lookup order:
          1. .credentials/{channel_key}_token.json  (channel-specific)
          2. .credentials/default_token.json        (fallback)
        """
        from google.oauth2.credentials import Credentials  # lazy import

        token_path = self.credentials_dir / f"{self.channel_key}_token.json"
        logger.info("Loading credentials: checking %s", token_path.resolve())
        if not token_path.exists():
            fallback = self.credentials_dir / "default_token.json"
            if fallback.exists():
                logger.debug(
                    "Channel token not found (%s) — falling back to %s",
                    token_path.name, fallback.name,
                )
                token_path = fallback
            else:
                raise FileNotFoundError(
                    f"Credentials file not found: {token_path} "
                    f"(also tried {fallback})"
                )

        with open(token_path) as f:
            data = json.load(f)

        return Credentials(
            token=data.get("token"),
            refresh_token=data.get("refresh_token"),
            token_uri=data.get("token_uri", "https://oauth2.googleapis.com/token"),
            client_id=data.get("client_id"),
            client_secret=data.get("client_secret"),
            scopes=data.get("scopes", [YOUTUBE_UPLOAD_SCOPE]),
        )

    def _build_service(self, credentials):
        """Build the YouTube Data API v3 service resource."""
        from googleapiclient.discovery import build  # lazy import
        return build("youtube", "v3", credentials=credentials)

    def _build_body(self, metadata: dict[str, Any], publish_at: str | None) -> dict:
        """Construct the videos.insert request body."""
        status: dict[str, Any] = {
            "privacyStatus": "private" if publish_at else "public",
        }
        if publish_at:
            status["publishAt"] = publish_at

        return {
            "snippet": {
                "title":       metadata.get("title", ""),
                "description": metadata.get("description", ""),
                "tags":        metadata.get("tags", []),
                "categoryId":  metadata.get("category_id", DEFAULT_CATEGORY_ID),
            },
            "status": status,
        }

    def _execute_upload(self, service, body: dict, video_path: Path) -> str:
        """
        Execute a resumable upload in 5 MB chunks with retry on transient errors.

        Returns:
            YouTube video ID string on success.

        Raises:
            HttpError: On quota exceeded (403) or unrecoverable errors.
            RuntimeError: If MAX_RETRIES is exhausted on transient errors.
        """
        from googleapiclient.errors import HttpError    # lazy import
        from googleapiclient.http import MediaFileUpload  # lazy import

        media = MediaFileUpload(
            str(video_path),
            mimetype="video/mp4",
            resumable=True,
            chunksize=CHUNK_SIZE,
        )

        request = service.videos().insert(
            part="snippet,status",
            body=body,
            media_body=media,
        )

        response = None
        retries = 0

        while response is None:
            try:
                _, response = request.next_chunk()
            except HttpError as exc:
                status_code = exc.resp.status
                if status_code in _RETRYABLE_STATUSES:
                    retries += 1
                    if retries > MAX_RETRIES:
                        raise RuntimeError(
                            f"Upload failed after {MAX_RETRIES} retries (last: HTTP {status_code})"
                        ) from exc
                    wait = RETRY_DELAY_SECONDS * retries
                    logger.warning(
                        "Transient upload error %s — retry %d/%d in %ds",
                        status_code, retries, MAX_RETRIES, wait,
                    )
                    time.sleep(wait)
                elif status_code == 403:
                    logger.error("YouTube quota/permission error (403): %s", exc)
                    raise
                else:
                    raise

        return response["id"]

    def _upload_thumbnail(self, service, video_id: str, thumb_path: Path) -> None:
        """Upload a custom thumbnail. Logs and swallows errors silently."""
        try:
            from googleapiclient.http import MediaFileUpload  # lazy
            media = MediaFileUpload(str(thumb_path), mimetype="image/jpeg")
            service.thumbnails().set(
                videoId=video_id,
                media_body=media,
            ).execute()
            logger.info("Thumbnail uploaded for video_id=%s", video_id)
        except Exception as exc:
            logger.warning(
                "Thumbnail saved locally — verify channel at youtube.com/verify to enable. "
                "Error: %s", exc,
            )

    def _queue_for_next_day(
        self,
        topic_id: str,
        video_path: Path,
        metadata: dict[str, Any],
        publish_at: str | None,
        thumbnail_path: str | Path,
    ) -> None:
        """Save upload payload to quota_queue/ for next-day retry."""
        _QUOTA_QUEUE_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        payload = {
            "topic_id":       topic_id,
            "video_path":     str(video_path),
            "metadata":       metadata,
            "publish_at":     publish_at or "",
            "thumbnail_path": str(thumbnail_path),
            "queued_at":      datetime.now(timezone.utc).isoformat(),
            "channel_key":    self.channel_key,
        }
        queue_file = _QUOTA_QUEUE_DIR / f"{topic_id}_{ts}.json"
        queue_file.write_text(json.dumps(payload, indent=2))
        logger.warning(
            "YouTube quota exceeded — upload queued for next day: %s", queue_file
        )

    @staticmethod
    def _validate_inputs(video_path: Path, metadata: dict) -> list[str]:
        errors: list[str] = []
        if not video_path.exists():
            errors.append(f"video file not found: {video_path}")
        if not metadata.get("title", "").strip():
            errors.append("metadata title is empty")
        return errors

"""
analytics_tracker.py — AnalyticsTracker

Fetches video performance metrics from YouTube Analytics API v2,
computes engagement/virality scores, assigns performance tiers,
and saves results to the video_metrics SQLite table.

Performance tiers:
    S  —  views > 50 000 AND engagement_rate > 8%
    A  —  views > 20 000 OR  engagement_rate > 6%
    B  —  views > 5 000  OR  engagement_rate > 3%
    C  —  views > 1 000
    F  —  views <= 1 000

Usage:
    tracker = AnalyticsTracker()
    tracker.register_video("YT_abc123", "default", "stoic_001", "Stoic Quote")
    metrics = tracker.track_video("YT_abc123", "default")
    print(metrics.tier, metrics.engagement_rate)
"""

import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_PATH = Path("data/processed/channel_forge.db")
CREDENTIALS_DIR = Path(".credentials")

_METRICS_DDL = """
CREATE TABLE IF NOT EXISTS video_metrics (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id            TEXT    NOT NULL,
    channel_key         TEXT    NOT NULL DEFAULT 'default',
    views               INTEGER NOT NULL DEFAULT 0,
    watch_time_minutes  REAL    NOT NULL DEFAULT 0,
    likes               INTEGER NOT NULL DEFAULT 0,
    comments            INTEGER NOT NULL DEFAULT 0,
    shares              INTEGER NOT NULL DEFAULT 0,
    impressions         INTEGER NOT NULL DEFAULT 0,
    ctr                 REAL    NOT NULL DEFAULT 0,
    subscribers_gained  INTEGER NOT NULL DEFAULT 0,
    subscribers_lost    INTEGER NOT NULL DEFAULT 0,
    engagement_rate     REAL    NOT NULL DEFAULT 0,
    virality_score      REAL    NOT NULL DEFAULT 0,
    tier                TEXT    NOT NULL DEFAULT 'F',
    fetched_at          TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_metrics_video   ON video_metrics (video_id);
CREATE INDEX IF NOT EXISTS idx_metrics_tier    ON video_metrics (tier);
CREATE INDEX IF NOT EXISTS idx_metrics_fetched ON video_metrics (fetched_at);
"""

_UPLOADED_DDL = """
CREATE TABLE IF NOT EXISTS uploaded_videos (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id    TEXT    NOT NULL UNIQUE,
    channel_key TEXT    NOT NULL DEFAULT 'default',
    topic_id    TEXT,
    title       TEXT,
    uploaded_at TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_uploaded_channel ON uploaded_videos (channel_key);
"""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class VideoMetrics:
    """Performance metrics for a single YouTube video."""

    video_id: str
    channel_key: str = "default"
    views: int = 0
    watch_time_minutes: float = 0.0
    likes: int = 0
    comments: int = 0
    shares: int = 0
    impressions: int = 0
    ctr: float = 0.0
    subscribers_gained: int = 0
    subscribers_lost: int = 0
    engagement_rate: float = 0.0    # (likes+comments+shares)/views*100
    virality_score: float = 0.0     # (shares*3+comments*2+likes)/views*100
    tier: str = "F"                 # S/A/B/C/F
    fetched_at: str = ""

    def __post_init__(self) -> None:
        if not self.fetched_at:
            self.fetched_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id":           self.video_id,
            "channel_key":        self.channel_key,
            "views":              self.views,
            "watch_time_minutes": self.watch_time_minutes,
            "likes":              self.likes,
            "comments":           self.comments,
            "shares":             self.shares,
            "impressions":        self.impressions,
            "ctr":                self.ctr,
            "subscribers_gained": self.subscribers_gained,
            "subscribers_lost":   self.subscribers_lost,
            "engagement_rate":    round(self.engagement_rate, 4),
            "virality_score":     round(self.virality_score, 4),
            "tier":               self.tier,
            "fetched_at":         self.fetched_at,
        }


# ---------------------------------------------------------------------------
# AnalyticsTracker
# ---------------------------------------------------------------------------

class AnalyticsTracker:
    """
    Fetches YouTube Analytics API v2 metrics for uploaded videos,
    computes performance scores, and persists results to SQLite.

    Args:
        db_path:         Path to SQLite database.
        credentials_dir: Directory containing OAuth token files.
        lookback_days:   Days of history to pull per request (default 28).
    """

    def __init__(
        self,
        db_path: str | Path = DB_PATH,
        credentials_dir: str | Path = CREDENTIALS_DIR,
        lookback_days: int = 28,
    ) -> None:
        self.db_path = Path(db_path)
        self.credentials_dir = Path(credentials_dir)
        self.lookback_days = lookback_days
        self._ensure_tables()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_video(
        self,
        video_id: str,
        channel_key: str = "default",
        topic_id: str = "",
        title: str = "",
    ) -> None:
        """Add a video to the uploaded_videos table for future tracking."""
        now = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO uploaded_videos (video_id, channel_key, topic_id, title, uploaded_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(video_id) DO UPDATE SET
                    channel_key = excluded.channel_key,
                    topic_id    = excluded.topic_id,
                    title       = excluded.title
                """,
                (video_id, channel_key, topic_id, title, now),
            )
            conn.commit()
            logger.debug("Registered video: %s (channel=%s)", video_id, channel_key)
        finally:
            conn.close()

    def track_video(self, video_id: str, channel_key: str = "default") -> VideoMetrics:
        """
        Fetch metrics for one video from YouTube Analytics API and save to DB.

        Args:
            video_id:    YouTube video ID (e.g. "dQw4w9WgXcQ").
            channel_key: Key for loading OAuth credentials.

        Returns:
            VideoMetrics with all computed fields populated.
            On API failure, returns a default VideoMetrics (all zeros, tier=F).
        """
        logger.info("Tracking video: %s (channel=%s)", video_id, channel_key)
        try:
            raw = self._fetch_metrics(video_id, channel_key)
            metrics = self._build_metrics(video_id, channel_key, raw)
        except Exception as exc:
            logger.warning("Failed to fetch metrics for %s: %s", video_id, exc)
            metrics = VideoMetrics(video_id=video_id, channel_key=channel_key)
        self._save_metrics(metrics)
        return metrics

    def track_all(self, channel_key: str = "default") -> list[VideoMetrics]:
        """
        Fetch and save metrics for every video registered in uploaded_videos
        for the given channel.

        Returns:
            List of VideoMetrics, one per uploaded video.
        """
        video_ids = self._get_uploaded_video_ids(channel_key)
        logger.info("Tracking %d video(s) for channel=%s", len(video_ids), channel_key)
        results: list[VideoMetrics] = []
        for video_id in video_ids:
            metrics = self.track_video(video_id, channel_key)
            results.append(metrics)
        return results

    # ------------------------------------------------------------------
    # Calculations (pure functions — no I/O)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_engagement_rate(
        likes: int, comments: int, shares: int, views: int
    ) -> float:
        """
        engagement_rate = (likes + comments + shares) / views * 100.
        Returns 0.0 if views == 0.
        """
        if views <= 0:
            return 0.0
        return (likes + comments + shares) / views * 100

    @staticmethod
    def compute_virality_score(
        likes: int, comments: int, shares: int, views: int
    ) -> float:
        """
        virality_score = (shares*3 + comments*2 + likes) / views * 100.
        Returns 0.0 if views == 0.
        """
        if views <= 0:
            return 0.0
        return (shares * 3 + comments * 2 + likes) / views * 100

    @staticmethod
    def assign_tier(views: int, engagement_rate: float) -> str:
        """
        Assign performance tier based on views and engagement_rate.

            S  —  views > 50 000 AND engagement_rate > 8%
            A  —  views > 20 000 OR  engagement_rate > 6%
            B  —  views > 5 000  OR  engagement_rate > 3%
            C  —  views > 1 000
            F  —  views <= 1 000
        """
        if views > 50_000 and engagement_rate > 8.0:
            return "S"
        if views > 20_000 or engagement_rate > 6.0:
            return "A"
        if views > 5_000 or engagement_rate > 3.0:
            return "B"
        if views > 1_000:
            return "C"
        return "F"

    # ------------------------------------------------------------------
    # Private: API calls (lazy Google imports)
    # ------------------------------------------------------------------

    def _load_credentials(self, channel_key: str):
        """Load OAuth credentials from .credentials/{channel_key}_token.json."""
        from google.oauth2.credentials import Credentials  # lazy
        token_path = self.credentials_dir / f"{channel_key}_token.json"
        if not token_path.exists():
            raise FileNotFoundError(f"Credentials not found: {token_path}")
        data = json.loads(token_path.read_text())
        return Credentials(
            token=data.get("token"),
            refresh_token=data.get("refresh_token"),
            token_uri=data.get("token_uri", "https://oauth2.googleapis.com/token"),
            client_id=data.get("client_id"),
            client_secret=data.get("client_secret"),
            scopes=data.get("scopes"),
        )

    def _build_analytics_service(self, credentials):
        """Build YouTube Analytics API v2 service."""
        from googleapiclient.discovery import build  # lazy
        return build("youtubeAnalytics", "v2", credentials=credentials)

    def _fetch_metrics(self, video_id: str, channel_key: str) -> dict[str, Any]:
        """
        Call YouTube Analytics API v2 and return a raw metrics dict.

        Metrics requested (in order):
            views, estimatedMinutesWatched, likes, comments, shares,
            impressions, impressionClickThroughRate,
            subscribersGained, subscribersLost
        """
        credentials = self._load_credentials(channel_key)
        service = self._build_analytics_service(credentials)

        end_date = date.today()
        start_date = end_date - timedelta(days=self.lookback_days)

        response = service.reports().query(
            ids="channel==MINE",
            startDate=start_date.isoformat(),
            endDate=end_date.isoformat(),
            metrics=(
                "views,estimatedMinutesWatched,likes,comments,shares,"
                "impressions,impressionClickThroughRate,"
                "subscribersGained,subscribersLost"
            ),
            filters=f"video=={video_id}",
        ).execute()

        rows = response.get("rows", [])
        row = rows[0] if rows else []

        def _int(idx: int) -> int:
            return int(row[idx]) if len(row) > idx else 0

        def _float(idx: int) -> float:
            return float(row[idx]) if len(row) > idx else 0.0

        return {
            "views":              _int(0),
            "watch_time_minutes": _float(1),
            "likes":              _int(2),
            "comments":           _int(3),
            "shares":             _int(4),
            "impressions":        _int(5),
            "ctr":                _float(6),
            "subscribers_gained": _int(7),
            "subscribers_lost":   _int(8),
        }

    # ------------------------------------------------------------------
    # Private: metric computation
    # ------------------------------------------------------------------

    def _build_metrics(
        self, video_id: str, channel_key: str, raw: dict[str, Any]
    ) -> VideoMetrics:
        """Construct a VideoMetrics from a raw API response dict."""
        views    = raw.get("views", 0)
        likes    = raw.get("likes", 0)
        comments = raw.get("comments", 0)
        shares   = raw.get("shares", 0)

        engagement_rate = self.compute_engagement_rate(likes, comments, shares, views)
        virality_score  = self.compute_virality_score(likes, comments, shares, views)
        tier = self.assign_tier(views, engagement_rate)

        return VideoMetrics(
            video_id=video_id,
            channel_key=channel_key,
            views=views,
            watch_time_minutes=raw.get("watch_time_minutes", 0.0),
            likes=likes,
            comments=comments,
            shares=shares,
            impressions=raw.get("impressions", 0),
            ctr=raw.get("ctr", 0.0),
            subscribers_gained=raw.get("subscribers_gained", 0),
            subscribers_lost=raw.get("subscribers_lost", 0),
            engagement_rate=engagement_rate,
            virality_score=virality_score,
            tier=tier,
        )

    # ------------------------------------------------------------------
    # Private: database
    # ------------------------------------------------------------------

    def _ensure_tables(self) -> None:
        """Create video_metrics and uploaded_videos tables if they don't exist."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path)
            try:
                conn.executescript(_METRICS_DDL + _UPLOADED_DDL)
                conn.commit()
            finally:
                conn.close()
        except Exception as exc:
            logger.warning("Could not initialise analytics tables: %s", exc)

    def _save_metrics(self, metrics: VideoMetrics) -> None:
        """Insert a new row in video_metrics."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO video_metrics
                    (video_id, channel_key, views, watch_time_minutes, likes, comments,
                     shares, impressions, ctr, subscribers_gained, subscribers_lost,
                     engagement_rate, virality_score, tier, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metrics.video_id, metrics.channel_key,
                    metrics.views, metrics.watch_time_minutes,
                    metrics.likes, metrics.comments, metrics.shares,
                    metrics.impressions, metrics.ctr,
                    metrics.subscribers_gained, metrics.subscribers_lost,
                    metrics.engagement_rate, metrics.virality_score,
                    metrics.tier, metrics.fetched_at,
                ),
            )
            conn.commit()
            logger.debug("Saved metrics: video_id=%s tier=%s", metrics.video_id, metrics.tier)
        finally:
            conn.close()

    def _get_uploaded_video_ids(self, channel_key: str) -> list[str]:
        """Return all video_ids in uploaded_videos for the given channel_key."""
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(
                "SELECT video_id FROM uploaded_videos WHERE channel_key = ?",
                (channel_key,),
            ).fetchall()
            return [r[0] for r in rows]
        finally:
            conn.close()

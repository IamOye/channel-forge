"""
upload_scheduler.py — UploadScheduler

Staggers video uploads across daily time slots and records scheduled
publish times in the SQLite production_queue table.

Slots: 08:00, 12:00, 18:00 in the configured timezone.
MAX_DAILY_VIDEOS and UPLOAD_TIMEZONE are read from .env.

Usage:
    scheduler = UploadScheduler()
    items = scheduler.schedule(queue=[
        {"topic_id": "stoic_001", "video_path": "data/output/stoic_001_final.mp4"},
        {"topic_id": "stoic_002", "video_path": "data/output/stoic_002_final.mp4"},
    ])
    for item in items:
        print(item.topic_id, item.publish_at)
"""

import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DAILY_SLOTS: list[str] = ["08:00", "12:00", "18:00"]
DEFAULT_TIMEZONE = "Africa/Lagos"
DEFAULT_MAX_DAILY_VIDEOS = 3
DB_PATH = Path("data/processed/channel_forge.db")

_PRODUCTION_QUEUE_DDL = """
CREATE TABLE IF NOT EXISTS production_queue (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id         TEXT    NOT NULL UNIQUE,
    video_path       TEXT    NOT NULL,
    status           TEXT    NOT NULL DEFAULT 'pending',
    publish_at       TEXT,
    slot_index       INTEGER NOT NULL DEFAULT 0,
    scheduled_at     TEXT,
    youtube_video_id TEXT,
    created_at       TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_queue_status  ON production_queue (status);
CREATE INDEX IF NOT EXISTS idx_queue_publish ON production_queue (publish_at);
CREATE INDEX IF NOT EXISTS idx_queue_topic   ON production_queue (topic_id);
"""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ScheduledItem:
    """A single queue item assigned to a publish slot."""

    topic_id: str
    video_path: str
    publish_at: str       # ISO 8601 UTC datetime
    slot_index: int       # 0=08:00, 1=12:00, 2=18:00
    is_scheduled: bool
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic_id":     self.topic_id,
            "video_path":   self.video_path,
            "publish_at":   self.publish_at,
            "slot_index":   self.slot_index,
            "is_scheduled": self.is_scheduled,
            "error":        self.error,
        }


# ---------------------------------------------------------------------------
# UploadScheduler
# ---------------------------------------------------------------------------

class UploadScheduler:
    """
    Assigns daily upload slots to a queue of production items and persists
    the schedule in the SQLite production_queue table.

    Args:
        db_path:          Path to the SQLite database.
        timezone_name:    IANA timezone name. If None, reads UPLOAD_TIMEZONE
                          from env (default "Africa/Lagos").
        max_daily_videos: Max uploads per day. If None, reads MAX_DAILY_VIDEOS
                          from env (default 3).
        daily_slots:      List of "HH:MM" slot strings (default 08:00/12:00/18:00).
    """

    def __init__(
        self,
        db_path: str | Path = DB_PATH,
        timezone_name: str | None = None,
        max_daily_videos: int | None = None,
        daily_slots: list[str] | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.timezone_name = (
            timezone_name
            if timezone_name is not None
            else os.getenv("UPLOAD_TIMEZONE", DEFAULT_TIMEZONE)
        )
        self.max_daily_videos = (
            max_daily_videos
            if max_daily_videos is not None
            else int(os.getenv("MAX_DAILY_VIDEOS", str(DEFAULT_MAX_DAILY_VIDEOS)))
        )
        self.daily_slots = daily_slots or DAILY_SLOTS
        self._ensure_table()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def schedule(
        self,
        queue: list[dict[str, Any]],
        start_date: date | None = None,
    ) -> list[ScheduledItem]:
        """
        Assign each queue item to a publish slot and persist in the database.

        Items beyond max_daily_videos per day are pushed to the next day.

        Args:
            queue:       List of dicts, each with "topic_id" and "video_path".
            start_date:  First upload date (defaults to today in the configured tz).

        Returns:
            List of ScheduledItem, one per input item, in the same order.
        """
        if start_date is None:
            start_date = datetime.now(ZoneInfo(self.timezone_name)).date()

        results: list[ScheduledItem] = []
        day_offset = 0
        slot_in_day = 0
        slots_per_day = min(self.max_daily_videos, len(self.daily_slots))

        for item in queue:
            if slot_in_day >= slots_per_day:
                day_offset += 1
                slot_in_day = 0

            slot_index = slot_in_day % len(self.daily_slots)
            publish_dt = self._slot_to_utc(start_date, day_offset, slot_index)
            publish_at = publish_dt.isoformat()

            scheduled = ScheduledItem(
                topic_id=item.get("topic_id", ""),
                video_path=item.get("video_path", ""),
                publish_at=publish_at,
                slot_index=slot_index,
                is_scheduled=True,
            )

            try:
                self._mark_scheduled(scheduled)
            except Exception as exc:
                logger.warning("DB write failed for topic_id=%s: %s", scheduled.topic_id, exc)
                scheduled.is_scheduled = False
                scheduled.error = str(exc)

            results.append(scheduled)
            slot_in_day += 1

        logger.info(
            "Scheduled %d items across %d day(s) starting %s",
            len(results), day_offset + 1, start_date,
        )
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _slot_to_utc(self, base_date: date, day_offset: int, slot_index: int) -> datetime:
        """Return an aware UTC datetime for the given slot."""
        tz = ZoneInfo(self.timezone_name)
        slot_str = self.daily_slots[slot_index]
        hour, minute = map(int, slot_str.split(":"))
        target_date = base_date + timedelta(days=day_offset)
        dt_local = datetime(
            target_date.year, target_date.month, target_date.day,
            hour, minute, 0,
            tzinfo=tz,
        )
        return dt_local.astimezone(timezone.utc)

    def _ensure_table(self) -> None:
        """Create the production_queue table if it does not exist."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path)
            try:
                conn.executescript(_PRODUCTION_QUEUE_DDL)
                conn.commit()
            finally:
                conn.close()
        except Exception as exc:
            logger.warning("Could not initialise production_queue table: %s", exc)

    def _mark_scheduled(self, item: ScheduledItem) -> None:
        """Insert or update a row in production_queue for the given item."""
        now = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO production_queue
                    (topic_id, video_path, status, publish_at, slot_index, scheduled_at, created_at)
                VALUES (?, ?, 'scheduled', ?, ?, ?, ?)
                ON CONFLICT(topic_id) DO UPDATE SET
                    status       = 'scheduled',
                    publish_at   = excluded.publish_at,
                    slot_index   = excluded.slot_index,
                    scheduled_at = excluded.scheduled_at
                """,
                (item.topic_id, item.video_path, item.publish_at, item.slot_index, now, now),
            )
            conn.commit()
            logger.debug("Marked scheduled: topic_id=%s publish_at=%s", item.topic_id, item.publish_at)
        finally:
            conn.close()

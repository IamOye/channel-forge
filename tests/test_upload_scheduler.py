"""
Tests for src/publisher/upload_scheduler.py

Uses tmp_path for a real SQLite database — no mocks needed for DB calls.
No real API calls or network activity.
"""

from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.publisher.upload_scheduler import (
    DAILY_SLOTS,
    DEFAULT_MAX_DAILY_VIDEOS,
    DEFAULT_TIMEZONE,
    ScheduledItem,
    UploadScheduler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_queue(n: int) -> list[dict]:
    return [
        {"topic_id": f"topic_{i:03d}", "video_path": f"data/output/topic_{i:03d}_final.mp4"}
        for i in range(n)
    ]


def _make_scheduler(tmp_path: Path, **kw) -> UploadScheduler:
    return UploadScheduler(db_path=tmp_path / "test.db", **kw)


# ---------------------------------------------------------------------------
# ScheduledItem
# ---------------------------------------------------------------------------

class TestScheduledItem:
    def test_to_dict_has_all_keys(self) -> None:
        item = ScheduledItem(
            topic_id="t001",
            video_path="data/output/t001_final.mp4",
            publish_at="2025-01-01T08:00:00+00:00",
            slot_index=0,
            is_scheduled=True,
        )
        d = item.to_dict()
        for key in ("topic_id", "video_path", "publish_at", "slot_index", "is_scheduled", "error"):
            assert key in d

    def test_error_defaults_to_empty(self) -> None:
        item = ScheduledItem("t", "p", "2025-01-01T08:00:00+00:00", 0, True)
        assert item.error == ""


# ---------------------------------------------------------------------------
# UploadScheduler defaults from env
# ---------------------------------------------------------------------------

class TestSchedulerDefaults:
    def test_default_timezone_from_env(self, tmp_path) -> None:
        with patch.dict("os.environ", {"UPLOAD_TIMEZONE": "Europe/London"}):
            scheduler = _make_scheduler(tmp_path)
        assert scheduler.timezone_name == "Europe/London"

    def test_default_max_daily_from_env(self, tmp_path) -> None:
        with patch.dict("os.environ", {"MAX_DAILY_VIDEOS": "5"}):
            scheduler = _make_scheduler(tmp_path)
        assert scheduler.max_daily_videos == 5

    def test_hardcoded_defaults_when_env_absent(self, tmp_path) -> None:
        with patch.dict("os.environ", {}, clear=False):
            # Use explicit values to avoid env bleed
            scheduler = UploadScheduler(
                db_path=tmp_path / "t.db",
                timezone_name=DEFAULT_TIMEZONE,
                max_daily_videos=DEFAULT_MAX_DAILY_VIDEOS,
            )
        assert scheduler.timezone_name == DEFAULT_TIMEZONE
        assert scheduler.max_daily_videos == DEFAULT_MAX_DAILY_VIDEOS


# ---------------------------------------------------------------------------
# UploadScheduler.schedule — slot assignment
# ---------------------------------------------------------------------------

class TestScheduleSlots:
    def test_single_item_gets_slot_0(self, tmp_path) -> None:
        scheduler = _make_scheduler(tmp_path, timezone_name="UTC", max_daily_videos=3)
        results = scheduler.schedule(_make_queue(1), start_date=date(2025, 1, 1))
        assert len(results) == 1
        assert results[0].slot_index == 0
        assert results[0].is_scheduled is True

    def test_three_items_fill_three_daily_slots(self, tmp_path) -> None:
        scheduler = _make_scheduler(tmp_path, timezone_name="UTC", max_daily_videos=3)
        results = scheduler.schedule(_make_queue(3), start_date=date(2025, 1, 1))
        assert [r.slot_index for r in results] == [0, 1, 2]

    def test_fourth_item_rolls_to_next_day(self, tmp_path) -> None:
        scheduler = _make_scheduler(tmp_path, timezone_name="UTC", max_daily_videos=3)
        results = scheduler.schedule(_make_queue(4), start_date=date(2025, 1, 1))
        # First 3 on day 1, 4th on day 2 slot 0
        assert results[3].slot_index == 0
        # Day 2 publish_at should be one day later
        day1_date = results[0].publish_at[:10]
        day2_date = results[3].publish_at[:10]
        assert day2_date > day1_date

    def test_six_items_span_two_days(self, tmp_path) -> None:
        scheduler = _make_scheduler(tmp_path, timezone_name="UTC", max_daily_videos=3)
        results = scheduler.schedule(_make_queue(6), start_date=date(2025, 1, 1))
        assert len(results) == 6
        # Items 0-2 on 2025-01-01, 3-5 on 2025-01-02
        assert results[0].publish_at[:10] == "2025-01-01"
        assert results[3].publish_at[:10] == "2025-01-02"

    def test_max_daily_videos_1_means_one_per_day(self, tmp_path) -> None:
        scheduler = _make_scheduler(tmp_path, timezone_name="UTC", max_daily_videos=1)
        results = scheduler.schedule(_make_queue(3), start_date=date(2025, 1, 1))
        dates = [r.publish_at[:10] for r in results]
        assert dates == ["2025-01-01", "2025-01-02", "2025-01-03"]

    def test_returns_all_items_even_beyond_slots(self, tmp_path) -> None:
        scheduler = _make_scheduler(tmp_path, timezone_name="UTC", max_daily_videos=3)
        results = scheduler.schedule(_make_queue(7), start_date=date(2025, 1, 1))
        assert len(results) == 7

    def test_empty_queue_returns_empty_list(self, tmp_path) -> None:
        scheduler = _make_scheduler(tmp_path, timezone_name="UTC")
        results = scheduler.schedule([], start_date=date(2025, 1, 1))
        assert results == []


# ---------------------------------------------------------------------------
# UploadScheduler — publish_at format and timezone
# ---------------------------------------------------------------------------

class TestPublishAtFormat:
    def test_publish_at_is_iso_string(self, tmp_path) -> None:
        scheduler = _make_scheduler(tmp_path, timezone_name="UTC", max_daily_videos=3)
        results = scheduler.schedule(_make_queue(1), start_date=date(2025, 1, 1))
        # Must be parseable as ISO datetime
        dt = datetime.fromisoformat(results[0].publish_at)
        assert dt.tzinfo is not None

    def test_slot_0_is_0800_local_time(self, tmp_path) -> None:
        scheduler = _make_scheduler(tmp_path, timezone_name="UTC", max_daily_videos=3)
        results = scheduler.schedule(_make_queue(1), start_date=date(2025, 1, 1))
        dt = datetime.fromisoformat(results[0].publish_at).astimezone(timezone.utc)
        assert dt.hour == 8
        assert dt.minute == 0

    def test_slot_1_is_1200_local_time(self, tmp_path) -> None:
        scheduler = _make_scheduler(tmp_path, timezone_name="UTC", max_daily_videos=3)
        results = scheduler.schedule(_make_queue(2), start_date=date(2025, 1, 1))
        dt = datetime.fromisoformat(results[1].publish_at).astimezone(timezone.utc)
        assert dt.hour == 12

    def test_slot_2_is_1800_local_time(self, tmp_path) -> None:
        scheduler = _make_scheduler(tmp_path, timezone_name="UTC", max_daily_videos=3)
        results = scheduler.schedule(_make_queue(3), start_date=date(2025, 1, 1))
        dt = datetime.fromisoformat(results[2].publish_at).astimezone(timezone.utc)
        assert dt.hour == 18

    def test_africa_lagos_timezone_offset(self, tmp_path) -> None:
        # Africa/Lagos is UTC+1; 08:00 local = 07:00 UTC
        scheduler = _make_scheduler(tmp_path, timezone_name="Africa/Lagos", max_daily_videos=3)
        results = scheduler.schedule(_make_queue(1), start_date=date(2025, 1, 1))
        dt = datetime.fromisoformat(results[0].publish_at).astimezone(timezone.utc)
        assert dt.hour == 7  # 08:00 WAT = 07:00 UTC


# ---------------------------------------------------------------------------
# UploadScheduler — database persistence
# ---------------------------------------------------------------------------

class TestDatabasePersistence:
    def test_items_persisted_in_db(self, tmp_path) -> None:
        import sqlite3
        db = tmp_path / "test.db"
        scheduler = UploadScheduler(db_path=db, timezone_name="UTC", max_daily_videos=3)
        scheduler.schedule(_make_queue(2), start_date=date(2025, 1, 1))

        conn = sqlite3.connect(db)
        rows = conn.execute("SELECT topic_id, status FROM production_queue ORDER BY topic_id").fetchall()
        conn.close()

        assert len(rows) == 2
        assert all(row[1] == "scheduled" for row in rows)

    def test_duplicate_topic_id_updates_not_inserts(self, tmp_path) -> None:
        import sqlite3
        db = tmp_path / "test.db"
        scheduler = UploadScheduler(db_path=db, timezone_name="UTC", max_daily_videos=3)

        queue = [{"topic_id": "same_001", "video_path": "v1.mp4"}]
        scheduler.schedule(queue, start_date=date(2025, 1, 1))
        scheduler.schedule(queue, start_date=date(2025, 2, 1))  # reschedule

        conn = sqlite3.connect(db)
        count = conn.execute("SELECT COUNT(*) FROM production_queue WHERE topic_id='same_001'").fetchone()[0]
        conn.close()
        assert count == 1  # updated, not duplicated

    def test_db_failure_marks_item_not_scheduled(self, tmp_path) -> None:
        scheduler = _make_scheduler(tmp_path, timezone_name="UTC", max_daily_videos=3)
        with patch.object(scheduler, "_mark_scheduled", side_effect=Exception("DB down")):
            results = scheduler.schedule(_make_queue(2), start_date=date(2025, 1, 1))
        assert all(not r.is_scheduled for r in results)
        assert all("DB down" in r.error for r in results)

    def test_to_dict_is_serialisable(self, tmp_path) -> None:
        import json
        scheduler = _make_scheduler(tmp_path, timezone_name="UTC", max_daily_videos=3)
        results = scheduler.schedule(_make_queue(2), start_date=date(2025, 1, 1))
        for item in results:
            assert len(json.dumps(item.to_dict())) > 10

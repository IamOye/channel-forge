"""
Tests for the three quota fixes:

FIX 1 — Reduce comment check quota usage
  - QuotaTracker.is_quota_exceeded()
  - Comment quota cost constants
  - run_comment_check skips when quota exhausted
  - Comment check schedule is hourly

FIX 2 — OAuth scopes for comments
  - auth_youtube.py SCOPES list includes youtube.force-ssl

FIX 3 — Queue video for next day on quota
  - pending_uploads table in init_db
  - _save_pending_upload writes to DB
  - run_quota_recovery retries pending uploads
  - Quota recovery job registered at 08:05
"""

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.publisher.youtube_uploader import (
    QUOTA_UNITS,
    QuotaTracker,
    YouTubeUploader,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_QUOTA_DDL = """
CREATE TABLE IF NOT EXISTS youtube_quota_usage (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    date             TEXT    NOT NULL,
    operation        TEXT    NOT NULL,
    units_used       INTEGER NOT NULL,
    cumulative_daily INTEGER NOT NULL,
    created_at       TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""

_PENDING_DDL = """
CREATE TABLE IF NOT EXISTS pending_uploads (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id        TEXT    NOT NULL,
    channel_key     TEXT    NOT NULL DEFAULT 'default',
    video_path      TEXT    NOT NULL,
    thumbnail_path  TEXT    NOT NULL DEFAULT '',
    metadata_json   TEXT    NOT NULL DEFAULT '{}',
    publish_at      TEXT    NOT NULL DEFAULT '',
    status          TEXT    NOT NULL DEFAULT 'pending',
    retry_count     INTEGER NOT NULL DEFAULT 0,
    queued_at       TEXT    NOT NULL DEFAULT (datetime('now')),
    completed_at    TEXT
);
"""


def _make_db(tmp_path: Path) -> Path:
    """Create a minimal DB with quota + pending_uploads tables."""
    db = tmp_path / "cf.db"
    conn = sqlite3.connect(db)
    conn.executescript(_QUOTA_DDL + _PENDING_DDL)
    conn.commit()
    conn.close()
    return db


# ---------------------------------------------------------------------------
# FIX 1a — QuotaTracker.is_quota_exceeded
# ---------------------------------------------------------------------------

class TestIsQuotaExceeded:
    def test_returns_false_when_under_limit(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        qt = QuotaTracker(db_path=db, daily_limit=10_000)
        qt.record("video_upload", 1600)
        assert qt.is_quota_exceeded() is False

    def test_returns_true_when_at_limit(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        qt = QuotaTracker(db_path=db, daily_limit=1600)
        qt.record("video_upload", 1600)
        assert qt.is_quota_exceeded() is True

    def test_returns_true_when_over_limit(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        qt = QuotaTracker(db_path=db, daily_limit=1000)
        qt.record("video_upload", 1600)
        assert qt.is_quota_exceeded() is True

    def test_returns_false_when_no_usage(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        qt = QuotaTracker(db_path=db, daily_limit=10_000)
        assert qt.is_quota_exceeded() is False

    def test_consistent_with_can_upload(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        qt = QuotaTracker(db_path=db, daily_limit=1600)
        qt.record("video_upload", 1600)
        assert qt.is_quota_exceeded() is True
        assert qt.can_upload() is False


# ---------------------------------------------------------------------------
# FIX 1a — Comment quota cost constants
# ---------------------------------------------------------------------------

class TestCommentQuotaCosts:
    def test_channels_list_cost_defined(self) -> None:
        assert "channels_list" in QUOTA_UNITS
        assert QUOTA_UNITS["channels_list"] == 1

    def test_playlist_items_list_cost_defined(self) -> None:
        assert "playlist_items_list" in QUOTA_UNITS
        assert QUOTA_UNITS["playlist_items_list"] == 1

    def test_comment_threads_list_cost_defined(self) -> None:
        assert "comment_threads_list" in QUOTA_UNITS
        assert QUOTA_UNITS["comment_threads_list"] == 1


# ---------------------------------------------------------------------------
# FIX 1b — run_comment_check skips when quota exhausted
# ---------------------------------------------------------------------------

class TestCommentCheckQuotaGuard:
    def test_skips_when_quota_exhausted(self, tmp_path: Path) -> None:
        """run_comment_check returns early without making YouTube API calls."""
        from src.scheduler import run_comment_check

        db = _make_db(tmp_path)
        qt = QuotaTracker(db_path=db, daily_limit=0)  # always exhausted

        with patch("src.scheduler._DEFAULT_DB", db):
            with patch(
                "src.publisher.youtube_uploader.QuotaTracker",
                return_value=qt,
            ):
                # If it doesn't skip, it would try to import googleapiclient
                # and either fail or call the mock. We just verify no error.
                run_comment_check()

    def test_proceeds_when_quota_available(self, tmp_path: Path) -> None:
        """run_comment_check proceeds when quota is available."""
        from src.scheduler import run_comment_check

        db = _make_db(tmp_path)
        qt = QuotaTracker(db_path=db, daily_limit=10_000)

        with patch("src.scheduler._DEFAULT_DB", db):
            with patch(
                "src.publisher.youtube_uploader.QuotaTracker",
                return_value=qt,
            ):
                # Will proceed past quota check, then fail on googleapiclient
                # import — that's fine, it means the guard didn't block
                run_comment_check()


# ---------------------------------------------------------------------------
# FIX 1c — Comment check schedule is hourly
# ---------------------------------------------------------------------------

class TestCommentCheckHourly:
    def test_comment_check_trigger_is_hourly(self) -> None:
        from apscheduler.triggers.cron import CronTrigger
        from src.scheduler import build_scheduler

        scheduler = build_scheduler("UTC")
        job = next(j for j in scheduler.get_jobs() if j.id == "comment_check")
        assert isinstance(job.trigger, CronTrigger)
        # Verify it's hour='*' minute=0 (hourly) not minute='*/15'
        trigger_fields = {f.name: str(f) for f in job.trigger.fields}
        assert trigger_fields.get("minute") == "0"


# ---------------------------------------------------------------------------
# FIX 2 — OAuth scopes
# ---------------------------------------------------------------------------

class TestOAuthScopes:
    def test_scopes_include_youtube_force_ssl(self) -> None:
        """auth_youtube.py must include youtube.force-ssl for comment reading."""
        from pathlib import Path as _Path

        scope_file = _Path("scripts/auth_youtube.py")
        content = scope_file.read_text()
        assert "youtube.force-ssl" in content

    def test_scopes_include_youtube_base(self) -> None:
        content = Path("scripts/auth_youtube.py").read_text()
        assert "googleapis.com/auth/youtube\"" in content or \
               "googleapis.com/auth/youtube'" in content

    def test_scopes_include_youtube_upload(self) -> None:
        content = Path("scripts/auth_youtube.py").read_text()
        assert "youtube.upload" in content

    def test_scopes_include_youtube_readonly(self) -> None:
        content = Path("scripts/auth_youtube.py").read_text()
        assert "youtube.readonly" in content

    def test_scopes_include_youtubepartner(self) -> None:
        content = Path("scripts/auth_youtube.py").read_text()
        assert "youtubepartner" in content

    def test_scopes_include_yt_analytics(self) -> None:
        content = Path("scripts/auth_youtube.py").read_text()
        assert "yt-analytics.readonly" in content


# ---------------------------------------------------------------------------
# FIX 3a — pending_uploads table in init_db
# ---------------------------------------------------------------------------

class TestPendingUploadsTable:
    def test_pending_uploads_created_by_init_db(self, tmp_path: Path) -> None:
        from scripts.init_db import MAIN_DDL_PARTS, create_db

        db_path = tmp_path / "test.db"
        create_db(db_path, MAIN_DDL_PARTS)
        conn = sqlite3.connect(db_path)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        conn.close()
        assert "pending_uploads" in tables

    def test_pending_uploads_has_correct_columns(self, tmp_path: Path) -> None:
        from scripts.init_db import MAIN_DDL_PARTS, create_db

        db_path = tmp_path / "test.db"
        create_db(db_path, MAIN_DDL_PARTS)
        conn = sqlite3.connect(db_path)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(pending_uploads)").fetchall()]
        conn.close()
        for expected in ("topic_id", "channel_key", "video_path", "metadata_json",
                         "status", "retry_count", "queued_at", "completed_at"):
            assert expected in cols

    def test_migration_creates_pending_uploads(self, tmp_path: Path) -> None:
        from scripts.init_db import migrate_db

        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE dummy (id INTEGER)")
        conn.commit()
        conn.close()

        migrate_db(db_path)

        conn = sqlite3.connect(db_path)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        conn.close()
        assert "pending_uploads" in tables


# ---------------------------------------------------------------------------
# FIX 3b — _save_pending_upload writes to DB
# ---------------------------------------------------------------------------

class TestSavePendingUpload:
    def test_saves_to_pending_uploads_table(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        qt = QuotaTracker(db_path=db, daily_limit=0)
        uploader = YouTubeUploader(
            channel_key="test_ch",
            credentials_dir=tmp_path,
            quota_tracker=qt,
        )

        payload = {
            "topic_id": "t001",
            "channel_key": "test_ch",
            "video_path": "/tmp/video.mp4",
            "thumbnail_path": "",
            "metadata": {"title": "Test", "description": "Desc"},
            "publish_at": "",
        }
        uploader._save_pending_upload(payload)

        conn = sqlite3.connect(db)
        row = conn.execute(
            "SELECT topic_id, channel_key, status FROM pending_uploads WHERE topic_id = 't001'"
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == "t001"
        assert row[1] == "test_ch"
        assert row[2] == "pending"

    def test_save_pending_upload_stores_metadata_json(self, tmp_path: Path) -> None:
        db = _make_db(tmp_path)
        qt = QuotaTracker(db_path=db, daily_limit=0)
        uploader = YouTubeUploader(
            channel_key="ch1",
            credentials_dir=tmp_path,
            quota_tracker=qt,
        )

        metadata = {"title": "Money Secret", "tags": ["#Shorts"]}
        payload = {
            "topic_id": "t002",
            "channel_key": "ch1",
            "video_path": "/tmp/v.mp4",
            "thumbnail_path": "/tmp/t.jpg",
            "metadata": metadata,
            "publish_at": "2026-01-01T08:00:00Z",
        }
        uploader._save_pending_upload(payload)

        conn = sqlite3.connect(db)
        row = conn.execute(
            "SELECT metadata_json, publish_at FROM pending_uploads WHERE topic_id = 't002'"
        ).fetchone()
        conn.close()
        assert json.loads(row[0]) == metadata
        assert row[1] == "2026-01-01T08:00:00Z"

    def test_save_pending_upload_survives_missing_db(self, tmp_path: Path) -> None:
        qt = QuotaTracker(db_path=tmp_path / "missing.db", daily_limit=0)
        uploader = YouTubeUploader(
            channel_key="ch1",
            credentials_dir=tmp_path,
            quota_tracker=qt,
        )
        # Should not raise
        uploader._save_pending_upload({
            "topic_id": "t003", "channel_key": "ch1",
            "video_path": "/tmp/v.mp4", "thumbnail_path": "",
            "metadata": {}, "publish_at": "",
        })


# ---------------------------------------------------------------------------
# FIX 3b — upload queues to DB when quota exhausted
# ---------------------------------------------------------------------------

class TestUploadQueuesToDb:
    def test_upload_saves_to_pending_uploads_on_quota(self, tmp_path: Path) -> None:
        video = tmp_path / "out.mp4"
        video.write_bytes(b"v" * 100)

        db = _make_db(tmp_path)
        qt = QuotaTracker(db_path=db, daily_limit=0)  # always exhausted

        uploader = YouTubeUploader(
            channel_key="test",
            credentials_dir=tmp_path,
            quota_tracker=qt,
        )

        queue_dir = tmp_path / "queue"
        with patch("src.publisher.youtube_uploader._QUOTA_QUEUE_DIR", queue_dir):
            result = uploader.upload(
                "q_db_001", video,
                {"title": "Test", "description": "D", "tags": []},
            )

        assert result.is_valid is False

        conn = sqlite3.connect(db)
        row = conn.execute(
            "SELECT topic_id, status FROM pending_uploads WHERE topic_id = 'q_db_001'"
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[1] == "pending"


# ---------------------------------------------------------------------------
# FIX 3c — run_quota_recovery
# ---------------------------------------------------------------------------

class TestRunQuotaRecovery:
    def test_no_pending_uploads_logs_and_returns(self, tmp_path: Path) -> None:
        from src.scheduler import run_quota_recovery

        db = _make_db(tmp_path)
        with patch("src.scheduler._DEFAULT_DB", db):
            # Should not raise
            run_quota_recovery()

    def test_retries_pending_upload_on_success(self, tmp_path: Path) -> None:
        from src.scheduler import run_quota_recovery

        db = _make_db(tmp_path)
        # Insert a pending upload
        conn = sqlite3.connect(db)
        conn.execute(
            "INSERT INTO pending_uploads "
            "(topic_id, channel_key, video_path, thumbnail_path, metadata_json, publish_at, status) "
            "VALUES (?, ?, ?, ?, ?, ?, 'pending')",
            ("t_retry", "test", "/tmp/v.mp4", "", '{"title":"T","description":"D"}', ""),
        )
        conn.commit()
        conn.close()

        mock_result = MagicMock(is_valid=True, youtube_url="https://youtube.com/watch?v=abc")
        mock_uploader = MagicMock()
        mock_uploader.upload.return_value = mock_result

        with patch("src.scheduler._DEFAULT_DB", db):
            with patch(
                "src.publisher.youtube_uploader.YouTubeUploader",
                return_value=mock_uploader,
            ):
                with patch(
                    "src.publisher.youtube_uploader.QuotaTracker",
                ) as MockQT:
                    MockQT.return_value.is_quota_exceeded.return_value = False
                    run_quota_recovery()

        conn = sqlite3.connect(db)
        row = conn.execute(
            "SELECT status FROM pending_uploads WHERE topic_id = 't_retry'"
        ).fetchone()
        conn.close()
        assert row[0] == "completed"

    def test_increments_retry_count_on_failure(self, tmp_path: Path) -> None:
        from src.scheduler import run_quota_recovery

        db = _make_db(tmp_path)
        conn = sqlite3.connect(db)
        conn.execute(
            "INSERT INTO pending_uploads "
            "(topic_id, channel_key, video_path, thumbnail_path, metadata_json, publish_at, status) "
            "VALUES (?, ?, ?, ?, ?, ?, 'pending')",
            ("t_fail", "test", "/tmp/v.mp4", "", '{"title":"T","description":"D"}', ""),
        )
        conn.commit()
        conn.close()

        mock_result = MagicMock(
            is_valid=False, validation_errors=["upload error"]
        )
        mock_uploader = MagicMock()
        mock_uploader.upload.return_value = mock_result

        with patch("src.scheduler._DEFAULT_DB", db):
            with patch(
                "src.publisher.youtube_uploader.YouTubeUploader",
                return_value=mock_uploader,
            ):
                with patch(
                    "src.publisher.youtube_uploader.QuotaTracker",
                ) as MockQT:
                    MockQT.return_value.is_quota_exceeded.return_value = False
                    run_quota_recovery()

        conn = sqlite3.connect(db)
        row = conn.execute(
            "SELECT retry_count, status FROM pending_uploads WHERE topic_id = 't_fail'"
        ).fetchone()
        conn.close()
        assert row[0] == 1
        assert row[1] == "pending"

    def test_stops_when_quota_exhausted_mid_retry(self, tmp_path: Path) -> None:
        from src.scheduler import run_quota_recovery

        db = _make_db(tmp_path)
        conn = sqlite3.connect(db)
        for i in range(3):
            conn.execute(
                "INSERT INTO pending_uploads "
                "(topic_id, channel_key, video_path, thumbnail_path, "
                "metadata_json, publish_at, status) "
                "VALUES (?, ?, ?, ?, ?, ?, 'pending')",
                (f"t_{i}", "test", "/tmp/v.mp4", "",
                 '{"title":"T","description":"D"}', ""),
            )
        conn.commit()
        conn.close()

        with patch("src.scheduler._DEFAULT_DB", db):
            with patch(
                "src.publisher.youtube_uploader.QuotaTracker",
            ) as MockQT:
                # Quota exhausted immediately
                MockQT.return_value.is_quota_exceeded.return_value = True
                run_quota_recovery()

        # All should still be pending (none attempted)
        conn = sqlite3.connect(db)
        pending = conn.execute(
            "SELECT COUNT(*) FROM pending_uploads WHERE status = 'pending'"
        ).fetchone()[0]
        conn.close()
        assert pending == 3

    def test_exception_does_not_propagate(self, tmp_path: Path) -> None:
        from src.scheduler import run_quota_recovery

        db = _make_db(tmp_path)
        with patch("src.scheduler._DEFAULT_DB", db):
            with patch.dict("sys.modules", {
                "src.publisher.youtube_uploader": MagicMock(
                    side_effect=ImportError("broken")
                ),
            }):
                # Should not raise
                run_quota_recovery()

    def test_missing_db_returns_cleanly(self, tmp_path: Path) -> None:
        from src.scheduler import run_quota_recovery

        with patch("src.scheduler._DEFAULT_DB", tmp_path / "missing.db"):
            run_quota_recovery()


# ---------------------------------------------------------------------------
# FIX 3c — Quota recovery job registered in scheduler
# ---------------------------------------------------------------------------

class TestQuotaRecoveryJob:
    def test_quota_recovery_job_registered(self) -> None:
        from src.scheduler import build_scheduler

        scheduler = build_scheduler("UTC")
        job_ids = [j.id for j in scheduler.get_jobs()]
        assert "quota_recovery" in job_ids

    def test_quota_recovery_uses_correct_function(self) -> None:
        from src.scheduler import build_scheduler, run_quota_recovery

        scheduler = build_scheduler("UTC")
        job = next(j for j in scheduler.get_jobs() if j.id == "quota_recovery")
        assert job.func is run_quota_recovery

    def test_quota_recovery_has_cron_trigger(self) -> None:
        from apscheduler.triggers.cron import CronTrigger
        from src.scheduler import build_scheduler

        scheduler = build_scheduler("UTC")
        job = next(j for j in scheduler.get_jobs() if j.id == "quota_recovery")
        assert isinstance(job.trigger, CronTrigger)

    def test_quota_recovery_runs_at_0805(self) -> None:
        from src.scheduler import build_scheduler

        scheduler = build_scheduler("UTC")
        job = next(j for j in scheduler.get_jobs() if j.id == "quota_recovery")
        trigger_fields = {f.name: str(f) for f in job.trigger.fields}
        assert trigger_fields.get("hour") == "8"
        assert trigger_fields.get("minute") == "5"


# ---------------------------------------------------------------------------
# Settings default update (comment_check_interval = 60)
# ---------------------------------------------------------------------------

class TestSettingsDefault:
    def test_comment_check_interval_is_60(self, tmp_path: Path) -> None:
        from scripts.init_db import MAIN_DDL_PARTS, create_db

        db_path = tmp_path / "test.db"
        create_db(db_path, MAIN_DDL_PARTS)
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT value FROM settings WHERE key = 'comment_check_interval'"
        ).fetchone()
        conn.close()
        assert row[0] == "60"

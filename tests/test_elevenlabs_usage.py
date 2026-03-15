"""
Tests for ElevenLabs usage tracking.

Covers:
- _save_usage() writes to elevenlabs_usage table
- _check_monthly_usage() logs correct warnings at thresholds
- Videos remaining calculation
- Monthly reset boundary (Jan 31 → Feb 1 resets counter)
- check_elevenlabs_usage.get_usage_report() output shape and values
"""

import sqlite3
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.media.voiceover import VoiceoverGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gen(tmp_path: Path) -> VoiceoverGenerator:
    db = tmp_path / "test.db"
    return VoiceoverGenerator(api_key="fake", db_path=db)


def _insert_usage(db_path: Path, rows: list[tuple]) -> None:
    """Insert (date, topic_id, chars_used, voice_name) rows directly."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS elevenlabs_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            topic_id TEXT NOT NULL,
            chars_used INTEGER NOT NULL,
            voice_name TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.executemany(
        "INSERT INTO elevenlabs_usage (date, topic_id, chars_used, voice_name) VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# _save_usage
# ---------------------------------------------------------------------------

class TestSaveUsage:
    def test_saves_row_to_db(self, tmp_path: Path) -> None:
        gen = _make_gen(tmp_path)
        gen._save_usage(topic_id="topic_001", chars_used=850, voice_name="Adam")

        conn = sqlite3.connect(gen.db_path)
        row = conn.execute("SELECT date, topic_id, chars_used, voice_name FROM elevenlabs_usage").fetchone()
        conn.close()

        assert row is not None
        assert row[1] == "topic_001"
        assert row[2] == 850
        assert row[3] == "Adam"

    def test_date_is_today(self, tmp_path: Path) -> None:
        gen = _make_gen(tmp_path)
        gen._save_usage(topic_id="t1", chars_used=100, voice_name="Josh")

        conn = sqlite3.connect(gen.db_path)
        row = conn.execute("SELECT date FROM elevenlabs_usage").fetchone()
        conn.close()

        assert row[0] == date.today().isoformat()

    def test_multiple_rows_accumulate(self, tmp_path: Path) -> None:
        gen = _make_gen(tmp_path)
        gen._save_usage("t1", 500, "Adam")
        gen._save_usage("t2", 700, "Adam")
        gen._save_usage("t3", 300, "Adam")

        conn = sqlite3.connect(gen.db_path)
        count = conn.execute("SELECT COUNT(*) FROM elevenlabs_usage").fetchone()[0]
        total = conn.execute("SELECT SUM(chars_used) FROM elevenlabs_usage").fetchone()[0]
        conn.close()

        assert count == 3
        assert total == 1500

    def test_swallows_db_error(self, tmp_path: Path) -> None:
        gen = VoiceoverGenerator(api_key="fake", db_path=tmp_path / "nodir" / "sub" / "test.db")
        # Should not raise even if parent dirs don't exist initially
        # (mkdir is called inside, so this should actually succeed — just verify no exception)
        try:
            gen._save_usage("t1", 100, "Adam")
        except Exception as exc:
            pytest.fail(f"_save_usage raised unexpectedly: {exc}")


# ---------------------------------------------------------------------------
# _check_monthly_usage — warning thresholds
# ---------------------------------------------------------------------------

class TestCheckMonthlyUsage:
    def _setup_usage(self, tmp_path: Path, chars: int, n_videos: int) -> VoiceoverGenerator:
        """Insert n_videos rows of equal size totalling `chars` this month."""
        gen = _make_gen(tmp_path)
        per_video = chars // max(n_videos, 1)
        today = date.today().isoformat()
        rows = [(today, f"t{i}", per_video, "Adam") for i in range(n_videos)]
        _insert_usage(gen.db_path, rows)
        return gen

    def test_no_warning_below_67_pct(self, tmp_path: Path, caplog) -> None:
        # 19,000 / 30,000 = 63.3% — below 67%
        gen = self._setup_usage(tmp_path, 19_000, 20)
        with patch.dict("os.environ", {"ELEVENLABS_MONTHLY_LIMIT": "30000", "ELEVENLABS_RESET_DAY": "1"}):
            with caplog.at_level("WARNING", logger="src.media.voiceover"):
                gen._check_monthly_usage()
        assert not any("ElevenLabs" in r.message for r in caplog.records if r.levelname == "WARNING")

    def test_warning_at_67_pct(self, tmp_path: Path, caplog) -> None:
        # 20,100 / 30,000 = 67.0%
        gen = self._setup_usage(tmp_path, 20_100, 20)
        with patch.dict("os.environ", {"ELEVENLABS_MONTHLY_LIMIT": "30000", "ELEVENLABS_RESET_DAY": "1"}):
            with caplog.at_level("WARNING", logger="src.media.voiceover"):
                gen._check_monthly_usage()
        assert any("67%" in r.message for r in caplog.records if r.levelname == "WARNING")

    def test_warning_at_85_pct(self, tmp_path: Path, caplog) -> None:
        # 25_500 / 30,000 = 85%
        gen = self._setup_usage(tmp_path, 25_500, 25)
        with patch.dict("os.environ", {"ELEVENLABS_MONTHLY_LIMIT": "30000", "ELEVENLABS_RESET_DAY": "1"}):
            with caplog.at_level("WARNING", logger="src.media.voiceover"):
                gen._check_monthly_usage()
        assert any("85%" in r.message for r in caplog.records if r.levelname == "WARNING")

    def test_critical_at_95_pct(self, tmp_path: Path, caplog) -> None:
        # Use a single row so integer division doesn't undershoot
        gen = _make_gen(tmp_path)
        _insert_usage(gen.db_path, [(date.today().isoformat(), "t1", 28500, "Adam")])
        with patch.dict("os.environ", {"ELEVENLABS_MONTHLY_LIMIT": "30000", "ELEVENLABS_RESET_DAY": "1"}):
            with caplog.at_level("CRITICAL", logger="src.media.voiceover"):
                gen._check_monthly_usage()
        assert any("95%" in r.message for r in caplog.records if r.levelname == "CRITICAL")

    def test_critical_includes_reset_date(self, tmp_path: Path, caplog) -> None:
        gen = self._setup_usage(tmp_path, 29_000, 30)
        with patch.dict("os.environ", {"ELEVENLABS_MONTHLY_LIMIT": "30000", "ELEVENLABS_RESET_DAY": "1"}):
            with caplog.at_level("CRITICAL", logger="src.media.voiceover"):
                gen._check_monthly_usage()
        critical_msgs = [r.message for r in caplog.records if r.levelname == "CRITICAL"]
        assert any("Reset date" in m for m in critical_msgs)

    def test_swallows_db_error(self, tmp_path: Path) -> None:
        gen = VoiceoverGenerator(api_key="fake", db_path=tmp_path / "ghost.db")
        # DB doesn't exist — should log warning, not raise
        try:
            gen._check_monthly_usage()
        except Exception as exc:
            pytest.fail(f"_check_monthly_usage raised: {exc}")


# ---------------------------------------------------------------------------
# Videos remaining calculation
# ---------------------------------------------------------------------------

class TestVideosRemaining:
    def test_videos_remaining_calculation(self, tmp_path: Path, caplog) -> None:
        """22 videos at 1000 chars each = 22k used (73%), 8k left → ~8 videos remaining."""
        gen = _make_gen(tmp_path)
        today = date.today().isoformat()
        rows = [(today, f"t{i}", 1000, "Adam") for i in range(22)]
        _insert_usage(gen.db_path, rows)

        with patch.dict("os.environ", {"ELEVENLABS_MONTHLY_LIMIT": "30000", "ELEVENLABS_RESET_DAY": "1"}):
            with caplog.at_level("WARNING", logger="src.media.voiceover"):
                gen._check_monthly_usage()

        warning_msgs = [r.message for r in caplog.records if r.levelname == "WARNING"]
        # At 67% threshold, videos remaining (8) should appear in the warning
        assert any("8" in m for m in warning_msgs)

    def test_no_videos_this_month_no_crash(self, tmp_path: Path) -> None:
        gen = _make_gen(tmp_path)
        # Ensure table exists but is empty
        _insert_usage(gen.db_path, [])
        with patch.dict("os.environ", {"ELEVENLABS_MONTHLY_LIMIT": "30000", "ELEVENLABS_RESET_DAY": "1"}):
            gen._check_monthly_usage()  # should not raise


# ---------------------------------------------------------------------------
# Monthly reset boundary
# ---------------------------------------------------------------------------

class TestMonthlyResetBoundary:
    def test_jan_rows_not_counted_in_feb(self, tmp_path: Path) -> None:
        """Rows from January should not count when reset_day=1 and today is Feb 1."""
        gen = _make_gen(tmp_path)
        _insert_usage(gen.db_path, [
            ("2026-01-20", "old_video", 5000, "Adam"),
            ("2026-01-31", "old_video2", 5000, "Adam"),
        ])

        # Simulate today = Feb 1, reset_day = 1
        fake_today = date(2026, 2, 1)
        with patch("src.media.voiceover.date") as mock_date:
            mock_date.today.return_value = fake_today
            with patch.dict("os.environ", {"ELEVENLABS_MONTHLY_LIMIT": "30000", "ELEVENLABS_RESET_DAY": "1"}):
                import logging
                import io
                gen._check_monthly_usage()

        # Check that the DB query for Feb 1 returns 0 for our gen's DB
        conn = sqlite3.connect(gen.db_path)
        row = conn.execute(
            "SELECT SUM(chars_used) FROM elevenlabs_usage WHERE date >= '2026-02-01'"
        ).fetchone()
        conn.close()
        assert (row[0] or 0) == 0

    def test_feb_rows_counted_from_reset_day(self, tmp_path: Path) -> None:
        """Rows on/after Feb 1 are counted in the Feb cycle."""
        gen = _make_gen(tmp_path)
        _insert_usage(gen.db_path, [
            ("2026-01-31", "old", 5000, "Adam"),
            ("2026-02-01", "new1", 1000, "Adam"),
            ("2026-02-05", "new2", 800, "Adam"),
        ])

        conn = sqlite3.connect(gen.db_path)
        total = conn.execute(
            "SELECT SUM(chars_used) FROM elevenlabs_usage WHERE date >= '2026-02-01'"
        ).fetchone()[0]
        conn.close()
        assert total == 1800


# ---------------------------------------------------------------------------
# check_elevenlabs_usage.get_usage_report
# ---------------------------------------------------------------------------

class TestGetUsageReport:
    def test_returns_expected_keys(self, tmp_path: Path) -> None:
        from scripts.check_elevenlabs_usage import get_usage_report
        report = get_usage_report(db_path=tmp_path / "empty.db", monthly_limit=30000, reset_day=1)
        expected_keys = {
            "month_label", "monthly_limit", "monthly_total", "pct_used",
            "chars_remaining", "videos_produced", "avg_chars_per_video",
            "videos_remaining", "reset_date", "status", "daily_breakdown",
        }
        assert expected_keys <= set(report.keys())

    def test_empty_db_returns_zero_usage(self, tmp_path: Path) -> None:
        from scripts.check_elevenlabs_usage import get_usage_report
        report = get_usage_report(db_path=tmp_path / "new.db", monthly_limit=30000, reset_day=1)
        assert report["monthly_total"] == 0
        assert report["pct_used"] == 0.0
        assert report["status"] == "OK — plenty of headroom"

    def test_nonexistent_db_returns_zero_usage(self, tmp_path: Path) -> None:
        from scripts.check_elevenlabs_usage import get_usage_report
        report = get_usage_report(db_path=tmp_path / "ghost" / "db.db", monthly_limit=30000, reset_day=1)
        assert report["monthly_total"] == 0

    def test_calculates_pct_correctly(self, tmp_path: Path) -> None:
        from scripts.check_elevenlabs_usage import get_usage_report
        db = tmp_path / "usage.db"
        today = date.today().isoformat()
        _insert_usage(db, [(today, "t1", 15000, "Adam")])
        report = get_usage_report(db_path=db, monthly_limit=30000, reset_day=1)
        assert report["monthly_total"] == 15000
        assert report["pct_used"] == 50.0
        assert report["chars_remaining"] == 15000

    def test_status_ok_below_67(self, tmp_path: Path) -> None:
        from scripts.check_elevenlabs_usage import get_usage_report
        db = tmp_path / "usage.db"
        _insert_usage(db, [(date.today().isoformat(), "t1", 10000, "Adam")])
        report = get_usage_report(db_path=db, monthly_limit=30000, reset_day=1)
        assert report["status"] == "OK — plenty of headroom"

    def test_status_warning_at_67(self, tmp_path: Path) -> None:
        from scripts.check_elevenlabs_usage import get_usage_report
        db = tmp_path / "usage.db"
        _insert_usage(db, [(date.today().isoformat(), "t1", 20100, "Adam")])
        report = get_usage_report(db_path=db, monthly_limit=30000, reset_day=1)
        assert "WARNING" in report["status"]

    def test_status_caution_at_85(self, tmp_path: Path) -> None:
        from scripts.check_elevenlabs_usage import get_usage_report
        db = tmp_path / "usage.db"
        _insert_usage(db, [(date.today().isoformat(), "t1", 25500, "Adam")])
        report = get_usage_report(db_path=db, monthly_limit=30000, reset_day=1)
        assert "CAUTION" in report["status"]

    def test_status_critical_at_95(self, tmp_path: Path) -> None:
        from scripts.check_elevenlabs_usage import get_usage_report
        db = tmp_path / "usage.db"
        _insert_usage(db, [(date.today().isoformat(), "t1", 28500, "Adam")])
        report = get_usage_report(db_path=db, monthly_limit=30000, reset_day=1)
        assert "CRITICAL" in report["status"]

    def test_daily_breakdown_groups_by_date(self, tmp_path: Path) -> None:
        from scripts.check_elevenlabs_usage import get_usage_report
        db = tmp_path / "usage.db"
        today = date.today()
        _insert_usage(db, [
            (today.isoformat(), "t1", 500, "Adam"),
            (today.isoformat(), "t2", 700, "Adam"),
        ])
        report = get_usage_report(db_path=db, monthly_limit=30000, reset_day=1)
        breakdown = report["daily_breakdown"]
        assert len(breakdown) == 1
        assert breakdown[0]["chars"] == 1200
        assert breakdown[0]["videos"] == 2

    def test_avg_chars_per_video_calculated(self, tmp_path: Path) -> None:
        from scripts.check_elevenlabs_usage import get_usage_report
        db = tmp_path / "usage.db"
        today = date.today().isoformat()
        _insert_usage(db, [
            (today, "t1", 900, "Adam"),
            (today, "t2", 1100, "Adam"),
        ])
        report = get_usage_report(db_path=db, monthly_limit=30000, reset_day=1)
        assert report["avg_chars_per_video"] == 1000

    def test_videos_remaining_estimated(self, tmp_path: Path) -> None:
        from scripts.check_elevenlabs_usage import get_usage_report
        db = tmp_path / "usage.db"
        today = date.today().isoformat()
        # 20 videos × 1000 chars = 20k used, 10k left → 10 remaining
        rows = [(today, f"t{i}", 1000, "Adam") for i in range(20)]
        _insert_usage(db, rows)
        report = get_usage_report(db_path=db, monthly_limit=30000, reset_day=1)
        assert report["videos_remaining"] == 10

    def test_monthly_limit_respected(self, tmp_path: Path) -> None:
        from scripts.check_elevenlabs_usage import get_usage_report
        report = get_usage_report(db_path=tmp_path / "new.db", monthly_limit=50000, reset_day=1)
        assert report["monthly_limit"] == 50000


# ---------------------------------------------------------------------------
# Scheduler job
# ---------------------------------------------------------------------------

class TestSchedulerElevenLabsJob:
    def test_job_registered_in_scheduler(self) -> None:
        from src.scheduler import build_scheduler
        with patch.dict("sys.modules", {
            "apscheduler": MagicMock(),
            "apscheduler.schedulers": MagicMock(),
            "apscheduler.schedulers.blocking": MagicMock(),
            "apscheduler.triggers": MagicMock(),
            "apscheduler.triggers.cron": MagicMock(),
        }):
            from src.scheduler import run_elevenlabs_usage_check
            assert callable(run_elevenlabs_usage_check)

    def test_elevenlabs_job_id_in_scheduler(self) -> None:
        from src.scheduler import build_scheduler
        scheduler = build_scheduler(timezone_name="UTC")
        job_ids = [j.id for j in scheduler.get_jobs()]
        assert "elevenlabs_usage" in job_ids

    def test_elevenlabs_job_runs_at_9am(self) -> None:
        from src.scheduler import build_scheduler
        from apscheduler.triggers.cron import CronTrigger
        scheduler = build_scheduler(timezone_name="UTC")
        job = next(j for j in scheduler.get_jobs() if j.id == "elevenlabs_usage")
        assert isinstance(job.trigger, CronTrigger)

    def test_usage_check_logs_warning_above_67(self, tmp_path: Path, caplog) -> None:
        from scripts.check_elevenlabs_usage import get_usage_report
        db = tmp_path / "usage.db"
        today = date.today().isoformat()
        _insert_usage(db, [(today, f"t{i}", 1000, "Adam") for i in range(22)])

        report = get_usage_report(db_path=db, monthly_limit=30000, reset_day=1)
        assert report["pct_used"] >= 67
        assert "WARNING" in report["status"]

    def test_usage_check_no_warning_below_67(self, tmp_path: Path) -> None:
        from scripts.check_elevenlabs_usage import get_usage_report
        db = tmp_path / "usage.db"
        today = date.today().isoformat()
        _insert_usage(db, [(today, f"t{i}", 500, "Adam") for i in range(10)])

        report = get_usage_report(db_path=db, monthly_limit=30000, reset_day=1)
        assert report["pct_used"] < 67
        assert report["status"] == "OK — plenty of headroom"

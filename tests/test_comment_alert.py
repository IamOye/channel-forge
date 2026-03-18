"""
Tests for the comment alert + approval system additions:

- TelegramNotifier.fmt_new_comment_alert / send_new_comment_alert
- CommentResponder.detect_and_alert (saves to DB, sends Telegram, no autopost)
- Scheduler comment_check job registration
"""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.notifications.telegram_notifier import TelegramNotifier
from src.publisher.comment_responder import CommentResponder, ReplyResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db(tmp_path: Path) -> Path:
    """Create a fresh SQLite DB with comment_states + settings tables."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS comment_states (
            comment_id          TEXT PRIMARY KEY,
            video_id            TEXT,
            commenter           TEXT,
            comment_text        TEXT,
            suggested_reply     TEXT,
            edited_reply        TEXT,
            state               TEXT NOT NULL DEFAULT 'PENDING_APPROVAL',
            telegram_message_id INTEGER,
            created_at          TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at          TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS settings (
            key        TEXT PRIMARY KEY,
            value      TEXT,
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        INSERT OR IGNORE INTO settings (key, value) VALUES ('telegram_automode', 'on');
    """)
    conn.commit()
    conn.close()
    return db_path


# ---------------------------------------------------------------------------
# TelegramNotifier — fmt_new_comment_alert
# ---------------------------------------------------------------------------

class TestFmtNewCommentAlert:
    def test_contains_commenter(self) -> None:
        msg = TelegramNotifier.fmt_new_comment_alert(
            commenter="Jordan", comment_text="Great!", video_title="V",
            comment_id="abc123", suggested_reply="Thanks!"
        )
        assert "Jordan" in msg

    def test_contains_comment_text(self) -> None:
        msg = TelegramNotifier.fmt_new_comment_alert(
            commenter="J", comment_text="This changed my life",
            video_title="V", comment_id="x", suggested_reply="S"
        )
        assert "This changed my life" in msg

    def test_contains_video_title(self) -> None:
        msg = TelegramNotifier.fmt_new_comment_alert(
            commenter="J", comment_text="C", video_title="Rich People Secrets",
            comment_id="x", suggested_reply="S"
        )
        assert "Rich People Secrets" in msg

    def test_contains_comment_id_in_commands(self) -> None:
        msg = TelegramNotifier.fmt_new_comment_alert(
            commenter="J", comment_text="C", video_title="V",
            comment_id="abc123", suggested_reply="S"
        )
        assert "/approve_abc123" in msg
        assert "/edit_abc123" in msg
        assert "/skip_abc123" in msg

    def test_contains_suggested_reply(self) -> None:
        msg = TelegramNotifier.fmt_new_comment_alert(
            commenter="J", comment_text="C", video_title="V",
            comment_id="x", suggested_reply="Glad you found it useful!"
        )
        assert "Glad you found it useful!" in msg

    def test_contains_new_comment_header(self) -> None:
        msg = TelegramNotifier.fmt_new_comment_alert(
            commenter="J", comment_text="C", video_title="V",
            comment_id="x", suggested_reply="S"
        )
        assert "New Comment" in msg


class TestSendNewCommentAlert:
    def test_calls_send_with_formatted_message(self) -> None:
        n = TelegramNotifier(token="tok", chat_id="123")
        sent = []
        n.send = lambda msg: sent.append(msg) or True

        result = n.send_new_comment_alert(
            commenter="Jordan", comment_text="Yes!",
            video_title="Money Tips", video_id="v1",
            comment_id="c1", suggested_reply="Here you go!"
        )
        assert result is True
        assert len(sent) == 1
        assert "Jordan" in sent[0]
        assert "/approve_c1" in sent[0]


# ---------------------------------------------------------------------------
# CommentResponder.detect_and_alert
# ---------------------------------------------------------------------------

class TestDetectAndAlert:
    def _make_comment(self, comment_id: str = "c1") -> dict:
        return {
            "comment_id": comment_id,
            "video_id": "v1",
            "commenter": "Jordan",
            "comment_text": "Yes!",
            "video_title": "Money Tips",
            "category": "money",
            "trigger_type": "YES",
        }

    @patch("src.publisher.comment_responder.anthropic.Anthropic")
    @patch("src.notifications.telegram_notifier.TelegramNotifier")
    def test_saves_comment_to_db(self, mock_tg_cls, mock_api_cls, db: Path) -> None:
        mock_client = MagicMock()
        msg = MagicMock()
        msg.content = [MagicMock(text="Thanks for watching!")]
        mock_client.messages.create.return_value = msg
        mock_api_cls.return_value = mock_client

        mock_tg = MagicMock()
        mock_tg.send_new_comment_alert.return_value = True
        mock_tg_cls.return_value = mock_tg

        responder = CommentResponder(api_key="fake")
        results = responder.detect_and_alert([self._make_comment()], db_path=db)

        assert len(results) == 1
        conn = sqlite3.connect(db)
        row = conn.execute("SELECT * FROM comment_states WHERE comment_id = 'c1'").fetchone()
        conn.close()
        assert row is not None

    @patch("src.publisher.comment_responder.anthropic.Anthropic")
    @patch("src.notifications.telegram_notifier.TelegramNotifier")
    def test_sends_telegram_alert(self, mock_tg_cls, mock_api_cls, db: Path) -> None:
        mock_client = MagicMock()
        msg = MagicMock()
        msg.content = [MagicMock(text="Thanks!")]
        mock_client.messages.create.return_value = msg
        mock_api_cls.return_value = mock_client

        mock_tg = MagicMock()
        mock_tg.send_new_comment_alert.return_value = True
        mock_tg_cls.return_value = mock_tg

        responder = CommentResponder(api_key="fake")
        responder.detect_and_alert([self._make_comment()], db_path=db)

        mock_tg.send_new_comment_alert.assert_called_once()

    @patch("src.publisher.comment_responder.anthropic.Anthropic")
    @patch("src.notifications.telegram_notifier.TelegramNotifier")
    def test_state_is_pending_approval(self, mock_tg_cls, mock_api_cls, db: Path) -> None:
        mock_client = MagicMock()
        msg = MagicMock()
        msg.content = [MagicMock(text="Reply text")]
        mock_client.messages.create.return_value = msg
        mock_api_cls.return_value = mock_client
        mock_tg_cls.return_value = MagicMock()

        responder = CommentResponder(api_key="fake")
        responder.detect_and_alert([self._make_comment()], db_path=db)

        conn = sqlite3.connect(db)
        state = conn.execute(
            "SELECT state FROM comment_states WHERE comment_id = 'c1'"
        ).fetchone()[0]
        conn.close()
        assert state == "PENDING_APPROVAL"

    @patch("src.publisher.comment_responder.anthropic.Anthropic")
    @patch("src.notifications.telegram_notifier.TelegramNotifier")
    def test_does_not_post_to_youtube(self, mock_tg_cls, mock_api_cls, db: Path) -> None:
        """The detect_and_alert method must never post to YouTube."""
        mock_client = MagicMock()
        msg = MagicMock()
        msg.content = [MagicMock(text="Reply")]
        mock_client.messages.create.return_value = msg
        mock_api_cls.return_value = mock_client
        mock_tg_cls.return_value = MagicMock()

        with patch.object(CommentResponder, "post_reply") as mock_post:
            responder = CommentResponder(api_key="fake")
            responder.detect_and_alert([self._make_comment()], db_path=db)
            mock_post.assert_not_called()

    @patch("src.publisher.comment_responder.anthropic.Anthropic")
    @patch("src.notifications.telegram_notifier.TelegramNotifier")
    def test_skips_duplicate_comment(self, mock_tg_cls, mock_api_cls, db: Path) -> None:
        # Pre-insert the comment
        conn = sqlite3.connect(db)
        conn.execute(
            "INSERT INTO comment_states (comment_id, state) VALUES ('c1', 'APPROVED')"
        )
        conn.commit()
        conn.close()

        mock_client = MagicMock()
        mock_api_cls.return_value = mock_client
        mock_tg_cls.return_value = MagicMock()

        responder = CommentResponder(api_key="fake")
        results = responder.detect_and_alert([self._make_comment()], db_path=db)
        assert len(results) == 0

    @patch("src.publisher.comment_responder.anthropic.Anthropic")
    @patch("src.notifications.telegram_notifier.TelegramNotifier")
    def test_skips_reply_generation_when_automode_off(self, mock_tg_cls, mock_api_cls, db: Path) -> None:
        # Set automode to off
        conn = sqlite3.connect(db)
        conn.execute("UPDATE settings SET value = 'off' WHERE key = 'telegram_automode'")
        conn.commit()
        conn.close()

        mock_client = MagicMock()
        mock_api_cls.return_value = mock_client
        mock_tg = MagicMock()
        mock_tg.send_new_comment_alert.return_value = True
        mock_tg_cls.return_value = mock_tg

        responder = CommentResponder(api_key="fake")
        results = responder.detect_and_alert([self._make_comment()], db_path=db)

        # Should not have called Claude API
        mock_client.messages.create.assert_not_called()
        # But should still save and alert
        assert len(results) == 1

    @patch("src.publisher.comment_responder.anthropic.Anthropic")
    @patch("src.notifications.telegram_notifier.TelegramNotifier")
    def test_processes_multiple_comments(self, mock_tg_cls, mock_api_cls, db: Path) -> None:
        mock_client = MagicMock()
        msg = MagicMock()
        msg.content = [MagicMock(text="Reply")]
        mock_client.messages.create.return_value = msg
        mock_api_cls.return_value = mock_client
        mock_tg_cls.return_value = MagicMock()

        comments = [self._make_comment("c1"), self._make_comment("c2"), self._make_comment("c3")]
        responder = CommentResponder(api_key="fake")
        results = responder.detect_and_alert(comments, db_path=db)
        assert len(results) == 3

    @patch("src.publisher.comment_responder.anthropic.Anthropic")
    @patch("src.notifications.telegram_notifier.TelegramNotifier")
    def test_skips_comment_without_id(self, mock_tg_cls, mock_api_cls, db: Path) -> None:
        mock_api_cls.return_value = MagicMock()
        mock_tg_cls.return_value = MagicMock()

        responder = CommentResponder(api_key="fake")
        results = responder.detect_and_alert([{"comment_id": ""}], db_path=db)
        assert len(results) == 0


# ---------------------------------------------------------------------------
# Scheduler — comment_check job
# ---------------------------------------------------------------------------

class TestSchedulerCommentCheck:
    def test_comment_check_job_registered(self) -> None:
        from src.scheduler import build_scheduler
        scheduler = build_scheduler("UTC")
        job_ids = [j.id for j in scheduler.get_jobs()]
        assert "comment_check" in job_ids

    def test_comment_check_uses_correct_function(self) -> None:
        from src.scheduler import build_scheduler, run_comment_check
        scheduler = build_scheduler("UTC")
        job = next(j for j in scheduler.get_jobs() if j.id == "comment_check")
        assert job.func is run_comment_check

    def test_comment_check_runs_hourly(self) -> None:
        from apscheduler.triggers.cron import CronTrigger
        from src.scheduler import build_scheduler
        scheduler = build_scheduler("UTC")
        job = next(j for j in scheduler.get_jobs() if j.id == "comment_check")
        assert isinstance(job.trigger, CronTrigger)


# ---------------------------------------------------------------------------
# init_db — new tables
# ---------------------------------------------------------------------------

class TestInitDbNewTables:
    def test_comment_states_table_created(self, tmp_path: Path) -> None:
        from scripts.init_db import create_db, MAIN_DDL_PARTS
        db_path = tmp_path / "test_init.db"
        create_db(db_path, MAIN_DDL_PARTS)
        conn = sqlite3.connect(db_path)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        conn.close()
        assert "comment_states" in tables

    def test_settings_table_created(self, tmp_path: Path) -> None:
        from scripts.init_db import create_db, MAIN_DDL_PARTS
        db_path = tmp_path / "test_init.db"
        create_db(db_path, MAIN_DDL_PARTS)
        conn = sqlite3.connect(db_path)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        conn.close()
        assert "settings" in tables

    def test_settings_has_automode_default(self, tmp_path: Path) -> None:
        from scripts.init_db import create_db, MAIN_DDL_PARTS
        db_path = tmp_path / "test_init.db"
        create_db(db_path, MAIN_DDL_PARTS)
        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT value FROM settings WHERE key = 'telegram_automode'").fetchone()
        conn.close()
        assert row[0] == "on"

    def test_settings_has_check_interval_default(self, tmp_path: Path) -> None:
        from scripts.init_db import create_db, MAIN_DDL_PARTS
        db_path = tmp_path / "test_init.db"
        create_db(db_path, MAIN_DDL_PARTS)
        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT value FROM settings WHERE key = 'comment_check_interval'").fetchone()
        conn.close()
        assert row[0] == "60"

"""
Tests for src/publisher/telegram_reply_handler.py

All YouTube API calls and Telegram sends are mocked — no real API activity.
"""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.publisher.telegram_reply_handler import (
    VALID_STATES,
    TelegramReplyHandler,
)


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


def _handler(db: Path) -> TelegramReplyHandler:
    return TelegramReplyHandler(token="tok", chat_id="123", db_path=db)


def _insert_comment(db: Path, comment_id: str = "c1", state: str = "PENDING_APPROVAL", **kw) -> None:
    defaults = dict(
        video_id="v1", commenter="Jordan", comment_text="Great video!",
        suggested_reply="Thanks for watching!", edited_reply=None,
    )
    defaults.update(kw)
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO comment_states (comment_id, video_id, commenter, comment_text, "
        "suggested_reply, edited_reply, state) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (comment_id, defaults["video_id"], defaults["commenter"],
         defaults["comment_text"], defaults["suggested_reply"],
         defaults["edited_reply"], state),
    )
    conn.commit()
    conn.close()


def _get_state(db: Path, comment_id: str) -> str | None:
    conn = sqlite3.connect(db)
    row = conn.execute("SELECT state FROM comment_states WHERE comment_id = ?", (comment_id,)).fetchone()
    conn.close()
    return row[0] if row else None


# ---------------------------------------------------------------------------
# VALID_STATES
# ---------------------------------------------------------------------------

class TestValidStates:
    def test_contains_all_expected_states(self) -> None:
        expected = {"PENDING_APPROVAL", "PENDING_CONFIRMATION", "PENDING_EDIT",
                    "PENDING_EDIT_CONFIRMATION", "APPROVED", "SKIPPED"}
        assert VALID_STATES == expected


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

class TestGetCommentState:
    def test_returns_dict_for_existing_comment(self, db: Path) -> None:
        _insert_comment(db)
        h = _handler(db)
        row = h.get_comment_state("c1")
        assert row is not None
        assert row["comment_id"] == "c1"

    def test_returns_none_for_missing_comment(self, db: Path) -> None:
        h = _handler(db)
        assert h.get_comment_state("nonexistent") is None


class TestUpdateState:
    def test_updates_state(self, db: Path) -> None:
        _insert_comment(db)
        h = _handler(db)
        assert h.update_state("c1", "PENDING_CONFIRMATION") is True
        assert _get_state(db, "c1") == "PENDING_CONFIRMATION"

    def test_rejects_invalid_state(self, db: Path) -> None:
        _insert_comment(db)
        h = _handler(db)
        assert h.update_state("c1", "INVALID_STATE") is False

    def test_updates_extra_columns(self, db: Path) -> None:
        _insert_comment(db)
        h = _handler(db)
        h.update_state("c1", "PENDING_EDIT_CONFIRMATION", edited_reply="New text")
        row = h.get_comment_state("c1")
        assert row["edited_reply"] == "New text"

    def test_returns_false_for_missing_comment(self, db: Path) -> None:
        h = _handler(db)
        assert h.update_state("nonexistent", "APPROVED") is False


class TestGetPendingComments:
    def test_returns_pending_comments(self, db: Path) -> None:
        _insert_comment(db, "c1", "PENDING_APPROVAL")
        _insert_comment(db, "c2", "APPROVED")
        _insert_comment(db, "c3", "PENDING_APPROVAL")
        h = _handler(db)
        pending = h.get_pending_comments()
        ids = [r["comment_id"] for r in pending]
        assert "c1" in ids
        assert "c3" in ids
        assert "c2" not in ids

    def test_returns_empty_when_none_pending(self, db: Path) -> None:
        h = _handler(db)
        assert h.get_pending_comments() == []


class TestSettings:
    def test_get_setting_returns_value(self, db: Path) -> None:
        h = _handler(db)
        assert h.get_setting("telegram_automode") == "on"

    def test_get_setting_returns_default_for_missing_key(self, db: Path) -> None:
        h = _handler(db)
        assert h.get_setting("nonexistent", "fallback") == "fallback"

    def test_set_setting_inserts_new_key(self, db: Path) -> None:
        h = _handler(db)
        h.set_setting("new_key", "new_value")
        assert h.get_setting("new_key") == "new_value"

    def test_set_setting_updates_existing_key(self, db: Path) -> None:
        h = _handler(db)
        h.set_setting("telegram_automode", "off")
        assert h.get_setting("telegram_automode") == "off"


# ---------------------------------------------------------------------------
# handle_approve
# ---------------------------------------------------------------------------

class TestHandleApprove:
    def test_sends_confirmation_prompt(self, db: Path) -> None:
        _insert_comment(db)
        h = _handler(db)
        resp = h.handle_approve("c1")
        assert "Confirm Post" in resp
        assert "Thanks for watching!" in resp
        assert "YES" in resp

    def test_transitions_to_pending_confirmation(self, db: Path) -> None:
        _insert_comment(db)
        h = _handler(db)
        h.handle_approve("c1")
        assert _get_state(db, "c1") == "PENDING_CONFIRMATION"

    def test_returns_error_for_missing_comment(self, db: Path) -> None:
        h = _handler(db)
        resp = h.handle_approve("nope")
        assert "not found" in resp

    def test_returns_error_for_wrong_state(self, db: Path) -> None:
        _insert_comment(db, state="APPROVED")
        h = _handler(db)
        resp = h.handle_approve("c1")
        assert "cannot approve" in resp

    def test_uses_edited_reply_when_available(self, db: Path) -> None:
        _insert_comment(db, edited_reply="Edited text here")
        h = _handler(db)
        resp = h.handle_approve("c1")
        assert "Edited text here" in resp


# ---------------------------------------------------------------------------
# handle_edit
# ---------------------------------------------------------------------------

class TestHandleEdit:
    def test_enters_edit_mode(self, db: Path) -> None:
        _insert_comment(db)
        h = _handler(db)
        resp = h.handle_edit("c1")
        assert "Edit Reply" in resp
        assert "Thanks for watching!" in resp

    def test_transitions_to_pending_edit(self, db: Path) -> None:
        _insert_comment(db)
        h = _handler(db)
        h.handle_edit("c1")
        assert _get_state(db, "c1") == "PENDING_EDIT"

    def test_returns_error_for_missing_comment(self, db: Path) -> None:
        h = _handler(db)
        resp = h.handle_edit("nope")
        assert "not found" in resp

    def test_returns_error_for_wrong_state(self, db: Path) -> None:
        _insert_comment(db, state="APPROVED")
        h = _handler(db)
        resp = h.handle_edit("c1")
        assert "cannot edit" in resp


# ---------------------------------------------------------------------------
# handle_skip
# ---------------------------------------------------------------------------

class TestHandleSkip:
    def test_marks_comment_skipped(self, db: Path) -> None:
        _insert_comment(db)
        h = _handler(db)
        resp = h.handle_skip("c1")
        assert "skipped" in resp.lower()
        assert _get_state(db, "c1") == "SKIPPED"

    def test_returns_error_for_missing_comment(self, db: Path) -> None:
        h = _handler(db)
        resp = h.handle_skip("nope")
        assert "not found" in resp


# ---------------------------------------------------------------------------
# handle_pending
# ---------------------------------------------------------------------------

class TestHandlePending:
    def test_lists_pending_comments(self, db: Path) -> None:
        _insert_comment(db, "c1")
        _insert_comment(db, "c2", commenter="Alex", comment_text="Nice!")
        h = _handler(db)
        resp = h.handle_pending()
        assert "Pending Comments (2)" in resp
        assert "/approve_c1" in resp
        assert "/edit_c2" in resp
        assert "/skip_c1" in resp

    def test_shows_no_pending_when_empty(self, db: Path) -> None:
        h = _handler(db)
        resp = h.handle_pending()
        assert "No pending comments" in resp


# ---------------------------------------------------------------------------
# handle_automode
# ---------------------------------------------------------------------------

class TestHandleAutomode:
    def test_automode_on(self, db: Path) -> None:
        h = _handler(db)
        resp = h.handle_automode("on")
        assert "Automode ON" in resp
        assert h.get_setting("telegram_automode") == "on"

    def test_automode_off(self, db: Path) -> None:
        h = _handler(db)
        resp = h.handle_automode("off")
        assert "Automode OFF" in resp
        assert h.get_setting("telegram_automode") == "off"

    def test_automode_invalid(self, db: Path) -> None:
        h = _handler(db)
        resp = h.handle_automode("maybe")
        assert "Usage" in resp

    def test_manual_approval_always_required_message(self, db: Path) -> None:
        h = _handler(db)
        resp = h.handle_automode("on")
        assert "Manual approval" in resp or "manual approval" in resp.lower()


# ---------------------------------------------------------------------------
# handle_yes
# ---------------------------------------------------------------------------

class TestHandleYes:
    @patch("src.publisher.telegram_reply_handler.TelegramReplyHandler.post_reply_to_youtube", return_value=True)
    def test_yes_posts_to_youtube(self, mock_post, db: Path) -> None:
        _insert_comment(db, state="PENDING_CONFIRMATION")
        h = _handler(db)
        resp = h.handle_yes()
        assert "posted successfully" in resp
        mock_post.assert_called_once_with("c1", "Thanks for watching!")

    @patch("src.publisher.telegram_reply_handler.TelegramReplyHandler.post_reply_to_youtube", return_value=True)
    def test_yes_transitions_to_approved(self, mock_post, db: Path) -> None:
        _insert_comment(db, state="PENDING_CONFIRMATION")
        h = _handler(db)
        h.handle_yes()
        assert _get_state(db, "c1") == "APPROVED"

    @patch("src.publisher.telegram_reply_handler.TelegramReplyHandler.post_reply_to_youtube", return_value=True)
    def test_yes_uses_edited_reply_when_available(self, mock_post, db: Path) -> None:
        _insert_comment(db, state="PENDING_EDIT_CONFIRMATION", edited_reply="Custom reply")
        h = _handler(db)
        h.handle_yes()
        mock_post.assert_called_once_with("c1", "Custom reply")

    @patch("src.publisher.telegram_reply_handler.TelegramReplyHandler.post_reply_to_youtube", return_value=False)
    def test_yes_reverts_on_youtube_failure(self, mock_post, db: Path) -> None:
        _insert_comment(db, state="PENDING_CONFIRMATION")
        h = _handler(db)
        resp = h.handle_yes()
        assert "Failed" in resp
        assert _get_state(db, "c1") == "PENDING_APPROVAL"

    def test_yes_no_pending_confirmation(self, db: Path) -> None:
        h = _handler(db)
        resp = h.handle_yes()
        assert "No comment awaiting" in resp


# ---------------------------------------------------------------------------
# handle_no
# ---------------------------------------------------------------------------

class TestHandleNo:
    def test_no_cancels_confirmation(self, db: Path) -> None:
        _insert_comment(db, state="PENDING_CONFIRMATION")
        h = _handler(db)
        resp = h.handle_no()
        assert "Cancelled" in resp
        assert _get_state(db, "c1") == "PENDING_APPROVAL"

    def test_no_returns_to_edit_after_edit_confirmation(self, db: Path) -> None:
        _insert_comment(db, state="PENDING_EDIT_CONFIRMATION", edited_reply="Draft")
        h = _handler(db)
        resp = h.handle_no()
        assert "Edit Reply" in resp
        assert _get_state(db, "c1") == "PENDING_EDIT"

    def test_no_when_nothing_pending(self, db: Path) -> None:
        h = _handler(db)
        resp = h.handle_no()
        assert "No comment awaiting" in resp


# ---------------------------------------------------------------------------
# handle_free_text (edit flow)
# ---------------------------------------------------------------------------

class TestHandleFreeText:
    def test_stores_edited_reply(self, db: Path) -> None:
        _insert_comment(db, state="PENDING_EDIT")
        h = _handler(db)
        resp = h.handle_free_text("My custom reply here")
        assert resp is not None
        assert "Confirm Edited Reply" in resp
        assert "My custom reply here" in resp

    def test_transitions_to_pending_edit_confirmation(self, db: Path) -> None:
        _insert_comment(db, state="PENDING_EDIT")
        h = _handler(db)
        h.handle_free_text("Edited text")
        assert _get_state(db, "c1") == "PENDING_EDIT_CONFIRMATION"

    def test_returns_none_when_no_edit_pending(self, db: Path) -> None:
        h = _handler(db)
        assert h.handle_free_text("random text") is None

    def test_stores_edited_reply_in_db(self, db: Path) -> None:
        _insert_comment(db, state="PENDING_EDIT")
        h = _handler(db)
        h.handle_free_text("New version")
        row = h.get_comment_state("c1")
        assert row["edited_reply"] == "New version"


# ---------------------------------------------------------------------------
# route_message
# ---------------------------------------------------------------------------

class TestRouteMessage:
    def test_routes_approve_command(self, db: Path) -> None:
        _insert_comment(db)
        h = _handler(db)
        resp = h.route_message("/approve_c1")
        assert "Confirm Post" in resp

    def test_routes_edit_command(self, db: Path) -> None:
        _insert_comment(db)
        h = _handler(db)
        resp = h.route_message("/edit_c1")
        assert "Edit Reply" in resp

    def test_routes_skip_command(self, db: Path) -> None:
        _insert_comment(db)
        h = _handler(db)
        resp = h.route_message("/skip_c1")
        assert "skipped" in resp.lower()

    def test_routes_pending_command(self, db: Path) -> None:
        h = _handler(db)
        resp = h.route_message("/pending")
        assert "pending" in resp.lower()

    def test_routes_automode_command(self, db: Path) -> None:
        h = _handler(db)
        resp = h.route_message("/automode on")
        assert "Automode ON" in resp

    def test_routes_yes(self, db: Path) -> None:
        h = _handler(db)
        resp = h.route_message("YES")
        assert resp is not None

    def test_routes_no(self, db: Path) -> None:
        h = _handler(db)
        resp = h.route_message("NO")
        assert resp is not None

    def test_routes_yes_case_insensitive(self, db: Path) -> None:
        h = _handler(db)
        resp = h.route_message("yes")
        assert resp is not None

    def test_routes_free_text_for_edit(self, db: Path) -> None:
        _insert_comment(db, state="PENDING_EDIT")
        h = _handler(db)
        resp = h.route_message("My edited reply")
        assert "Confirm Edited Reply" in resp

    def test_returns_none_for_unknown_command(self, db: Path) -> None:
        h = _handler(db)
        resp = h.route_message("/unknown_command")
        assert resp is None

    def test_returns_none_for_free_text_no_edit(self, db: Path) -> None:
        h = _handler(db)
        resp = h.route_message("just some text")
        assert resp is None


# ---------------------------------------------------------------------------
# State transitions — full flows
# ---------------------------------------------------------------------------

class TestFullApproveFlow:
    """Test the complete approve flow: /approve -> YES -> posted."""

    @patch("src.publisher.telegram_reply_handler.TelegramReplyHandler.post_reply_to_youtube", return_value=True)
    def test_approve_then_yes_posts(self, mock_post, db: Path) -> None:
        _insert_comment(db)
        h = _handler(db)

        # Step 1: /approve
        resp1 = h.route_message("/approve_c1")
        assert "Confirm Post" in resp1
        assert _get_state(db, "c1") == "PENDING_CONFIRMATION"

        # Step 2: YES
        resp2 = h.route_message("YES")
        assert "posted successfully" in resp2
        assert _get_state(db, "c1") == "APPROVED"

    @patch("src.publisher.telegram_reply_handler.TelegramReplyHandler.post_reply_to_youtube", return_value=True)
    def test_approve_then_no_cancels(self, mock_post, db: Path) -> None:
        _insert_comment(db)
        h = _handler(db)

        h.route_message("/approve_c1")
        resp = h.route_message("NO")
        assert "Cancelled" in resp
        assert _get_state(db, "c1") == "PENDING_APPROVAL"
        mock_post.assert_not_called()


class TestFullEditFlow:
    """Test the complete edit flow: /edit -> text -> YES -> posted."""

    @patch("src.publisher.telegram_reply_handler.TelegramReplyHandler.post_reply_to_youtube", return_value=True)
    def test_edit_then_text_then_yes(self, mock_post, db: Path) -> None:
        _insert_comment(db)
        h = _handler(db)

        # Step 1: /edit
        resp1 = h.route_message("/edit_c1")
        assert "Edit Reply" in resp1
        assert _get_state(db, "c1") == "PENDING_EDIT"

        # Step 2: send edited text
        resp2 = h.route_message("Better reply text")
        assert "Confirm Edited Reply" in resp2
        assert _get_state(db, "c1") == "PENDING_EDIT_CONFIRMATION"

        # Step 3: YES
        resp3 = h.route_message("YES")
        assert "posted successfully" in resp3
        mock_post.assert_called_once_with("c1", "Better reply text")
        assert _get_state(db, "c1") == "APPROVED"

    @patch("src.publisher.telegram_reply_handler.TelegramReplyHandler.post_reply_to_youtube", return_value=True)
    def test_edit_then_text_then_no_returns_to_edit(self, mock_post, db: Path) -> None:
        _insert_comment(db)
        h = _handler(db)

        h.route_message("/edit_c1")
        h.route_message("Draft reply")
        resp = h.route_message("NO")
        assert "Edit Reply" in resp
        assert _get_state(db, "c1") == "PENDING_EDIT"
        mock_post.assert_not_called()


class TestNoYouTubePostWithoutYes:
    """Verify no YouTube post happens without explicit YES."""

    @patch("src.publisher.telegram_reply_handler.TelegramReplyHandler.post_reply_to_youtube")
    def test_approve_alone_does_not_post(self, mock_post, db: Path) -> None:
        _insert_comment(db)
        h = _handler(db)
        h.route_message("/approve_c1")
        mock_post.assert_not_called()

    @patch("src.publisher.telegram_reply_handler.TelegramReplyHandler.post_reply_to_youtube")
    def test_edit_alone_does_not_post(self, mock_post, db: Path) -> None:
        _insert_comment(db)
        h = _handler(db)
        h.route_message("/edit_c1")
        mock_post.assert_not_called()

    @patch("src.publisher.telegram_reply_handler.TelegramReplyHandler.post_reply_to_youtube")
    def test_skip_does_not_post(self, mock_post, db: Path) -> None:
        _insert_comment(db)
        h = _handler(db)
        h.route_message("/skip_c1")
        mock_post.assert_not_called()


# ---------------------------------------------------------------------------
# Telegram _send helper
# ---------------------------------------------------------------------------

class TestSendHelper:
    def test_returns_false_when_not_configured(self, db: Path) -> None:
        h = TelegramReplyHandler(token="", chat_id="", db_path=db)
        assert h._send("test") is False

    def test_returns_true_on_success(self, db: Path) -> None:
        mock_httpx = MagicMock()
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        mock_httpx.post.return_value = resp
        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            h = _handler(db)
            assert h._send("hello") is True

    def test_returns_false_on_exception(self, db: Path) -> None:
        mock_httpx = MagicMock()
        mock_httpx.post.side_effect = Exception("network error")
        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            h = _handler(db)
            assert h._send("hello") is False

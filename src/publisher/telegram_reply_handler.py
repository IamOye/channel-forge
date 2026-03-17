"""
telegram_reply_handler.py — Telegram long-polling handler for comment approval.

Listens for incoming Telegram messages and routes them through the
comment approval flow:

    /approve_{comment_id}  -> confirmation prompt -> YES/NO -> post to YouTube
    /edit_{comment_id}     -> enter edit mode -> receive text -> confirm -> post
    /skip_{comment_id}     -> mark skipped
    /pending               -> list all PENDING_APPROVAL comments
    /automode on|off       -> toggle auto-generate suggested replies

State machine per comment:
    PENDING_APPROVAL -> PENDING_CONFIRMATION (via /approve)
    PENDING_APPROVAL -> PENDING_EDIT (via /edit)
    PENDING_CONFIRMATION -> APPROVED (via YES) | PENDING_APPROVAL (via NO)
    PENDING_EDIT -> PENDING_EDIT_CONFIRMATION (via free text)
    PENDING_EDIT_CONFIRMATION -> APPROVED (via YES) | PENDING_EDIT (via NO)
    any -> SKIPPED (via /skip)

Usage:
    from src.publisher.telegram_reply_handler import TelegramReplyHandler
    handler = TelegramReplyHandler()
    # In a background thread:
    asyncio.run(handler.poll())
"""

import asyncio
import logging
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_STATES = {
    "PENDING_APPROVAL",
    "PENDING_CONFIRMATION",
    "PENDING_EDIT",
    "PENDING_EDIT_CONFIRMATION",
    "APPROVED",
    "SKIPPED",
}

_DEFAULT_DB = Path(os.getenv("DB_PATH", "data/processed/channel_forge.db"))
_API_BASE = "https://api.telegram.org/bot{token}"


# ---------------------------------------------------------------------------
# TelegramReplyHandler
# ---------------------------------------------------------------------------


class TelegramReplyHandler:
    """
    Polls Telegram for incoming messages and manages the comment approval flow.

    Args:
        token:   Telegram bot token.  Defaults to TELEGRAM_BOT_TOKEN env var.
        chat_id: Authorised chat ID.  Only messages from this chat are handled.
        db_path: Path to the SQLite database.
    """

    def __init__(
        self,
        token: str | None = None,
        chat_id: str | None = None,
        db_path: Path | None = None,
    ) -> None:
        self.token = token if token is not None else os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id if chat_id is not None else os.getenv("TELEGRAM_CHAT_ID", "")
        self.db_path = db_path or _DEFAULT_DB

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        """Return a new SQLite connection with WAL mode."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_comment_state(self, comment_id: str) -> dict | None:
        """Fetch a single comment_states row as a dict, or None."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM comment_states WHERE comment_id = ?",
                (comment_id,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def update_state(self, comment_id: str, new_state: str, **extra: str) -> bool:
        """
        Transition a comment to *new_state*.

        Extra keyword args are written as column updates (e.g. edited_reply=...).
        Returns True if a row was actually updated.
        """
        if new_state not in VALID_STATES:
            logger.error("Invalid state: %s", new_state)
            return False

        sets = ["state = ?", "updated_at = ?"]
        vals: list[str] = [new_state, datetime.now(timezone.utc).isoformat()]
        for col, val in extra.items():
            sets.append(f"{col} = ?")
            vals.append(val)
        vals.append(comment_id)

        conn = self._get_conn()
        try:
            cur = conn.execute(
                f"UPDATE comment_states SET {', '.join(sets)} WHERE comment_id = ?",
                vals,
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def get_pending_comments(self) -> list[dict]:
        """Return all comments in PENDING_APPROVAL state."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM comment_states WHERE state = 'PENDING_APPROVAL' "
                "ORDER BY created_at DESC",
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_setting(self, key: str, default: str = "") -> str:
        """Read a value from the settings table."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT value FROM settings WHERE key = ?", (key,)
            ).fetchone()
            return row["value"] if row else default
        finally:
            conn.close()

    def set_setting(self, key: str, value: str) -> None:
        """Upsert a value in the settings table."""
        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, datetime('now')) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                (key, value),
            )
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Telegram send helper
    # ------------------------------------------------------------------

    def _send(self, text: str) -> bool:
        """Send an HTML message to the configured Telegram chat (sync)."""
        if not self.token or not self.chat_id:
            return False
        try:
            import httpx

            url = f"{_API_BASE.format(token=self.token)}/sendMessage"
            resp = httpx.post(
                url,
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": "HTML",
                },
                timeout=10.0,
            )
            resp.raise_for_status()
            return True
        except Exception as exc:
            logger.warning("Telegram send failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # YouTube reply posting
    # ------------------------------------------------------------------

    def post_reply_to_youtube(self, comment_id: str, text: str) -> bool:
        """
        Post a reply to a YouTube comment via the YouTube Data API v3.

        Returns True on success, False on failure (never raises).
        """
        try:
            from googleapiclient.discovery import build as yt_build
            from google.oauth2.credentials import Credentials

            creds_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent / ".credentials"
            token_path = creds_dir / "money_debate_token.json"

            if not token_path.exists():
                logger.error("YouTube token not found: %s", token_path)
                return False

            import json
            token_data = json.loads(token_path.read_text())
            creds = Credentials.from_authorized_user_info(token_data)

            service = yt_build("youtube", "v3", credentials=creds)
            service.comments().insert(
                part="snippet",
                body={
                    "snippet": {
                        "parentId": comment_id,
                        "textOriginal": text,
                    }
                },
            ).execute()
            logger.info("YouTube reply posted for comment_id=%s", comment_id)
            return True
        except Exception as exc:
            logger.error("YouTube reply failed for comment_id=%s: %s", comment_id, exc)
            return False

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    def handle_approve(self, comment_id: str) -> str:
        """Handle /approve_{comment_id} — send confirmation prompt."""
        row = self.get_comment_state(comment_id)
        if not row:
            return f"Comment {comment_id} not found."
        if row["state"] not in ("PENDING_APPROVAL", "PENDING_EDIT"):
            return f"Comment {comment_id} is in state {row['state']}, cannot approve."

        reply_text = row.get("edited_reply") or row["suggested_reply"]
        self.update_state(comment_id, "PENDING_CONFIRMATION")
        return (
            f"\u26a0\ufe0f <b>Confirm Post</b>\n\n"
            f"Posting to YouTube:\n<i>{reply_text}</i>\n\n"
            f"Reply YES to confirm\nReply NO to cancel"
        )

    def handle_edit(self, comment_id: str) -> str:
        """Handle /edit_{comment_id} — enter edit mode."""
        row = self.get_comment_state(comment_id)
        if not row:
            return f"Comment {comment_id} not found."
        if row["state"] not in ("PENDING_APPROVAL", "PENDING_EDIT_CONFIRMATION"):
            return f"Comment {comment_id} is in state {row['state']}, cannot edit."

        current = row.get("edited_reply") or row["suggested_reply"]
        self.update_state(comment_id, "PENDING_EDIT")
        return (
            f"\u270f\ufe0f <b>Edit Reply</b>\n\n"
            f"Current reply:\n<i>{current}</i>\n\n"
            f"Send your edited version now.\n"
            f"(Next message you send will be used)"
        )

    def handle_skip(self, comment_id: str) -> str:
        """Handle /skip_{comment_id} — mark as skipped."""
        row = self.get_comment_state(comment_id)
        if not row:
            return f"Comment {comment_id} not found."

        self.update_state(comment_id, "SKIPPED")
        return "\u23ed Comment skipped"

    def handle_pending(self) -> str:
        """Handle /pending — list all comments awaiting approval."""
        pending = self.get_pending_comments()
        if not pending:
            return "\U0001f4cb <b>No pending comments</b>"

        lines = [f"\U0001f4cb <b>Pending Comments ({len(pending)})</b>\n"]
        for i, row in enumerate(pending, 1):
            preview = (row["comment_text"] or "")[:50]
            cid = row["comment_id"]
            commenter = row.get("commenter") or "unknown"
            lines.append(
                f"{i}. @{commenter}: '{preview}...'\n"
                f"   /approve_{cid} | /edit_{cid} | /skip_{cid}"
            )
        return "\n".join(lines)

    def handle_automode(self, mode: str) -> str:
        """Handle /automode on|off — toggle auto-generate."""
        mode_lower = mode.strip().lower()
        if mode_lower not in ("on", "off"):
            return "Usage: /automode on or /automode off"
        self.set_setting("telegram_automode", mode_lower)
        if mode_lower == "on":
            return (
                "\u2705 Automode ON\n"
                "Claude will auto-generate suggested replies.\n"
                "Manual approval is still required before posting."
            )
        return (
            "\u274c Automode OFF\n"
            "You will receive comment alerts only.\n"
            "No suggested replies will be generated."
        )

    def handle_yes(self) -> str:
        """Handle YES — confirm and post the pending reply."""
        # Find the comment in PENDING_CONFIRMATION or PENDING_EDIT_CONFIRMATION
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM comment_states "
                "WHERE state IN ('PENDING_CONFIRMATION', 'PENDING_EDIT_CONFIRMATION') "
                "ORDER BY updated_at DESC LIMIT 1",
            ).fetchone()
        finally:
            conn.close()

        if not row:
            return "No comment awaiting confirmation."

        row = dict(row)
        reply_text = row.get("edited_reply") or row["suggested_reply"]
        comment_id = row["comment_id"]

        posted = self.post_reply_to_youtube(comment_id, reply_text)
        if posted:
            self.update_state(comment_id, "APPROVED")
            return "\u2705 Reply posted successfully"
        else:
            # Revert to PENDING_APPROVAL on failure
            self.update_state(comment_id, "PENDING_APPROVAL")
            return "\u274c Failed to post reply to YouTube. Comment returned to pending."

    def handle_no(self) -> str:
        """Handle NO — cancel confirmation, return to pending or edit prompt."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM comment_states "
                "WHERE state IN ('PENDING_CONFIRMATION', 'PENDING_EDIT_CONFIRMATION') "
                "ORDER BY updated_at DESC LIMIT 1",
            ).fetchone()
        finally:
            conn.close()

        if not row:
            return "No comment awaiting confirmation."

        row = dict(row)
        comment_id = row["comment_id"]

        if row["state"] == "PENDING_EDIT_CONFIRMATION":
            # Return to edit prompt
            self.update_state(comment_id, "PENDING_EDIT")
            current = row.get("edited_reply") or row["suggested_reply"]
            return (
                f"\u270f\ufe0f <b>Edit Reply</b>\n\n"
                f"Current reply:\n<i>{current}</i>\n\n"
                f"Send your edited version now.\n"
                f"(Next message you send will be used)"
            )
        else:
            self.update_state(comment_id, "PENDING_APPROVAL")
            return "\u274c Cancelled \u2014 comment still pending"

    def handle_free_text(self, text: str) -> str | None:
        """
        Handle free-text messages (not slash commands, not YES/NO).

        If a comment is in PENDING_EDIT state, treat the text as the edited reply.
        Returns the response message, or None if nothing to do.
        """
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM comment_states WHERE state = 'PENDING_EDIT' "
                "ORDER BY updated_at DESC LIMIT 1",
            ).fetchone()
        finally:
            conn.close()

        if not row:
            return None

        row = dict(row)
        comment_id = row["comment_id"]
        self.update_state(comment_id, "PENDING_EDIT_CONFIRMATION", edited_reply=text)

        return (
            f"\U0001f4dd <b>Confirm Edited Reply</b>\n\n"
            f"Posting to YouTube:\n<i>{text}</i>\n\n"
            f"Reply YES to confirm\nReply NO to cancel or re-edit"
        )

    # ------------------------------------------------------------------
    # Message router
    # ------------------------------------------------------------------

    def route_message(self, text: str) -> str | None:
        """
        Route an incoming Telegram message to the appropriate handler.

        Returns the response text, or None if the message is not recognised.
        """
        text = text.strip()

        # /approve_{id}
        m = re.match(r"^/approve_(\S+)$", text)
        if m:
            return self.handle_approve(m.group(1))

        # /edit_{id}
        m = re.match(r"^/edit_(\S+)$", text)
        if m:
            return self.handle_edit(m.group(1))

        # /skip_{id}
        m = re.match(r"^/skip_(\S+)$", text)
        if m:
            return self.handle_skip(m.group(1))

        # /pending
        if text == "/pending":
            return self.handle_pending()

        # /automode on|off
        m = re.match(r"^/automode\s+(\S+)$", text)
        if m:
            return self.handle_automode(m.group(1))

        # YES / NO (case-insensitive)
        upper = text.upper()
        if upper == "YES":
            return self.handle_yes()
        if upper == "NO":
            return self.handle_no()

        # Free text (for edit mode)
        if not text.startswith("/"):
            return self.handle_free_text(text)

        return None

    # ------------------------------------------------------------------
    # Long-polling loop
    # ------------------------------------------------------------------

    async def poll(self) -> None:
        """
        Long-poll Telegram for updates and route messages.

        Runs indefinitely — intended to be launched in a background thread
        via ``asyncio.run(handler.poll())``.
        """
        import httpx

        if not self.token or not self.chat_id:
            logger.warning("Telegram not configured — reply handler not starting")
            return

        offset = 0
        logger.info("Telegram reply handler started (chat_id=%s)", self.chat_id)

        async with httpx.AsyncClient(timeout=35.0) as client:
            while True:
                try:
                    url = f"{_API_BASE.format(token=self.token)}/getUpdates"
                    resp = await client.get(
                        url,
                        params={
                            "offset": offset,
                            "timeout": 30,
                            "allowed_updates": '["message"]',
                        },
                    )
                    data = resp.json()
                    updates = data.get("result", [])

                    for update in updates:
                        offset = update["update_id"] + 1
                        msg = update.get("message", {})
                        chat = msg.get("chat", {})
                        msg_text = msg.get("text", "")

                        # Only handle messages from our authorised chat
                        if str(chat.get("id")) != str(self.chat_id):
                            continue

                        if not msg_text:
                            continue

                        response = self.route_message(msg_text)
                        if response:
                            self._send(response)

                except Exception as exc:
                    logger.error("Telegram poll error: %s", exc)
                    await asyncio.sleep(5)

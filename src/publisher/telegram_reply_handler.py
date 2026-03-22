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
            import time

            url = f"{_API_BASE.format(token=self.token)}/sendMessage"
            for attempt in range(2):
                resp = httpx.post(
                    url,
                    json={
                        "chat_id": self.chat_id,
                        "text": text,
                        "parse_mode": "HTML",
                    },
                    timeout=10.0,
                )
                if resp.status_code == 429:
                    retry_after = resp.json().get("parameters", {}).get("retry_after", 5)
                    logger.warning("Telegram 429 — sleeping %ss before retry", retry_after)
                    time.sleep(retry_after)
                    continue
                resp.raise_for_status()
                return True
            logger.warning("Telegram send failed after 429 retry")
            return False
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

    def handle_held(self) -> str:
        """Handle /held — list all videos held by the quality gate."""
        conn = self._get_conn()
        try:
            # Auto-create table if it doesn't exist yet
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS quality_holds (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic_id        TEXT NOT NULL DEFAULT '',
                    video_path      TEXT NOT NULL,
                    failure_reason  TEXT NOT NULL,
                    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
            rows = conn.execute(
                "SELECT * FROM quality_holds ORDER BY created_at DESC LIMIT 20"
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            return "✅ <b>No held videos</b>\nAll videos passed quality gate."

        lines = [f"🚫 <b>Held Videos ({len(rows)})</b>\n"]
        for i, row in enumerate(rows, 1):
            topic = row["topic_id"] or "unknown"
            path = row["video_path"]
            reason = (row["failure_reason"] or "")[:80]
            created = row["created_at"] or ""
            lines.append(
                f"{i}. <b>{topic}</b>\n"
                f"   📁 {path}\n"
                f"   ❌ {reason}\n"
                f"   🕐 {created}"
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
    # Topic queue commands
    # ------------------------------------------------------------------

    def handle_addtopic(self, args: str) -> str:
        """Handle /addtopic [category] [title] — add a topic to the queue."""
        parts = args.strip().split(None, 1)
        if len(parts) < 2:
            return "Usage: /addtopic [category] [title]\nExample: /addtopic money Why banks want you in debt"

        category = parts[0].lower()
        title = parts[1].strip()

        if category not in ("money", "career", "success"):
            return f"Invalid category '{category}'. Use: money, career, success"

        # Insert into local DB
        conn = self._get_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS manual_topics (
                    seq INTEGER PRIMARY KEY, title TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'money',
                    hook_angle TEXT NOT NULL DEFAULT '', priority TEXT NOT NULL DEFAULT 'MEDIUM',
                    notes TEXT NOT NULL DEFAULT '', status TEXT NOT NULL DEFAULT 'QUEUED',
                    loaded_at TEXT NOT NULL DEFAULT (datetime('now')),
                    used_at TEXT, video_id TEXT NOT NULL DEFAULT ''
                )
            """)
            # Get next seq
            row = conn.execute("SELECT MAX(seq) FROM manual_topics").fetchone()
            next_seq = (row[0] or 0) + 1 if row else 1
            conn.execute(
                "INSERT INTO manual_topics (seq, title, category, status, notes) "
                "VALUES (?, ?, ?, 'QUEUED', 'via Telegram')",
                (next_seq, title, category),
            )
            conn.commit()
        finally:
            conn.close()

        # Also try to add to Google Sheet
        try:
            from src.crawler.gsheet_topic_sync import GSheetTopicSync
            sync = GSheetTopicSync()
            sheet_seq = sync.append_topic(title=title, category=category, notes="via Telegram")
            return f"✅ Added SEQ #{sheet_seq}: {title}\n(category: {category}, added to Sheet + DB)"
        except Exception as exc:
            logger.warning("[telegram] Sheet append failed: %s", exc)
            return f"✅ Added SEQ #{next_seq}: {title}\n(category: {category}, DB only — Sheet sync failed)"

    def handle_listtopics(self) -> str:
        """Handle /listtopics — show next 7 queued topics."""
        conn = self._get_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS manual_topics (
                    seq INTEGER PRIMARY KEY, title TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'money',
                    hook_angle TEXT NOT NULL DEFAULT '', priority TEXT NOT NULL DEFAULT 'MEDIUM',
                    notes TEXT NOT NULL DEFAULT '', status TEXT NOT NULL DEFAULT 'QUEUED',
                    loaded_at TEXT NOT NULL DEFAULT (datetime('now')),
                    used_at TEXT, video_id TEXT NOT NULL DEFAULT ''
                )
            """)
            rows = conn.execute(
                "SELECT seq, title, category FROM manual_topics "
                "WHERE status = 'QUEUED' ORDER BY seq ASC LIMIT 7"
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            return "📋 <b>No queued topics</b>\nAdd topics via /addtopic or Google Sheet."

        lines = [f"📋 <b>Next {len(rows)} queued topics:</b>\n"]
        for seq, title, cat in rows:
            lines.append(f"[{seq}] {title} ({cat})")
        return "\n".join(lines)

    def handle_weeklystatus(self) -> str:
        """Handle /weeklystatus — show production stats for this week."""
        conn = self._get_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS manual_topics (
                    seq INTEGER PRIMARY KEY, title TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'money',
                    hook_angle TEXT NOT NULL DEFAULT '', priority TEXT NOT NULL DEFAULT 'MEDIUM',
                    notes TEXT NOT NULL DEFAULT '', status TEXT NOT NULL DEFAULT 'QUEUED',
                    loaded_at TEXT NOT NULL DEFAULT (datetime('now')),
                    used_at TEXT, video_id TEXT NOT NULL DEFAULT ''
                )
            """)
            queued = conn.execute(
                "SELECT COUNT(*) FROM manual_topics WHERE status = 'QUEUED'"
            ).fetchone()[0]
            used_week = conn.execute(
                "SELECT COUNT(*) FROM manual_topics WHERE status = 'USED' "
                "AND used_at >= date('now', 'weekday 0', '-6 days')"
            ).fetchone()[0]
            next_row = conn.execute(
                "SELECT seq, title FROM manual_topics "
                "WHERE status = 'QUEUED' ORDER BY seq ASC LIMIT 1"
            ).fetchone()
        finally:
            conn.close()

        next_str = f"[{next_row[0]}] {next_row[1]}" if next_row else "(none)"
        return (
            f"📊 <b>Week status:</b>\n"
            f"✅ Produced: {used_week}\n"
            f"📋 Queued: {queued}\n"
            f"Next: {next_str}"
        )

    def handle_skiptopic(self, seq_str: str) -> str:
        """Handle /skiptopic [seq] — mark topic as SKIP."""
        try:
            seq = int(seq_str)
        except ValueError:
            return "Usage: /skiptopic [seq number]"

        conn = self._get_conn()
        try:
            cur = conn.execute(
                "UPDATE manual_topics SET status = 'SKIP' WHERE seq = ? AND status = 'QUEUED'",
                (seq,),
            )
            conn.commit()
        finally:
            conn.close()

        if cur.rowcount == 0:
            return f"SEQ #{seq} not found or not QUEUED."

        # Try to update Sheet too
        try:
            from src.crawler.gsheet_topic_sync import GSheetTopicSync
            GSheetTopicSync().set_status(seq, "SKIP")
        except Exception:
            pass

        return f"⏭ Skipped SEQ #{seq}"

    def handle_holdtopic(self, seq_str: str) -> str:
        """Handle /holdtopic [seq] — put topic on hold."""
        try:
            seq = int(seq_str)
        except ValueError:
            return "Usage: /holdtopic [seq number]"

        conn = self._get_conn()
        try:
            cur = conn.execute(
                "UPDATE manual_topics SET status = 'HOLD' WHERE seq = ? AND status = 'QUEUED'",
                (seq,),
            )
            conn.commit()
        finally:
            conn.close()

        if cur.rowcount == 0:
            return f"SEQ #{seq} not found or not QUEUED."

        try:
            from src.crawler.gsheet_topic_sync import GSheetTopicSync
            GSheetTopicSync().set_status(seq, "HOLD")
        except Exception:
            pass

        return f"⏸ SEQ #{seq} on hold"

    def handle_synctopics(self) -> str:
        """Handle /synctopics — manually trigger topic queue sync."""
        try:
            from src.scheduler import run_topic_queue_sync
            run_topic_queue_sync()
            return "🔄 Topic queue sync completed — check logs for details."
        except Exception as exc:
            return f"❌ Sync failed: {exc}"

    def handle_produce(self) -> str:
        """Handle /produce — manually trigger production run in background (force mode)."""
        import threading

        def _run() -> None:
            try:
                from src.pipeline.multi_channel_orchestrator import MultiChannelOrchestrator
                orchestrator = MultiChannelOrchestrator()
                orchestrator.run_all(force=True)
            except Exception as exc:
                logger.error("[telegram] /produce failed: %s", exc)
                try:
                    self._send(f"❌ Production failed: {exc}")
                except Exception:
                    pass

        threading.Thread(target=_run, daemon=True).start()
        return "🎬 Manual production starting... I'll notify you when complete."

    def handle_clearpending(self) -> str:
        """Handle /clearpending — clear all stale pending_uploads entries."""
        conn = self._get_conn()
        try:
            count = conn.execute("SELECT COUNT(*) FROM pending_uploads").fetchone()[0]
            conn.execute("DELETE FROM pending_uploads")
            conn.commit()
            return f"🗑 Cleared {count} pending uploads."
        except Exception as exc:
            return f"❌ Failed: {exc}"
        finally:
            conn.close()

    def handle_queue(self) -> str:
        """Handle /queue — show videos with status='pending' awaiting upload."""
        try:
            conn = self._get_conn()
            try:
                col_rows = conn.execute(
                    "PRAGMA table_info(pending_uploads)"
                ).fetchall()
                col_names = [r[1] for r in col_rows]
                rows = conn.execute(
                    "SELECT topic_id, channel_key, queued_at "
                    "FROM pending_uploads WHERE status = 'pending' ORDER BY queued_at ASC"
                ).fetchall()
                total = conn.execute(
                    "SELECT COUNT(*) FROM pending_uploads"
                ).fetchone()[0]
            finally:
                conn.close()
            schema_line = f"<i>Schema: {', '.join(col_names)}</i>"
            if not rows:
                return (
                    f"📭 No videos pending. ({total} completed entries in history)\n"
                    + schema_line
                )
            lines = [f"📬 <b>{len(rows)} video(s) queued for upload at 08:00 WAT:</b>\n"]
            for i, (tid, channel, queued) in enumerate(rows, 1):
                lines.append(f"{i}. <code>{tid}</code> [{channel}] (queued: {queued})")
            lines.append(f"\n{schema_line}")
            return "\n".join(lines)
        except Exception as exc:
            return f"❌ Queue check failed: {exc}"

    def handle_diagnose(self) -> str:
        """Handle /diagnose — DB inspection for Railway debugging."""
        sections: list[str] = []

        try:
            conn = self._get_conn()
            try:
                # 1. Next 10 QUEUED manual topics
                try:
                    rows = conn.execute(
                        "SELECT seq, title, category, status FROM manual_topics "
                        "WHERE status = 'QUEUED' ORDER BY seq ASC LIMIT 10"
                    ).fetchall()
                    lines = ["📋 <b>Manual Queue — Next 10 QUEUED:</b>"]
                    if rows:
                        for r in rows:
                            lines.append(
                                f"  SEQ {r[0]}: {r[1][:40]} [{r[2]}]"
                            )
                    else:
                        lines.append("  (none)")
                    sections.append("\n".join(lines))
                except sqlite3.OperationalError:
                    sections.append("📋 <b>Manual Queue:</b> table not found")

                # 2. Manual topics counts by status
                try:
                    rows = conn.execute(
                        "SELECT status, COUNT(*) FROM manual_topics GROUP BY status "
                        "ORDER BY status"
                    ).fetchall()
                    total = sum(c for _, c in rows)
                    lines = [f"📊 <b>Manual Topics by Status ({total} total):</b>"]
                    for status, count in rows:
                        lines.append(f"  {status}: {count}")
                    sections.append("\n".join(lines))
                except sqlite3.OperationalError:
                    pass

                # 3. Clip history by source
                try:
                    rows = conn.execute(
                        "SELECT source, COUNT(*) FROM clip_history GROUP BY source "
                        "ORDER BY source"
                    ).fetchall()
                    total = sum(c for _, c in rows)
                    lines = [f"🎬 <b>Clip History ({total} total):</b>"]
                    for source, count in rows:
                        lines.append(f"  {source}: {count}")
                    sections.append("\n".join(lines))
                except sqlite3.OperationalError:
                    sections.append("🎬 <b>Clip History:</b> table not found")

                # 4. Pending uploads by status
                try:
                    rows = conn.execute(
                        "SELECT status, COUNT(*) FROM pending_uploads GROUP BY status "
                        "ORDER BY status"
                    ).fetchall()
                    lines = ["📬 <b>Pending Uploads by Status:</b>"]
                    for status, count in rows:
                        lines.append(f"  {status}: {count}")
                    sections.append("\n".join(lines))
                except sqlite3.OperationalError:
                    sections.append("📬 <b>Pending Uploads:</b> table not found")

                # 5. ElevenLabs usage this month
                try:
                    rows = conn.execute(
                        "SELECT SUM(characters_used) FROM elevenlabs_usage "
                        "WHERE strftime('%Y-%m', created_at) = strftime('%Y-%m', 'now')"
                    ).fetchone()
                    chars = rows[0] or 0
                    remaining = max(0, 30000 - chars)
                    sections.append(
                        f"🎙️ <b>ElevenLabs (this month):</b>\n"
                        f"  Used: {chars:,} / 30,000 chars\n"
                        f"  Remaining: ~{remaining:,} chars (~{remaining // 1500} videos)"
                    )
                except sqlite3.OperationalError:
                    sections.append("🎙️ <b>ElevenLabs:</b> table not found")

            finally:
                conn.close()
        except Exception as exc:
            return f"❌ Diagnose failed: {exc}"

        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Research commands
    # ------------------------------------------------------------------

    def _get_engine(self):
        """Get a ResearchEngine instance (exclude Reddit on Railway)."""
        from src.research.research_engine import ResearchEngine
        return ResearchEngine(exclude_reddit=True, db_path=self.db_path)

    def handle_research(self, args: str) -> str:
        """Handle /research [source] [category] — start research session."""
        import threading
        from src.research.research_engine import ResearchEngine

        parts = args.strip().split() if args.strip() else []
        source = parts[0].lower() if parts else None
        category = parts[1].lower() if len(parts) > 1 else None

        if source == "reddit":
            return (
                "⚠️ Reddit research must run locally to avoid IP blocks.\n"
                "Run: python tools/research.py --source reddit\n"
                "Then: python tools/research.py --sync\n"
                "Then send /syncreviewed here to sync reviewed topics."
            )

        valid_sources = ("competitor", "autocomplete", "trends")
        if source and source not in valid_sources:
            return f"Invalid source '{source}'. Use: {', '.join(valid_sources)} or omit for all."

        engine = self._get_engine()
        existing = engine.get_session(self.chat_id)
        if existing and existing.get("status") == "active":
            return "⏳ Research already running. Check /researchstatus or /cancelsession"

        # Run in background
        def _run(chat_id: str, src: str | None, cat: str | None) -> None:
            try:
                eng = ResearchEngine(exclude_reddit=True, db_path=self.db_path)

                def cb(msg: str) -> None:
                    try:
                        self._send(msg)
                    except Exception:
                        pass

                topics = eng.run(source=src, category=cat, progress_callback=cb)
                if topics:
                    session_id = eng.create_session(chat_id, topics, source=src or "all", category=cat or "")
                    # Send first 5
                    first_5 = topics[:5]
                    for i, t in enumerate(first_5):
                        self._send(self._format_topic(t, i + 1))
                    self._send(
                        f"✅ Ready! {len(topics)} topics ranked. Showing 1–{len(first_5)}.\n"
                        "Use /next for more, /add 1,3 to add, /skip 2 to skip."
                    )
                else:
                    self._send("⚠️ No topics found after scoring.")
            except Exception as exc:
                logger.error("[research] Background run failed: %s", exc)
                self._send(f"❌ Research failed: {exc}")

        threading.Thread(target=_run, args=(self.chat_id, source, category), daemon=True).start()
        src_label = source or "competitor + autocomplete + trends"
        return f"🔍 Researching {src_label}... I'll update you as each phase completes."

    def _format_topic(self, t, num: int) -> str:
        """Format a single topic for Telegram display."""
        hook_str = f" (H:{t.hook_strength:.1f})" if t.hook_strength > 0 else ""
        rewrite_tag = "✏️ " if t.original_title else ""
        lines = [
            "━━━━━━━━━━━━━━━━━━━━",
            f"#{num} · Score: {t.score:.1f}{hook_str} · {t.category}",
            "",
            f"{rewrite_tag}<b>{t.title}</b>",
        ]
        if t.original_title and t.original_title != t.title:
            lines.append(f"📝 Original: {t.original_title}")
        if t.hook_angle:
            lines.append(f"🎯 Hook: {t.hook_angle}")
        source_str = t.source or "unknown"
        if t.source_detail:
            source_str += f"/{t.source_detail}"
        if t.score_hint > 0 and t.source == "reddit":
            source_str += f" · {int(t.score_hint)} upvotes"
        lines.append(f"📌 {source_str}")
        lines.append("━━━━━━━━━━━━━━━━━━━━")
        return "\n".join(lines)

    def handle_next(self) -> str:
        """Handle /next — show next 5 topics."""
        return self._show_topics(5)

    def handle_more(self, args: str) -> str:
        """Handle /more [n] — show next n topics."""
        try:
            n = min(int(args.strip()), 20) if args.strip() else 5
        except ValueError:
            n = 5
        return self._show_topics(n)

    def _show_topics(self, n: int) -> str:
        """Show next n topics from active session."""
        engine = self._get_engine()
        session = engine.get_session(self.chat_id)
        if not session:
            return "No active session. Use /research to start."

        topics = engine.get_session_topics(session)
        idx = session.get("current_index", 0)

        if idx >= len(topics):
            return "📋 No more topics. Use /done to finish."

        batch = topics[idx:idx + n]
        for i, t in enumerate(batch):
            self._send(self._format_topic(t, idx + i + 1))

        new_idx = idx + len(batch)
        engine.update_session(session["id"], current_index=new_idx)

        return f"Showing {idx + 1}–{new_idx} of {len(topics)}. /next for more, /add to add, /done to finish."

    def handle_research_add(self, args: str) -> str:
        """Handle /add [numbers] — add topics to Google Sheet."""
        engine = self._get_engine()
        session = engine.get_session(self.chat_id)
        if not session:
            return "No active session. Use /research to start."

        topics = engine.get_session_topics(session)
        indices = self._parse_numbers(args)
        if not indices:
            return "Usage: /add 1,3,5-8"

        to_add = [topics[i - 1] for i in indices if 1 <= i <= len(topics)]
        if not to_add:
            return "No valid topic numbers."

        try:
            from src.crawler.gsheet_topic_sync import GSheetTopicSync
            sync = GSheetTopicSync()
            lines = ["✅ Added to Google Sheet:"]
            for t in to_add:
                seq = sync.append_topic(title=t.title, category=t.category,
                                        hook_angle=t.hook_angle, notes=f"via Telegram | Score: {t.score:.1f}")
                engine.mark_reviewed(title=t.title, action="added", session_id=session["id"],
                                     score=t.score, category=t.category, source=t.source,
                                     original_title=t.original_title)
                lines.append(f"  SEQ #{seq}: {t.title}")

            added = session.get("topics_added", 0) + len(to_add)
            engine.update_session(session["id"], topics_added=added)
            return "\n".join(lines)
        except Exception as exc:
            return f"❌ Sheet error: {exc}"

    def handle_research_skip(self, args: str) -> str:
        """Handle /skip [numbers] — skip topics."""
        engine = self._get_engine()
        session = engine.get_session(self.chat_id)
        if not session:
            return "No active session."

        topics = engine.get_session_topics(session)
        indices = self._parse_numbers(args)
        if not indices:
            return "Usage: /skip 1,3,5-8"

        to_skip = [topics[i - 1] for i in indices if 1 <= i <= len(topics)]
        engine.mark_batch_reviewed(to_skip, "skipped", session["id"])
        skipped = session.get("topics_skipped", 0) + len(to_skip)
        engine.update_session(session["id"], topics_skipped=skipped)
        return f"⏭️ Skipped {len(to_skip)} topics — won't appear again"

    def handle_research_edit(self, args: str) -> str:
        """Handle /edit [number] — enter edit mode for a topic."""
        # Store the topic number for the next free-text message
        # For simplicity, return instructions
        return f"Send me the new title as a reply. Then /add the number to add it."

    def handle_done(self) -> str:
        """Handle /done — end research session."""
        engine = self._get_engine()
        session = engine.get_session(self.chat_id)
        if not session:
            return "No active session."

        topics = engine.get_session_topics(session)
        # Mark unreviewed topics as skipped
        idx = session.get("current_index", 0)
        unacted = topics[idx:]
        if unacted:
            engine.mark_batch_reviewed(unacted, "skipped", session["id"])

        added = session.get("topics_added", 0)
        skipped = session.get("topics_skipped", 0) + len(unacted)
        engine.update_session(session["id"], status="done", topics_skipped=skipped)

        # Count queued topics
        conn = self._get_conn()
        try:
            try:
                queued = conn.execute(
                    "SELECT COUNT(*) FROM manual_topics WHERE status = 'QUEUED'"
                ).fetchone()[0]
            except Exception:
                queued = "?"
        finally:
            conn.close()

        return (
            f"📊 Session complete:\n"
            f"✅ Added: {added} topics\n"
            f"⏭️ Skipped: {skipped} topics\n"
            f"📋 Queue has {queued} READY topics\n"
            f"Next sync: Monday 05:00 WAT"
        )

    def handle_researchstatus(self) -> str:
        """Handle /researchstatus."""
        engine = self._get_engine()
        session = engine.get_session(self.chat_id)
        if not session:
            return "No active session. Use /research to start."
        status = session.get("status", "unknown")
        total = session.get("total_topics", 0)
        idx = session.get("current_index", 0)
        added = session.get("topics_added", 0)
        if status == "active":
            return f"✅ {total} topics ready. Viewed {idx}/{total}. Added {added}. Use /next to continue."
        return f"Session {status}. Added {added} of {total}."

    def handle_cancelsession(self) -> str:
        """Handle /cancelsession."""
        engine = self._get_engine()
        session = engine.get_session(self.chat_id)
        if not session:
            return "No active session."
        engine.update_session(session["id"], status="cancelled")
        return "Session cancelled."

    def handle_syncreviewed(self) -> str:
        """Handle /syncreviewed — import reviewed topics from Google Sheet."""
        try:
            engine = self._get_engine()
            count = engine.import_reviewed_from_sheet()
            return f"✅ Synced {count} reviewed topics from local sessions. Future research will exclude these."
        except Exception as exc:
            return f"❌ Sync failed: {exc}"

    @staticmethod
    def _parse_numbers(text: str) -> list[int]:
        """Parse '1,3,5-8' into [1, 3, 5, 6, 7, 8]."""
        nums: list[int] = []
        if not text or not text.strip():
            return nums
        try:
            for part in text.split(","):
                part = part.strip()
                if "-" in part:
                    lo, hi = part.split("-", 1)
                    nums.extend(range(int(lo), int(hi) + 1))
                else:
                    nums.append(int(part))
        except ValueError:
            pass
        return sorted(set(nums))

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

        # /held
        if text == "/held":
            return self.handle_held()

        # /addtopic [category] [title]
        m = re.match(r"^/addtopic\s+(.+)$", text)
        if m:
            return self.handle_addtopic(m.group(1))

        # /listtopics
        if text == "/listtopics":
            return self.handle_listtopics()

        # /weeklystatus
        if text == "/weeklystatus":
            return self.handle_weeklystatus()

        # /status (alias for /weeklystatus)
        if text == "/status":
            return self.handle_weeklystatus()

        # /skiptopic [seq]
        m = re.match(r"^/skiptopic\s+(\S+)$", text)
        if m:
            return self.handle_skiptopic(m.group(1))

        # /holdtopic [seq]
        m = re.match(r"^/holdtopic\s+(\S+)$", text)
        if m:
            return self.handle_holdtopic(m.group(1))

        # /synctopics
        if text == "/synctopics":
            return self.handle_synctopics()

        # /produce
        if text == "/produce":
            return self.handle_produce()

        # /clearpending
        if text == "/clearpending":
            return self.handle_clearpending()

        # /queue
        if text == "/queue":
            return self.handle_queue()

        # /diagnose
        if text == "/diagnose":
            return self.handle_diagnose()

        # Research commands
        m = re.match(r"^/research(?:\s+(.*))?$", text)
        if m:
            return self.handle_research(m.group(1) or "")

        if text == "/next":
            return self.handle_next()

        m = re.match(r"^/more(?:\s+(.*))?$", text)
        if m:
            return self.handle_more(m.group(1) or "")

        m = re.match(r"^/add\s+(.+)$", text)
        if m:
            return self.handle_research_add(m.group(1))

        m = re.match(r"^/skip\s+(\d.*)$", text)
        if m:
            return self.handle_research_skip(m.group(1))

        if text == "/done":
            return self.handle_done()

        if text == "/researchstatus":
            return self.handle_researchstatus()

        if text == "/cancelsession":
            return self.handle_cancelsession()

        if text == "/syncreviewed":
            return self.handle_syncreviewed()

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
                    if resp.status_code == 429:
                        retry_after = resp.json().get("parameters", {}).get("retry_after", 5)
                        logger.warning("Telegram poll 429 — sleeping %ss", retry_after)
                        await asyncio.sleep(retry_after)
                        continue
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

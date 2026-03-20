"""
gsheet_topic_sync.py — Google Sheets Topic Queue Sync

Reads READY topics from a Google Sheet "Topic Queue" tab and syncs them
to the local manual_topics SQLite table.  After a video is produced from
a manual topic, marks the Sheet row as USED with the YouTube video ID.

Configuration (.env):
    GOOGLE_SHEET_ID=<spreadsheet ID from URL>
    GOOGLE_CREDENTIALS_B64=<base64-encoded service-account JSON>

Usage:
    from src.crawler.gsheet_topic_sync import GSheetTopicSync
    sync = GSheetTopicSync(sheet_id="...", credentials_b64="...")
    batch = sync.get_next_batch(last_seq=0, count=28)
"""

import base64
import json
import logging
import os
from datetime import date
from typing import Any

logger = logging.getLogger(__name__)

# Google API scopes required for Sheets + Drive access
_SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]


class GSheetTopicSync:
    """
    Two-way sync between a Google Sheet Topic Queue and the local SQLite DB.

    The Sheet must have a "Topic Queue" tab with columns:
      #, SEQ, Title / Topic, Category, Hook Angle (optional),
      Status, Priority, Date Added, Date Used, Video ID, Notes

    And a "Sync Log" tab for audit entries.

    Args:
        sheet_id:        Google Spreadsheet ID (from the URL).
        credentials_b64: Base64-encoded service-account JSON credentials.
    """

    def __init__(
        self,
        sheet_id: str | None = None,
        credentials_b64: str | None = None,
    ) -> None:
        self.sheet_id = sheet_id or os.getenv("GOOGLE_SHEET_ID", "")
        self.credentials_b64 = credentials_b64 or os.getenv("GOOGLE_CREDENTIALS_B64", "")

        self._sheet = None
        self._queue_tab = None
        self._log_tab = None

    # ------------------------------------------------------------------
    # Lazy connection (import gspread only when needed)
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        """Establish gspread connection on first use."""
        if self._sheet is not None:
            return

        import gspread
        from google.oauth2.service_account import Credentials

        if not self.sheet_id:
            raise ValueError("GOOGLE_SHEET_ID not set")
        if not self.credentials_b64:
            raise ValueError("GOOGLE_CREDENTIALS_B64 not set")

        b64 = self.credentials_b64.strip()
        missing = len(b64) % 4
        if missing:
            b64 += "=" * (4 - missing)
        creds_json = json.loads(base64.b64decode(b64))
        creds = Credentials.from_service_account_info(creds_json, scopes=_SCOPES)
        client = gspread.authorize(creds)
        self._sheet = client.open_by_key(self.sheet_id)
        self._queue_tab = self._sheet.worksheet("Topic Queue")
        self._log_tab = self._sheet.worksheet("Sync Log")
        logger.info("[gsheet] Connected to sheet %s", self.sheet_id)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_next_batch(
        self,
        last_seq: int,
        count: int = 28,
    ) -> list[dict[str, Any]]:
        """Return up to ``count`` READY rows with SEQ > ``last_seq``.

        Args:
            last_seq: Last synced SEQ number (rows with SEQ <= this are skipped).
            count:    Maximum rows to return.

        Returns:
            List of dicts with keys: seq, title, category, hook_angle,
            priority, notes, row_number.
        """
        self._connect()
        rows = self._queue_tab.get_all_records()
        results: list[dict[str, Any]] = []

        for i, row in enumerate(rows):
            try:
                seq = int(row.get("SEQ", 0))
            except (ValueError, TypeError):
                continue

            if seq <= last_seq:
                continue

            status = str(row.get("Status", "")).strip().upper()
            if status != "READY":
                continue

            results.append({
                "seq":        seq,
                "title":      str(row.get("Title / Topic", "") or row.get("Title", "")),
                "category":   str(row.get("Category", "money")).lower(),
                "hook_angle": str(row.get("Hook Angle (optional)", "") or row.get("Hook Angle", "")),
                "priority":   str(row.get("Priority", "MEDIUM")).upper(),
                "notes":      str(row.get("Notes", "")),
                "row_number": i + 2,  # +1 for header, +1 for 1-based
            })

        results.sort(key=lambda x: x["seq"])
        batch = results[:count]
        logger.info(
            "[gsheet] get_next_batch: %d READY topics after SEQ %d (returning %d)",
            len(results), last_seq, len(batch),
        )
        return batch

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def mark_used(
        self,
        seq: int,
        video_id: str = "",
        date_used: str = "",
    ) -> bool:
        """Mark a Topic Queue row as USED and fill date_used + video_id.

        Args:
            seq:       SEQ number to update.
            video_id:  YouTube video ID (optional).
            date_used: Date string (defaults to today).

        Returns:
            True if a row was found and updated.
        """
        self._connect()
        if not date_used:
            date_used = date.today().strftime("%d-%b-%y")

        rows = self._queue_tab.get_all_records()
        for i, row in enumerate(rows):
            try:
                row_seq = int(row.get("SEQ", -1))
            except (ValueError, TypeError):
                continue
            if row_seq == seq:
                sheet_row = i + 2
                # Col F=Status, Col I=Date Used, Col J=Video ID
                self._queue_tab.update_cell(sheet_row, 6, "USED")
                self._queue_tab.update_cell(sheet_row, 9, date_used)
                if video_id:
                    self._queue_tab.update_cell(sheet_row, 10, video_id)
                logger.info("[gsheet] Marked SEQ %d as USED (video=%s)", seq, video_id)
                return True

        logger.warning("[gsheet] SEQ %d not found in sheet", seq)
        return False

    def set_status(self, seq: int, status: str) -> bool:
        """Set any status (SKIP, HOLD, etc.) by SEQ number.

        Returns True if found and updated.
        """
        self._connect()
        rows = self._queue_tab.get_all_records()
        for i, row in enumerate(rows):
            try:
                row_seq = int(row.get("SEQ", -1))
            except (ValueError, TypeError):
                continue
            if row_seq == seq:
                self._queue_tab.update_cell(i + 2, 6, status.upper())
                logger.info("[gsheet] SEQ %d status → %s", seq, status.upper())
                return True
        return False

    def append_topic(
        self,
        title: str,
        category: str,
        hook_angle: str = "",
        notes: str = "",
    ) -> int:
        """Append a new READY row to Topic Queue. Returns the new SEQ number."""
        self._connect()
        rows = self._queue_tab.get_all_records()
        seqs = []
        for r in rows:
            try:
                seqs.append(int(r.get("SEQ", 0)))
            except (ValueError, TypeError):
                pass
        new_seq = max(seqs, default=0) + 1
        today = date.today().strftime("%d-%b-%y")

        new_row = [
            len(rows) + 1,   # # column
            new_seq,          # SEQ
            title,            # Title / Topic
            category,         # Category
            hook_angle,       # Hook Angle
            "READY",          # Status
            "MEDIUM",         # Priority
            today,            # Date Added
            "",               # Date Used
            "",               # Video ID
            notes,            # Notes
        ]
        self._queue_tab.append_row(new_row, value_input_option="USER_ENTERED")
        logger.info("[gsheet] Appended SEQ %d: %s", new_seq, title)
        return new_seq

    def update_sync_log(
        self,
        topics_loaded: int,
        seq_from: int,
        seq_to: int,
        notes: str = "",
    ) -> None:
        """Append an entry to the Sync Log tab."""
        self._connect()
        today = date.today().strftime("%d-%b-%y")
        self._log_tab.append_row(
            [today, topics_loaded, seq_from, seq_to, notes],
            value_input_option="USER_ENTERED",
        )
        logger.info("[gsheet] Sync log updated: %d topics, SEQ %d→%d", topics_loaded, seq_from, seq_to)

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

# Google API scopes required for Sheets read/write + Drive access
_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


def get_gsheet_client(
    sheet_id: str | None = None,
    credentials_b64: str | None = None,
):
    """Return (gspread_client, spreadsheet) using service-account credentials.

    Reusable helper — avoids duplicating auth logic across modules.
    """
    import gspread
    from google.oauth2.service_account import Credentials

    sid = sheet_id or os.getenv("GOOGLE_SHEET_ID", "")
    creds_b64 = credentials_b64 or os.getenv("GOOGLE_CREDENTIALS_B64", "")

    if not sid:
        raise ValueError("GOOGLE_SHEET_ID not set")
    if not creds_b64:
        raise ValueError("GOOGLE_CREDENTIALS_B64 not set")

    b64 = creds_b64.strip()
    missing = len(b64) % 4
    if missing:
        b64 += "=" * (4 - missing)
    creds_json = json.loads(base64.b64decode(b64))
    creds = Credentials.from_service_account_info(creds_json, scopes=_SCOPES)
    client = gspread.authorize(creds)
    spreadsheet = client.open_by_key(sid)
    return client, spreadsheet


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

        _, self._sheet = get_gsheet_client(self.sheet_id, self.credentials_b64)
        self._queue_tab = self._sheet.worksheet("Topic Queue")
        self._log_tab = self._sheet.worksheet("Sync Log")
        logger.info("[gsheet] Connected to sheet %s", self.sheet_id)

    # ------------------------------------------------------------------
    # Sheet reading helper
    # ------------------------------------------------------------------

    def _read_queue_rows(self) -> list[dict[str, Any]]:
        """Read all rows from Topic Queue tab using raw values.

        Uses get_all_values() + manual dict building instead of
        get_all_records() to avoid gspread header validation issues
        with empty or duplicate column headers.

        Headers are in row 3 (rows 1-2 are title/subtitle).
        Data starts at row 4.
        """
        self._connect()
        try:
            all_values = self._queue_tab.get_all_values()
            if not all_values:
                return []

            # Row 3 is headers (index 2), data starts row 4 (index 3)
            headers = all_values[2] if len(all_values) > 2 else []
            data_rows = all_values[3:] if len(all_values) > 3 else []

            # Clean headers — strip whitespace
            headers = [h.strip() for h in headers]

            # Build list of dicts, skipping empty rows
            # _sheet_row tracks the 1-based sheet row number for each entry
            results: list[dict[str, Any]] = []
            for idx, row in enumerate(data_rows):
                # Pad row to header length if shorter
                padded = row + [""] * (len(headers) - len(row))
                row_dict = {
                    headers[i]: padded[i]
                    for i in range(len(headers))
                    if headers[i]  # skip empty header columns
                }
                # Skip completely empty rows
                if not any(str(v).strip() for v in row_dict.values()):
                    continue
                # idx=0 is all_values[3] = sheet row 4 (1-based)
                row_dict["_sheet_row"] = idx + 4
                results.append(row_dict)

            return results

        except Exception as exc:
            logger.error("[gsheet] Failed to read sheet: %s", exc)
            logger.error(
                "[gsheet] Expected headers in row 3: "
                "#, SEQ, Title / Topic, Category, "
                "Hook Angle (optional), Status, "
                "Priority, Date Added, Date Used, "
                "Video ID, Notes"
            )
            raise

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
        rows = self._read_queue_rows()
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
                "row_number": row["_sheet_row"],
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
        if not date_used:
            date_used = date.today().strftime("%d-%b-%y")

        rows = self._read_queue_rows()
        for row in rows:
            try:
                row_seq = int(row.get("SEQ", -1))
            except (ValueError, TypeError):
                continue
            if row_seq == seq:
                sheet_row = row["_sheet_row"]
                logger.info("[gsheet] Writing USED for SEQ %d (sheet row %d)...", seq, sheet_row)
                # Col F=Status, Col I=Date Used, Col J=Video ID
                self._queue_tab.update_cell(sheet_row, 6, "USED")
                self._queue_tab.update_cell(sheet_row, 9, date_used)
                if video_id:
                    self._queue_tab.update_cell(sheet_row, 10, video_id)
                logger.info("[gsheet] Writeback complete for SEQ %d", seq)
                return True

        logger.warning("[gsheet] SEQ %d not found in sheet — cannot write USED", seq)
        return False

    def set_status(self, seq: int, status: str) -> bool:
        """Set any status (SKIP, HOLD, etc.) by SEQ number.

        Returns True if found and updated.
        """
        rows = self._read_queue_rows()
        for row in rows:
            try:
                row_seq = int(row.get("SEQ", -1))
            except (ValueError, TypeError):
                continue
            if row_seq == seq:
                sheet_row = row["_sheet_row"]
                self._queue_tab.update_cell(sheet_row, 6, status.upper())
                logger.info("[gsheet] SEQ %d status → %s (sheet row %d)", seq, status.upper(), sheet_row)
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
        rows = self._read_queue_rows()
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

    def write_youtube_title(self, seq: int, youtube_title: str, col: int = 12) -> None:
        """Write the YouTube-published title to col L (column 12) of the matching row.

        Auto-creates the column header in L3 if it is currently blank.
        """
        self._connect()

        # Ensure header exists in L3
        header_val = self._queue_tab.cell(3, col).value
        if not header_val or not str(header_val).strip():
            self._queue_tab.update_cell(3, col, "YouTube Title")
            logger.info("[gsheet] Created 'YouTube Title' header in col L (row 3)")

        rows = self._read_queue_rows()
        for row in rows:
            try:
                row_seq = int(row.get("SEQ", -1))
            except (ValueError, TypeError):
                continue
            if row_seq == seq:
                sheet_row = row["_sheet_row"]
                self._queue_tab.update_cell(sheet_row, col, youtube_title)
                logger.info(
                    "[gsheet] YouTube title written to col L for SEQ %d: %s",
                    seq, youtube_title,
                )
                return

        logger.warning("[gsheet] SEQ %d not found — cannot write YouTube title", seq)

    def sync_scraped_topics(self, rows: list[dict]) -> None:
        """Write competitor topic rows to the 'Scraped Topics' tab.

        Clears and rewrites on each call. Creates the tab if it does not exist.

        Tab columns (in order):
        Source | Channel | Original Title | Topic | Views | Category | Score | Date Scraped
        """
        self._connect()
        import gspread

        try:
            ws = self._sheet.worksheet("Scraped Topics")
        except gspread.WorksheetNotFound:
            ws = self._sheet.add_worksheet(title="Scraped Topics", rows=600, cols=8)

        ws.clear()

        headers = [
            "Source", "Channel", "Original Title", "Topic",
            "Views", "Category", "Score", "Date Scraped",
        ]
        ws.append_row(headers, value_input_option="USER_ENTERED")

        data_rows = []
        for r in rows:
            data_rows.append([
                str(r.get("source", "")),
                str(r.get("channel_name", "")),
                str(r.get("original_title", "")),
                str(r.get("extracted_topic", "")),
                r.get("view_count", 0),
                str(r.get("category", "")),
                str(r.get("score", "")),
                str(r.get("scraped_at", "")),
            ])

        if data_rows:
            ws.append_rows(data_rows, value_input_option="USER_ENTERED")

        logger.info("[gsheet] Scraped Topics tab synced — %d rows", len(data_rows))

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

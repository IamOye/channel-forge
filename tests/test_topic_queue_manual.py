"""
Tests for manual topic queue integration:
  - manual_topics consumed before AI topics
  - QUEUED → USED transition
  - SEQ ordering (lower first)
  - HOLD and SKIP ignored
  - Fallback to AI when manual_topics empty
  - Graceful degradation when env vars missing
  - run_topic_queue_sync inserts correct row count
"""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.topic_queue import TopicQueue


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_db(db_path: Path) -> None:
    """Create manual_topics + scored_topics + settings tables."""
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS manual_topics (
            seq         INTEGER PRIMARY KEY,
            title       TEXT NOT NULL,
            category    TEXT NOT NULL DEFAULT 'money',
            hook_angle  TEXT NOT NULL DEFAULT '',
            priority    TEXT NOT NULL DEFAULT 'MEDIUM',
            notes       TEXT NOT NULL DEFAULT '',
            status      TEXT NOT NULL DEFAULT 'QUEUED',
            loaded_at   TEXT NOT NULL DEFAULT (datetime('now')),
            used_at     TEXT,
            video_id    TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS scored_topics (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            keyword  TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT 'money',
            score    REAL NOT NULL DEFAULT 0,
            source   TEXT NOT NULL DEFAULT 'manual',
            used     INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS competitor_topics (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            channel_name    TEXT NOT NULL DEFAULT '',
            original_title  TEXT NOT NULL DEFAULT '',
            extracted_topic TEXT NOT NULL,
            view_count      INTEGER NOT NULL DEFAULT 0,
            category        TEXT NOT NULL DEFAULT 'money',
            source          TEXT NOT NULL DEFAULT 'competitor',
            used            INTEGER NOT NULL DEFAULT 0,
            scraped_at      TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        INSERT OR IGNORE INTO settings (key, value) VALUES ('last_manual_seq', '0');
    """)
    conn.commit()
    conn.close()


def _insert_manual(db_path: Path, seq: int, title: str,
                   category: str = "money", status: str = "QUEUED") -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO manual_topics (seq, title, category, status) VALUES (?, ?, ?, ?)",
        (seq, title, category, status),
    )
    conn.commit()
    conn.close()


def _insert_scored(db_path: Path, keyword: str, category: str = "money",
                   score: float = 50.0) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO scored_topics (keyword, category, score, source) "
        "VALUES (?, ?, ?, 'GOOGLE_TRENDS')",
        (keyword, category, score),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Tests: Manual topics consumed before AI topics
# ---------------------------------------------------------------------------

class TestManualTopicPriority:
    def test_manual_topic_returned_first(self, tmp_path) -> None:
        """Manual topics must be returned before scored_topics."""
        db = tmp_path / "test.db"
        _init_db(db)
        _insert_manual(db, 1, "Why banks want you in debt")
        _insert_scored(db, "passive income ideas", score=90.0)

        queue = TopicQueue(db_path=db, anthropic_api_key="")
        topics = queue.get_next_topics("money", max_count=3)

        assert len(topics) >= 1
        assert topics[0]["keyword"] == "Why banks want you in debt"
        assert topics[0]["source"] == "MANUAL"

    def test_manual_topic_has_manual_seq(self, tmp_path) -> None:
        """Manual topics must include manual_seq for Sheet callback."""
        db = tmp_path / "test.db"
        _init_db(db)
        _insert_manual(db, 7, "Salary negotiation script")

        queue = TopicQueue(db_path=db, anthropic_api_key="")
        topics = queue.get_next_topics("money", max_count=1)

        assert topics[0]["manual_seq"] == 7


# ---------------------------------------------------------------------------
# Tests: QUEUED → USED transition
# ---------------------------------------------------------------------------

class TestStatusTransition:
    def test_consumed_topic_marked_used(self, tmp_path) -> None:
        """After get_next_topics, the manual topic status must be USED."""
        db = tmp_path / "test.db"
        _init_db(db)
        _insert_manual(db, 1, "Why inflation is a hidden tax")

        queue = TopicQueue(db_path=db, anthropic_api_key="")
        queue.get_next_topics("money", max_count=1)

        conn = sqlite3.connect(db)
        row = conn.execute(
            "SELECT status FROM manual_topics WHERE seq = 1"
        ).fetchone()
        conn.close()
        assert row[0] == "USED"

    def test_used_topic_not_returned_again(self, tmp_path) -> None:
        """A USED topic must not be returned on subsequent calls."""
        db = tmp_path / "test.db"
        _init_db(db)
        _insert_manual(db, 1, "Hidden tax of inflation")

        queue = TopicQueue(db_path=db, anthropic_api_key="")
        first = queue.get_next_topics("money", max_count=1)
        assert len(first) == 1

        second = queue.get_next_topics("money", max_count=1)
        # Should not contain the same manual topic again
        manual_in_second = [t for t in second if t.get("source") == "MANUAL"]
        assert len(manual_in_second) == 0


# ---------------------------------------------------------------------------
# Tests: SEQ ordering
# ---------------------------------------------------------------------------

class TestSeqOrdering:
    def test_lower_seq_first(self, tmp_path) -> None:
        """Lower SEQ numbers must be consumed before higher ones."""
        db = tmp_path / "test.db"
        _init_db(db)
        _insert_manual(db, 5, "Topic five")
        _insert_manual(db, 2, "Topic two")
        _insert_manual(db, 9, "Topic nine")

        queue = TopicQueue(db_path=db, anthropic_api_key="")
        topics = queue.get_next_topics("money", max_count=3)

        manual_topics = [t for t in topics if t.get("source") == "MANUAL"]
        assert len(manual_topics) == 3
        assert manual_topics[0]["manual_seq"] == 2
        assert manual_topics[1]["manual_seq"] == 5
        assert manual_topics[2]["manual_seq"] == 9


# ---------------------------------------------------------------------------
# Tests: HOLD and SKIP ignored
# ---------------------------------------------------------------------------

class TestHoldAndSkip:
    def test_hold_topics_not_returned(self, tmp_path) -> None:
        db = tmp_path / "test.db"
        _init_db(db)
        _insert_manual(db, 1, "On hold topic", status="HOLD")
        _insert_manual(db, 2, "Ready topic", status="QUEUED")

        queue = TopicQueue(db_path=db, anthropic_api_key="")
        topics = queue.get_next_topics("money", max_count=3)

        manual_topics = [t for t in topics if t.get("source") == "MANUAL"]
        assert len(manual_topics) == 1
        assert manual_topics[0]["manual_seq"] == 2

    def test_skip_topics_not_returned(self, tmp_path) -> None:
        db = tmp_path / "test.db"
        _init_db(db)
        _insert_manual(db, 1, "Skipped topic", status="SKIP")

        queue = TopicQueue(db_path=db, anthropic_api_key="")
        topics = queue.get_next_topics("money", max_count=3)

        manual_topics = [t for t in topics if t.get("source") == "MANUAL"]
        assert len(manual_topics) == 0


# ---------------------------------------------------------------------------
# Tests: Fallback to AI when manual_topics empty
# ---------------------------------------------------------------------------

class TestFallbackToAI:
    def test_returns_scored_topics_when_no_manual(self, tmp_path) -> None:
        """When manual_topics is empty, scored_topics must be used."""
        db = tmp_path / "test.db"
        _init_db(db)
        _insert_scored(db, "passive income myths", score=80.0)

        queue = TopicQueue(db_path=db, anthropic_api_key="")
        topics = queue.get_next_topics("money", max_count=1)

        assert len(topics) >= 1
        assert topics[0]["keyword"] == "passive income myths"
        assert topics[0]["source"] != "MANUAL"

    def test_empty_db_returns_fallback(self, tmp_path) -> None:
        """Completely empty DB uses FALLBACK_TOPICS."""
        db = tmp_path / "test.db"
        _init_db(db)

        queue = TopicQueue(db_path=db, anthropic_api_key="")
        topics = queue.get_next_topics("money", max_count=1)

        # Should get at least one fallback topic
        assert len(topics) >= 1


# ---------------------------------------------------------------------------
# Tests: Graceful degradation
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    def test_sync_skips_when_env_vars_missing(self) -> None:
        """run_topic_queue_sync must skip silently without Sheet creds."""
        with patch.dict("os.environ", {"GOOGLE_SHEET_ID": "", "GOOGLE_CREDENTIALS_B64": ""}, clear=False):
            from src.scheduler import run_topic_queue_sync
            # Should not raise — just log warning and return
            run_topic_queue_sync()

    def test_queue_works_without_manual_topics_table(self, tmp_path) -> None:
        """TopicQueue must work even if manual_topics table doesn't exist."""
        db = tmp_path / "test.db"
        conn = sqlite3.connect(db)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS scored_topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                keyword TEXT NOT NULL, category TEXT NOT NULL DEFAULT 'money',
                score REAL NOT NULL DEFAULT 0, source TEXT NOT NULL DEFAULT 'manual',
                used INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS competitor_topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_name TEXT NOT NULL DEFAULT '', original_title TEXT NOT NULL DEFAULT '',
                extracted_topic TEXT NOT NULL, view_count INTEGER NOT NULL DEFAULT 0,
                category TEXT NOT NULL DEFAULT 'money', source TEXT NOT NULL DEFAULT 'competitor',
                used INTEGER NOT NULL DEFAULT 0,
                scraped_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
        """)
        conn.execute(
            "INSERT INTO scored_topics (keyword, category, score) VALUES ('test topic', 'money', 50)"
        )
        conn.commit()
        conn.close()

        queue = TopicQueue(db_path=db, anthropic_api_key="")
        topics = queue.get_next_topics("money", max_count=1)
        assert len(topics) >= 1


# ---------------------------------------------------------------------------
# Tests: run_topic_queue_sync inserts correct row count
# ---------------------------------------------------------------------------

class TestTopicQueueSync:
    def test_sync_inserts_topics_from_sheet(self, tmp_path) -> None:
        """run_topic_queue_sync must insert READY topics into manual_topics."""
        db = tmp_path / "test.db"
        _init_db(db)

        mock_batch = [
            {"seq": 1, "title": "Topic A", "category": "money",
             "hook_angle": "Hook A", "priority": "HIGH", "notes": ""},
            {"seq": 2, "title": "Topic B", "category": "career",
             "hook_angle": "", "priority": "MEDIUM", "notes": "test"},
            {"seq": 3, "title": "Topic C", "category": "success",
             "hook_angle": "Hook C", "priority": "LOW", "notes": ""},
        ]

        mock_sync = MagicMock()
        mock_sync.get_next_batch.return_value = mock_batch
        mock_sync.update_sync_log.return_value = None

        with patch.dict("os.environ", {
            "GOOGLE_SHEET_ID": "fake_id",
            "GOOGLE_CREDENTIALS_B64": "fake_creds",
            "DB_PATH": str(db),
        }, clear=False):
            with patch("src.scheduler._DEFAULT_DB", db):
                with patch("src.crawler.gsheet_topic_sync.GSheetTopicSync", return_value=mock_sync):
                    with patch("src.notifications.telegram_notifier.TelegramNotifier"):
                        from src.scheduler import run_topic_queue_sync
                        run_topic_queue_sync()

        conn = sqlite3.connect(db)
        rows = conn.execute("SELECT * FROM manual_topics ORDER BY seq").fetchall()
        conn.close()

        assert len(rows) == 3
        assert rows[0][0] == 1  # seq
        assert rows[0][1] == "Topic A"  # title
        assert rows[1][2] == "career"   # category

    def test_sync_updates_last_manual_seq(self, tmp_path) -> None:
        """After sync, last_manual_seq in settings must match highest SEQ."""
        db = tmp_path / "test.db"
        _init_db(db)

        mock_sync = MagicMock()
        mock_sync.get_next_batch.return_value = [
            {"seq": 10, "title": "Topic X", "category": "money",
             "hook_angle": "", "priority": "MEDIUM", "notes": ""},
        ]

        with patch.dict("os.environ", {
            "GOOGLE_SHEET_ID": "fake_id",
            "GOOGLE_CREDENTIALS_B64": "fake_creds",
        }, clear=False):
            with patch("src.scheduler._DEFAULT_DB", db):
                with patch("src.crawler.gsheet_topic_sync.GSheetTopicSync", return_value=mock_sync):
                    with patch("src.notifications.telegram_notifier.TelegramNotifier"):
                        from src.scheduler import run_topic_queue_sync
                        run_topic_queue_sync()

        conn = sqlite3.connect(db)
        row = conn.execute(
            "SELECT value FROM settings WHERE key = 'last_manual_seq'"
        ).fetchone()
        conn.close()
        assert row[0] == "10"


# ---------------------------------------------------------------------------
# Tests: GSheet USED writeback on consume
# ---------------------------------------------------------------------------

class TestGSheetWriteback:
    @patch("src.crawler.gsheet_topic_sync.GSheetTopicSync")
    def test_gsheet_mark_used_called_on_consume(self, mock_sync_cls, tmp_path) -> None:
        """When a manual topic is consumed, mark_used must be called on GSheet."""
        db = tmp_path / "test.db"
        _init_db(db)
        _insert_manual(db, 5, "Why your 401k is a trap")

        mock_sync = MagicMock()
        mock_sync_cls.return_value = mock_sync

        queue = TopicQueue(db_path=db, anthropic_api_key="")
        topics = queue.get_next_topics("money", max_count=1)

        assert len(topics) >= 1
        assert topics[0]["manual_seq"] == 5
        mock_sync.mark_used.assert_called_once_with(seq=5)

    @patch("src.crawler.gsheet_topic_sync.GSheetTopicSync")
    def test_gsheet_failure_does_not_block_topic(self, mock_sync_cls, tmp_path) -> None:
        """GSheet writeback failure must not prevent topic from being returned."""
        db = tmp_path / "test.db"
        _init_db(db)
        _insert_manual(db, 3, "Credit cards are designed to trap you")

        mock_sync = MagicMock()
        mock_sync_cls.return_value = mock_sync
        mock_sync.mark_used.side_effect = Exception("network error")

        queue = TopicQueue(db_path=db, anthropic_api_key="")
        topics = queue.get_next_topics("money", max_count=1)

        # Topic still returned despite GSheet failure
        assert len(topics) >= 1
        assert topics[0]["manual_seq"] == 3


# ---------------------------------------------------------------------------
# Tests: All categories returned (no category filter)
# ---------------------------------------------------------------------------

class TestNoCategoryFilter:
    def test_all_categories_returned_regardless_of_channel(self, tmp_path) -> None:
        """Manual topics of any category must be returned for any channel."""
        db = tmp_path / "test.db"
        _init_db(db)
        _insert_manual(db, 1, "Money topic", category="money")
        _insert_manual(db, 2, "Career topic", category="career")
        _insert_manual(db, 3, "Success topic", category="success")

        queue = TopicQueue(db_path=db, anthropic_api_key="")
        # Request as "money" channel — should still get all 3
        topics = queue.get_next_topics("money", max_count=3)

        manual_topics = [t for t in topics if t.get("source") == "MANUAL"]
        assert len(manual_topics) == 3
        categories = {t["category"] for t in manual_topics}
        assert categories == {"money", "career", "success"}

    def test_seq_order_preserved_across_categories(self, tmp_path) -> None:
        """SEQ order must be preserved even with mixed categories."""
        db = tmp_path / "test.db"
        _init_db(db)
        _insert_manual(db, 3, "Success first", category="success")
        _insert_manual(db, 7, "Money second", category="money")
        _insert_manual(db, 8, "Career third", category="career")

        queue = TopicQueue(db_path=db, anthropic_api_key="")
        topics = queue.get_next_topics("money", max_count=3)

        manual_topics = [t for t in topics if t.get("source") == "MANUAL"]
        assert manual_topics[0]["manual_seq"] == 3
        assert manual_topics[1]["manual_seq"] == 7
        assert manual_topics[2]["manual_seq"] == 8

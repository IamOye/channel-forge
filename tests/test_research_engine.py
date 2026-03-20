"""
Tests for src/research/research_engine.py

Tests the ResearchEngine class: scoring, rewriting, dedup, session management,
and reviewed topics memory. All API calls are mocked.
"""

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.research.research_engine import (
    ResearchEngine,
    RawTopic,
    ScoredTopic,
    _normalise_title,
    _safe_float,
    _safe_int,
    _safe_str,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS uploaded_videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT NOT NULL UNIQUE,
            channel_key TEXT NOT NULL DEFAULT 'default',
            topic_id TEXT NOT NULL DEFAULT '',
            title TEXT NOT NULL DEFAULT '',
            uploaded_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS production_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic_id TEXT NOT NULL, keyword TEXT NOT NULL,
            youtube_video_id TEXT, youtube_url TEXT,
            is_valid INTEGER NOT NULL DEFAULT 0,
            steps_json TEXT, validation_errors_json TEXT,
            completed_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS manual_topics (
            seq INTEGER PRIMARY KEY, title TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT 'money',
            hook_angle TEXT NOT NULL DEFAULT '', priority TEXT NOT NULL DEFAULT 'MEDIUM',
            notes TEXT NOT NULL DEFAULT '', status TEXT NOT NULL DEFAULT 'QUEUED',
            loaded_at TEXT NOT NULL DEFAULT (datetime('now')),
            used_at TEXT, video_id TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS research_reviewed (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL, normalised_title TEXT NOT NULL,
            original_title TEXT NOT NULL DEFAULT '',
            score REAL NOT NULL DEFAULT 0, category TEXT NOT NULL DEFAULT 'money',
            source TEXT NOT NULL DEFAULT '', action TEXT NOT NULL,
            session_id TEXT NOT NULL DEFAULT '',
            reviewed_at TEXT NOT NULL DEFAULT (datetime('now')),
            synced_to_sheet INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS research_sessions (
            id TEXT PRIMARY KEY, chat_id TEXT NOT NULL DEFAULT '',
            source TEXT NOT NULL DEFAULT 'all', category TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'active',
            current_index INTEGER NOT NULL DEFAULT 0,
            total_topics INTEGER NOT NULL DEFAULT 0,
            topics_added INTEGER NOT NULL DEFAULT 0,
            topics_skipped INTEGER NOT NULL DEFAULT 0,
            topics_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Deduplication tests
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_removes_exact_duplicates(self, tmp_path) -> None:
        db = tmp_path / "test.db"
        _init_db(db)
        engine = ResearchEngine(db_path=db, api_key="")

        topics = [
            RawTopic(title="Why banks want you in debt forever", source="reddit"),
            RawTopic(title="Why banks want you in debt forever", source="trends"),
        ]
        result = engine.deduplicate(topics)
        assert len(result) == 1

    def test_removes_already_uploaded(self, tmp_path) -> None:
        db = tmp_path / "test.db"
        _init_db(db)
        conn = sqlite3.connect(db)
        conn.execute("INSERT INTO uploaded_videos (video_id, title) VALUES ('v1', 'Why banks want you in debt forever')")
        conn.commit()
        conn.close()

        engine = ResearchEngine(db_path=db, api_key="")
        topics = [RawTopic(title="Why banks want you in debt forever", source="reddit")]
        result = engine.deduplicate(topics)
        assert len(result) == 0

    def test_removes_already_reviewed(self, tmp_path) -> None:
        db = tmp_path / "test.db"
        _init_db(db)
        conn = sqlite3.connect(db)
        conn.execute(
            "INSERT INTO research_reviewed (title, normalised_title, action) "
            "VALUES ('Why banks want you broke', 'why banks want you broke', 'skipped')"
        )
        conn.commit()
        conn.close()

        engine = ResearchEngine(db_path=db, api_key="")
        topics = [RawTopic(title="Why banks want you broke!", source="reddit")]
        result = engine.deduplicate(topics)
        assert len(result) == 0

    def test_removes_short_titles(self, tmp_path) -> None:
        db = tmp_path / "test.db"
        _init_db(db)
        engine = ResearchEngine(db_path=db, api_key="")
        topics = [RawTopic(title="Save more money", source="trends")]  # 3 words
        result = engine.deduplicate(topics)
        assert len(result) == 0

    def test_keeps_valid_topics(self, tmp_path) -> None:
        db = tmp_path / "test.db"
        _init_db(db)
        engine = ResearchEngine(db_path=db, api_key="")
        topics = [RawTopic(title="Why your emergency fund is making banks rich", source="reddit")]
        result = engine.deduplicate(topics)
        assert len(result) == 1

    def test_does_not_check_scored_topics(self, tmp_path) -> None:
        """scored_topics is the research pool, NOT a filter."""
        db = tmp_path / "test.db"
        _init_db(db)
        conn = sqlite3.connect(db)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scored_topics (
                id INTEGER PRIMARY KEY, keyword TEXT, category TEXT, score REAL,
                source TEXT, used INTEGER DEFAULT 0, created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("INSERT INTO scored_topics (keyword, category, score) VALUES ('Why banks want you in debt forever', 'money', 80)")
        conn.commit()
        conn.close()

        engine = ResearchEngine(db_path=db, api_key="")
        topics = [RawTopic(title="Why banks want you in debt forever", source="reddit")]
        result = engine.deduplicate(topics)
        assert len(result) == 1  # NOT filtered by scored_topics


# ---------------------------------------------------------------------------
# Scoring tests
# ---------------------------------------------------------------------------

class TestScoring:
    def test_unscored_when_no_api_key(self, tmp_path) -> None:
        engine = ResearchEngine(db_path=tmp_path / "test.db", api_key="")
        topics = [RawTopic(title="Why your pension will not be enough for retirement", source="reddit")]
        scored = engine.score(topics)
        assert len(scored) == 1
        assert scored[0].score == 0.0
        assert scored[0].reason == "unscored"

    def test_weighted_score_calculation(self, tmp_path) -> None:
        """Claude returns sub-scores; engine computes weighted average."""
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text=json.dumps([{
            "title": "Why banks want you broke forever and always",
            "hook_strength": 9, "contrarian": 8, "specificity": 7,
            "brand_fit": 8, "search_demand": 6,
            "score": 8.1, "category": "money",
            "hook_angle": "The system profits from your ignorance",
            "reason": "Strong hook, contrarian, specific"
        }]))]

        engine = ResearchEngine(db_path=tmp_path / "test.db", api_key="fake")
        topics = [RawTopic(title="Why banks want you broke forever and always", source="reddit")]

        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.create.return_value = mock_msg
            scored = engine.score(topics)

        assert len(scored) == 1
        expected = 9*0.35 + 8*0.25 + 7*0.20 + 8*0.10 + 6*0.10
        assert scored[0].score == round(expected, 1)
        assert scored[0].hook_strength == 9.0


# ---------------------------------------------------------------------------
# Rewrite tests
# ---------------------------------------------------------------------------

class TestRewrite:
    def test_rewrite_sets_original_title(self, tmp_path) -> None:
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text=json.dumps([{
            "original": "Why banks want you broke forever and always",
            "rewritten": "Banks profit when you stay broke",
            "rewrite_score": 9,
            "improvement_reason": "Shorter, more direct"
        }]))]

        engine = ResearchEngine(db_path=tmp_path / "test.db", api_key="fake")
        scored = [ScoredTopic(
            title="Why banks want you broke forever and always", score=8.0,
            category="money", hook_angle="", reason="", source="reddit",
            source_detail="r/personalfinance",
        )]

        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.create.return_value = mock_msg
            rewritten = engine.rewrite(scored)

        assert len(rewritten) == 1
        assert rewritten[0].title == "Banks profit when you stay broke"
        assert rewritten[0].original_title == "Why banks want you broke forever and always"

    def test_rewrite_preserves_topics_on_api_failure(self, tmp_path) -> None:
        engine = ResearchEngine(db_path=tmp_path / "test.db", api_key="fake")
        scored = [ScoredTopic(
            title="Original topic title with enough words here", score=7.0,
            category="career", hook_angle="", reason="", source="trends",
            source_detail="",
        )]

        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.create.side_effect = Exception("API down")
            rewritten = engine.rewrite(scored)

        assert len(rewritten) == 1
        assert rewritten[0].title == "Original topic title with enough words here"


# ---------------------------------------------------------------------------
# Reviewed topics memory
# ---------------------------------------------------------------------------

class TestReviewedMemory:
    def test_mark_reviewed_inserts_row(self, tmp_path) -> None:
        db = tmp_path / "test.db"
        _init_db(db)
        engine = ResearchEngine(db_path=db, api_key="")

        engine.mark_reviewed("Why banks want you broke", action="added", score=8.0)

        conn = sqlite3.connect(db)
        rows = conn.execute("SELECT * FROM research_reviewed").fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0][1] == "Why banks want you broke"  # title
        assert rows[0][7] == "added"  # action

    def test_mark_batch_reviewed(self, tmp_path) -> None:
        db = tmp_path / "test.db"
        _init_db(db)
        engine = ResearchEngine(db_path=db, api_key="")

        topics = [
            ScoredTopic(title="Topic A with enough words here", score=8.0,
                        category="money", hook_angle="", reason="", source="reddit", source_detail=""),
            ScoredTopic(title="Topic B with enough words here", score=7.0,
                        category="career", hook_angle="", reason="", source="trends", source_detail=""),
        ]
        engine.mark_batch_reviewed(topics, "skipped", "session1")

        conn = sqlite3.connect(db)
        count = conn.execute("SELECT COUNT(*) FROM research_reviewed").fetchone()[0]
        conn.close()
        assert count == 2


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

class TestSessions:
    def test_create_and_get_session(self, tmp_path) -> None:
        db = tmp_path / "test.db"
        _init_db(db)
        engine = ResearchEngine(db_path=db, api_key="")

        topics = [ScoredTopic(
            title="Test topic title that is long enough", score=8.0,
            category="money", hook_angle="Hook", reason="reason",
            source="reddit", source_detail="r/test",
        )]
        session_id = engine.create_session("chat123", topics)
        assert session_id

        session = engine.get_session("chat123")
        assert session is not None
        assert session["status"] == "active"
        assert session["total_topics"] == 1

    def test_get_session_topics(self, tmp_path) -> None:
        db = tmp_path / "test.db"
        _init_db(db)
        engine = ResearchEngine(db_path=db, api_key="")

        topics = [ScoredTopic(
            title="Test topic with enough words in it", score=8.0,
            category="money", hook_angle="Hook", reason="reason",
            source="reddit", source_detail="r/test",
        )]
        engine.create_session("chat123", topics)
        session = engine.get_session("chat123")
        loaded = engine.get_session_topics(session)
        assert len(loaded) == 1
        assert loaded[0].title == "Test topic with enough words in it"

    def test_update_session(self, tmp_path) -> None:
        db = tmp_path / "test.db"
        _init_db(db)
        engine = ResearchEngine(db_path=db, api_key="")

        topics = [ScoredTopic(title="Topic A B C D E", score=8.0,
                  category="money", hook_angle="", reason="", source="", source_detail="")]
        sid = engine.create_session("chat123", topics)
        engine.update_session(sid, current_index=5, topics_added=3)

        session = engine.get_session("chat123")
        assert session["current_index"] == 5
        assert session["topics_added"] == 3


# ---------------------------------------------------------------------------
# ScoredTopic serialization
# ---------------------------------------------------------------------------

class TestScoredTopicSerde:
    def test_to_dict_and_from_dict_roundtrip(self) -> None:
        original = ScoredTopic(
            title="Test title", score=8.5, category="money",
            hook_angle="Hook", reason="reason", source="reddit",
            source_detail="r/test", score_hint=100.0,
            hook_strength=9.0, contrarian=8.0, specificity=7.0,
            brand_fit=8.0, search_demand=6.0,
            original_title="Original", rewritten_score=9.2,
        )
        d = original.to_dict()
        restored = ScoredTopic.from_dict(d)
        assert restored.title == original.title
        assert restored.score == original.score
        assert restored.hook_strength == original.hook_strength
        assert restored.original_title == original.original_title


# ---------------------------------------------------------------------------
# Exclude Reddit on Railway
# ---------------------------------------------------------------------------

class TestExcludeReddit:
    def test_exclude_reddit_skips_reddit_scraper(self, tmp_path) -> None:
        engine = ResearchEngine(db_path=tmp_path / "test.db", api_key="", exclude_reddit=True)
        # Mock all scrapers to track which are called
        with patch.object(engine, "_scrape_reddit") as mock_reddit, \
             patch.object(engine, "_scrape_autocomplete", return_value=[]), \
             patch.object(engine, "_scrape_trends", return_value=[]), \
             patch.object(engine, "_scrape_competitors", return_value=[]):
            engine.scrape()
            mock_reddit.assert_not_called()

    def test_include_reddit_calls_reddit_scraper(self, tmp_path) -> None:
        engine = ResearchEngine(db_path=tmp_path / "test.db", api_key="", exclude_reddit=False)
        with patch.object(engine, "_scrape_reddit", return_value=[]) as mock_reddit, \
             patch.object(engine, "_scrape_autocomplete", return_value=[]), \
             patch.object(engine, "_scrape_trends", return_value=[]), \
             patch.object(engine, "_scrape_competitors", return_value=[]):
            engine.scrape()
            mock_reddit.assert_called_once()


# ---------------------------------------------------------------------------
# Safe conversion helpers
# ---------------------------------------------------------------------------

class TestSafeHelpers:
    def test_safe_str_none(self) -> None:
        assert _safe_str(None) == ""

    def test_safe_int_none(self) -> None:
        assert _safe_int(None) == 0

    def test_safe_float_none(self) -> None:
        assert _safe_float(None) == 0.0

    def test_safe_int_invalid(self) -> None:
        assert _safe_int("abc") == 0

    def test_normalise_title_none(self) -> None:
        assert _normalise_title(None) == ""

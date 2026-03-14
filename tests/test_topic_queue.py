"""
Tests for src/pipeline/topic_queue.py

Uses real SQLite via tmp_path for DB tests.
All Claude API calls are mocked.
"""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.topic_queue import TopicQueue


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_queue(tmp_path: Path, **kw) -> TopicQueue:
    return TopicQueue(
        db_path=tmp_path / "test.db",
        anthropic_api_key=kw.get("anthropic_api_key", ""),
    )


def _insert_scored_topics(db_path: Path, rows: list[tuple]) -> None:
    """Insert (keyword, category, score, source) into scored_topics."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scored_topics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            keyword TEXT, category TEXT,
            score REAL, source TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.executemany(
        "INSERT INTO scored_topics (keyword, category, score, source) VALUES (?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()


def _insert_competitor_topics(db_path: Path, rows: list[tuple]) -> None:
    """Insert (extracted_topic, category, view_count, source) into competitor_topics."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS competitor_topics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel_name TEXT, original_title TEXT,
            extracted_topic TEXT, view_count INTEGER,
            category TEXT, source TEXT, used INTEGER DEFAULT 0,
            scraped_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.executemany(
        """INSERT INTO competitor_topics
           (channel_name, original_title, extracted_topic, view_count, category, source)
           VALUES ('test','test',?,?,?,?)""",
        rows,
    )
    conn.commit()
    conn.close()


def _insert_uploaded(db_path: Path, keywords: list[str]) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS production_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic_id TEXT, keyword TEXT, category TEXT,
            hook TEXT DEFAULT '', script TEXT DEFAULT '',
            voiceover_path TEXT DEFAULT '', video_path TEXT DEFAULT '',
            youtube_video_id TEXT DEFAULT '', is_valid INTEGER DEFAULT 0,
            validation_errors TEXT DEFAULT '[]', created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    for kw in keywords:
        conn.execute(
            "INSERT INTO production_results (topic_id, keyword, category, created_at) VALUES (?,?,?,datetime('now'))",
            (f"t_{kw[:5]}", kw, "money"),
        )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# get_uploaded_topics
# ---------------------------------------------------------------------------

class TestGetUploadedTopics:
    def test_returns_empty_when_db_missing(self, tmp_path) -> None:
        queue = _make_queue(tmp_path)
        result = queue.get_uploaded_topics()
        assert result == []

    def test_returns_keywords_from_production_results(self, tmp_path) -> None:
        db = tmp_path / "test.db"
        _insert_uploaded(db, ["why saving money keeps you poor", "the debt trap"])
        queue = TopicQueue(db_path=db)
        result = queue.get_uploaded_topics()
        assert "why saving money keeps you poor" in result
        assert "the debt trap" in result

    def test_handles_missing_tables_gracefully(self, tmp_path) -> None:
        db = tmp_path / "test.db"
        db.parent.mkdir(parents=True, exist_ok=True)
        # Create DB with no tables
        sqlite3.connect(db).close()
        queue = TopicQueue(db_path=db)
        result = queue.get_uploaded_topics()
        assert result == []


# ---------------------------------------------------------------------------
# get_next_topics — priority ordering
# ---------------------------------------------------------------------------

class TestGetNextTopicsPriority:
    def test_competitor_topics_ranked_above_scored(self, tmp_path) -> None:
        db = tmp_path / "test.db"
        _insert_scored_topics(db, [
            ("google trends topic", "money", 70.0, "GOOGLE_TRENDS"),
        ])
        _insert_competitor_topics(db, [
            ("competitor viral topic", 200_000, "money", "COMPETITOR_HIGH_SIGNAL"),
        ])

        queue = TopicQueue(db_path=db)
        topics = queue.get_next_topics(category="money", max_count=5)
        keywords = [t["keyword"] for t in topics]
        assert keywords.index("competitor viral topic") < keywords.index("google trends topic")

    def test_viewer_requested_ranked_first(self, tmp_path) -> None:
        db = tmp_path / "test.db"
        _insert_competitor_topics(db, [
            ("viewer question topic", 0, "money", "VIEWER_REQUESTED"),
            ("trending topic", 50_000, "money", "YOUTUBE_TRENDING"),
        ])

        queue = TopicQueue(db_path=db)
        topics = queue.get_next_topics(category="money", max_count=5)
        assert topics[0]["keyword"] == "viewer question topic"

    def test_returns_at_most_max_count(self, tmp_path) -> None:
        db = tmp_path / "test.db"
        _insert_scored_topics(db, [
            (f"topic {i}", "money", float(80 - i), "GOOGLE_TRENDS")
            for i in range(10)
        ])
        queue = TopicQueue(db_path=db)
        topics = queue.get_next_topics(category="money", max_count=3)
        assert len(topics) <= 3

    def test_fallback_used_when_db_empty(self, tmp_path) -> None:
        queue = _make_queue(tmp_path)
        topics = queue.get_next_topics(category="money", max_count=2)
        assert len(topics) >= 1
        assert topics[0]["source"] == "FALLBACK"

    def test_fallback_category_respected(self, tmp_path) -> None:
        queue = _make_queue(tmp_path)
        topics = queue.get_next_topics(category="career", max_count=1)
        assert len(topics) == 1
        # Career fallback topics contain career-related words
        career_words = {"salary", "career", "degree", "paycheck", "loyal", "broke"}
        kw_words = set(topics[0]["keyword"].lower().split())
        assert kw_words & career_words  # at least one overlap


# ---------------------------------------------------------------------------
# get_next_topics — deduplication
# ---------------------------------------------------------------------------

class TestGetNextTopicsDedup:
    def test_uploaded_topic_excluded(self, tmp_path) -> None:
        db = tmp_path / "test.db"
        _insert_scored_topics(db, [
            ("why saving money keeps you poor", "money", 70.0, "GOOGLE_TRENDS"),
            ("the debt trap nobody talks about", "money", 65.0, "GOOGLE_TRENDS"),
        ])

        queue = TopicQueue(db_path=db)
        # Near-identical to first scored topic (only last word differs → similarity > 0.70)
        topics = queue.get_next_topics(
            category="money",
            max_count=5,
            uploaded_topics=["why saving money keeps you broke"],
        )
        keywords = [t["keyword"] for t in topics]
        # Near-duplicate must be filtered
        assert "why saving money keeps you poor" not in keywords
        # Unrelated topic must still appear
        assert "the debt trap nobody talks about" in keywords

    def test_no_duplicate_topics_within_results(self, tmp_path) -> None:
        db = tmp_path / "test.db"
        # Insert near-identical topics from two sources (only last word differs)
        _insert_scored_topics(db, [
            ("why saving money keeps you poor", "money", 70.0, "GOOGLE_TRENDS"),
        ])
        _insert_competitor_topics(db, [
            ("why saving money keeps you broke", 50_000, "money", "YOUTUBE_TRENDING"),
        ])

        queue = TopicQueue(db_path=db)
        topics = queue.get_next_topics(category="money", max_count=5)
        keywords = [t["keyword"] for t in topics]
        # Both near-identical topics must NOT both appear in results
        both_present = (
            "why saving money keeps you poor" in keywords
            and "why saving money keeps you broke" in keywords
        )
        assert not both_present, "Near-duplicate topics both appeared — dedup failed"


# ---------------------------------------------------------------------------
# get_next_topics — Claude fresh topic last resort
# ---------------------------------------------------------------------------

class TestGetNextTopicsClaudeFallback:
    @patch("src.pipeline.topic_queue.TopicQueue._generate_fresh_topic")
    def test_calls_claude_when_all_exhausted(self, mock_gen, tmp_path) -> None:
        mock_gen.return_value = "a brand new unique topic"
        # DB empty, all fallbacks exhausted via high threshold
        queue = TopicQueue(
            db_path=tmp_path / "test.db",
            anthropic_api_key="fake",
        )
        # Exhaust fallbacks by marking all as uploaded
        from config.constants import FALLBACK_TOPICS
        all_fallbacks = [t for ts in FALLBACK_TOPICS.values() for t in ts]

        topics = queue.get_next_topics(
            category="money",
            max_count=1,
            uploaded_topics=all_fallbacks,
        )
        mock_gen.assert_called_once()

    def test_returns_empty_list_when_no_api_key_and_no_topics(self, tmp_path) -> None:
        from config.constants import FALLBACK_TOPICS
        all_fallbacks = [t for ts in FALLBACK_TOPICS.values() for t in ts]

        queue = TopicQueue(
            db_path=tmp_path / "test.db",
            anthropic_api_key="",
        )
        topics = queue.get_next_topics(
            category="money",
            max_count=1,
            uploaded_topics=all_fallbacks,
        )
        assert topics == []


# ---------------------------------------------------------------------------
# Priority queue ordering — all 9 levels
# ---------------------------------------------------------------------------

class TestPriorityOrdering:
    def test_all_9_priority_levels_present(self) -> None:
        from config.constants import SOURCE_PRIORITIES
        expected = [
            "VIEWER_REQUESTED", "COMPETITOR_HIGH_SIGNAL",
            "AUTOCOMPLETE", "TRENDING_SEARCH", "YOUTUBE_TRENDING",
            "RISING_GOOGLE_TRENDS", "GOOGLE_TRENDS", "YOUTUBE_KEYWORD", "FALLBACK",
        ]
        for src in expected:
            assert src in SOURCE_PRIORITIES, f"Missing priority source: {src}"

    def test_priority_ordering_top_to_bottom(self) -> None:
        from config.constants import SOURCE_PRIORITIES
        p = SOURCE_PRIORITIES
        assert p["VIEWER_REQUESTED"]       > p["COMPETITOR_HIGH_SIGNAL"]
        assert p["COMPETITOR_HIGH_SIGNAL"] > p["AUTOCOMPLETE"]
        assert p["AUTOCOMPLETE"]           > p["RISING_GOOGLE_TRENDS"]
        assert p["RISING_GOOGLE_TRENDS"]   > p["GOOGLE_TRENDS"]
        assert p["GOOGLE_TRENDS"]          > p["YOUTUBE_KEYWORD"]
        assert p["YOUTUBE_KEYWORD"]        > p["FALLBACK"]

    def test_autocomplete_at_or_above_youtube_trending(self) -> None:
        from config.constants import SOURCE_PRIORITIES
        assert SOURCE_PRIORITIES["AUTOCOMPLETE"] >= SOURCE_PRIORITIES["YOUTUBE_TRENDING"]

    def test_trending_search_equal_to_youtube_trending(self) -> None:
        from config.constants import SOURCE_PRIORITIES
        assert SOURCE_PRIORITIES["TRENDING_SEARCH"] == SOURCE_PRIORITIES["YOUTUBE_TRENDING"]

    def test_rising_google_trends_above_google_trends(self) -> None:
        from config.constants import SOURCE_PRIORITIES
        assert SOURCE_PRIORITIES["RISING_GOOGLE_TRENDS"] > SOURCE_PRIORITIES["GOOGLE_TRENDS"]

    def test_queue_ranks_autocomplete_before_fallback(self, tmp_path) -> None:
        """AUTOCOMPLETE topics rank above FALLBACK in get_next_topics output."""
        db = tmp_path / "test.db"
        _insert_competitor_topics(db, [
            ("autocomplete suggestion topic", 0, "money", "AUTOCOMPLETE"),
        ])
        queue = TopicQueue(db_path=db)
        topics = queue.get_next_topics(category="money", max_count=5)
        keywords = [t["keyword"] for t in topics]
        assert "autocomplete suggestion topic" in keywords
        ac_idx = keywords.index("autocomplete suggestion topic")
        # All fallback topics (if any) must appear after the autocomplete topic
        for i, t in enumerate(topics):
            if t.get("source") == "FALLBACK":
                assert ac_idx < i


# ---------------------------------------------------------------------------
# Scheduler integration — competitor_research job exists
# ---------------------------------------------------------------------------

class TestSchedulerCompetitorJob:
    def test_competitor_research_job_registered(self) -> None:
        from apscheduler.triggers.cron import CronTrigger
        from src.scheduler import build_scheduler, run_competitor_research

        scheduler = build_scheduler(timezone_name="UTC")
        job_ids = [j.id for j in scheduler.get_jobs()]
        assert "competitor_research" in job_ids

    def test_competitor_research_job_is_correct_function(self) -> None:
        from src.scheduler import build_scheduler, run_competitor_research

        scheduler = build_scheduler(timezone_name="UTC")
        job = next(j for j in scheduler.get_jobs() if j.id == "competitor_research")
        assert job.func is run_competitor_research

    def test_scheduler_now_has_five_jobs(self) -> None:
        from src.scheduler import build_scheduler
        scheduler = build_scheduler(timezone_name="UTC")
        assert len(scheduler.get_jobs()) == 5

    def test_run_competitor_research_does_not_raise(self) -> None:
        """run_competitor_research must swallow all exceptions."""
        with patch("src.crawler.competitor_scraper.CompetitorScraper") as mock_cls:
            mock_cls.side_effect = Exception("import failure")
            from src.scheduler import run_competitor_research
            # Should not raise
            run_competitor_research()

    def import_src_scheduler(self):
        from src.scheduler import build_scheduler
        return build_scheduler

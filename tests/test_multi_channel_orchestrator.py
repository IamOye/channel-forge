"""
Tests for src/pipeline/multi_channel_orchestrator.py

All pipeline calls and file I/O are mocked — no real API calls.
Uses tmp_path for real SQLite topic loading tests.
"""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from config.channels import ChannelConfig
from src.pipeline.multi_channel_orchestrator import (
    ChannelRunResult,
    MultiChannelOrchestrator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_orchestrator(tmp_path: Path, **kw) -> MultiChannelOrchestrator:
    return MultiChannelOrchestrator(
        base_output_dir=tmp_path / "output",
        base_db_dir=tmp_path / "processed",
        **kw,
    )


def _make_channel(**overrides) -> ChannelConfig:
    defaults = dict(channel_key="test_ch", name="Test Channel")
    defaults.update(overrides)
    return ChannelConfig(**defaults)


def _make_mock_pipeline(is_valid: bool = True) -> MagicMock:
    """Return a mock ProductionPipeline whose run() returns a mock result."""
    mock_result = MagicMock()
    mock_result.is_valid = is_valid
    mock_result.validation_errors = [] if is_valid else ["step failed"]
    pipeline = MagicMock()
    pipeline.run.return_value = mock_result
    return pipeline


# ---------------------------------------------------------------------------
# ChannelConfig
# ---------------------------------------------------------------------------

class TestChannelConfig:
    def test_default_output_dir_auto_set(self) -> None:
        cfg = ChannelConfig(channel_key="mychan", name="My Channel")
        assert cfg.output_dir == "data/output/mychan"

    def test_custom_output_dir_preserved(self) -> None:
        cfg = ChannelConfig(
            channel_key="mychan", name="My Channel", output_dir="/custom/path"
        )
        assert cfg.output_dir == "/custom/path"

    def test_default_daily_quota(self) -> None:
        cfg = ChannelConfig(channel_key="c", name="C")
        assert cfg.daily_quota == 3

    def test_default_category(self) -> None:
        cfg = ChannelConfig(channel_key="c", name="C")
        assert cfg.category == "success"

    def test_default_timezone(self) -> None:
        cfg = ChannelConfig(channel_key="c", name="C")
        assert cfg.timezone == "Africa/Lagos"


# ---------------------------------------------------------------------------
# ChannelRunResult
# ---------------------------------------------------------------------------

class TestChannelRunResult:
    def test_to_dict_has_all_keys(self) -> None:
        r = ChannelRunResult(channel_key="ch1", channel_name="Ch One")
        d = r.to_dict()
        for key in (
            "channel_key", "channel_name", "topics_processed",
            "topics_succeeded", "topics_failed",
            "is_valid", "error", "completed_at",
        ):
            assert key in d

    def test_completed_at_auto_set(self) -> None:
        r = ChannelRunResult(channel_key="ch1", channel_name="Ch One")
        assert r.completed_at != ""

    def test_default_is_valid_true(self) -> None:
        r = ChannelRunResult(channel_key="ch1", channel_name="Ch One")
        assert r.is_valid is True

    def test_to_dict_serializable(self) -> None:
        import json
        r = ChannelRunResult(channel_key="ch1", channel_name="Ch One", topics_succeeded=2)
        assert len(json.dumps(r.to_dict())) > 10


# ---------------------------------------------------------------------------
# run_channel — success path
# ---------------------------------------------------------------------------

class TestRunChannel:
    def test_successful_run_counts_topics(self, tmp_path) -> None:
        orchestrator = _make_orchestrator(tmp_path)
        channel = _make_channel()
        mock_pipeline = _make_mock_pipeline(is_valid=True)

        with patch.object(orchestrator, "_load_topics", return_value=[
            {"topic_id": "t001", "keyword": "stoic", "category": "success", "score": 85},
            {"topic_id": "t002", "keyword": "wisdom", "category": "success", "score": 80},
        ]):
            with patch.object(orchestrator, "_build_pipeline", return_value=mock_pipeline):
                with patch.object(orchestrator, "_setup_channel_output"):
                    result = orchestrator.run_channel(channel)

        assert result.topics_processed == 2
        assert result.topics_succeeded == 2
        assert result.topics_failed == 0
        assert result.is_valid is True

    def test_failed_pipeline_counted_in_topics_failed(self, tmp_path) -> None:
        orchestrator = _make_orchestrator(tmp_path)
        channel = _make_channel()
        mock_pipeline = _make_mock_pipeline(is_valid=False)

        with patch.object(orchestrator, "_load_topics", return_value=[
            {"topic_id": "t001", "keyword": "stoic", "category": "success", "score": 85},
        ]):
            with patch.object(orchestrator, "_build_pipeline", return_value=mock_pipeline):
                with patch.object(orchestrator, "_setup_channel_output"):
                    result = orchestrator.run_channel(channel)

        assert result.topics_failed == 1
        assert result.topics_succeeded == 0

    def test_exception_in_topic_run_counted_as_failed(self, tmp_path) -> None:
        orchestrator = _make_orchestrator(tmp_path)
        channel = _make_channel()
        mock_pipeline = MagicMock()
        mock_pipeline.run.side_effect = Exception("pipeline crashed")

        with patch.object(orchestrator, "_load_topics", return_value=[
            {"topic_id": "t001", "keyword": "stoic", "category": "success", "score": 85},
        ]):
            with patch.object(orchestrator, "_build_pipeline", return_value=mock_pipeline):
                with patch.object(orchestrator, "_setup_channel_output"):
                    result = orchestrator.run_channel(channel)

        assert result.topics_failed == 1
        assert result.is_valid is True  # channel itself still valid

    def test_setup_failure_marks_channel_invalid(self, tmp_path) -> None:
        orchestrator = _make_orchestrator(tmp_path)
        channel = _make_channel()

        with patch.object(orchestrator, "_load_topics", side_effect=Exception("DB error")):
            result = orchestrator.run_channel(channel)

        assert result.is_valid is False
        assert "DB error" in result.error

    def test_channel_key_and_name_preserved(self, tmp_path) -> None:
        orchestrator = _make_orchestrator(tmp_path)
        channel = _make_channel(channel_key="stoic_ch", name="Stoic Channel")
        mock_pipeline = _make_mock_pipeline()

        with patch.object(orchestrator, "_load_topics", return_value=[]):
            with patch.object(orchestrator, "_build_pipeline", return_value=mock_pipeline):
                with patch.object(orchestrator, "_setup_channel_output"):
                    result = orchestrator.run_channel(channel)

        assert result.channel_key == "stoic_ch"
        assert result.channel_name == "Stoic Channel"

    def test_respects_daily_quota_via_load_topics(self, tmp_path) -> None:
        orchestrator = _make_orchestrator(tmp_path, topics_per_channel=5)
        channel = _make_channel(daily_quota=2)

        captured = {}

        def capture_load(cfg, max_topics):
            captured["max_topics"] = max_topics
            return []

        with patch.object(orchestrator, "_load_topics", side_effect=capture_load):
            with patch.object(orchestrator, "_build_pipeline", return_value=MagicMock()):
                with patch.object(orchestrator, "_setup_channel_output"):
                    orchestrator.run_channel(channel)

        # daily_quota from channel config (2) takes priority over topics_per_channel (5)
        assert captured["max_topics"] == 2


# ---------------------------------------------------------------------------
# run_all — channel isolation
# ---------------------------------------------------------------------------

class TestRunAll:
    def test_runs_all_channels(self, tmp_path) -> None:
        orchestrator = _make_orchestrator(tmp_path)

        channels = [
            _make_channel(channel_key="ch1", name="Channel 1"),
            _make_channel(channel_key="ch2", name="Channel 2"),
        ]

        with patch("config.channels.CHANNELS", channels):
            with patch.object(orchestrator, "_load_topics", return_value=[]):
                with patch.object(orchestrator, "_build_pipeline", return_value=MagicMock()):
                    with patch.object(orchestrator, "_setup_channel_output"):
                        results = orchestrator.run_all()

        assert len(results) == 2
        keys = {r.channel_key for r in results}
        assert keys == {"ch1", "ch2"}

    def test_one_channel_failure_does_not_stop_others(self, tmp_path) -> None:
        orchestrator = _make_orchestrator(tmp_path)

        channels = [
            _make_channel(channel_key="ch1", name="Channel 1"),
            _make_channel(channel_key="ch2", name="Channel 2"),
            _make_channel(channel_key="ch3", name="Channel 3"),
        ]

        call_count = {"n": 0}
        def run_channel_side_effect(cfg):
            call_count["n"] += 1
            if cfg.channel_key == "ch2":
                return ChannelRunResult(
                    channel_key="ch2", channel_name="Channel 2",
                    is_valid=False, error="fatal error",
                )
            return ChannelRunResult(
                channel_key=cfg.channel_key, channel_name=cfg.name
            )

        with patch("config.channels.CHANNELS", channels):
            with patch.object(orchestrator, "run_channel", side_effect=run_channel_side_effect):
                results = orchestrator.run_all()

        # All 3 channels attempted
        assert call_count["n"] == 3
        assert len(results) == 3

        # ch1 and ch3 succeed, ch2 fails
        ch2_result = next(r for r in results if r.channel_key == "ch2")
        assert ch2_result.is_valid is False
        ch1_result = next(r for r in results if r.channel_key == "ch1")
        assert ch1_result.is_valid is True

    def test_empty_channels_returns_empty_list(self, tmp_path) -> None:
        orchestrator = _make_orchestrator(tmp_path)

        with patch("config.channels.CHANNELS", []):
            results = orchestrator.run_all()

        assert results == []


# ---------------------------------------------------------------------------
# _load_topics — real SQLite
# ---------------------------------------------------------------------------

class TestLoadTopics:
    def _insert_topics(self, db_path: Path, topics: list[tuple]) -> None:
        """Insert (keyword, category, score) rows into scored_topics."""
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scored_topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                keyword TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT 'success',
                score REAL NOT NULL DEFAULT 0,
                source TEXT NOT NULL DEFAULT 'manual',
                created_at TEXT NOT NULL DEFAULT '2025-01-01'
            )
            """
        )
        for kw, cat, score in topics:
            conn.execute(
                "INSERT INTO scored_topics (keyword, category, score, created_at) VALUES (?, ?, ?, '2025-01-01')",
                (kw, cat, score),
            )
        conn.commit()
        conn.close()

    def test_loads_topics_from_channel_db(self, tmp_path) -> None:
        orchestrator = _make_orchestrator(tmp_path)
        channel = _make_channel(channel_key="testch")
        db_path = tmp_path / "processed" / "testch.db"
        self._insert_topics(db_path, [
            ("stoic topic", "success", 90.0),
            ("career topic", "career", 85.0),
        ])

        topics = orchestrator._load_topics(channel, max_topics=5)
        assert len(topics) >= 1
        # DB topic (GOOGLE_TRENDS priority=70) ranks above fallbacks (priority=50)
        assert topics[0]["keyword"] == "stoic topic"

    def test_limits_to_max_topics(self, tmp_path) -> None:
        orchestrator = _make_orchestrator(tmp_path)
        channel = _make_channel(channel_key="testch")
        db_path = tmp_path / "processed" / "testch.db"
        self._insert_topics(db_path, [
            ("t1", "success", 95.0),
            ("t2", "success", 90.0),
            ("t3", "success", 85.0),
        ])

        topics = orchestrator._load_topics(channel, max_topics=2)
        assert len(topics) == 2

    def test_missing_db_returns_fallback_topic(self, tmp_path) -> None:
        orchestrator = _make_orchestrator(tmp_path)
        channel = _make_channel(channel_key="nodb")
        # No DB file — TopicQueue fills with FALLBACK_TOPICS up to max_topics
        topics = orchestrator._load_topics(channel, max_topics=5)
        assert len(topics) >= 1
        assert topics[0]["keyword"] != ""
        assert topics[0]["category"] == "success"
        assert "fallback" in topics[0]["topic_id"]

    def test_topic_has_expected_keys(self, tmp_path) -> None:
        orchestrator = _make_orchestrator(tmp_path)
        channel = _make_channel(channel_key="testch2")
        db_path = tmp_path / "processed" / "testch2.db"
        self._insert_topics(db_path, [("my keyword", "success", 80.0)])

        topics = orchestrator._load_topics(channel, max_topics=5)
        assert len(topics) >= 1
        topic = topics[0]
        for key in ("topic_id", "keyword", "category", "score"):
            assert key in topic

    def test_topic_id_contains_channel_key(self, tmp_path) -> None:
        orchestrator = _make_orchestrator(tmp_path)
        channel = _make_channel(channel_key="mych")
        db_path = tmp_path / "processed" / "mych.db"
        self._insert_topics(db_path, [("test", "success", 80.0)])

        topics = orchestrator._load_topics(channel, max_topics=5)
        assert "mych" in topics[0]["topic_id"]

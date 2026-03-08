"""
Tests for src/optimizer/optimization_loop.py

All Claude API calls are mocked. SQLite uses tmp_path for real DB assertions.
No real API calls or network activity.
"""

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.optimizer.optimization_loop import (
    OptimizationLoop,
    OptimizationResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_loop(tmp_path: Path, **kw) -> OptimizationLoop:
    return OptimizationLoop(
        api_key="fake_key",
        db_path=tmp_path / "test.db",
        **kw,
    )


def _mock_client(response_text: str) -> MagicMock:
    mock = MagicMock()
    mock.messages.create.return_value.content[0].text = response_text
    return mock


def _insert_metric(db_path: Path, video_id: str, tier: str, views: int = 1000) -> None:
    """Insert a row into video_metrics for pull_performers tests."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS video_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL,
                channel_key TEXT NOT NULL DEFAULT 'default',
                views INTEGER NOT NULL DEFAULT 0,
                watch_time_minutes REAL NOT NULL DEFAULT 0,
                likes INTEGER NOT NULL DEFAULT 0,
                comments INTEGER NOT NULL DEFAULT 0,
                shares INTEGER NOT NULL DEFAULT 0,
                impressions INTEGER NOT NULL DEFAULT 0,
                ctr REAL NOT NULL DEFAULT 0,
                subscribers_gained INTEGER NOT NULL DEFAULT 0,
                subscribers_lost INTEGER NOT NULL DEFAULT 0,
                engagement_rate REAL NOT NULL DEFAULT 0,
                virality_score REAL NOT NULL DEFAULT 0,
                tier TEXT NOT NULL DEFAULT 'F',
                fetched_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO video_metrics
                (video_id, tier, views, engagement_rate, virality_score, fetched_at)
            VALUES (?, ?, ?, 0, 0, '2025-01-01T00:00:00+00:00')
            """,
            (video_id, tier, views),
        )
        conn.commit()
    finally:
        conn.close()


_PATTERN_RESPONSE = json.dumps({
    "winning_patterns": ["short hooks", "stoic quotes"],
    "losing_patterns": ["long intros"],
    "content_insights": "Stoic content dominates",
    "recommended_category": "success",
})

_TOPICS_RESPONSE = json.dumps([
    {"keyword": "stoic morning routine", "category": "success", "predicted_tier": "A", "rationale": "High demand"},
    {"keyword": "career stoic secrets", "category": "career",  "predicted_tier": "S", "rationale": "Viral potential"},
    {"keyword": "mediocre topic",        "category": "success", "predicted_tier": "C", "rationale": "Low ceiling"},
])


# ---------------------------------------------------------------------------
# OptimizationResult
# ---------------------------------------------------------------------------

class TestOptimizationResult:
    def test_to_dict_has_all_keys(self) -> None:
        r = OptimizationResult()
        d = r.to_dict()
        for key in (
            "winners_count", "losers_count", "pattern_analysis",
            "topics_generated", "topics_injected", "is_valid", "error", "run_at",
        ):
            assert key in d

    def test_run_at_auto_set(self) -> None:
        r = OptimizationResult()
        assert r.run_at != ""

    def test_default_is_valid_true(self) -> None:
        r = OptimizationResult()
        assert r.is_valid is True

    def test_to_dict_serializable(self) -> None:
        import json as _json
        r = OptimizationResult(winners_count=5, losers_count=2, topics_injected=3)
        assert len(_json.dumps(r.to_dict())) > 10


# ---------------------------------------------------------------------------
# pull_performers
# ---------------------------------------------------------------------------

class TestPullPerformers:
    def test_pulls_s_and_a_winners(self, tmp_path) -> None:
        loop = _make_loop(tmp_path)
        db = tmp_path / "test.db"
        _insert_metric(db, "v1", "S", views=100_000)
        _insert_metric(db, "v2", "A", views=25_000)
        _insert_metric(db, "v3", "F", views=100)

        winners = loop.pull_performers(tiers=["S", "A"])
        assert len(winners) == 2
        tiers = {w["tier"] for w in winners}
        assert tiers == {"S", "A"}

    def test_pulls_f_losers(self, tmp_path) -> None:
        loop = _make_loop(tmp_path)
        db = tmp_path / "test.db"
        _insert_metric(db, "v1", "S")
        _insert_metric(db, "v2", "F", views=50)

        losers = loop.pull_performers(tiers=["F"])
        assert len(losers) == 1
        assert losers[0]["tier"] == "F"

    def test_empty_table_returns_empty_list(self, tmp_path) -> None:
        loop = _make_loop(tmp_path)
        result = loop.pull_performers(tiers=["S", "A"])
        assert result == []

    def test_empty_tiers_arg_returns_empty(self, tmp_path) -> None:
        loop = _make_loop(tmp_path)
        result = loop.pull_performers(tiers=[])
        assert result == []

    def test_result_has_expected_keys(self, tmp_path) -> None:
        loop = _make_loop(tmp_path)
        db = tmp_path / "test.db"
        _insert_metric(db, "v1", "A")

        rows = loop.pull_performers(tiers=["A"])
        assert len(rows) == 1
        row = rows[0]
        for key in ("video_id", "tier", "views", "engagement_rate", "virality_score"):
            assert key in row


# ---------------------------------------------------------------------------
# analyze_patterns
# ---------------------------------------------------------------------------

class TestAnalyzePatterns:
    def test_returns_dict_with_expected_keys(self, tmp_path) -> None:
        loop = _make_loop(tmp_path)
        mock_client = _mock_client(_PATTERN_RESPONSE)

        with patch.object(loop, "_get_client", return_value=mock_client):
            result = loop.analyze_patterns(
                [{"video_id": "v1", "tier": "A"}],
                [{"video_id": "v2", "tier": "F"}],
            )

        for key in ("winning_patterns", "losing_patterns", "content_insights"):
            assert key in result

    def test_winning_patterns_is_list(self, tmp_path) -> None:
        loop = _make_loop(tmp_path)
        mock_client = _mock_client(_PATTERN_RESPONSE)

        with patch.object(loop, "_get_client", return_value=mock_client):
            result = loop.analyze_patterns([], [])

        assert isinstance(result["winning_patterns"], list)

    def test_handles_invalid_json_returns_default(self, tmp_path) -> None:
        loop = _make_loop(tmp_path)
        mock_client = _mock_client("not json at all {{{")

        with patch.object(loop, "_get_client", return_value=mock_client):
            result = loop.analyze_patterns([], [])

        assert isinstance(result, dict)
        assert "winning_patterns" in result


# ---------------------------------------------------------------------------
# generate_topics
# ---------------------------------------------------------------------------

class TestGenerateTopics:
    def test_filters_to_s_and_a_tiers_only(self, tmp_path) -> None:
        loop = _make_loop(tmp_path)
        mock_client = _mock_client(_TOPICS_RESPONSE)

        with patch.object(loop, "_get_client", return_value=mock_client):
            topics = loop.generate_topics({"winning_patterns": []})

        # _TOPICS_RESPONSE has 3 items, but only S and A qualify
        assert len(topics) == 2
        for t in topics:
            assert t["predicted_tier"] in ("S", "A")

    def test_non_list_response_returns_empty(self, tmp_path) -> None:
        loop = _make_loop(tmp_path)
        mock_client = _mock_client(json.dumps({"error": "bad output"}))

        with patch.object(loop, "_get_client", return_value=mock_client):
            topics = loop.generate_topics({})

        assert topics == []

    def test_invalid_json_returns_empty(self, tmp_path) -> None:
        loop = _make_loop(tmp_path)
        mock_client = _mock_client("garbage response +++")

        with patch.object(loop, "_get_client", return_value=mock_client):
            topics = loop.generate_topics({})

        assert topics == []

    def test_strips_markdown_fences(self, tmp_path) -> None:
        loop = _make_loop(tmp_path)
        fenced = "```json\n" + _TOPICS_RESPONSE + "\n```"
        mock_client = _mock_client(fenced)

        with patch.object(loop, "_get_client", return_value=mock_client):
            topics = loop.generate_topics({})

        # After stripping fences and filtering, should get S and A items
        assert len(topics) == 2


# ---------------------------------------------------------------------------
# inject_topics
# ---------------------------------------------------------------------------

class TestInjectTopics:
    def test_injects_topics_into_db(self, tmp_path) -> None:
        loop = _make_loop(tmp_path)
        topics = [
            {"keyword": "stoic morning", "category": "success", "predicted_tier": "A"},
            {"keyword": "career growth",  "category": "career",  "predicted_tier": "S"},
        ]
        count = loop.inject_topics(topics)

        assert count == 2
        conn = sqlite3.connect(tmp_path / "test.db")
        rows = conn.execute(
            "SELECT keyword, score, source FROM scored_topics ORDER BY keyword"
        ).fetchall()
        conn.close()

        assert len(rows) == 2
        for row in rows:
            assert row[1] == loop.inject_score
            assert row[2] == "optimization_loop"

    def test_empty_topics_returns_zero(self, tmp_path) -> None:
        loop = _make_loop(tmp_path)
        count = loop.inject_topics([])
        assert count == 0

    def test_skips_blank_keywords(self, tmp_path) -> None:
        loop = _make_loop(tmp_path)
        topics = [
            {"keyword": "", "category": "success", "predicted_tier": "A"},
            {"keyword": "  ", "category": "success", "predicted_tier": "A"},
            {"keyword": "valid topic", "category": "success", "predicted_tier": "S"},
        ]
        count = loop.inject_topics(topics)
        assert count == 1

    def test_inject_score_default_is_85(self, tmp_path) -> None:
        loop = _make_loop(tmp_path)
        loop.inject_topics([{"keyword": "test topic", "category": "success"}])

        conn = sqlite3.connect(tmp_path / "test.db")
        score = conn.execute(
            "SELECT score FROM scored_topics WHERE keyword='test topic'"
        ).fetchone()[0]
        conn.close()
        assert score == 85.0

    def test_custom_inject_score_respected(self, tmp_path) -> None:
        loop = _make_loop(tmp_path, inject_score=90.0)
        loop.inject_topics([{"keyword": "high score topic", "category": "success"}])

        conn = sqlite3.connect(tmp_path / "test.db")
        score = conn.execute(
            "SELECT score FROM scored_topics WHERE keyword='high score topic'"
        ).fetchone()[0]
        conn.close()
        assert score == 90.0


# ---------------------------------------------------------------------------
# run() — full orchestration
# ---------------------------------------------------------------------------

class TestRun:
    def _mock_two_clients(self, tmp_path) -> tuple[OptimizationLoop, MagicMock]:
        loop = _make_loop(tmp_path)
        mock_client = MagicMock()
        # First call: analyze_patterns → pattern JSON
        # Second call: generate_topics → topics JSON
        mock_client.messages.create.side_effect = [
            MagicMock(**{"content.__getitem__.return_value.text": _PATTERN_RESPONSE}),
            MagicMock(**{"content.__getitem__.return_value.text": _TOPICS_RESPONSE}),
        ]
        return loop, mock_client

    def test_full_run_with_no_metrics_succeeds(self, tmp_path) -> None:
        loop = _make_loop(tmp_path)
        mock_client = MagicMock()
        mock_client.messages.create.return_value.content[0].text = _PATTERN_RESPONSE

        # patch both analyze_patterns and generate_topics to avoid double client call
        with patch.object(loop, "analyze_patterns", return_value={"winning_patterns": []}):
            with patch.object(loop, "generate_topics", return_value=[]):
                result = loop.run()

        assert result.is_valid is True
        assert result.topics_injected == 0

    def test_run_with_metrics_injects_topics(self, tmp_path) -> None:
        loop = _make_loop(tmp_path)
        db = tmp_path / "test.db"
        _insert_metric(db, "v1", "A")
        _insert_metric(db, "v2", "F")

        pattern_data = {"winning_patterns": ["short"]}
        topics = [{"keyword": "winner topic", "category": "success", "predicted_tier": "A"}]

        with patch.object(loop, "analyze_patterns", return_value=pattern_data):
            with patch.object(loop, "generate_topics", return_value=topics):
                result = loop.run()

        assert result.winners_count == 1
        assert result.losers_count == 1
        assert result.topics_injected == 1

    def test_run_saves_to_optimization_log(self, tmp_path) -> None:
        loop = _make_loop(tmp_path)

        with patch.object(loop, "analyze_patterns", return_value={}):
            with patch.object(loop, "generate_topics", return_value=[]):
                loop.run()

        conn = sqlite3.connect(tmp_path / "test.db")
        count = conn.execute("SELECT COUNT(*) FROM optimization_log").fetchone()[0]
        conn.close()
        assert count == 1

    def test_run_api_failure_returns_invalid_result(self, tmp_path) -> None:
        loop = _make_loop(tmp_path)

        with patch.object(
            loop, "analyze_patterns", side_effect=Exception("Claude down")
        ):
            result = loop.run()

        assert result.is_valid is False
        assert "Claude down" in result.error

    def test_run_api_failure_still_saves_log(self, tmp_path) -> None:
        loop = _make_loop(tmp_path)

        with patch.object(
            loop, "analyze_patterns", side_effect=Exception("fail")
        ):
            loop.run()

        conn = sqlite3.connect(tmp_path / "test.db")
        count = conn.execute("SELECT COUNT(*) FROM optimization_log").fetchone()[0]
        conn.close()
        assert count == 1


# ---------------------------------------------------------------------------
# _parse_json
# ---------------------------------------------------------------------------

class TestParseJson:
    def test_parses_valid_json_object(self) -> None:
        result = OptimizationLoop._parse_json('{"key": "value"}', default={})
        assert result == {"key": "value"}

    def test_parses_valid_json_array(self) -> None:
        result = OptimizationLoop._parse_json('[1, 2, 3]', default=[])
        assert result == [1, 2, 3]

    def test_invalid_json_returns_default(self) -> None:
        result = OptimizationLoop._parse_json("bad json", default={"fallback": True})
        assert result == {"fallback": True}

    def test_strips_markdown_fences(self) -> None:
        raw = "```json\n{\"a\": 1}\n```"
        result = OptimizationLoop._parse_json(raw, default={})
        assert result == {"a": 1}

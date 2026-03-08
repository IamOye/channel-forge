"""
Tests for src/analytics/sentiment_analyzer.py

All YouTube API and Claude API calls are mocked — no real network activity.
Uses tmp_path for real SQLite DB assertions.
"""

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.analytics.sentiment_analyzer import (
    SentimentAnalyzer,
    SentimentResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_analyzer(tmp_path: Path, **kw) -> SentimentAnalyzer:
    return SentimentAnalyzer(
        api_key="fake_key",
        db_path=tmp_path / "test.db",
        **kw,
    )


def _mock_claude_response(data: dict) -> MagicMock:
    """Build a mock anthropic client whose messages.create returns data as JSON."""
    mock_client = MagicMock()
    mock_client.messages.create.return_value.content[0].text = json.dumps(data)
    return mock_client


def _valid_sentiment_data(**overrides) -> dict:
    base = {
        "sentiment_score": 0.75,
        "dominant_reaction": "inspired",
        "debate_intensity": "low",
        "summary": "Viewers loved the stoic wisdom.",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# SentimentResult
# ---------------------------------------------------------------------------

class TestSentimentResult:
    def test_to_dict_has_all_keys(self) -> None:
        r = SentimentResult(
            video_id="v1",
            video_title="Test",
            comment_count=10,
            sentiment_score=0.5,
            dominant_reaction="inspired",
            debate_intensity="low",
        )
        d = r.to_dict()
        for key in (
            "video_id", "video_title", "comment_count",
            "sentiment_score", "dominant_reaction", "debate_intensity",
            "summary", "followup_injected", "analyzed_at", "is_valid", "error",
        ):
            assert key in d

    def test_analyzed_at_auto_set(self) -> None:
        r = SentimentResult(
            video_id="v1", video_title="T", comment_count=0,
            sentiment_score=0.0, dominant_reaction="unknown", debate_intensity="low",
        )
        assert r.analyzed_at != ""

    def test_default_is_valid_true(self) -> None:
        r = SentimentResult(
            video_id="v1", video_title="T", comment_count=0,
            sentiment_score=0.0, dominant_reaction="unknown", debate_intensity="low",
        )
        assert r.is_valid is True

    def test_to_dict_serializable(self) -> None:
        import json as _json
        r = SentimentResult(
            video_id="v1", video_title="T", comment_count=5,
            sentiment_score=0.8, dominant_reaction="excited", debate_intensity="high",
        )
        assert len(_json.dumps(r.to_dict())) > 10


# ---------------------------------------------------------------------------
# _parse_result
# ---------------------------------------------------------------------------

class TestParseResult:
    def test_parses_valid_json(self, tmp_path) -> None:
        analyzer = _make_analyzer(tmp_path)
        raw = json.dumps(_valid_sentiment_data())
        result = analyzer._parse_result("v1", "Title", 10, raw)

        assert result.is_valid is True
        assert abs(result.sentiment_score - 0.75) < 0.001
        assert result.dominant_reaction == "inspired"
        assert result.debate_intensity == "low"

    def test_sentiment_clamped_to_minus_one(self, tmp_path) -> None:
        analyzer = _make_analyzer(tmp_path)
        raw = json.dumps(_valid_sentiment_data(sentiment_score=-5.0))
        result = analyzer._parse_result("v1", "T", 5, raw)
        assert result.sentiment_score == -1.0

    def test_sentiment_clamped_to_plus_one(self, tmp_path) -> None:
        analyzer = _make_analyzer(tmp_path)
        raw = json.dumps(_valid_sentiment_data(sentiment_score=99.0))
        result = analyzer._parse_result("v1", "T", 5, raw)
        assert result.sentiment_score == 1.0

    def test_invalid_debate_intensity_defaults_to_low(self, tmp_path) -> None:
        analyzer = _make_analyzer(tmp_path)
        raw = json.dumps(_valid_sentiment_data(debate_intensity="extreme"))
        result = analyzer._parse_result("v1", "T", 5, raw)
        assert result.debate_intensity == "low"

    def test_invalid_json_returns_error_result(self, tmp_path) -> None:
        analyzer = _make_analyzer(tmp_path)
        result = analyzer._parse_result("v1", "T", 5, "not json {{{")
        assert result.is_valid is False
        assert "Parse error" in result.error

    def test_strips_markdown_fences(self, tmp_path) -> None:
        analyzer = _make_analyzer(tmp_path)
        raw = "```json\n" + json.dumps(_valid_sentiment_data()) + "\n```"
        result = analyzer._parse_result("v1", "T", 5, raw)
        assert result.is_valid is True


# ---------------------------------------------------------------------------
# _analyze_sentiment (mocks Claude client)
# ---------------------------------------------------------------------------

class TestAnalyzeSentiment:
    def test_returns_valid_result_with_comments(self, tmp_path) -> None:
        analyzer = _make_analyzer(tmp_path)
        mock_client = _mock_claude_response(_valid_sentiment_data())

        with patch.object(analyzer, "_get_client", return_value=mock_client):
            result = analyzer._analyze_sentiment(
                "v1", "Stoic Wisdom", ["Great!", "Amazing"]
            )

        assert result.is_valid is True
        assert result.comment_count == 2

    def test_no_comments_returns_low_debate_without_api_call(self, tmp_path) -> None:
        analyzer = _make_analyzer(tmp_path)

        with patch.object(analyzer, "_get_client") as mock_get:
            result = analyzer._analyze_sentiment("v1", "T", [])

        mock_get.assert_not_called()
        assert result.debate_intensity == "low"
        assert result.comment_count == 0

    def test_viral_debate_intensity_detected(self, tmp_path) -> None:
        analyzer = _make_analyzer(tmp_path)
        mock_client = _mock_claude_response(
            _valid_sentiment_data(debate_intensity="viral")
        )

        with patch.object(analyzer, "_get_client", return_value=mock_client):
            result = analyzer._analyze_sentiment("v1", "Controversial", ["🔥", "🔥"])

        assert result.debate_intensity == "viral"


# ---------------------------------------------------------------------------
# _fetch_comments (mocks YouTube API)
# ---------------------------------------------------------------------------

class TestFetchComments:
    def _make_comment_response(self, texts: list[str]) -> dict:
        items = []
        for text in texts:
            items.append({
                "snippet": {
                    "topLevelComment": {
                        "snippet": {"textDisplay": text}
                    }
                }
            })
        return {"items": items}

    def test_fetch_comments_extracts_text(self, tmp_path) -> None:
        analyzer = _make_analyzer(tmp_path)
        mock_service = MagicMock()
        mock_service.commentThreads.return_value.list.return_value.execute.return_value = (
            self._make_comment_response(["Great!", "Love it"])
        )

        with patch.object(analyzer, "_load_credentials", return_value=MagicMock()):
            with patch.object(analyzer, "_build_youtube_service", return_value=mock_service):
                comments = analyzer._fetch_comments("vid001", "default")

        assert comments == ["Great!", "Love it"]

    def test_fetch_comments_empty_response(self, tmp_path) -> None:
        analyzer = _make_analyzer(tmp_path)
        mock_service = MagicMock()
        mock_service.commentThreads.return_value.list.return_value.execute.return_value = (
            {"items": []}
        )

        with patch.object(analyzer, "_load_credentials", return_value=MagicMock()):
            with patch.object(analyzer, "_build_youtube_service", return_value=mock_service):
                comments = analyzer._fetch_comments("vid001", "default")

        assert comments == []

    def test_fetch_comments_skips_empty_text(self, tmp_path) -> None:
        analyzer = _make_analyzer(tmp_path)
        mock_service = MagicMock()
        mock_service.commentThreads.return_value.list.return_value.execute.return_value = (
            self._make_comment_response(["Good", "", "  "])
        )

        with patch.object(analyzer, "_load_credentials", return_value=MagicMock()):
            with patch.object(analyzer, "_build_youtube_service", return_value=mock_service):
                comments = analyzer._fetch_comments("vid001", "default")

        assert "Good" in comments
        # empty strings and whitespace-only should not be included
        assert "" not in comments


# ---------------------------------------------------------------------------
# Viral followup injection
# ---------------------------------------------------------------------------

class TestViralInjection:
    def test_viral_injects_followup_topic_to_db(self, tmp_path) -> None:
        analyzer = _make_analyzer(tmp_path)
        analyzer._inject_followup_topic("Stoic Wisdom Exposed")

        conn = sqlite3.connect(tmp_path / "test.db")
        rows = conn.execute(
            "SELECT keyword, score, source FROM scored_topics"
        ).fetchall()
        conn.close()

        assert len(rows) == 1
        assert "Stoic Wisdom Exposed" in rows[0][0]
        assert rows[0][1] == 92.0
        assert rows[0][2] == "sentiment_viral"

    def test_non_viral_does_not_inject(self, tmp_path) -> None:
        analyzer = _make_analyzer(tmp_path)
        mock_client = _mock_claude_response(_valid_sentiment_data(debate_intensity="medium"))

        with patch.object(analyzer, "_fetch_comments", return_value=["Nice vid"]):
            with patch.object(analyzer, "_get_client", return_value=mock_client):
                result = analyzer.analyze("vid001", "Test", "default")

        assert result.followup_injected is False

        conn = sqlite3.connect(tmp_path / "test.db")
        count = conn.execute("SELECT COUNT(*) FROM scored_topics").fetchone()[0]
        conn.close()
        assert count == 0

    def test_viral_sets_followup_injected_flag(self, tmp_path) -> None:
        analyzer = _make_analyzer(tmp_path)
        mock_client = _mock_claude_response(_valid_sentiment_data(debate_intensity="viral"))

        with patch.object(analyzer, "_fetch_comments", return_value=["🔥🔥🔥"]):
            with patch.object(analyzer, "_get_client", return_value=mock_client):
                result = analyzer.analyze("vid001", "Hot Topic", "default")

        assert result.followup_injected is True

    def test_viral_topic_score_is_92(self, tmp_path) -> None:
        analyzer = _make_analyzer(tmp_path)
        mock_client = _mock_claude_response(_valid_sentiment_data(debate_intensity="viral"))

        with patch.object(analyzer, "_fetch_comments", return_value=["fight me"]):
            with patch.object(analyzer, "_get_client", return_value=mock_client):
                analyzer.analyze("vid001", "Controversial Title", "default")

        conn = sqlite3.connect(tmp_path / "test.db")
        score = conn.execute(
            "SELECT score FROM scored_topics WHERE source='sentiment_viral'"
        ).fetchone()[0]
        conn.close()
        assert score == 92.0


# ---------------------------------------------------------------------------
# analyze() — top-level method
# ---------------------------------------------------------------------------

class TestAnalyzeMethod:
    def test_analyze_success(self, tmp_path) -> None:
        analyzer = _make_analyzer(tmp_path)
        mock_client = _mock_claude_response(_valid_sentiment_data())

        with patch.object(analyzer, "_fetch_comments", return_value=["Love it"]):
            with patch.object(analyzer, "_get_client", return_value=mock_client):
                result = analyzer.analyze("vid001", "Title", "default")

        assert isinstance(result, SentimentResult)
        assert result.is_valid is True
        assert result.video_id == "vid001"

    def test_analyze_exception_returns_invalid(self, tmp_path) -> None:
        analyzer = _make_analyzer(tmp_path)

        with patch.object(
            analyzer, "_fetch_comments", side_effect=Exception("network error")
        ):
            result = analyzer.analyze("vid001", "Title", "default")

        assert result.is_valid is False
        assert "network error" in result.error

    def test_analyze_passes_title_to_result(self, tmp_path) -> None:
        analyzer = _make_analyzer(tmp_path)
        mock_client = _mock_claude_response(_valid_sentiment_data())

        with patch.object(analyzer, "_fetch_comments", return_value=["ok"]):
            with patch.object(analyzer, "_get_client", return_value=mock_client):
                result = analyzer.analyze("vid001", "My Video Title", "default")

        assert result.video_title == "My Video Title"

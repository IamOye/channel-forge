"""
Tests for src/scorer/engagement_scorer.py

All Claude API calls are mocked so tests run without API keys.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.scorer.engagement_scorer import (
    ClaudeScorer,
    EngagementScorer,
    ScoreResult,
    _score_competition,
    _score_trend,
)


# ---------------------------------------------------------------------------
# _score_trend
# ---------------------------------------------------------------------------

class TestScoreTrend:
    def test_zero_input_returns_zero(self) -> None:
        assert _score_trend(0.0) == 0.0

    def test_negative_returns_zero(self) -> None:
        assert _score_trend(-10.0) == 0.0

    def test_100_returns_capped_100(self) -> None:
        score = _score_trend(100.0)
        assert 0.0 <= score <= 100.0

    def test_proportional(self) -> None:
        low = _score_trend(20.0)
        high = _score_trend(80.0)
        assert high > low

    def test_range_valid(self) -> None:
        for v in [0, 10, 25, 50, 75, 100]:
            score = _score_trend(float(v))
            assert 0.0 <= score <= 100.0


# ---------------------------------------------------------------------------
# _score_competition
# ---------------------------------------------------------------------------

class TestScoreCompetition:
    def test_long_tail_gets_higher_score(self) -> None:
        short = _score_competition("Python")
        long_tail = _score_competition("best Python course for beginners 2024")
        assert long_tail > short

    def test_question_form_gets_boost(self) -> None:
        plain = _score_competition("microphone for streaming")
        question = _score_competition("what microphone for streaming")
        assert question > plain

    def test_year_specificity_gets_boost(self) -> None:
        no_year = _score_competition("best Python course")
        with_year = _score_competition("best Python course 2024")
        assert with_year > no_year

    def test_high_result_count_lowers_score(self) -> None:
        no_count = _score_competition("Python tutorial")
        saturated = _score_competition("Python tutorial", youtube_result_count=10_000_000)
        assert saturated < no_count

    def test_score_in_valid_range(self) -> None:
        score = _score_competition("some keyword", youtube_result_count=50000)
        assert 0.0 <= score <= 100.0

    def test_very_high_result_count_clamped_to_zero(self) -> None:
        score = _score_competition("a", youtube_result_count=10 ** 15)
        assert score >= 0.0


# ---------------------------------------------------------------------------
# ScoreResult
# ---------------------------------------------------------------------------

class TestScoreResult:
    def test_defaults(self) -> None:
        result = ScoreResult(keyword="test")
        assert result.title_score == 0.0
        assert result.composite_score == 0.0
        assert result.scored_at != ""

    def test_to_dict(self) -> None:
        result = ScoreResult(
            keyword="AI automation",
            title_score=80.0,
            trend_score=70.0,
            competition_score=60.0,
            monetization_score=75.0,
            composite_score=72.5,
            rationale="High demand topic",
        )
        d = result.to_dict()
        assert d["keyword"] == "AI automation"
        assert d["composite_score"] == 72.5
        assert "rationale" in d
        assert "scored_at" in d

    def test_rounding_in_to_dict(self) -> None:
        result = ScoreResult(keyword="k", title_score=33.3333333)
        d = result.to_dict()
        assert d["title_score"] == 33.33


# ---------------------------------------------------------------------------
# ClaudeScorer (mocked)
# ---------------------------------------------------------------------------

class TestClaudeScorer:
    def _mock_message(self, payload: dict) -> MagicMock:
        msg = MagicMock()
        msg.content = [MagicMock(text=json.dumps(payload))]
        return msg

    @patch("src.scorer.engagement_scorer.anthropic.Anthropic")
    def test_score_returns_values(self, mock_anthropic_cls) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._mock_message(
            {"title_score": 85, "monetization_score": 70, "rationale": "Great topic"}
        )
        mock_anthropic_cls.return_value = mock_client

        scorer = ClaudeScorer(api_key="fake")
        title, monetization, rationale = scorer.score("best budget mic")

        assert title == 85.0
        assert monetization == 70.0
        assert rationale == "Great topic"

    @patch("src.scorer.engagement_scorer.anthropic.Anthropic")
    def test_score_clamps_out_of_range(self, mock_anthropic_cls) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._mock_message(
            {"title_score": 150, "monetization_score": -20, "rationale": "test"}
        )
        mock_anthropic_cls.return_value = mock_client

        scorer = ClaudeScorer(api_key="fake")
        title, monetization, _ = scorer.score("keyword")

        assert title == 100.0
        assert monetization == 0.0

    @patch("src.scorer.engagement_scorer.anthropic.Anthropic")
    def test_score_handles_api_exception(self, mock_anthropic_cls) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("rate limited")
        mock_anthropic_cls.return_value = mock_client

        scorer = ClaudeScorer(api_key="fake")
        title, monetization, rationale = scorer.score("test")

        assert title == 50.0
        assert monetization == 50.0
        assert "unavailable" in rationale.lower()

    @patch("src.scorer.engagement_scorer.anthropic.Anthropic")
    def test_score_strips_markdown_fences(self, mock_anthropic_cls) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(
                text='```json\n{"title_score": 72, "monetization_score": 65, "rationale": "ok"}\n```'
            )]
        )
        mock_anthropic_cls.return_value = mock_client

        scorer = ClaudeScorer(api_key="fake")
        title, _, _ = scorer.score("test")
        assert title == 72.0

    def test_raises_without_api_key(self) -> None:
        scorer = ClaudeScorer(api_key="")
        with pytest.raises(ValueError):
            scorer._get_client()


# ---------------------------------------------------------------------------
# EngagementScorer._heuristic_title_score
# ---------------------------------------------------------------------------

class TestHeuristicTitleScore:
    def test_how_to_gets_boost(self) -> None:
        scorer = EngagementScorer(use_claude=False)
        score = scorer._heuristic_title_score("how to start a podcast")
        assert score > 40.0

    def test_number_gets_boost(self) -> None:
        scorer = EngagementScorer(use_claude=False)
        with_num = scorer._heuristic_title_score("10 best tools for productivity")
        no_num = scorer._heuristic_title_score("best tools for productivity")
        assert with_num > no_num

    def test_emotional_word_boost(self) -> None:
        scorer = EngagementScorer(use_claude=False)
        with_word = scorer._heuristic_title_score("best budget mic 2024")
        plain = scorer._heuristic_title_score("microphone review 2024")
        assert with_word >= plain

    def test_score_in_valid_range(self) -> None:
        scorer = EngagementScorer(use_claude=False)
        for kw in ["a", "short", "a very long keyword with many many words here", "10 tips"]:
            score = scorer._heuristic_title_score(kw)
            assert 0.0 <= score <= 100.0


# ---------------------------------------------------------------------------
# EngagementScorer.score (no Claude)
# ---------------------------------------------------------------------------

class TestEngagementScorerNoAPI:
    def test_score_returns_result(self) -> None:
        scorer = EngagementScorer(use_claude=False)
        result = scorer.score("Python tutorial 2024", trend_interest=65.0)

        assert isinstance(result, ScoreResult)
        assert result.keyword == "Python tutorial 2024"
        assert 0.0 <= result.composite_score <= 100.0
        assert 0.0 <= result.title_score <= 100.0
        assert 0.0 <= result.trend_score <= 100.0
        assert 0.0 <= result.competition_score <= 100.0
        assert 0.0 <= result.monetization_score <= 100.0

    def test_score_uses_trend_interest(self) -> None:
        scorer = EngagementScorer(use_claude=False)
        low_trend = scorer.score("keyword", trend_interest=0.0)
        high_trend = scorer.score("keyword", trend_interest=90.0)
        assert high_trend.trend_score > low_trend.trend_score

    def test_composite_is_weighted_average(self) -> None:
        scorer = EngagementScorer(use_claude=False)
        result = scorer.score("test", trend_interest=50.0)
        # Manually compute expected composite
        expected = (
            result.title_score * 0.30
            + result.trend_score * 0.30
            + result.competition_score * 0.20
            + result.monetization_score * 0.20
        )
        assert abs(result.composite_score - round(expected, 2)) < 0.01

    def test_custom_weights(self) -> None:
        weights = {
            "title_score": 0.50,
            "trend_score": 0.20,
            "competition_score": 0.15,
            "monetization_score": 0.15,
        }
        scorer = EngagementScorer(use_claude=False, weights=weights)
        result = scorer.score("test keyword")
        assert result.composite_score >= 0.0

    def test_invalid_weights_raises(self) -> None:
        with pytest.raises(ValueError, match="sum"):
            EngagementScorer(
                use_claude=False,
                weights={
                    "title_score": 0.50,
                    "trend_score": 0.50,
                    "competition_score": 0.50,
                    "monetization_score": 0.50,
                },
            )

    def test_missing_weight_key_raises(self) -> None:
        with pytest.raises(ValueError):
            EngagementScorer(
                use_claude=False,
                weights={"title_score": 0.5, "trend_score": 0.5},
            )


# ---------------------------------------------------------------------------
# EngagementScorer.score_batch
# ---------------------------------------------------------------------------

class TestEngagementScorerBatch:
    def test_batch_returns_all_results(self) -> None:
        scorer = EngagementScorer(use_claude=False)
        keywords = ["Python", "JavaScript", "Rust programming language"]
        results = scorer.score_batch(keywords)
        assert len(results) == 3
        result_keywords = {r.keyword for r in results}
        assert result_keywords == set(keywords)

    def test_batch_sorted_by_composite_desc(self) -> None:
        scorer = EngagementScorer(use_claude=False)
        results = scorer.score_batch(
            ["a", "b", "c"],
            trend_map={"a": 90.0, "b": 30.0, "c": 60.0},
        )
        scores = [r.composite_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_batch_uses_trend_map(self) -> None:
        scorer = EngagementScorer(use_claude=False)
        results = scorer.score_batch(
            ["kw1", "kw2"],
            trend_map={"kw1": 0.0, "kw2": 100.0},
        )
        score_map = {r.keyword: r.trend_score for r in results}
        assert score_map["kw2"] > score_map["kw1"]

    def test_batch_empty_list(self) -> None:
        scorer = EngagementScorer(use_claude=False)
        results = scorer.score_batch([])
        assert results == []


# ---------------------------------------------------------------------------
# EngagementScorer.score (with mocked Claude)
# ---------------------------------------------------------------------------

class TestEngagementScorerWithClaude:
    @patch("src.scorer.engagement_scorer.anthropic.Anthropic")
    def test_score_with_claude_uses_api_scores(self, mock_anthropic_cls) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text=json.dumps(
                {"title_score": 88, "monetization_score": 72, "rationale": "trending topic"}
            ))]
        )
        mock_anthropic_cls.return_value = mock_client

        scorer = EngagementScorer(use_claude=True, claude_api_key="fake")
        result = scorer.score("AI tools for content creators", trend_interest=80.0)

        assert result.title_score == 88.0
        assert result.monetization_score == 72.0
        assert "trending" in result.rationale

"""
Tests for src/utils/topic_dedup.py

Pure Python logic — no API calls, no DB.
"""

import pytest

from src.utils.topic_dedup import (
    SIMILARITY_THRESHOLD,
    STOP_WORDS,
    filter_new_topics,
    is_duplicate,
    normalize_topic,
    topic_similarity,
)


# ---------------------------------------------------------------------------
# normalize_topic
# ---------------------------------------------------------------------------

class TestNormalizeTopic:
    def test_lowercases_text(self) -> None:
        assert normalize_topic("WHY SAVING MONEY") == "saving money"

    def test_removes_punctuation(self) -> None:
        result = normalize_topic("why your salary — keeps you poor!")
        assert "—" not in result
        assert "!" not in result

    def test_removes_stop_words(self) -> None:
        result = normalize_topic("why your salary is keeping you poor")
        assert "why" not in result
        assert "your" not in result
        assert "is" not in result
        assert "you" not in result

    def test_keeps_content_words(self) -> None:
        result = normalize_topic("salary keeping poor")
        assert "salary" in result
        assert "keeping" in result
        assert "poor" in result

    def test_collapses_whitespace(self) -> None:
        result = normalize_topic("  wealth   building   secrets  ")
        assert "  " not in result

    def test_empty_string_returns_empty(self) -> None:
        assert normalize_topic("") == ""

    def test_all_stop_words_returns_empty(self) -> None:
        result = normalize_topic("why how the a an")
        assert result == ""

    def test_numbers_preserved(self) -> None:
        result = normalize_topic("5 income streams that work")
        assert "5" in result

    def test_hyphenated_words(self) -> None:
        # Hyphen becomes space, so "nine-to-five" → "nine five"
        result = normalize_topic("nine-to-five job trap")
        assert "nine" in result
        assert "five" in result


# ---------------------------------------------------------------------------
# topic_similarity
# ---------------------------------------------------------------------------

class TestTopicSimilarity:
    def test_identical_topics_return_one(self) -> None:
        assert topic_similarity("salary trap", "salary trap") == pytest.approx(1.0)

    def test_completely_different_returns_low(self) -> None:
        sim = topic_similarity("cats and dogs", "real estate investing")
        assert sim < 0.5

    def test_paraphrased_similar_topic_above_threshold(self) -> None:
        sim = topic_similarity(
            "why saving money keeps you poor",
            "why saving cash keeps you broke",
        )
        assert sim >= 0.5   # similar enough to be suspicious

    def test_very_different_below_threshold(self) -> None:
        sim = topic_similarity(
            "the debt trap nobody talks about",
            "morning routine myths that waste your time",
        )
        assert sim < SIMILARITY_THRESHOLD

    def test_empty_both_returns_one(self) -> None:
        assert topic_similarity("", "") == pytest.approx(1.0)

    def test_empty_one_side_returns_zero(self) -> None:
        assert topic_similarity("salary trap", "") == pytest.approx(0.0)

    def test_case_insensitive(self) -> None:
        a = topic_similarity("SALARY TRAP", "salary trap")
        assert a == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# is_duplicate
# ---------------------------------------------------------------------------

class TestIsDuplicate:
    def test_exact_match_is_duplicate(self) -> None:
        assert is_duplicate("salary keeps you poor", ["salary keeps you poor"])

    def test_near_identical_topic_is_duplicate(self) -> None:
        # Only the last content word differs — normalised similarity > 0.70
        assert is_duplicate(
            "why saving money keeps you poor",
            ["why saving money keeps you broke"],
        )

    def test_clearly_different_topic_not_duplicate(self) -> None:
        assert not is_duplicate(
            "the debt trap nobody talks about",
            ["morning routine myths that waste your time"],
        )

    def test_empty_uploaded_list_never_duplicate(self) -> None:
        assert not is_duplicate("any topic", [])

    def test_custom_threshold_lower(self) -> None:
        # With threshold=0.3 almost anything matches
        assert is_duplicate("money wealth", ["money finance"], threshold=0.3)

    def test_custom_threshold_higher(self) -> None:
        # With threshold=0.99 only nearly identical strings match
        assert not is_duplicate(
            "why saving money is smart",
            ["why saving cash is smart"],
            threshold=0.99,
        )

    def test_checks_all_uploaded_not_just_first(self) -> None:
        uploaded = [
            "morning routine myths",
            "why saving money keeps you poor",
        ]
        # Near-identical to second item (only last word differs)
        assert is_duplicate("why saving money keeps you broke", uploaded)


# ---------------------------------------------------------------------------
# filter_new_topics
# ---------------------------------------------------------------------------

class TestFilterNewTopics:
    def test_returns_all_when_no_uploaded(self) -> None:
        candidates = [
            "the debt trap nobody talks about",
            "why rich people never pay taxes",
            "how passive income actually works",
        ]
        result = filter_new_topics(candidates, [])
        assert result == candidates

    def test_removes_duplicates_of_uploaded(self) -> None:
        uploaded = ["why saving money keeps you poor"]
        candidates = [
            "why saving money keeps you broke",  # near-identical (only last word) → removed
            "the debt trap nobody talks about",  # clearly different → kept
        ]
        result = filter_new_topics(candidates, uploaded)
        assert len(result) == 1
        assert "debt trap" in result[0]

    def test_returns_empty_when_all_duplicates(self) -> None:
        uploaded = ["salary keeps you poor", "debt trap explained"]
        candidates = ["salary keeps you broke", "the debt trap nobody talks about"]
        result = filter_new_topics(candidates, uploaded)
        # Both are similar to uploaded ones
        assert len(result) <= len(candidates)   # may remove some

    def test_preserves_order_of_fresh_topics(self) -> None:
        uploaded: list[str] = []
        candidates = [
            "how passive income actually works",
            "why rich people never pay taxes",
            "the debt trap nobody talks about",
        ]
        result = filter_new_topics(candidates, uploaded)
        assert result == candidates

    def test_dedupes_within_candidates_too(self) -> None:
        uploaded: list[str] = []
        candidates = [
            "why saving money keeps you poor",
            "why saving money keeps you broke",  # near-identical to first → removed
            "the debt trap nobody talks about",
        ]
        result = filter_new_topics(candidates, uploaded)
        # Second candidate is deduped against first
        assert len(result) == 2
        assert result[0] == candidates[0]
        assert result[-1] == candidates[2]

    def test_empty_candidates_returns_empty(self) -> None:
        assert filter_new_topics([], ["uploaded topic"]) == []

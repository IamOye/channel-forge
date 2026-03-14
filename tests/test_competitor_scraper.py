"""
Tests for src/crawler/competitor_scraper.py

All HTTP and Claude API calls are mocked — no network traffic.
"""

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.crawler.competitor_scraper import CompetitorScraper, CompetitorTopic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scraper(tmp_path: Path, api_key: str = "yt-fake", anthropic_key: str = "ant-fake") -> CompetitorScraper:
    return CompetitorScraper(
        api_key=api_key,
        anthropic_api_key=anthropic_key,
        db_path=tmp_path / "test.db",
    )


def _yt_search_response(titles: list[str]) -> dict:
    """Fake YouTube search API response."""
    items = []
    for i, title in enumerate(titles):
        items.append({
            "id": {"videoId": f"vid_{i:03d}"},
            "snippet": {"title": title, "channelTitle": "TestChannel"},
        })
    return {"items": items}


def _yt_stats_response(video_ids: list[str], view_count: int = 150_000) -> dict:
    """Fake YouTube video statistics API response."""
    return {
        "items": [
            {"id": vid_id, "statistics": {"viewCount": str(view_count)}}
            for vid_id in video_ids
        ]
    }


def _mock_httpx_client(search_titles: list[str], view_count: int = 150_000) -> MagicMock:
    """Return a mock httpx.Client that returns fake YouTube responses."""
    client = MagicMock()

    search_resp = MagicMock()
    search_resp.json.return_value = _yt_search_response(search_titles)
    search_resp.raise_for_status = MagicMock()

    stats_ids = [f"vid_{i:03d}" for i in range(len(search_titles))]
    stats_resp = MagicMock()
    stats_resp.json.return_value = _yt_stats_response(stats_ids, view_count)
    stats_resp.raise_for_status = MagicMock()

    client.get.side_effect = [search_resp, stats_resp]
    return client


# ---------------------------------------------------------------------------
# CompetitorTopic
# ---------------------------------------------------------------------------

class TestCompetitorTopic:
    def test_to_dict_has_required_keys(self) -> None:
        ct = CompetitorTopic(
            channel_name="GrahamStephan",
            original_title="Why The Middle Class Is Disappearing",
            extracted_topic="why the middle class is disappearing",
            view_count=250_000,
            category="money",
            source="COMPETITOR_HIGH_SIGNAL",
        )
        d = ct.to_dict()
        for key in ("channel_name", "original_title", "extracted_topic",
                    "view_count", "category", "source", "scraped_at"):
            assert key in d

    def test_scraped_at_auto_set(self) -> None:
        ct = CompetitorTopic("ch", "title", "topic")
        assert ct.scraped_at != ""

    def test_defaults(self) -> None:
        ct = CompetitorTopic("ch", "title", "topic")
        assert ct.view_count == 0
        assert ct.category == "money"
        assert ct.source == "competitor"


# ---------------------------------------------------------------------------
# _heuristic_extract
# ---------------------------------------------------------------------------

class TestHeuristicExtract:
    def test_extracts_up_to_10_words(self) -> None:
        title = "word " * 15
        result = CompetitorScraper._heuristic_extract(title)
        assert len(result.split()) <= 10

    def test_removes_hashtags(self) -> None:
        result = CompetitorScraper._heuristic_extract("Why Saving Money Fails #shorts #finance")
        assert "#" not in result

    def test_lowercases(self) -> None:
        result = CompetitorScraper._heuristic_extract("WHY SAVING MONEY FAILS")
        assert result == result.lower()

    def test_empty_title_returns_empty(self) -> None:
        assert CompetitorScraper._heuristic_extract("") == ""


# ---------------------------------------------------------------------------
# _extract_topic (Claude path)
# ---------------------------------------------------------------------------

class TestExtractTopic:
    def test_returns_heuristic_when_no_api_key(self, tmp_path) -> None:
        scraper = CompetitorScraper(api_key="yt-fake", anthropic_api_key="", db_path=tmp_path / "t.db")
        result = scraper._extract_topic("Why The Middle Class Is Shrinking", "money")
        assert result != ""   # heuristic always returns something for non-empty title

    @patch("src.crawler.competitor_scraper.CompetitorScraper._get_anthropic_client")
    def test_calls_claude_when_key_set(self, mock_get_client, tmp_path) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="why the middle class is disappearing")]
        )
        mock_get_client.return_value = mock_client

        scraper = _make_scraper(tmp_path)
        result = scraper._extract_topic("Why The Middle Class Is Disappearing", "money")
        assert result == "why the middle class is disappearing"

    @patch("src.crawler.competitor_scraper.CompetitorScraper._get_anthropic_client")
    def test_falls_back_to_heuristic_on_claude_error(self, mock_get_client, tmp_path) -> None:
        mock_get_client.side_effect = Exception("API error")
        scraper = _make_scraper(tmp_path)
        result = scraper._extract_topic("Why The Middle Class Is Disappearing", "money")
        # heuristic result: non-empty
        assert result != ""

    @patch("src.crawler.competitor_scraper.CompetitorScraper._get_anthropic_client")
    def test_rejects_topic_with_too_few_words(self, mock_get_client, tmp_path) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="ok")]   # 1 word — rejected
        )
        mock_get_client.return_value = mock_client

        scraper = _make_scraper(tmp_path)
        result = scraper._extract_topic("Some Long Title About Money", "money")
        assert result == ""   # heuristic returns empty or something valid


# ---------------------------------------------------------------------------
# _save_to_db
# ---------------------------------------------------------------------------

class TestSaveToDb:
    def test_creates_table_and_inserts(self, tmp_path) -> None:
        scraper = _make_scraper(tmp_path)
        topics = [
            CompetitorTopic("Chan", "Title A", "topic a", 200_000, "money", "COMPETITOR_HIGH_SIGNAL"),
            CompetitorTopic("Chan", "Title B", "topic b", 150_000, "money", "COMPETITOR_HIGH_SIGNAL"),
        ]
        scraper._save_to_db(topics)

        conn = sqlite3.connect(scraper.db_path)
        rows = conn.execute("SELECT extracted_topic FROM competitor_topics").fetchall()
        conn.close()
        assert len(rows) == 2
        assert rows[0][0] == "topic a"

    def test_empty_list_does_nothing(self, tmp_path) -> None:
        scraper = _make_scraper(tmp_path)
        scraper._save_to_db([])   # should not raise
        assert not scraper.db_path.exists()   # DB not created for empty list


# ---------------------------------------------------------------------------
# scrape_competitor_topics
# ---------------------------------------------------------------------------

class TestScrapeCompetitorTopics:
    def test_returns_empty_when_no_api_key(self, tmp_path) -> None:
        scraper = CompetitorScraper(api_key="", db_path=tmp_path / "t.db")
        result = scraper.scrape_competitor_topics("money")
        assert result == []

    def test_returns_empty_for_unknown_category(self, tmp_path) -> None:
        scraper = _make_scraper(tmp_path)
        result = scraper.scrape_competitor_topics("unknown_category_xyz")
        assert result == []

    @patch("src.crawler.competitor_scraper.CompetitorScraper._extract_topic", return_value="why saving money fails")
    @patch("src.crawler.competitor_scraper.CompetitorScraper._fetch_video_stats", return_value={"vid_000": 200_000})
    @patch("src.crawler.competitor_scraper.CompetitorScraper._fetch_channel_videos")
    def test_returns_topics_for_high_signal_videos(
        self, mock_fetch, mock_stats, mock_extract, tmp_path
    ) -> None:
        mock_fetch.return_value = [
            {"title": "Why Saving Money Fails", "view_count": 200_000},
        ]
        scraper = _make_scraper(tmp_path)
        result = scraper.scrape_competitor_topics("money")
        assert "why saving money fails" in result

    @patch("src.crawler.competitor_scraper.CompetitorScraper._extract_topic", return_value="low view topic")
    @patch("src.crawler.competitor_scraper.CompetitorScraper._fetch_channel_videos")
    def test_ignores_low_view_count_videos(self, mock_fetch, mock_extract, tmp_path) -> None:
        mock_fetch.return_value = [
            {"title": "Low Views Video", "view_count": 5_000},  # below 100k threshold
        ]
        scraper = _make_scraper(tmp_path)
        result = scraper.scrape_competitor_topics("money")
        assert result == []   # below threshold → not included


# ---------------------------------------------------------------------------
# mine_comment_topics
# ---------------------------------------------------------------------------

class TestMineCommentTopics:
    def test_returns_empty_when_no_api_key(self, tmp_path) -> None:
        scraper = CompetitorScraper(api_key="", db_path=tmp_path / "t.db")
        result = scraper.mine_comment_topics(["vid123"])
        assert result == []

    @patch("src.crawler.competitor_scraper.CompetitorScraper._extract_topic_from_comment",
           return_value="why investing early beats saving")
    @patch("src.crawler.competitor_scraper.CompetitorScraper._fetch_comments")
    def test_extracts_questions_from_comments(self, mock_fetch, mock_extract, tmp_path) -> None:
        mock_fetch.return_value = [
            "Why does investing early matter so much?",
            "Great video!",   # no question pattern → skipped
        ]
        scraper = _make_scraper(tmp_path)
        result = scraper.mine_comment_topics(["vid123"])
        # Only the question comment produces a topic
        assert len(result) == 1
        assert result[0] == "why investing early beats saving"

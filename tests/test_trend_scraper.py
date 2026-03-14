"""
Tests for src/crawler/trend_scraper.py

All tests mock external API calls so no internet access is required.
"""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.crawler.trend_scraper import (
    GoogleTrendsScraper,
    TrendScrapingEngine,
    TrendSignal,
    YouTubeTrendsScraper,
)


# ---------------------------------------------------------------------------
# TrendSignal
# ---------------------------------------------------------------------------

class TestTrendSignal:
    def test_defaults_populated(self) -> None:
        sig = TrendSignal(keyword="python tutorial", source="google")
        assert sig.keyword == "python tutorial"
        assert sig.source == "google"
        assert sig.region == "US"
        assert sig.interest_score == 0.0
        assert sig.related_queries == []
        assert sig.fetched_at != ""
        # fetched_at should be a valid ISO timestamp
        datetime.fromisoformat(sig.fetched_at)

    def test_to_dict_shape(self) -> None:
        sig = TrendSignal(
            keyword="AI tools",
            source="youtube",
            interest_score=75.5,
            related_queries=["best AI tools 2024"],
        )
        d = sig.to_dict()
        assert d["keyword"] == "AI tools"
        assert d["source"] == "youtube"
        assert d["interest_score"] == 75.5
        assert json.loads(d["related_query"]) == ["best AI tools 2024"]
        assert "fetched_at" in d
        assert "raw_json" in d

    def test_to_dict_raw_json_serialisable(self) -> None:
        sig = TrendSignal(keyword="k", source="google", raw_json={"data": [1, 2, 3]})
        d = sig.to_dict()
        parsed = json.loads(d["raw_json"])
        assert parsed == {"data": [1, 2, 3]}


# ---------------------------------------------------------------------------
# GoogleTrendsScraper
# ---------------------------------------------------------------------------

class TestGoogleTrendsScraper:
    """Mock pytrends so no real HTTP calls are made."""

    def _make_iot_df(self, keywords: list[str]) -> pd.DataFrame:
        """Build a fake interest_over_time DataFrame."""
        data = {kw: [50.0, 60.0, 55.0] for kw in keywords}
        data["isPartial"] = [False, False, False]
        return pd.DataFrame(data)

    def _make_related(self, keywords: list[str]) -> dict:
        return {
            kw: {
                "top": pd.DataFrame({"query": [f"{kw} tutorial", f"learn {kw}"], "value": [100, 80]}),
                "rising": pd.DataFrame({"query": [f"{kw} 2024"], "value": [500]}),
            }
            for kw in keywords
        }

    @patch("src.crawler.trend_scraper.TrendReq")
    def test_fetch_returns_signals(self, mock_trend_req_cls) -> None:
        keywords = ["Python", "JavaScript"]

        mock_pt = MagicMock()
        mock_pt.interest_over_time.return_value = self._make_iot_df(keywords)
        mock_pt.related_queries.return_value = self._make_related(keywords)
        mock_trend_req_cls.return_value = mock_pt

        scraper = GoogleTrendsScraper()
        signals = scraper.fetch(keywords)

        assert len(signals) == 2
        kws = {s.keyword for s in signals}
        assert kws == {"Python", "JavaScript"}
        for sig in signals:
            assert sig.source == "google"
            assert sig.interest_score > 0
            assert len(sig.related_queries) > 0

    @patch("src.crawler.trend_scraper.TrendReq")
    def test_fetch_handles_empty_iot(self, mock_trend_req_cls) -> None:
        mock_pt = MagicMock()
        mock_pt.interest_over_time.return_value = pd.DataFrame()
        mock_pt.related_queries.return_value = {}
        mock_trend_req_cls.return_value = mock_pt

        scraper = GoogleTrendsScraper()
        signals = scraper.fetch(["some keyword"])

        assert len(signals) == 1
        assert signals[0].interest_score == 0.0

    @patch("src.crawler.trend_scraper.TrendReq")
    def test_fetch_chunks_more_than_5(self, mock_trend_req_cls) -> None:
        """pytrends can only handle 5 keywords at a time; ensure chunking works."""
        keywords = [f"kw{i}" for i in range(7)]
        iot = self._make_iot_df(keywords[:5])

        mock_pt = MagicMock()
        mock_pt.interest_over_time.return_value = iot
        mock_pt.related_queries.return_value = self._make_related(keywords[:5])
        mock_trend_req_cls.return_value = mock_pt

        scraper = GoogleTrendsScraper()
        signals = scraper.fetch(keywords)

        # Should have called build_payload twice (5 + 2)
        assert mock_pt.build_payload.call_count == 2
        assert len(signals) == 7

    @patch("src.crawler.trend_scraper.TrendReq")
    def test_fetch_retries_on_exception(self, mock_trend_req_cls) -> None:
        mock_pt = MagicMock()
        # Fail twice, succeed on third attempt
        mock_pt.interest_over_time.side_effect = [
            Exception("timeout"),
            Exception("timeout"),
            self._make_iot_df(["keyword"]),
        ]
        mock_pt.related_queries.return_value = {}
        mock_trend_req_cls.return_value = mock_pt

        scraper = GoogleTrendsScraper(retries=3)
        # patch sleep so test is instant
        with patch("src.crawler.trend_scraper.time.sleep"):
            signals = scraper.fetch(["keyword"])

        assert len(signals) == 1

    @patch("src.crawler.trend_scraper.TrendReq")
    def test_fetch_returns_empty_on_exhausted_retries(self, mock_trend_req_cls) -> None:
        mock_pt = MagicMock()
        mock_pt.interest_over_time.side_effect = Exception("always fails")
        mock_trend_req_cls.return_value = mock_pt

        scraper = GoogleTrendsScraper(retries=2)
        with patch("src.crawler.trend_scraper.time.sleep"):
            signals = scraper.fetch(["fail"])

        assert signals == []


# ---------------------------------------------------------------------------
# GoogleTrendsScraper — fetch_rising
# ---------------------------------------------------------------------------

class TestGoogleTrendsFetchRising:
    """Tests for the rising-query discovery method."""

    def _make_related_with_rising(self, keywords: list[str]) -> dict:
        return {
            kw: {
                "top":    pd.DataFrame({"query": [f"{kw} tutorial"], "value": [100]}),
                "rising": pd.DataFrame({"query": [f"{kw} 2026", f"best {kw}"], "value": [500, 300]}),
            }
            for kw in keywords
        }

    @patch("src.crawler.trend_scraper.TrendReq")
    def test_fetch_rising_returns_signals(self, mock_trend_req_cls) -> None:
        keywords = ["money"]
        mock_pt = MagicMock()
        mock_pt.related_queries.return_value = self._make_related_with_rising(keywords)
        mock_trend_req_cls.return_value = mock_pt

        scraper = GoogleTrendsScraper()
        with patch("src.crawler.trend_scraper.time.sleep"):
            signals = scraper.fetch_rising(keywords)

        assert len(signals) >= 1
        assert all(s.source == "rising_google" for s in signals)

    @patch("src.crawler.trend_scraper.TrendReq")
    def test_rising_queries_scored_higher_than_top_queries(self, mock_trend_req_cls) -> None:
        """Rising signals always carry score 75; regular Google Trends interest
        is based on mean IOT which for this fixture is 55 — so rising wins."""
        keywords = ["salary"]
        iot_df = pd.DataFrame({"salary": [50.0, 55.0, 60.0], "isPartial": [False]*3})
        related = {
            "salary": {
                "top":    pd.DataFrame({"query": ["salary negotiation"], "value": [80]}),
                "rising": pd.DataFrame({"query": ["salary transparency 2026"], "value": [600]}),
            }
        }

        mock_pt = MagicMock()
        mock_pt.interest_over_time.return_value = iot_df
        mock_pt.related_queries.return_value = related
        mock_trend_req_cls.return_value = mock_pt

        scraper = GoogleTrendsScraper()
        with patch("src.crawler.trend_scraper.time.sleep"):
            top_signals    = scraper.fetch(keywords)
            rising_signals = scraper.fetch_rising(keywords)

        assert len(top_signals) == 1
        assert len(rising_signals) >= 1
        top_score    = top_signals[0].interest_score     # ~55
        rising_score = rising_signals[0].interest_score  # always 75
        assert rising_score > top_score

    @patch("src.crawler.trend_scraper.TrendReq")
    def test_fetch_rising_handles_missing_rising_data(self, mock_trend_req_cls) -> None:
        """When rising DataFrame is absent, fetch_rising returns empty list."""
        mock_pt = MagicMock()
        mock_pt.related_queries.return_value = {
            "money": {"top": pd.DataFrame({"query": ["money tips"], "value": [100]})}
            # "rising" key absent
        }
        mock_trend_req_cls.return_value = mock_pt

        scraper = GoogleTrendsScraper()
        with patch("src.crawler.trend_scraper.time.sleep"):
            signals = scraper.fetch_rising(["money"])

        assert signals == []

    @patch("src.crawler.trend_scraper.TrendReq")
    def test_fetch_rising_handles_exception_gracefully(self, mock_trend_req_cls) -> None:
        mock_pt = MagicMock()
        mock_pt.related_queries.side_effect = Exception("rate limited")
        mock_trend_req_cls.return_value = mock_pt

        scraper = GoogleTrendsScraper(retries=2)
        with patch("src.crawler.trend_scraper.time.sleep"):
            signals = scraper.fetch_rising(["money"])

        assert signals == []


# ---------------------------------------------------------------------------
# YouTubeTrendsScraper
# ---------------------------------------------------------------------------

FAKE_VIDEO_ITEMS = [
    {
        "id": "abc123",
        "snippet": {
            "title": "10 AI Tools You Need in 2024",
            "channelTitle": "Tech Channel",
            "categoryId": "28",
        },
        "statistics": {"viewCount": "2500000", "likeCount": "50000"},
    },
    {
        "id": "def456",
        "snippet": {
            "title": "Make Money Online Tutorial",
            "channelTitle": "Finance Guru",
            "categoryId": "22",
        },
        "statistics": {"viewCount": "800000", "likeCount": "20000"},
    },
]

FAKE_SEARCH_RESPONSE = {
    "items": [
        {"snippet": {"title": "Python tutorial for beginners 2024"}},
        {"snippet": {"title": "Learn Python in 1 hour"}},
    ],
    "pageInfo": {"totalResults": 450000},
}


class TestYouTubeTrendsScraper:
    @patch("src.crawler.trend_scraper.httpx.Client")
    def test_fetch_trending_returns_signals(self, mock_client_cls) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"items": FAKE_VIDEO_ITEMS}
        mock_resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        scraper = YouTubeTrendsScraper(api_key="fake_key")
        signals = scraper.fetch_trending()

        assert len(signals) == 2
        assert all(s.source == "youtube" for s in signals)
        assert all(s.interest_score > 0 for s in signals)

    @patch("src.crawler.trend_scraper.httpx.Client")
    def test_fetch_for_keywords_returns_signals(self, mock_client_cls) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = FAKE_SEARCH_RESPONSE
        mock_resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        scraper = YouTubeTrendsScraper(api_key="fake_key")
        with patch("src.crawler.trend_scraper.time.sleep"):
            signals = scraper.fetch_for_keywords(["Python tutorial"])

        assert len(signals) == 1
        assert signals[0].keyword == "Python tutorial"
        assert signals[0].interest_score > 0
        assert len(signals[0].related_queries) > 0

    def test_skips_when_no_api_key(self) -> None:
        scraper = YouTubeTrendsScraper(api_key="")
        signals = scraper.fetch_trending()
        assert signals == []
        signals2 = scraper.fetch_for_keywords(["test"])
        assert signals2 == []

    @patch("src.crawler.trend_scraper.httpx.Client")
    def test_fetch_for_keywords_handles_http_error(self, mock_client_cls) -> None:
        import httpx

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "403", request=MagicMock(), response=MagicMock()
        )
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        scraper = YouTubeTrendsScraper(api_key="fake_key", retries=2)
        with patch("src.crawler.trend_scraper.time.sleep"):
            signals = scraper.fetch_for_keywords(["test"])

        assert signals == []


# ---------------------------------------------------------------------------
# TrendScrapingEngine
# ---------------------------------------------------------------------------

class TestTrendScrapingEngine:
    @patch("src.crawler.trend_scraper.YouTubeTrendsScraper")
    @patch("src.crawler.trend_scraper.GoogleTrendsScraper")
    def test_fetch_all_combines_sources(
        self, mock_google_cls, mock_yt_cls
    ) -> None:
        mock_google = MagicMock()
        mock_google.fetch.return_value = [
            TrendSignal(keyword="Python", source="google", interest_score=80.0)
        ]
        mock_google.fetch_rising.return_value = []  # no rising signals in this test
        mock_google_cls.return_value = mock_google

        mock_yt = MagicMock()
        mock_yt.fetch_for_keywords.return_value = [
            TrendSignal(keyword="Python", source="youtube", interest_score=65.0)
        ]
        mock_yt.fetch_trending.return_value = []
        mock_yt_cls.return_value = mock_yt

        engine = TrendScrapingEngine()
        engine.google = mock_google
        engine.youtube = mock_yt

        signals = engine.fetch_all(["Python"])
        assert len(signals) == 2
        sources = {s.source for s in signals}
        assert "google" in sources
        assert "youtube" in sources

    def test_engine_disabled_sources(self) -> None:
        engine = TrendScrapingEngine(google_enabled=False, youtube_enabled=False)
        assert engine.google is None
        assert engine.youtube is None
        signals = engine.fetch_all(["anything"])
        assert signals == []

    @patch("src.crawler.trend_scraper.YouTubeTrendsScraper")
    @patch("src.crawler.trend_scraper.GoogleTrendsScraper")
    def test_fetch_all_includes_rising_signals(
        self, mock_google_cls, mock_yt_cls
    ) -> None:
        """fetch_all propagates rising_google signals from fetch_rising."""
        mock_google = MagicMock()
        mock_google.fetch.return_value = [
            TrendSignal(keyword="Python", source="google", interest_score=55.0)
        ]
        mock_google.fetch_rising.return_value = [
            TrendSignal(keyword="Python 2026", source="rising_google", interest_score=75.0)
        ]
        mock_google_cls.return_value = mock_google

        mock_yt = MagicMock()
        mock_yt.fetch_for_keywords.return_value = []
        mock_yt.fetch_trending.return_value = []
        mock_yt_cls.return_value = mock_yt

        engine = TrendScrapingEngine(youtube_enabled=False)
        engine.google = mock_google
        engine.youtube = None

        signals = engine.fetch_all(["Python"])
        sources = {s.source for s in signals}
        assert "rising_google" in sources

    @patch("src.crawler.trend_scraper.GoogleTrendsScraper")
    def test_engine_handles_google_exception(self, mock_google_cls) -> None:
        mock_google = MagicMock()
        mock_google.fetch.side_effect = RuntimeError("network failure")
        mock_google_cls.return_value = mock_google

        engine = TrendScrapingEngine(youtube_enabled=False)
        engine.google = mock_google
        # Should not raise — just return empty
        signals = engine.fetch_all(["test"])
        assert signals == []

"""
tests/test_reddit_scraper.py

All HTTP requests and Claude API calls are mocked — no real network I/O.
Uses tmp_path for real SQLite DB tests.
"""

import json
import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from src.crawler.reddit_scraper import (
    CATEGORY_MAP,
    SUBREDDITS,
    RedditScraper,
    RedditTopic,
    _DEDUP_THRESHOLD,
    _RATE_LIMIT_DELAY,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_post(
    title: str = "How I paid off $40k in debt in under two years",
    score: int = 500,
    num_comments: int = 50,
    stickied: bool = False,
    selftext: str = "",
) -> dict:
    return {
        "title": title,
        "score": score,
        "num_comments": num_comments,
        "stickied": stickied,
        "selftext": selftext,
    }


def _mock_reddit_response(posts: list[dict]) -> MagicMock:
    m = MagicMock()
    m.raise_for_status = MagicMock()
    m.json.return_value = {
        "data": {
            "children": [{"data": p} for p in posts]
        }
    }
    return m


def _make_scraper(tmp_path, api_key: str = "") -> RedditScraper:
    return RedditScraper(anthropic_api_key=api_key, db_path=tmp_path / "test.db")


# ---------------------------------------------------------------------------
# Post filtering
# ---------------------------------------------------------------------------


class TestPostFiltering:
    def test_accepts_valid_post(self, tmp_path) -> None:
        scraper = _make_scraper(tmp_path)
        result = scraper._filter_posts([_make_post()])
        assert len(result) == 1

    def test_rejects_stickied_post(self, tmp_path) -> None:
        scraper = _make_scraper(tmp_path)
        result = scraper._filter_posts([_make_post(stickied=True)])
        assert result == []

    def test_rejects_removed_selftext(self, tmp_path) -> None:
        scraper = _make_scraper(tmp_path)
        result = scraper._filter_posts([_make_post(selftext="[removed]")])
        assert result == []

    def test_rejects_deleted_selftext(self, tmp_path) -> None:
        scraper = _make_scraper(tmp_path)
        result = scraper._filter_posts([_make_post(selftext="[deleted]")])
        assert result == []

    def test_rejects_low_score(self, tmp_path) -> None:
        scraper = _make_scraper(tmp_path)
        result = scraper._filter_posts([_make_post(score=50)])
        assert result == []

    def test_rejects_low_comments(self, tmp_path) -> None:
        scraper = _make_scraper(tmp_path)
        result = scraper._filter_posts([_make_post(num_comments=5)])
        assert result == []

    def test_rejects_weekly_mod_prefix(self, tmp_path) -> None:
        scraper = _make_scraper(tmp_path)
        post = _make_post(title="[Weekly] What are your wins this week?")
        assert scraper._filter_posts([post]) == []

    def test_rejects_daily_mod_prefix(self, tmp_path) -> None:
        scraper = _make_scraper(tmp_path)
        post = _make_post(title="[Daily] Discussion thread for today")
        assert scraper._filter_posts([post]) == []

    def test_rejects_meta_mod_prefix(self, tmp_path) -> None:
        scraper = _make_scraper(tmp_path)
        post = _make_post(title="[META] Announcement about the subreddit rules")
        assert scraper._filter_posts([post]) == []

    def test_rejects_title_too_short(self, tmp_path) -> None:
        scraper = _make_scraper(tmp_path)
        result = scraper._filter_posts([_make_post(title="Short")])
        assert result == []

    def test_rejects_title_too_long(self, tmp_path) -> None:
        scraper = _make_scraper(tmp_path)
        result = scraper._filter_posts([_make_post(title="x" * 201)])
        assert result == []

    def test_accepts_post_at_score_boundary(self, tmp_path) -> None:
        """score=100 (exactly _MIN_SCORE) → accepted."""
        scraper = _make_scraper(tmp_path)
        result = scraper._filter_posts([_make_post(score=100)])
        assert len(result) == 1

    def test_mixed_posts_only_valid_returned(self, tmp_path) -> None:
        scraper = _make_scraper(tmp_path)
        posts = [
            _make_post(),               # valid
            _make_post(score=10),       # too low
            _make_post(stickied=True),  # stickied
        ]
        result = scraper._filter_posts(posts)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Score calculation
# ---------------------------------------------------------------------------


class TestScoreCalculation:
    def test_score_tier_10k_upvotes_capped(self, tmp_path) -> None:
        scraper = _make_scraper(tmp_path)
        score = scraper._calculate_score(10000, 10)
        # 98 * 1.0 * 1.2 = 117.6 → capped at 99
        assert score == 99.0

    def test_score_tier_5k_upvotes(self, tmp_path) -> None:
        scraper = _make_scraper(tmp_path)
        score = scraper._calculate_score(5000, 5)
        # 95 * 0.5 * 1.2 = 57.0
        assert abs(score - 57.0) < 0.5

    def test_score_tier_1k_upvotes(self, tmp_path) -> None:
        scraper = _make_scraper(tmp_path)
        score = scraper._calculate_score(1000, 8)
        # 92 * 0.8 * 1.2 = 88.32
        assert abs(score - 88.3) < 0.5

    def test_score_tier_500_upvotes(self, tmp_path) -> None:
        scraper = _make_scraper(tmp_path)
        score = scraper._calculate_score(500, 7)
        # 88 * 0.7 * 1.2 = 73.92
        assert abs(score - 73.9) < 0.5

    def test_score_tier_100_upvotes(self, tmp_path) -> None:
        scraper = _make_scraper(tmp_path)
        score = scraper._calculate_score(100, 6)
        # 75 * 0.6 * 1.2 = 54.0
        assert abs(score - 54.0) < 0.5

    def test_score_never_exceeds_99(self, tmp_path) -> None:
        scraper = _make_scraper(tmp_path)
        assert scraper._calculate_score(999999, 10) <= 99.0

    def test_score_positive_for_min_pain(self, tmp_path) -> None:
        scraper = _make_scraper(tmp_path)
        assert scraper._calculate_score(100, 1) > 0

    def test_higher_pain_yields_higher_score(self, tmp_path) -> None:
        scraper = _make_scraper(tmp_path)
        low = scraper._calculate_score(500, 3)
        high = scraper._calculate_score(500, 8)
        assert high > low


# ---------------------------------------------------------------------------
# Category mapping
# ---------------------------------------------------------------------------


class TestCategoryMapping:
    def test_all_subreddits_in_category_map(self) -> None:
        all_subs = [sub for subs in SUBREDDITS.values() for sub in subs]
        for sub in all_subs:
            assert sub in CATEGORY_MAP, f"Missing CATEGORY_MAP entry for r/{sub}"

    def test_specific_money_mappings(self) -> None:
        assert CATEGORY_MAP["personalfinance"] == "money"
        assert CATEGORY_MAP["wallstreetbets"] == "money"
        assert CATEGORY_MAP["Fire"] == "money"

    def test_specific_career_mappings(self) -> None:
        assert CATEGORY_MAP["antiwork"] == "career"
        assert CATEGORY_MAP["salary"] == "career"

    def test_specific_success_mappings(self) -> None:
        assert CATEGORY_MAP["getmotivated"] == "success"
        assert CATEGORY_MAP["Entrepreneur"] == "success"

    def test_subreddits_has_three_categories(self) -> None:
        assert set(SUBREDDITS.keys()) == {"money", "career", "success"}

    def test_money_has_7_subreddits(self) -> None:
        assert len(SUBREDDITS["money"]) == 7

    def test_career_has_5_subreddits(self) -> None:
        assert len(SUBREDDITS["career"]) == 5

    def test_success_has_4_subreddits(self) -> None:
        assert len(SUBREDDITS["success"]) == 4


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_exact_match_is_duplicate(self, tmp_path) -> None:
        scraper = _make_scraper(tmp_path)
        existing = ["you are living paycheck to paycheck"]
        assert scraper._is_duplicate("you are living paycheck to paycheck", existing)

    def test_high_similarity_is_duplicate(self, tmp_path) -> None:
        scraper = _make_scraper(tmp_path)
        existing = ["living paycheck to paycheck every month"]
        # Very close phrasing → above 70% threshold
        assert scraper._is_duplicate("you live paycheck to paycheck every month", existing)

    def test_different_topic_not_duplicate(self, tmp_path) -> None:
        scraper = _make_scraper(tmp_path)
        existing = ["stock market investing tips for beginners"]
        assert not scraper._is_duplicate("how to negotiate your salary raise today", existing)

    def test_empty_existing_never_duplicate(self, tmp_path) -> None:
        scraper = _make_scraper(tmp_path)
        assert not scraper._is_duplicate("any topic here", [])

    def test_dedup_threshold_constant_is_correct(self) -> None:
        assert _DEDUP_THRESHOLD == 0.70


# ---------------------------------------------------------------------------
# Batch Claude extraction
# ---------------------------------------------------------------------------


class TestBatchExtraction:
    def test_handles_batch_of_10_posts(self, tmp_path) -> None:
        scraper = RedditScraper(
            anthropic_api_key="test-key", db_path=tmp_path / "t.db"
        )
        posts = [_make_post(title=f"Post about money habit number {i} that matters") for i in range(10)]
        result_data = [
            {"post_index": i + 1, "topic": f"money habit {i}", "pain_level": 5}
            for i in range(10)
        ]
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text=json.dumps(result_data))]

        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.create.return_value = mock_msg
            results = scraper._extract_topics_batch(posts, "personalfinance")

        assert len(results) == 10

    def test_api_failure_falls_back_to_title(self, tmp_path) -> None:
        scraper = RedditScraper(
            anthropic_api_key="test-key", db_path=tmp_path / "t.db"
        )
        posts = [_make_post(title="How I paid off forty thousand dollars in debt")]

        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.create.side_effect = Exception("API error")
            results = scraper._extract_topics_batch(posts, "personalfinance")

        assert len(results) == 1
        assert results[0]["pain_level"] == 5  # fallback default

    def test_no_api_key_uses_fallback(self, tmp_path) -> None:
        scraper = RedditScraper(anthropic_api_key="", db_path=tmp_path / "t.db")
        posts = [_make_post(title="How I saved money on absolutely everything in life")]
        results = scraper._extract_topics_batch(posts, "personalfinance")
        assert len(results) == 1
        assert results[0]["topic"]  # non-empty

    def test_handles_markdown_fenced_json(self, tmp_path) -> None:
        scraper = RedditScraper(
            anthropic_api_key="test-key", db_path=tmp_path / "t.db"
        )
        posts = [_make_post(title="How I saved money on everything around the house")]
        data = [{"post_index": 1, "topic": "save money on everyday expenses now", "pain_level": 6}]
        raw = f"```json\n{json.dumps(data)}\n```"
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text=raw)]

        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.create.return_value = mock_msg
            results = scraper._extract_topics_batch(posts, "personalfinance")

        assert len(results) == 1
        assert results[0]["topic"] == "save money on everyday expenses now"

    def test_fallback_trims_to_12_words(self, tmp_path) -> None:
        scraper = _make_scraper(tmp_path)
        long_title = " ".join([f"word{i}" for i in range(20)])
        posts = [_make_post(title=long_title)]
        results = scraper._fallback_extraction(posts)
        assert len(results[0]["topic"].split()) <= 12


# ---------------------------------------------------------------------------
# Scheduler integration
# ---------------------------------------------------------------------------


class TestSchedulerReddit:
    def test_reddit_job_id_in_scheduler(self) -> None:
        from src.scheduler import build_scheduler
        scheduler = build_scheduler(timezone_name="UTC")
        job_ids = [j.id for j in scheduler.get_jobs()]
        assert "reddit_scraper" in job_ids

    def test_run_reddit_scraper_is_callable(self) -> None:
        from src.scheduler import run_reddit_scraper
        assert callable(run_reddit_scraper)

    def test_run_reddit_scraper_handles_exception(self) -> None:
        """Exception inside scraper must not propagate."""
        from src.scheduler import run_reddit_scraper
        with patch("src.crawler.reddit_scraper.RedditScraper") as MockScraper:
            MockScraper.side_effect = Exception("network error")
            run_reddit_scraper()  # must not raise

    def test_run_reddit_scraper_calls_all_three_categories(self) -> None:
        from src.scheduler import run_reddit_scraper
        mock_scraper = MagicMock()
        mock_scraper.scrape_finance_subreddits.return_value = []
        with patch("src.crawler.reddit_scraper.RedditScraper", return_value=mock_scraper):
            run_reddit_scraper()
        calls = mock_scraper.scrape_finance_subreddits.call_args_list
        categories_called = {c[1]["category"] for c in calls}
        assert categories_called == {"money", "career", "success"}

    def test_reddit_job_trigger_is_cron(self) -> None:
        from apscheduler.triggers.cron import CronTrigger
        from src.scheduler import build_scheduler
        scheduler = build_scheduler(timezone_name="UTC")
        job = next(j for j in scheduler.get_jobs() if j.id == "reddit_scraper")
        assert isinstance(job.trigger, CronTrigger)


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimit:
    @patch("src.crawler.reddit_scraper.httpx.get")
    @patch("src.crawler.reddit_scraper.time.sleep")
    def test_sleep_called_between_requests(
        self, mock_sleep, mock_get, tmp_path
    ) -> None:
        """Multiple requests → time.sleep(_RATE_LIMIT_DELAY) called at least once."""
        mock_get.return_value = _mock_reddit_response([])
        scraper = RedditScraper(anthropic_api_key="", db_path=tmp_path / "t.db")
        scraper.scrape_finance_subreddits(category="money")
        # Multiple subreddits × 2 kinds → many sleeps
        assert mock_sleep.call_count >= 1

    @patch("src.crawler.reddit_scraper.httpx.get")
    @patch("src.crawler.reddit_scraper.time.sleep")
    def test_sleep_duration_is_2_seconds(
        self, mock_sleep, mock_get, tmp_path
    ) -> None:
        mock_get.return_value = _mock_reddit_response([])
        scraper = RedditScraper(anthropic_api_key="", db_path=tmp_path / "t.db")
        scraper.scrape_finance_subreddits(category="money")
        for call_args in mock_sleep.call_args_list:
            assert call_args[0][0] == _RATE_LIMIT_DELAY


# ---------------------------------------------------------------------------
# Timeout / network error handling
# ---------------------------------------------------------------------------


class TestTimeoutHandling:
    @patch("src.crawler.reddit_scraper.httpx.get")
    def test_timeout_returns_empty_list(self, mock_get, tmp_path) -> None:
        import httpx as _httpx
        mock_get.side_effect = _httpx.TimeoutException("timed out")
        scraper = _make_scraper(tmp_path)
        result = scraper._fetch_subreddit("personalfinance", "hot")
        assert result == []

    @patch("src.crawler.reddit_scraper.httpx.get")
    def test_network_error_returns_empty_list(self, mock_get, tmp_path) -> None:
        import httpx as _httpx
        mock_get.side_effect = _httpx.RequestError("connection refused")
        scraper = _make_scraper(tmp_path)
        result = scraper._fetch_subreddit("antiwork", "top")
        assert result == []


# ---------------------------------------------------------------------------
# Save to DB
# ---------------------------------------------------------------------------


class TestSaveToDb:
    def _make_topic(self, keyword: str = "paycheck trap exposed now") -> RedditTopic:
        return RedditTopic(
            keyword=keyword,
            category="money",
            score=88.0,
            subreddit="personalfinance",
            upvotes=500,
            pain_level=7,
            source="reddit/personalfinance",
        )

    def test_saves_topics_to_scored_topics(self, tmp_path) -> None:
        scraper = _make_scraper(tmp_path)
        count = scraper._save_to_db([self._make_topic()])
        assert count == 1
        conn = sqlite3.connect(tmp_path / "test.db")
        rows = conn.execute("SELECT keyword FROM scored_topics").fetchall()
        conn.close()
        assert len(rows) == 1

    def test_source_contains_reddit(self, tmp_path) -> None:
        scraper = _make_scraper(tmp_path)
        scraper._save_to_db([self._make_topic()])
        conn = sqlite3.connect(tmp_path / "test.db")
        row = conn.execute("SELECT source FROM scored_topics").fetchone()
        conn.close()
        assert "reddit" in row[0]

    def test_used_column_defaults_to_zero(self, tmp_path) -> None:
        scraper = _make_scraper(tmp_path)
        scraper._save_to_db([self._make_topic()])
        conn = sqlite3.connect(tmp_path / "test.db")
        row = conn.execute("SELECT used FROM scored_topics").fetchone()
        conn.close()
        assert row[0] == 0

    def test_empty_topics_returns_zero(self, tmp_path) -> None:
        scraper = _make_scraper(tmp_path)
        assert scraper._save_to_db([]) == 0

    def test_multiple_topics_all_saved(self, tmp_path) -> None:
        scraper = _make_scraper(tmp_path)
        topics = [self._make_topic(f"topic {i}") for i in range(5)]
        count = scraper._save_to_db(topics)
        assert count == 5
        conn = sqlite3.connect(tmp_path / "test.db")
        rows = conn.execute("SELECT COUNT(*) FROM scored_topics").fetchone()
        conn.close()
        assert rows[0] == 5


# ---------------------------------------------------------------------------
# _fetch_subreddit — URL construction
# ---------------------------------------------------------------------------


class TestFetchSubreddit:
    @patch("src.crawler.reddit_scraper.httpx.get")
    def test_hot_uses_hot_json_url(self, mock_get, tmp_path) -> None:
        mock_get.return_value = _mock_reddit_response([_make_post()])
        scraper = _make_scraper(tmp_path)
        result = scraper._fetch_subreddit("personalfinance", "hot")
        assert len(result) == 1
        called_url = mock_get.call_args[0][0]
        assert "personalfinance" in called_url
        assert "hot.json" in called_url

    @patch("src.crawler.reddit_scraper.httpx.get")
    def test_top_week_uses_top_json_with_t_week(self, mock_get, tmp_path) -> None:
        mock_get.return_value = _mock_reddit_response([_make_post()])
        scraper = _make_scraper(tmp_path)
        result = scraper._fetch_subreddit("investing", "top")
        assert len(result) == 1
        called_url = mock_get.call_args[0][0]
        assert "top.json" in called_url
        assert "t=week" in called_url

    @patch("src.crawler.reddit_scraper.httpx.get")
    def test_api_error_returns_empty(self, mock_get, tmp_path) -> None:
        import httpx as _httpx
        mock_get.side_effect = _httpx.RequestError("error")
        scraper = _make_scraper(tmp_path)
        result = scraper._fetch_subreddit("jobs", "hot")
        assert result == []

    @patch("src.crawler.reddit_scraper.httpx.get")
    def test_user_agent_header_sent(self, mock_get, tmp_path) -> None:
        mock_get.return_value = _mock_reddit_response([])
        scraper = _make_scraper(tmp_path)
        scraper._fetch_subreddit("salary", "hot")
        call_kwargs = mock_get.call_args[1]
        assert "headers" in call_kwargs
        assert "ChannelForge" in call_kwargs["headers"]["User-Agent"]

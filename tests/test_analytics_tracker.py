"""
Tests for src/analytics/analytics_tracker.py

Uses tmp_path for real SQLite — no mocking of DB calls.
All YouTube Analytics API calls are mocked via patch.object.
"""

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.analytics.analytics_tracker import (
    AnalyticsTracker,
    VideoMetrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tracker(tmp_path: Path, **kw) -> AnalyticsTracker:
    return AnalyticsTracker(db_path=tmp_path / "test.db", **kw)


def _raw_metrics(**overrides) -> dict:
    base = {
        "views": 10_000,
        "watch_time_minutes": 500.0,
        "likes": 300,
        "comments": 50,
        "shares": 20,
        "impressions": 50_000,
        "ctr": 0.06,
        "subscribers_gained": 100,
        "subscribers_lost": 5,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# VideoMetrics
# ---------------------------------------------------------------------------

class TestVideoMetrics:
    def test_to_dict_has_all_keys(self) -> None:
        m = VideoMetrics(video_id="vid001")
        d = m.to_dict()
        for key in (
            "video_id", "channel_key", "views", "watch_time_minutes",
            "likes", "comments", "shares", "impressions", "ctr",
            "subscribers_gained", "subscribers_lost",
            "engagement_rate", "virality_score", "tier", "fetched_at",
        ):
            assert key in d

    def test_fetched_at_auto_set(self) -> None:
        m = VideoMetrics(video_id="vid001")
        assert m.fetched_at != ""

    def test_to_dict_serializable(self) -> None:
        import json as _json
        m = VideoMetrics(video_id="v1", views=1000, tier="C")
        assert len(_json.dumps(m.to_dict())) > 10


# ---------------------------------------------------------------------------
# Tier assignment (pure function)
# ---------------------------------------------------------------------------

class TestTierAssignment:
    def test_s_tier(self) -> None:
        assert AnalyticsTracker.assign_tier(60_000, 9.0) == "S"

    def test_s_tier_exact_boundary(self) -> None:
        # exactly 50001 views AND 8.01% engagement → S
        assert AnalyticsTracker.assign_tier(50_001, 8.01) == "S"

    def test_not_s_when_views_low(self) -> None:
        # 80k views but only 5% engagement → not S
        assert AnalyticsTracker.assign_tier(80_000, 5.0) != "S"

    def test_not_s_when_engagement_low(self) -> None:
        # great views but engagement at 7% → not S
        assert AnalyticsTracker.assign_tier(60_000, 7.0) != "S"

    def test_a_tier_by_views(self) -> None:
        assert AnalyticsTracker.assign_tier(25_000, 2.0) == "A"

    def test_a_tier_by_engagement(self) -> None:
        assert AnalyticsTracker.assign_tier(1_000, 7.0) == "A"

    def test_b_tier_by_views(self) -> None:
        assert AnalyticsTracker.assign_tier(7_000, 1.0) == "B"

    def test_b_tier_by_engagement(self) -> None:
        assert AnalyticsTracker.assign_tier(500, 4.0) == "B"

    def test_c_tier(self) -> None:
        assert AnalyticsTracker.assign_tier(2_000, 1.0) == "C"

    def test_f_tier_zero_views(self) -> None:
        assert AnalyticsTracker.assign_tier(0, 0.0) == "F"

    def test_f_tier_exactly_1000_views(self) -> None:
        # 1000 is not > 1000 so should be F
        assert AnalyticsTracker.assign_tier(1_000, 0.0) == "F"


# ---------------------------------------------------------------------------
# Engagement calculations (pure functions)
# ---------------------------------------------------------------------------

class TestEngagementCalculations:
    def test_engagement_rate_formula(self) -> None:
        # (300+50+20) / 10000 * 100 = 3.7
        rate = AnalyticsTracker.compute_engagement_rate(300, 50, 20, 10_000)
        assert abs(rate - 3.7) < 0.001

    def test_virality_score_formula(self) -> None:
        # (20*3 + 50*2 + 300) / 10000 * 100 = (60+100+300)/10000*100 = 4.6
        score = AnalyticsTracker.compute_virality_score(300, 50, 20, 10_000)
        assert abs(score - 4.6) < 0.001

    def test_zero_views_engagement_returns_zero(self) -> None:
        assert AnalyticsTracker.compute_engagement_rate(100, 50, 20, 0) == 0.0

    def test_zero_views_virality_returns_zero(self) -> None:
        assert AnalyticsTracker.compute_virality_score(100, 50, 20, 0) == 0.0

    def test_engagement_all_shares(self) -> None:
        # 1000 shares / 10000 views * 100 = 10%
        rate = AnalyticsTracker.compute_engagement_rate(0, 0, 1_000, 10_000)
        assert abs(rate - 10.0) < 0.001

    def test_virality_weights_shares_highest(self) -> None:
        # shares=10 → contribution 30; likes=10 → contribution 10
        # so same count of shares vs likes should give higher virality
        v_shares = AnalyticsTracker.compute_virality_score(0, 0, 10, 1_000)
        v_likes  = AnalyticsTracker.compute_virality_score(10, 0, 0, 1_000)
        assert v_shares > v_likes


# ---------------------------------------------------------------------------
# _fetch_metrics (API is mocked via _build_analytics_service)
# ---------------------------------------------------------------------------

class TestFetchMetrics:
    def test_fetch_metrics_maps_response_columns(self, tmp_path) -> None:
        tracker = _make_tracker(tmp_path)
        mock_service = MagicMock()
        mock_service.reports.return_value.query.return_value.execute.return_value = {
            "rows": [[25000, 1500.5, 500, 100, 50, 100_000, 0.045, 200, 10]]
        }

        with patch.object(tracker, "_load_credentials", return_value=MagicMock()):
            with patch.object(tracker, "_build_analytics_service", return_value=mock_service):
                raw = tracker._fetch_metrics("vid001", "default")

        assert raw["views"] == 25_000
        assert abs(raw["watch_time_minutes"] - 1500.5) < 0.01
        assert raw["likes"] == 500
        assert raw["comments"] == 100
        assert raw["shares"] == 50
        assert raw["impressions"] == 100_000
        assert abs(raw["ctr"] - 0.045) < 0.001
        assert raw["subscribers_gained"] == 200
        assert raw["subscribers_lost"] == 10

    def test_empty_rows_returns_zeros(self, tmp_path) -> None:
        tracker = _make_tracker(tmp_path)
        mock_service = MagicMock()
        mock_service.reports.return_value.query.return_value.execute.return_value = {
            "rows": []
        }

        with patch.object(tracker, "_load_credentials", return_value=MagicMock()):
            with patch.object(tracker, "_build_analytics_service", return_value=mock_service):
                raw = tracker._fetch_metrics("vid001", "default")

        assert raw["views"] == 0
        assert raw["likes"] == 0

    def test_missing_rows_key_returns_zeros(self, tmp_path) -> None:
        tracker = _make_tracker(tmp_path)
        mock_service = MagicMock()
        mock_service.reports.return_value.query.return_value.execute.return_value = {}

        with patch.object(tracker, "_load_credentials", return_value=MagicMock()):
            with patch.object(tracker, "_build_analytics_service", return_value=mock_service):
                raw = tracker._fetch_metrics("vid001", "default")

        assert raw["views"] == 0


# ---------------------------------------------------------------------------
# track_video
# ---------------------------------------------------------------------------

class TestTrackVideo:
    def test_track_video_returns_metrics(self, tmp_path) -> None:
        tracker = _make_tracker(tmp_path)
        raw = _raw_metrics(views=25_000, likes=500, comments=100, shares=50)

        with patch.object(tracker, "_fetch_metrics", return_value=raw):
            metrics = tracker.track_video("vid001", "default")

        assert isinstance(metrics, VideoMetrics)
        assert metrics.video_id == "vid001"
        assert metrics.views == 25_000
        assert metrics.tier == "A"
        assert metrics.engagement_rate > 0

    def test_track_video_saves_to_db(self, tmp_path) -> None:
        tracker = _make_tracker(tmp_path)
        raw = _raw_metrics(views=60_000, likes=5_000, comments=500, shares=200)

        with patch.object(tracker, "_fetch_metrics", return_value=raw):
            tracker.track_video("vid001", "default")

        conn = sqlite3.connect(tmp_path / "test.db")
        rows = conn.execute("SELECT video_id, tier FROM video_metrics").fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0][0] == "vid001"
        assert rows[0][1] == "S"

    def test_track_video_api_failure_returns_default_metrics(self, tmp_path) -> None:
        tracker = _make_tracker(tmp_path)

        with patch.object(tracker, "_fetch_metrics", side_effect=Exception("API down")):
            metrics = tracker.track_video("vid_fail", "default")

        assert metrics.video_id == "vid_fail"
        assert metrics.views == 0
        assert metrics.tier == "F"

    def test_track_video_failure_still_saves_to_db(self, tmp_path) -> None:
        tracker = _make_tracker(tmp_path)

        with patch.object(tracker, "_fetch_metrics", side_effect=Exception("fail")):
            tracker.track_video("vid_fail", "default")

        conn = sqlite3.connect(tmp_path / "test.db")
        count = conn.execute("SELECT COUNT(*) FROM video_metrics").fetchone()[0]
        conn.close()
        assert count == 1


# ---------------------------------------------------------------------------
# track_all
# ---------------------------------------------------------------------------

class TestTrackAll:
    def test_track_all_calls_track_video_for_each_registered(self, tmp_path) -> None:
        tracker = _make_tracker(tmp_path)
        tracker.register_video("vid001", "default")
        tracker.register_video("vid002", "default")

        with patch.object(
            tracker, "_fetch_metrics", return_value=_raw_metrics()
        ) as mock_fetch:
            results = tracker.track_all("default")

        assert len(results) == 2
        assert mock_fetch.call_count == 2

    def test_track_all_empty_returns_empty_list(self, tmp_path) -> None:
        tracker = _make_tracker(tmp_path)
        results = tracker.track_all("default")
        assert results == []

    def test_track_all_only_returns_channel_videos(self, tmp_path) -> None:
        tracker = _make_tracker(tmp_path)
        tracker.register_video("vid001", "default")
        tracker.register_video("vid002", "career")

        with patch.object(tracker, "_fetch_metrics", return_value=_raw_metrics()):
            results = tracker.track_all("default")

        assert len(results) == 1


# ---------------------------------------------------------------------------
# register_video / DB persistence
# ---------------------------------------------------------------------------

class TestDatabasePersistence:
    def test_register_video_inserts_row(self, tmp_path) -> None:
        tracker = _make_tracker(tmp_path)
        tracker.register_video("vid001", "default", "stoic_001", "Stoic Title")

        conn = sqlite3.connect(tmp_path / "test.db")
        rows = conn.execute(
            "SELECT video_id, channel_key, topic_id, title FROM uploaded_videos"
        ).fetchall()
        conn.close()

        assert len(rows) == 1
        assert rows[0] == ("vid001", "default", "stoic_001", "Stoic Title")

    def test_register_video_updates_on_conflict(self, tmp_path) -> None:
        tracker = _make_tracker(tmp_path)
        tracker.register_video("vid001", "default", "t001", "Old Title")
        tracker.register_video("vid001", "career", "t002", "New Title")

        conn = sqlite3.connect(tmp_path / "test.db")
        count = conn.execute(
            "SELECT COUNT(*) FROM uploaded_videos WHERE video_id='vid001'"
        ).fetchone()[0]
        title = conn.execute(
            "SELECT title FROM uploaded_videos WHERE video_id='vid001'"
        ).fetchone()[0]
        conn.close()

        assert count == 1
        assert title == "New Title"

    def test_multiple_track_calls_insert_multiple_rows(self, tmp_path) -> None:
        tracker = _make_tracker(tmp_path)

        with patch.object(tracker, "_fetch_metrics", return_value=_raw_metrics()):
            tracker.track_video("vid001", "default")
            tracker.track_video("vid001", "default")  # same video, second tracking

        conn = sqlite3.connect(tmp_path / "test.db")
        count = conn.execute("SELECT COUNT(*) FROM video_metrics").fetchone()[0]
        conn.close()
        # Each track_video call inserts a new row (history, not upsert)
        assert count == 2

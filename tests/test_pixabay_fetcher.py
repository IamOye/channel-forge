"""
Tests for src/media/pixabay_fetcher.py

All httpx calls are mocked — no real network requests during tests.
"""

import json
from unittest.mock import MagicMock, patch, call

import pytest

from src.media.pixabay_fetcher import (
    KEYWORD_MAP,
    FetchResult,
    PixabayFetcher,
    PixabayVideo,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hit(
    vid_id: int = 1,
    duration: int = 15,
    url: str = "https://cdn.pixabay.com/video/sample.mp4",
    tags: str = "nature landscape",
) -> dict:
    """Build a minimal Pixabay API hit dict."""
    return {
        "id":      vid_id,
        "duration": duration,
        "pageURL": f"https://pixabay.com/videos/{vid_id}/",
        "tags":    tags,
        "videos": {
            "large":  {"url": url, "width": 1920, "height": 1080},
            "medium": {"url": url, "width": 1280, "height": 720},
            "small":  {"url": url, "width": 640,  "height": 360},
            "tiny":   {"url": url, "width": 320,  "height": 180},
        },
    }


def _mock_api_response(hits: list) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = {"hits": hits, "total": len(hits)}
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# PixabayVideo
# ---------------------------------------------------------------------------

class TestPixabayVideo:
    def test_to_dict_has_all_keys(self) -> None:
        v = PixabayVideo(
            pixabay_id=42,
            duration=15,
            width=1920,
            height=1080,
            download_url="https://example.com/vid.mp4",
            page_url="https://pixabay.com/videos/42/",
            tags="nature",
        )
        d = v.to_dict()
        for key in ("pixabay_id", "duration", "width", "height",
                    "download_url", "page_url", "tags"):
            assert key in d


# ---------------------------------------------------------------------------
# FetchResult
# ---------------------------------------------------------------------------

class TestFetchResult:
    def test_fetched_at_auto_set(self) -> None:
        r = FetchResult(topic_id="t1", video_path="", video_meta=None, is_valid=False)
        assert r.fetched_at != ""

    def test_to_dict_has_all_keys(self) -> None:
        r = FetchResult(topic_id="t1", video_path="", video_meta=None, is_valid=False)
        d = r.to_dict()
        for key in ("topic_id", "video_path", "video_meta", "is_valid",
                    "validation_errors", "fetched_at"):
            assert key in d

    def test_to_dict_with_video_meta(self) -> None:
        meta = PixabayVideo(1, 15, 1920, 1080, "url", "page", "tags")
        r = FetchResult(topic_id="t1", video_path="/data/raw/t1_stock.mp4",
                        video_meta=meta, is_valid=True)
        d = r.to_dict()
        assert d["video_meta"] is not None
        assert d["video_meta"]["pixabay_id"] == 1


# ---------------------------------------------------------------------------
# PixabayFetcher._best_url
# ---------------------------------------------------------------------------

class TestBestUrl:
    def test_prefers_large(self) -> None:
        videos = {
            "large":  {"url": "large_url"},
            "medium": {"url": "medium_url"},
        }
        assert PixabayFetcher._best_url(videos) == "large_url"

    def test_falls_back_to_medium(self) -> None:
        videos = {"medium": {"url": "medium_url"}}
        assert PixabayFetcher._best_url(videos) == "medium_url"

    def test_returns_empty_when_no_url(self) -> None:
        assert PixabayFetcher._best_url({}) == ""


# ---------------------------------------------------------------------------
# PixabayFetcher.fetch (fully mocked)
# ---------------------------------------------------------------------------

class TestPixabayFetcherFetch:
    @patch("src.media.pixabay_fetcher.httpx.get")
    @patch("src.media.pixabay_fetcher.PixabayFetcher._download_video")
    def test_returns_valid_result(self, mock_dl, mock_get) -> None:
        mock_get.return_value = _mock_api_response([_make_hit()])
        with patch("pathlib.Path.mkdir"):
            fetcher = PixabayFetcher(api_key="fake")
            result = fetcher.fetch(topic_id="test_001", category="success")

        assert isinstance(result, FetchResult)
        assert result.is_valid is True
        assert "test_001" in result.video_path
        assert result.video_path.endswith("_stock.mp4")
        assert result.video_meta is not None

    @patch("src.media.pixabay_fetcher.httpx.get")
    def test_returns_invalid_when_no_hits(self, mock_get) -> None:
        mock_get.return_value = _mock_api_response([])

        fetcher = PixabayFetcher(api_key="fake")
        result = fetcher.fetch(topic_id="no_hit", category="default")

        assert result.is_valid is False
        assert any("no suitable" in e for e in result.validation_errors)

    @patch("src.media.pixabay_fetcher.httpx.get")
    def test_skips_videos_shorter_than_min_duration(self, mock_get) -> None:
        hit_short = _make_hit(vid_id=1, duration=3)   # too short
        hit_ok    = _make_hit(vid_id=2, duration=20)  # ok
        mock_get.return_value = _mock_api_response([hit_short, hit_ok])

        with patch("src.media.pixabay_fetcher.PixabayFetcher._download_video"):
            with patch("pathlib.Path.mkdir"):
                fetcher = PixabayFetcher(api_key="fake", min_duration=10)
                result = fetcher.fetch(topic_id="dur_test", category="default")

        assert result.is_valid is True
        assert result.video_meta.pixabay_id == 2

    def test_raises_without_api_key(self) -> None:
        fetcher = PixabayFetcher(api_key="")
        with pytest.raises(ValueError, match="PIXABAY_API_KEY not set"):
            fetcher.fetch(topic_id="test")

    @patch("src.media.pixabay_fetcher.httpx.get")
    def test_api_error_returns_invalid(self, mock_get) -> None:
        import httpx
        mock_get.side_effect = httpx.RequestError("network error")

        fetcher = PixabayFetcher(api_key="fake")
        result = fetcher.fetch(topic_id="err_test")

        assert result.is_valid is False
        assert any("no suitable" in e for e in result.validation_errors)

    @patch("src.media.pixabay_fetcher.httpx.get")
    @patch("src.media.pixabay_fetcher.PixabayFetcher._download_video")
    def test_uses_keyword_map_for_category(self, mock_dl, mock_get) -> None:
        mock_get.return_value = _mock_api_response([_make_hit()])
        with patch("pathlib.Path.mkdir"):
            fetcher = PixabayFetcher(api_key="fake")
            fetcher.fetch(topic_id="kw_test", category="money")

        # The query param should contain one of the money keywords
        call_kwargs = mock_get.call_args[1]
        query_used = call_kwargs["params"]["q"]
        expected_keywords = KEYWORD_MAP["money"]
        assert any(phrase == query_used for phrase in expected_keywords)

    @patch("src.media.pixabay_fetcher.httpx.get")
    @patch("src.media.pixabay_fetcher.PixabayFetcher._download_video")
    def test_unknown_category_falls_back_to_default(self, mock_dl, mock_get) -> None:
        mock_get.return_value = _mock_api_response([_make_hit()])
        with patch("pathlib.Path.mkdir"):
            fetcher = PixabayFetcher(api_key="fake")
            result = fetcher.fetch(topic_id="def_test", category="nonexistent")

        assert result.is_valid is True

    @patch("src.media.pixabay_fetcher.httpx.get")
    @patch("src.media.pixabay_fetcher.PixabayFetcher._download_video")
    def test_to_dict_is_serialisable(self, mock_dl, mock_get) -> None:
        import json as _json
        mock_get.return_value = _mock_api_response([_make_hit()])
        with patch("pathlib.Path.mkdir"):
            fetcher = PixabayFetcher(api_key="fake")
            result = fetcher.fetch(topic_id="serial_001")

        assert len(_json.dumps(result.to_dict())) > 10

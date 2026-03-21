"""
Tests for src/media/unsplash_fetcher.py

All httpx calls are mocked — no real network requests.
"""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from src.media.unsplash_fetcher import (
    UnsplashFetcher,
    _clip_already_used,
    _clip_history_record,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db(tmp_path: Path) -> Path:
    """Create a minimal DB with the clip_history table."""
    db = tmp_path / "test.db"
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE clip_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            clip_id TEXT NOT NULL,
            source TEXT NOT NULL,
            query TEXT,
            topic_id TEXT,
            used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    return db


def _make_photo(
    photo_id: str = "abc123",
    width: int = 3000,
    height: int = 4000,
    download_location: str = "https://api.unsplash.com/photos/abc123/download",
) -> dict:
    return {
        "id": photo_id,
        "width": width,
        "height": height,
        "alt_description": "beautiful portrait",
        "urls": {
            "raw": "https://images.unsplash.com/photo-abc123",
        },
        "links": {
            "download_location": download_location,
        },
    }


def _mock_search_response(photos: list) -> MagicMock:
    m = MagicMock()
    m.raise_for_status = MagicMock()
    m.json.return_value = {"results": photos, "total": len(photos)}
    return m




# ---------------------------------------------------------------------------
# clip_history helpers
# ---------------------------------------------------------------------------

class TestClipHistoryHelpers:
    def test_clip_already_used_false_on_empty_db(self, tmp_path) -> None:
        db = _make_db(tmp_path)
        assert _clip_already_used(db, "unsplash", "abc") is False

    def test_clip_already_used_true_after_record(self, tmp_path) -> None:
        db = _make_db(tmp_path)
        _clip_history_record(db, "unsplash", "abc", "query", "t1")
        assert _clip_already_used(db, "unsplash", "abc") is True

    def test_source_scoped(self, tmp_path) -> None:
        db = _make_db(tmp_path)
        _clip_history_record(db, "pexels", "abc", "query", "t1")
        assert _clip_already_used(db, "unsplash", "abc") is False


# ---------------------------------------------------------------------------
# UnsplashFetcher.fetch_photos
# ---------------------------------------------------------------------------

class TestUnsplashFetcherFetchPhotos:
    def test_returns_empty_without_access_key(self, tmp_path) -> None:
        fetcher = UnsplashFetcher(access_key="", output_dir=tmp_path)
        assert fetcher.fetch_photos("finance") == []

    @patch("src.media.unsplash_fetcher.httpx.get")
    @patch("src.media.unsplash_fetcher.UnsplashFetcher._download_file", return_value=True)
    def test_downloads_portrait_photo(self, mock_dl, mock_get, tmp_path) -> None:
        mock_get.return_value = _mock_search_response([_make_photo()])

        fetcher = UnsplashFetcher(access_key="fake", output_dir=tmp_path)
        results = fetcher.fetch_photos("finance wealth", topic_id="t1")

        assert len(results) == 1
        assert results[0]["width"] == 1080
        assert results[0]["height"] == 1920
        assert "unsplash_0" in results[0]["local_path"]

    @patch("src.media.unsplash_fetcher.httpx.get")
    @patch("src.media.unsplash_fetcher.UnsplashFetcher._download_file", return_value=True)
    def test_download_location_called(self, mock_dl, mock_get, tmp_path) -> None:
        """Unsplash requires calling download_location for every used photo."""
        photo = _make_photo(download_location="https://api.unsplash.com/photos/abc/download")

        # First call = search, second call = download_location tracking
        mock_get.side_effect = [
            _mock_search_response([photo]),
            MagicMock(raise_for_status=MagicMock()),  # download_location response
        ]

        fetcher = UnsplashFetcher(access_key="fake", output_dir=tmp_path)
        fetcher.fetch_photos("finance", topic_id="t1")

        # Verify download_location was called (second httpx.get call)
        calls = mock_get.call_args_list
        assert len(calls) == 2
        assert "download" in calls[1][0][0]

    @patch("src.media.unsplash_fetcher.httpx.get")
    @patch("src.media.unsplash_fetcher.UnsplashFetcher._download_file", return_value=True)
    def test_clip_id_recorded_in_clip_history(self, mock_dl, mock_get, tmp_path) -> None:
        db = _make_db(tmp_path)
        photo = _make_photo(photo_id="xyz789")
        mock_get.side_effect = [
            _mock_search_response([photo]),
            MagicMock(raise_for_status=MagicMock()),
        ]

        fetcher = UnsplashFetcher(access_key="fake", output_dir=tmp_path)
        fetcher.fetch_photos("finance", topic_id="t1", db_path=db)

        assert _clip_already_used(db, "unsplash", "xyz789") is True

    @patch("src.media.unsplash_fetcher.httpx.get")
    def test_already_used_clip_skipped(self, mock_get, tmp_path) -> None:
        db = _make_db(tmp_path)
        _clip_history_record(db, "unsplash", "used_id", "q", "old")
        mock_get.return_value = _mock_search_response([_make_photo(photo_id="used_id")])

        fetcher = UnsplashFetcher(access_key="fake", output_dir=tmp_path)
        results = fetcher.fetch_photos("finance", db_path=db)

        assert results == []

    @patch("src.media.unsplash_fetcher.httpx.get")
    def test_rejects_landscape_photo(self, mock_get, tmp_path) -> None:
        """width >= height → rejected (landscape/square)."""
        photo = _make_photo(width=4000, height=3000)  # landscape
        mock_get.return_value = _mock_search_response([photo])
        fetcher = UnsplashFetcher(access_key="fake", output_dir=tmp_path)
        assert fetcher.fetch_photos("finance") == []

    @patch("src.media.unsplash_fetcher.httpx.get")
    def test_rejects_square_photo(self, mock_get, tmp_path) -> None:
        photo = _make_photo(width=2000, height=2000)
        mock_get.return_value = _mock_search_response([photo])
        fetcher = UnsplashFetcher(access_key="fake", output_dir=tmp_path)
        assert fetcher.fetch_photos("finance") == []

    @patch("src.media.unsplash_fetcher.httpx.get")
    def test_rejects_below_min_width(self, mock_get, tmp_path) -> None:
        photo = _make_photo(width=800, height=1200)
        mock_get.return_value = _mock_search_response([photo])
        fetcher = UnsplashFetcher(access_key="fake", output_dir=tmp_path)
        assert fetcher.fetch_photos("finance", min_width=1080) == []

    @patch("src.media.unsplash_fetcher.httpx.get")
    def test_returns_empty_on_api_error(self, mock_get, tmp_path) -> None:
        import httpx
        mock_get.side_effect = httpx.RequestError("network error")
        fetcher = UnsplashFetcher(access_key="fake", output_dir=tmp_path)
        assert fetcher.fetch_photos("finance") == []

    @patch("src.media.unsplash_fetcher.httpx.get")
    @patch("src.media.unsplash_fetcher.UnsplashFetcher._download_file", return_value=True)
    def test_download_location_failure_is_not_fatal(self, mock_dl, mock_get, tmp_path) -> None:
        """If download_location GET fails, fetch_photos should still succeed."""
        import httpx as _httpx
        photo = _make_photo()
        mock_get.side_effect = [
            _mock_search_response([photo]),
            _httpx.RequestError("tracking failed"),  # download_location call fails
        ]

        fetcher = UnsplashFetcher(access_key="fake", output_dir=tmp_path)
        results = fetcher.fetch_photos("finance", topic_id="t1")

        assert len(results) == 1  # photo still downloaded despite tracking failure

    @patch("src.media.unsplash_fetcher.httpx.get")
    @patch("src.media.unsplash_fetcher.UnsplashFetcher._download_file", return_value=True)
    def test_respects_count(self, mock_dl, mock_get, tmp_path) -> None:
        photos = [_make_photo(photo_id=f"p{i}") for i in range(5)]
        mock_get.side_effect = lambda *a, **kw: (
            _mock_search_response(photos)
            if "search" in a[0]
            else MagicMock(raise_for_status=MagicMock())
        )

        fetcher = UnsplashFetcher(access_key="fake", output_dir=tmp_path)
        results = fetcher.fetch_photos("finance", topic_id="t1", count=2)

        assert len(results) <= 2

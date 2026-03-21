"""
Tests for src/media/pexels_fetcher.py

All httpx calls are mocked — no real network requests.
"""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.media.pexels_fetcher import (
    MAX_PHOTO_PORTRAIT_RATIO,
    PexelsFetcher,
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


def _make_video_response(
    vid_id: int = 1,
    width: int = 1080,
    height: int = 1920,
    duration: int = 15,
) -> dict:
    """Build a minimal Pexels video API hit."""
    return {
        "id": vid_id,
        "duration": duration,
        "tags": [],
        "video_files": [
            {"link": f"https://cdn.pexels.com/v{vid_id}.mp4", "width": width, "height": height},
        ],
    }


def _mock_api(hits: list) -> MagicMock:
    m = MagicMock()
    m.raise_for_status = MagicMock()
    m.json.return_value = {"videos": hits, "total_results": len(hits)}
    return m


def _mock_photo_api(hits: list) -> MagicMock:
    m = MagicMock()
    m.raise_for_status = MagicMock()
    m.json.return_value = {"photos": hits, "total_results": len(hits)}
    return m




# ---------------------------------------------------------------------------
# clip_history helpers
# ---------------------------------------------------------------------------

class TestClipHistoryHelpers:
    def test_clip_already_used_false_on_empty_db(self, tmp_path) -> None:
        db = _make_db(tmp_path)
        assert _clip_already_used(db, "pexels", "999") is False

    def test_clip_already_used_true_after_record(self, tmp_path) -> None:
        db = _make_db(tmp_path)
        _clip_history_record(db, "pexels", "42", "query", "topic1")
        assert _clip_already_used(db, "pexels", "42") is True

    def test_clip_already_used_source_scoped(self, tmp_path) -> None:
        """Same clip_id from different source is not considered used."""
        db = _make_db(tmp_path)
        _clip_history_record(db, "pixabay", "42", "query", "topic1")
        assert _clip_already_used(db, "pexels", "42") is False

    def test_clip_history_record_swallows_error_when_no_db(self, tmp_path) -> None:
        """Should not raise when DB doesn't exist."""
        _clip_history_record(tmp_path / "nonexistent.db", "pexels", "1", "q", "t")

    def test_clip_already_used_false_when_no_db(self, tmp_path) -> None:
        assert _clip_already_used(tmp_path / "nonexistent.db", "pexels", "1") is False

    def test_clip_already_used_false_when_none(self) -> None:
        assert _clip_already_used(None, "pexels", "1") is False


# ---------------------------------------------------------------------------
# PexelsFetcher.fetch_clips
# ---------------------------------------------------------------------------

class TestPexelsFetcherFetchClips:
    def test_returns_empty_without_api_key(self, tmp_path) -> None:
        fetcher = PexelsFetcher(api_key="", output_dir=tmp_path)
        assert fetcher.fetch_clips("finance") == []

    @patch("src.media.pexels_fetcher.httpx.get")
    @patch("src.media.pexels_fetcher.PexelsFetcher._download_file", return_value=True)
    def test_downloads_portrait_clip(self, mock_dl, mock_get, tmp_path) -> None:
        mock_get.return_value = _mock_api([_make_video_response(vid_id=10)])

        fetcher = PexelsFetcher(api_key="fake", output_dir=tmp_path, anthropic_api_key="")
        paths = fetcher.fetch_clips("finance", topic_id="t1")

        assert len(paths) == 1
        assert paths[0].endswith(".mp4")
        assert "t1_pexels_0" in paths[0]

    @patch("src.media.pexels_fetcher.httpx.get")
    @patch("src.media.pexels_fetcher.PexelsFetcher._download_file", return_value=True)
    def test_clip_id_recorded_in_clip_history(self, mock_dl, mock_get, tmp_path) -> None:
        db = _make_db(tmp_path)
        mock_get.return_value = _mock_api([_make_video_response(vid_id=77)])

        fetcher = PexelsFetcher(api_key="fake", output_dir=tmp_path, anthropic_api_key="")
        fetcher.fetch_clips("finance", topic_id="t1", db_path=db)

        assert _clip_already_used(db, "pexels", "77") is True

    @patch("src.media.pexels_fetcher.httpx.get")
    def test_already_used_clip_skipped(self, mock_get, tmp_path) -> None:
        db = _make_db(tmp_path)
        _clip_history_record(db, "pexels", "99", "q", "old_topic")
        mock_get.return_value = _mock_api([_make_video_response(vid_id=99)])

        fetcher = PexelsFetcher(api_key="fake", output_dir=tmp_path, anthropic_api_key="")
        paths = fetcher.fetch_clips("finance", topic_id="t2", db_path=db)

        assert paths == []

    @patch("src.media.pexels_fetcher.httpx.get")
    def test_returns_empty_on_api_error(self, mock_get, tmp_path) -> None:
        import httpx
        mock_get.side_effect = httpx.RequestError("network error")
        fetcher = PexelsFetcher(api_key="fake", output_dir=tmp_path, anthropic_api_key="")
        assert fetcher.fetch_clips("finance") == []

    @patch("src.media.pexels_fetcher.httpx.get")
    def test_rejects_landscape_clip(self, mock_get, tmp_path) -> None:
        """Landscape clip (width > height) must be rejected."""
        hit = _make_video_response(vid_id=5, width=1920, height=1080)
        mock_get.return_value = _mock_api([hit])
        fetcher = PexelsFetcher(api_key="fake", output_dir=tmp_path, anthropic_api_key="")
        assert fetcher.fetch_clips("finance") == []

    @patch("src.media.pexels_fetcher.httpx.get")
    def test_rejects_clip_below_min_width(self, mock_get, tmp_path) -> None:
        """Clip narrower than min_width must be rejected."""
        hit = _make_video_response(vid_id=6, width=720, height=1280)
        mock_get.return_value = _mock_api([hit])
        fetcher = PexelsFetcher(api_key="fake", output_dir=tmp_path)
        assert fetcher.fetch_clips("finance", min_width=1080) == []

    @patch("src.media.pexels_fetcher.httpx.get")
    @patch("src.media.pexels_fetcher.PexelsFetcher._download_file", return_value=True)
    def test_respects_max_clips(self, mock_dl, mock_get, tmp_path) -> None:
        hits = [_make_video_response(vid_id=i) for i in range(1, 6)]
        mock_get.return_value = _mock_api(hits)

        fetcher = PexelsFetcher(api_key="fake", output_dir=tmp_path, anthropic_api_key="")
        paths = fetcher.fetch_clips("finance", max_clips=2, topic_id="t1")

        assert len(paths) <= 2


# ---------------------------------------------------------------------------
# PexelsFetcher._best_video_file
# ---------------------------------------------------------------------------

class TestBestVideoFile:
    def test_returns_highest_width_portrait_file(self) -> None:
        files = [
            {"link": "a.mp4", "width": 1080, "height": 1920},
            {"link": "b.mp4", "width": 720,  "height": 1280},
        ]
        result = PexelsFetcher._best_video_file(files, min_width=720)
        assert result["link"] == "a.mp4"

    def test_returns_none_when_no_portrait(self) -> None:
        files = [{"link": "a.mp4", "width": 1920, "height": 1080}]
        assert PexelsFetcher._best_video_file(files, min_width=1080) is None

    def test_returns_none_below_min_width(self) -> None:
        files = [{"link": "a.mp4", "width": 720, "height": 1280}]
        assert PexelsFetcher._best_video_file(files, min_width=1080) is None


# ---------------------------------------------------------------------------
# PexelsFetcher.fetch_photos
# ---------------------------------------------------------------------------

class TestPexelsFetcherFetchPhotos:
    def test_returns_empty_without_api_key(self, tmp_path) -> None:
        fetcher = PexelsFetcher(api_key="", output_dir=tmp_path)
        assert fetcher.fetch_photos("finance") == []

    @patch("src.media.pexels_fetcher.httpx.get")
    def test_rejects_landscape_photo(self, mock_get, tmp_path) -> None:
        photo = {"id": 1, "width": 1920, "height": 1080, "alt": "", "src": {"original": "url"}}
        mock_get.return_value = _mock_photo_api([photo])
        fetcher = PexelsFetcher(api_key="fake", output_dir=tmp_path)
        assert fetcher.fetch_photos("finance") == []

    @patch("src.media.pexels_fetcher.httpx.get")
    def test_rejects_near_square_photo(self, mock_get, tmp_path) -> None:
        photo = {"id": 2, "width": 1000, "height": 1200, "alt": "", "src": {"original": "url"}}
        # ratio = 0.833 >= MAX_PHOTO_PORTRAIT_RATIO (0.65) → rejected
        mock_get.return_value = _mock_photo_api([photo])
        fetcher = PexelsFetcher(api_key="fake", output_dir=tmp_path)
        assert fetcher.fetch_photos("finance") == []

    @patch("src.media.pexels_fetcher.httpx.get")
    def test_returns_empty_on_api_error(self, mock_get, tmp_path) -> None:
        import httpx
        mock_get.side_effect = httpx.RequestError("error")
        fetcher = PexelsFetcher(api_key="fake", output_dir=tmp_path)
        assert fetcher.fetch_photos("finance") == []

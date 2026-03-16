"""
Tests for src/publisher/youtube_uploader.py

All Google API client calls are mocked — no real OAuth or network activity.
"""

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

from src.publisher.youtube_uploader import (
    CHUNK_SIZE,
    DEFAULT_CATEGORY_ID,
    MAX_RETRIES,
    QUOTA_UNITS,
    QuotaTracker,
    UploadResult,
    YouTubeUploader,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

METADATA = {
    "title":       "Stoic Secret Most People Ignore",
    "description": "Ancient stoic wisdom. Comment below 👇",
    "tags":        ["#Shorts", "#Stoicism"],
    "category_id": "22",
}

CREDENTIALS_JSON = {
    "token":         "fake_token",
    "refresh_token": "fake_refresh",
    "token_uri":     "https://oauth2.googleapis.com/token",
    "client_id":     "fake_client_id",
    "client_secret": "fake_secret",
    "scopes":        ["https://www.googleapis.com/auth/youtube.upload"],
}


def _make_uploader(tmp_path: Path) -> YouTubeUploader:
    """Return an uploader pointed at tmp_path as credentials_dir."""
    return YouTubeUploader(channel_key="test", credentials_dir=tmp_path)


def _write_credentials(tmp_path: Path, data: dict = None) -> None:
    """Write a fake token JSON file to tmp_path/test_token.json."""
    cred_file = tmp_path / "test_token.json"
    cred_file.write_text(json.dumps(data or CREDENTIALS_JSON))


def _make_mock_service(video_id: str = "abc123") -> MagicMock:
    """Return a mock YouTube service whose videos.insert returns the given video ID."""
    service = MagicMock()
    insert_request = MagicMock()
    insert_request.next_chunk.return_value = (None, {"id": video_id})
    service.videos.return_value.insert.return_value = insert_request
    return service


# ---------------------------------------------------------------------------
# UploadResult
# ---------------------------------------------------------------------------

class TestUploadResult:
    def _make(self, **kw) -> UploadResult:
        defaults = dict(
            topic_id="t001",
            youtube_video_id="abc123",
            youtube_url="https://www.youtube.com/watch?v=abc123",
            title="Test Title",
            is_valid=True,
        )
        defaults.update(kw)
        return UploadResult(**defaults)

    def test_uploaded_at_auto_set(self) -> None:
        r = self._make()
        assert r.uploaded_at != ""

    def test_to_dict_has_all_keys(self) -> None:
        r = self._make()
        d = r.to_dict()
        for key in ("topic_id", "youtube_video_id", "youtube_url", "title",
                    "is_valid", "validation_errors", "uploaded_at", "publish_at"):
            assert key in d

    def test_invalid_result_has_errors(self) -> None:
        r = self._make(is_valid=False, validation_errors=["video not found"])
        assert not r.is_valid
        assert len(r.validation_errors) == 1

    def test_publish_at_stored(self) -> None:
        r = self._make(publish_at="2025-01-01T08:00:00+00:00")
        assert r.to_dict()["publish_at"] == "2025-01-01T08:00:00+00:00"


# ---------------------------------------------------------------------------
# YouTubeUploader._validate_inputs
# ---------------------------------------------------------------------------

class TestValidateInputs:
    def test_valid_file_and_title(self, tmp_path) -> None:
        video = tmp_path / "out.mp4"
        video.write_bytes(b"v" * 100)
        errors = YouTubeUploader._validate_inputs(video, METADATA)
        assert errors == []

    def test_missing_video_file(self, tmp_path) -> None:
        video = tmp_path / "missing.mp4"
        errors = YouTubeUploader._validate_inputs(video, METADATA)
        assert any("video file not found" in e for e in errors)

    def test_empty_title(self, tmp_path) -> None:
        video = tmp_path / "out.mp4"
        video.write_bytes(b"v")
        errors = YouTubeUploader._validate_inputs(video, {"title": "  "})
        assert any("title is empty" in e for e in errors)

    def test_both_missing_returns_two_errors(self, tmp_path) -> None:
        video = tmp_path / "missing.mp4"
        errors = YouTubeUploader._validate_inputs(video, {"title": ""})
        assert len(errors) == 2


# ---------------------------------------------------------------------------
# YouTubeUploader._build_body
# ---------------------------------------------------------------------------

class TestBuildBody:
    def setup_method(self) -> None:
        self.uploader = YouTubeUploader()

    def test_public_when_no_publish_at(self) -> None:
        body = self.uploader._build_body(METADATA, None)
        assert body["status"]["privacyStatus"] == "public"
        assert "publishAt" not in body["status"]

    def test_private_and_scheduled_when_publish_at_set(self) -> None:
        body = self.uploader._build_body(METADATA, "2025-06-01T08:00:00+01:00")
        assert body["status"]["privacyStatus"] == "private"
        assert body["status"]["publishAt"] == "2025-06-01T08:00:00+01:00"

    def test_snippet_fields_populated(self) -> None:
        body = self.uploader._build_body(METADATA, None)
        assert body["snippet"]["title"] == METADATA["title"]
        assert body["snippet"]["description"] == METADATA["description"]
        assert body["snippet"]["tags"] == METADATA["tags"]
        assert body["snippet"]["categoryId"] == "22"

    def test_default_category_id_used_when_missing(self) -> None:
        body = self.uploader._build_body({"title": "T", "description": "D"}, None)
        assert body["snippet"]["categoryId"] == DEFAULT_CATEGORY_ID


# ---------------------------------------------------------------------------
# YouTubeUploader.upload (mocked)
# ---------------------------------------------------------------------------

class TestUpload:
    def test_returns_valid_result(self, tmp_path) -> None:
        video = tmp_path / "out.mp4"
        video.write_bytes(b"v" * 100)
        _write_credentials(tmp_path)

        uploader = _make_uploader(tmp_path)
        mock_service = _make_mock_service("vid001")

        with patch.object(uploader, "_load_credentials", return_value=MagicMock()):
            with patch.object(uploader, "_build_service", return_value=mock_service):
                with patch.object(uploader, "_execute_upload", return_value="vid001"):
                    result = uploader.upload("t001", video, METADATA)

        assert isinstance(result, UploadResult)
        assert result.is_valid is True
        assert result.youtube_video_id == "vid001"
        assert "vid001" in result.youtube_url
        assert result.topic_id == "t001"

    def test_returns_invalid_when_video_missing(self, tmp_path) -> None:
        video = tmp_path / "missing.mp4"
        uploader = _make_uploader(tmp_path)

        result = uploader.upload("t002", video, METADATA)

        assert result.is_valid is False
        assert any("video file not found" in e for e in result.validation_errors)

    def test_returns_invalid_when_title_empty(self, tmp_path) -> None:
        video = tmp_path / "out.mp4"
        video.write_bytes(b"v")
        uploader = _make_uploader(tmp_path)

        result = uploader.upload("t003", video, {"title": "", "description": "d"})

        assert result.is_valid is False
        assert any("title is empty" in e for e in result.validation_errors)

    def test_returns_invalid_on_missing_credentials(self, tmp_path) -> None:
        video = tmp_path / "out.mp4"
        video.write_bytes(b"v" * 100)
        # No credentials file written
        uploader = _make_uploader(tmp_path)

        result = uploader.upload("t004", video, METADATA)

        assert result.is_valid is False
        assert len(result.validation_errors) > 0

    def test_publish_at_forwarded_to_result(self, tmp_path) -> None:
        video = tmp_path / "out.mp4"
        video.write_bytes(b"v" * 100)
        uploader = _make_uploader(tmp_path)
        mock_service = _make_mock_service("vid002")

        with patch.object(uploader, "_load_credentials", return_value=MagicMock()):
            with patch.object(uploader, "_build_service", return_value=mock_service):
                with patch.object(uploader, "_execute_upload", return_value="vid002"):
                    result = uploader.upload(
                        "t005", video, METADATA,
                        publish_at="2025-01-01T08:00:00+00:00",
                    )

        assert result.publish_at == "2025-01-01T08:00:00+00:00"

    def test_exception_during_upload_returns_invalid(self, tmp_path) -> None:
        video = tmp_path / "out.mp4"
        video.write_bytes(b"v" * 100)
        uploader = _make_uploader(tmp_path)

        with patch.object(uploader, "_load_credentials", side_effect=Exception("network error")):
            result = uploader.upload("t006", video, METADATA)

        assert result.is_valid is False
        assert any("network error" in e for e in result.validation_errors)

    def test_to_dict_is_serialisable(self, tmp_path) -> None:
        import json as _json
        video = tmp_path / "out.mp4"
        video.write_bytes(b"v" * 100)
        uploader = _make_uploader(tmp_path)
        mock_service = _make_mock_service("vid003")

        with patch.object(uploader, "_load_credentials", return_value=MagicMock()):
            with patch.object(uploader, "_build_service", return_value=mock_service):
                with patch.object(uploader, "_execute_upload", return_value="vid003"):
                    result = uploader.upload("serial_001", video, METADATA)

        assert len(_json.dumps(result.to_dict())) > 10


# ---------------------------------------------------------------------------
# YouTubeUploader._execute_upload — quota / transient errors
# ---------------------------------------------------------------------------

class TestExecuteUpload:
    """
    These tests use a real Exception subclass (FakeHttpError) that carries a
    .resp.status attribute, matching the googleapiclient HttpError interface.
    The googleapiclient modules are patched in sys.modules so the lazy imports
    inside _execute_upload pick up the fakes.
    """

    def _fake_http_error_class(self):
        """Return a real Exception subclass that mimics googleapiclient.errors.HttpError."""
        class FakeResp:
            def __init__(self, status: int) -> None:
                self.status = status

        class FakeHttpError(Exception):
            def __init__(self, status: int, msg: str = "") -> None:
                super().__init__(msg)
                self.resp = FakeResp(status)

        return FakeHttpError

    def _make_google_mocks(self, FakeHttpError):
        """Return (google_mock, google_errors, google_http) with HttpError wired up."""
        google_errors = MagicMock()
        google_errors.HttpError = FakeHttpError

        google_http = MagicMock()
        google_http.MediaFileUpload.return_value = MagicMock()

        google_mock = MagicMock()
        return google_mock, google_errors, google_http

    def test_retries_on_transient_error_then_succeeds(self, tmp_path) -> None:
        video = tmp_path / "out.mp4"
        video.write_bytes(b"v" * 100)

        FakeHttpError = self._fake_http_error_class()
        google_mock, google_errors, google_http = self._make_google_mocks(FakeHttpError)

        call_count = {"n": 0}

        def next_chunk_side_effect():
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise FakeHttpError(503, "server error")
            return (None, {"id": "retry_ok"})

        insert_request = MagicMock()
        insert_request.next_chunk.side_effect = next_chunk_side_effect
        service = MagicMock()
        service.videos.return_value.insert.return_value = insert_request

        uploader = YouTubeUploader()

        with patch.dict("sys.modules", {
            "googleapiclient":        google_mock,
            "googleapiclient.errors": google_errors,
            "googleapiclient.http":   google_http,
        }):
            with patch("time.sleep"):   # don't actually sleep during tests
                video_id = uploader._execute_upload(service, {}, video)

        assert video_id == "retry_ok"
        assert call_count["n"] == 2   # called twice: fail then succeed

    def test_quota_error_raises_immediately(self, tmp_path) -> None:
        video = tmp_path / "out.mp4"
        video.write_bytes(b"v" * 100)

        FakeHttpError = self._fake_http_error_class()
        google_mock, google_errors, google_http = self._make_google_mocks(FakeHttpError)

        insert_request = MagicMock()
        insert_request.next_chunk.side_effect = FakeHttpError(403, "quota exceeded")
        service = MagicMock()
        service.videos.return_value.insert.return_value = insert_request

        uploader = YouTubeUploader()

        with patch.dict("sys.modules", {
            "googleapiclient":        google_mock,
            "googleapiclient.errors": google_errors,
            "googleapiclient.http":   google_http,
        }):
            with pytest.raises(FakeHttpError):
                uploader._execute_upload(service, {}, video)


# ---------------------------------------------------------------------------
# QuotaTracker helpers
# ---------------------------------------------------------------------------

_QUOTA_DDL = """
CREATE TABLE IF NOT EXISTS youtube_quota_usage (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    date             TEXT    NOT NULL,
    operation        TEXT    NOT NULL,
    units_used       INTEGER NOT NULL,
    cumulative_daily INTEGER NOT NULL,
    created_at       TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""


def _make_quota_db(tmp_path: Path) -> Path:
    """Create a minimal DB with the youtube_quota_usage table."""
    db = tmp_path / "cf.db"
    conn = sqlite3.connect(db)
    conn.executescript(_QUOTA_DDL)
    conn.commit()
    conn.close()
    return db


# ---------------------------------------------------------------------------
# QuotaTracker unit tests
# ---------------------------------------------------------------------------

class TestQuotaTracker:
    def test_get_daily_usage_no_db_returns_zero(self, tmp_path) -> None:
        qt = QuotaTracker(db_path=tmp_path / "missing.db", daily_limit=10_000)
        assert qt.get_daily_usage() == 0

    def test_get_daily_usage_empty_table_returns_zero(self, tmp_path) -> None:
        db = _make_quota_db(tmp_path)
        qt = QuotaTracker(db_path=db, daily_limit=10_000)
        assert qt.get_daily_usage() == 0

    def test_record_inserts_row_and_returns_cumulative(self, tmp_path) -> None:
        db = _make_quota_db(tmp_path)
        qt = QuotaTracker(db_path=db, daily_limit=10_000)

        result = qt.record("video_upload", 1600)
        assert result == 1600
        assert qt.get_daily_usage() == 1600

    def test_record_accumulates_across_calls(self, tmp_path) -> None:
        db = _make_quota_db(tmp_path)
        qt = QuotaTracker(db_path=db, daily_limit=10_000)

        qt.record("video_upload", 1600)
        qt.record("thumbnail_upload", 50)
        assert qt.get_daily_usage() == 1650

    def test_can_upload_true_when_under_limit(self, tmp_path) -> None:
        db = _make_quota_db(tmp_path)
        qt = QuotaTracker(db_path=db, daily_limit=10_000)
        qt.record("video_upload", 1600)
        assert qt.can_upload() is True

    def test_can_upload_false_when_at_limit(self, tmp_path) -> None:
        db = _make_quota_db(tmp_path)
        qt = QuotaTracker(db_path=db, daily_limit=1600)
        qt.record("video_upload", 1600)
        assert qt.can_upload() is False

    def test_units_remaining_decreases_after_record(self, tmp_path) -> None:
        db = _make_quota_db(tmp_path)
        qt = QuotaTracker(db_path=db, daily_limit=10_000)
        qt.record("video_upload", 1600)
        assert qt.units_remaining() == 8_400

    def test_units_remaining_never_negative(self, tmp_path) -> None:
        db = _make_quota_db(tmp_path)
        qt = QuotaTracker(db_path=db, daily_limit=100)
        qt.record("video_upload", 1600)
        assert qt.units_remaining() == 0

    def test_warning_logged_at_80_percent(self, tmp_path) -> None:
        db = _make_quota_db(tmp_path)
        qt = QuotaTracker(db_path=db, daily_limit=10_000)
        # 8 000 units = exactly 80 %
        with patch("src.publisher.youtube_uploader.logger") as mock_log:
            qt.record("video_upload", 8_000)
            # warning should be called (80 % threshold)
            warning_msgs = [str(c) for c in mock_log.warning.call_args_list]
            assert any("80%" in m for m in warning_msgs)

    def test_critical_logged_at_95_percent(self, tmp_path) -> None:
        db = _make_quota_db(tmp_path)
        qt = QuotaTracker(db_path=db, daily_limit=10_000)
        # 9 500 units = exactly 95 %
        with patch("src.publisher.youtube_uploader.logger") as mock_log:
            qt.record("video_upload", 9_500)
            critical_msgs = [str(c) for c in mock_log.critical.call_args_list]
            assert any("critical" in m.lower() for m in critical_msgs)

    def test_record_missing_db_logs_warning_and_returns(self, tmp_path) -> None:
        qt = QuotaTracker(db_path=tmp_path / "missing.db", daily_limit=10_000)
        # Should not raise; returns 0 + units
        result = qt.record("video_upload", 1600)
        assert result == 1600

    def test_daily_limit_from_env(self, tmp_path) -> None:
        with patch.dict("os.environ", {"YOUTUBE_DAILY_QUOTA_LIMIT": "5000"}):
            qt = QuotaTracker(db_path=tmp_path / "x.db")
            assert qt.daily_limit == 5000


# ---------------------------------------------------------------------------
# YouTubeUploader quota integration tests
# ---------------------------------------------------------------------------

class TestUploadQuota:
    def test_upload_blocked_when_quota_exhausted(self, tmp_path) -> None:
        """When can_upload() is False the upload is skipped and queued."""
        video = tmp_path / "out.mp4"
        video.write_bytes(b"v" * 100)

        db = _make_quota_db(tmp_path)
        qt = QuotaTracker(db_path=db, daily_limit=0)   # limit=0 → always exhausted

        uploader = YouTubeUploader(
            channel_key="test",
            credentials_dir=tmp_path,
            quota_tracker=qt,
        )

        queue_dir = tmp_path / "queue"
        with patch("src.publisher.youtube_uploader._QUOTA_QUEUE_DIR", queue_dir):
            result = uploader.upload("q001", video, METADATA)

        assert result.is_valid is False
        assert any("quota exceeded" in e for e in result.validation_errors)
        # Queue file must have been written
        queue_files = list(queue_dir.glob("q001_*.json"))
        assert len(queue_files) == 1
        payload = json.loads(queue_files[0].read_text())
        assert payload["topic_id"] == "q001"

    def test_upload_records_video_units_on_success(self, tmp_path) -> None:
        """Successful upload must record 1600 units."""
        video = tmp_path / "out.mp4"
        video.write_bytes(b"v" * 100)
        _write_credentials(tmp_path)

        db = _make_quota_db(tmp_path)
        qt = QuotaTracker(db_path=db, daily_limit=10_000)
        uploader = _make_uploader(tmp_path)
        uploader.quota_tracker = qt

        with patch.object(uploader, "_load_credentials", return_value=MagicMock()):
            with patch.object(uploader, "_build_service", return_value=MagicMock()):
                with patch.object(uploader, "_execute_upload", return_value="vid_q"):
                    result = uploader.upload("q002", video, METADATA)

        assert result.is_valid is True
        assert qt.get_daily_usage() == QUOTA_UNITS["video_upload"]  # 1600

    def test_upload_records_thumbnail_units_when_provided(self, tmp_path) -> None:
        """Thumbnail upload adds 50 units on top of the 1600 video units."""
        video = tmp_path / "out.mp4"
        video.write_bytes(b"v" * 100)
        thumb = tmp_path / "thumb.jpg"
        thumb.write_bytes(b"j" * 100)
        _write_credentials(tmp_path)

        db = _make_quota_db(tmp_path)
        qt = QuotaTracker(db_path=db, daily_limit=10_000)
        uploader = _make_uploader(tmp_path)
        uploader.quota_tracker = qt

        with patch.object(uploader, "_load_credentials", return_value=MagicMock()):
            with patch.object(uploader, "_build_service", return_value=MagicMock()):
                with patch.object(uploader, "_execute_upload", return_value="vid_t"):
                    with patch.object(uploader, "_upload_thumbnail"):
                        result = uploader.upload("q003", video, METADATA, thumbnail_path=thumb)

        assert result.is_valid is True
        expected = QUOTA_UNITS["video_upload"] + QUOTA_UNITS["thumbnail_upload"]  # 1650
        assert qt.get_daily_usage() == expected

    def test_queue_file_contains_all_fields(self, tmp_path) -> None:
        """Queued JSON must contain all payload fields."""
        video = tmp_path / "out.mp4"
        video.write_bytes(b"v" * 100)

        db = _make_quota_db(tmp_path)
        qt = QuotaTracker(db_path=db, daily_limit=0)
        uploader = YouTubeUploader(
            channel_key="ch1",
            credentials_dir=tmp_path,
            quota_tracker=qt,
        )

        queue_dir = tmp_path / "queue"
        with patch("src.publisher.youtube_uploader._QUOTA_QUEUE_DIR", queue_dir):
            uploader.upload("q004", video, METADATA, publish_at="2025-06-01T08:00:00Z")

        payload = json.loads(list(queue_dir.glob("q004_*.json"))[0].read_text())
        for key in ("topic_id", "video_path", "metadata", "publish_at",
                    "thumbnail_path", "queued_at", "channel_key"):
            assert key in payload
        assert payload["channel_key"] == "ch1"
        assert payload["publish_at"] == "2025-06-01T08:00:00Z"

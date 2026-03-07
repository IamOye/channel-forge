"""
Tests for src/publisher/youtube_uploader.py

All Google API client calls are mocked — no real OAuth or network activity.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

from src.publisher.youtube_uploader import (
    CHUNK_SIZE,
    DEFAULT_CATEGORY_ID,
    MAX_RETRIES,
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

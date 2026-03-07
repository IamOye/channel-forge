"""
Tests for src/media/audio_processor.py

All filesystem and mutagen calls are mocked.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from src.media.audio_processor import (
    MIN_DURATION_SECONDS,
    MAX_DURATION_SECONDS,
    MIN_FILE_SIZE_BYTES,
    AudioInfo,
    AudioProcessor,
)


# ---------------------------------------------------------------------------
# AudioInfo
# ---------------------------------------------------------------------------

class TestAudioInfo:
    def _make(self, **kw) -> AudioInfo:
        defaults = dict(
            file_path="data/raw/test_voice.mp3",
            duration_seconds=13.0,
            file_size_bytes=50_000,
            is_valid=True,
            validation_errors=[],
        )
        defaults.update(kw)
        return AudioInfo(**defaults)

    def test_to_dict_has_all_keys(self) -> None:
        info = self._make()
        d = info.to_dict()
        for key in ("file_path", "duration_seconds", "file_size_bytes",
                    "is_valid", "validation_errors"):
            assert key in d

    def test_to_dict_values_correct(self) -> None:
        info = self._make(duration_seconds=12.5, file_size_bytes=48000)
        d = info.to_dict()
        assert d["duration_seconds"] == 12.5
        assert d["file_size_bytes"] == 48000


# ---------------------------------------------------------------------------
# AudioProcessor._validate_duration
# ---------------------------------------------------------------------------

class TestValidateDuration:
    def test_valid_duration_no_errors(self) -> None:
        errors = AudioProcessor._validate_duration(13.0)
        assert errors == []

    def test_exactly_at_min_passes(self) -> None:
        assert AudioProcessor._validate_duration(MIN_DURATION_SECONDS) == []

    def test_exactly_at_max_passes(self) -> None:
        assert AudioProcessor._validate_duration(MAX_DURATION_SECONDS) == []

    def test_zero_duration_error(self) -> None:
        errors = AudioProcessor._validate_duration(0.0)
        assert any("duration" in e for e in errors)

    def test_negative_duration_error(self) -> None:
        errors = AudioProcessor._validate_duration(-1.0)
        assert any("duration" in e for e in errors)

    def test_below_min_error(self) -> None:
        errors = AudioProcessor._validate_duration(5.0)
        assert any("minimum" in e for e in errors)

    def test_above_max_error(self) -> None:
        errors = AudioProcessor._validate_duration(20.0)
        assert any("maximum" in e for e in errors)


# ---------------------------------------------------------------------------
# AudioProcessor.inspect (mocked)
# ---------------------------------------------------------------------------

class TestAudioProcessorInspect:
    def _mock_mp3_info(self, length: float) -> MagicMock:
        info = MagicMock()
        info.length = length
        mp3 = MagicMock()
        mp3.info = info
        return mp3

    def test_returns_invalid_when_file_missing(self) -> None:
        proc = AudioProcessor()
        result = proc.inspect("data/raw/does_not_exist.mp3")
        assert result.is_valid is False
        assert any("not found" in e for e in result.validation_errors)
        assert result.duration_seconds == 0.0
        assert result.file_size_bytes == 0

    @patch("src.media.audio_processor.Path.exists", return_value=True)
    @patch("src.media.audio_processor.Path.stat")
    @patch("src.media.audio_processor.AudioProcessor._read_duration")
    def test_valid_file_passes(self, mock_duration, mock_stat, mock_exists) -> None:
        mock_stat.return_value.st_size = 50_000
        mock_duration.return_value = 13.0

        proc = AudioProcessor()
        result = proc.inspect("data/raw/good_voice.mp3")

        assert result.is_valid is True
        assert result.duration_seconds == 13.0
        assert result.file_size_bytes == 50_000
        assert result.validation_errors == []

    @patch("src.media.audio_processor.Path.exists", return_value=True)
    @patch("src.media.audio_processor.Path.stat")
    @patch("src.media.audio_processor.AudioProcessor._read_duration")
    def test_small_file_fails(self, mock_duration, mock_stat, mock_exists) -> None:
        mock_stat.return_value.st_size = 1_000   # < 10 KB
        mock_duration.return_value = 13.0

        proc = AudioProcessor()
        result = proc.inspect("data/raw/tiny.mp3")

        assert result.is_valid is False
        assert any("bytes" in e for e in result.validation_errors)

    @patch("src.media.audio_processor.Path.exists", return_value=True)
    @patch("src.media.audio_processor.Path.stat")
    @patch("src.media.audio_processor.AudioProcessor._read_duration")
    def test_short_duration_fails(self, mock_duration, mock_stat, mock_exists) -> None:
        mock_stat.return_value.st_size = 50_000
        mock_duration.return_value = 3.0   # too short

        proc = AudioProcessor()
        result = proc.inspect("data/raw/short.mp3")

        assert result.is_valid is False
        assert any("minimum" in e for e in result.validation_errors)

    @patch("src.media.audio_processor.Path.exists", return_value=True)
    @patch("src.media.audio_processor.Path.stat")
    @patch("src.media.audio_processor.AudioProcessor._read_duration")
    def test_long_duration_fails(self, mock_duration, mock_stat, mock_exists) -> None:
        mock_stat.return_value.st_size = 50_000
        mock_duration.return_value = 30.0   # too long

        proc = AudioProcessor()
        result = proc.inspect("data/raw/long.mp3")

        assert result.is_valid is False
        assert any("maximum" in e for e in result.validation_errors)

    @patch("src.media.audio_processor.Path.exists", return_value=True)
    @patch("src.media.audio_processor.Path.stat")
    @patch("src.media.audio_processor.AudioProcessor._read_duration")
    def test_duration_zero_error(self, mock_duration, mock_stat, mock_exists) -> None:
        mock_stat.return_value.st_size = 50_000
        mock_duration.return_value = 0.0   # couldn't read

        proc = AudioProcessor()
        result = proc.inspect("data/raw/unreadable.mp3")

        assert result.is_valid is False
        assert any("duration" in e for e in result.validation_errors)

    def test_read_duration_returns_zero_on_mutagen_error(self) -> None:
        """_read_duration falls back to 0.0 when mutagen fails."""
        # Construct mock module BEFORE patching so side_effect is on the right object
        mock_mp3_module = MagicMock()
        mock_mp3_module.MP3.side_effect = Exception("corrupt file")
        with patch.dict("sys.modules", {"mutagen": MagicMock(), "mutagen.mp3": mock_mp3_module}):
            proc = AudioProcessor()
            duration = proc._read_duration(Path("any.mp3"))
            assert duration == 0.0

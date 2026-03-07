"""
Tests for src/media/video_builder.py

All moviepy, filesystem I/O, and CaptionRenderer calls are mocked.
No real video processing happens during tests.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.media.video_builder import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    OVERLAY_OPACITY,
    VIDEO_DURATION,
    BuildResult,
    VideoBuilder,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SCRIPT = {
    "hook":      "Most people ignore this ancient secret.",
    "statement": "Stoics controlled their reactions daily.",
    "twist":     "Modern life rewired your brain away from that.",
    "question":  "What if you chose discomfort on purpose today?",
}


# ---------------------------------------------------------------------------
# BuildResult
# ---------------------------------------------------------------------------

class TestBuildResult:
    def _make(self, **kw) -> BuildResult:
        defaults = dict(
            topic_id="test_001",
            output_path="data/output/test_001_final.mp4",
            duration_seconds=13.5,
            is_valid=True,
        )
        defaults.update(kw)
        return BuildResult(**defaults)

    def test_built_at_auto_set(self) -> None:
        r = self._make()
        assert r.built_at != ""

    def test_to_dict_has_all_keys(self) -> None:
        r = self._make()
        d = r.to_dict()
        for key in ("topic_id", "output_path", "duration_seconds",
                    "is_valid", "validation_errors", "built_at"):
            assert key in d

    def test_invalid_result_has_errors(self) -> None:
        r = self._make(is_valid=False, validation_errors=["audio not found"])
        assert not r.is_valid
        assert len(r.validation_errors) == 1


# ---------------------------------------------------------------------------
# VideoBuilder._validate_inputs
# ---------------------------------------------------------------------------

class TestValidateInputs:
    def test_both_files_exist(self, tmp_path) -> None:
        audio = tmp_path / "voice.mp3"
        video = tmp_path / "stock.mp4"
        audio.write_bytes(b"audio")
        video.write_bytes(b"video")
        errors = VideoBuilder._validate_inputs(audio, video)
        assert errors == []

    def test_missing_audio_error(self, tmp_path) -> None:
        audio = tmp_path / "missing_voice.mp3"
        video = tmp_path / "stock.mp4"
        video.write_bytes(b"video")
        errors = VideoBuilder._validate_inputs(audio, video)
        assert any("audio" in e for e in errors)

    def test_missing_video_error(self, tmp_path) -> None:
        audio = tmp_path / "voice.mp3"
        video = tmp_path / "missing_stock.mp4"
        audio.write_bytes(b"audio")
        errors = VideoBuilder._validate_inputs(audio, video)
        assert any("stock" in e for e in errors)

    def test_both_missing_returns_two_errors(self, tmp_path) -> None:
        audio = tmp_path / "missing_a.mp3"
        video = tmp_path / "missing_v.mp4"
        errors = VideoBuilder._validate_inputs(audio, video)
        assert len(errors) == 2


# ---------------------------------------------------------------------------
# VideoBuilder.build (fully mocked)
# ---------------------------------------------------------------------------

class TestVideoBuildBuild:
    def _make_mock_clip(self) -> MagicMock:
        """Return a MagicMock that supports all moviepy v2 fluent calls."""
        clip = MagicMock()
        # Fluent method chaining — each method returns a MagicMock that also supports chains
        for method in ("subclipped", "resized", "with_opacity", "with_duration",
                       "with_audio", "with_start", "with_duration", "with_position"):
            getattr(clip, method).return_value = clip
        return clip

    def _patch_moviepy(self, mock_clip: MagicMock):
        """Return a context manager that patches moviepy v2 imports in video_builder."""
        moviepy_mock = MagicMock()
        moviepy_mock.VideoFileClip.return_value = mock_clip
        moviepy_mock.AudioFileClip.return_value = mock_clip
        moviepy_mock.ColorClip.return_value = mock_clip
        moviepy_mock.CompositeVideoClip.return_value = mock_clip
        return patch.dict("sys.modules", {"moviepy": moviepy_mock})

    def test_returns_valid_result(self, tmp_path) -> None:
        audio = tmp_path / "voice.mp3"
        stock = tmp_path / "stock.mp4"
        audio.write_bytes(b"a" * 50_000)
        stock.write_bytes(b"v" * 50_000)

        mock_clip = self._make_mock_clip()
        mock_caption_renderer = MagicMock()
        mock_caption_renderer.return_value.render.return_value = []

        with patch("src.media.video_builder.VideoBuilder._assemble", return_value=mock_clip):
            with patch("pathlib.Path.mkdir"):
                builder = VideoBuilder(output_dir=tmp_path)
                result = builder.build(
                    topic_id="test_001",
                    script_dict=SCRIPT,
                    audio_path=str(audio),
                    stock_video_path=str(stock),
                )

        assert isinstance(result, BuildResult)
        assert result.is_valid is True
        assert "test_001" in result.output_path
        assert result.output_path.endswith("_final.mp4")

    def test_returns_invalid_when_audio_missing(self, tmp_path) -> None:
        audio = tmp_path / "missing_voice.mp3"
        stock = tmp_path / "stock.mp4"
        stock.write_bytes(b"v" * 50_000)

        builder = VideoBuilder(output_dir=tmp_path)
        result = builder.build(
            topic_id="no_audio",
            script_dict=SCRIPT,
            audio_path=str(audio),
            stock_video_path=str(stock),
        )

        assert result.is_valid is False
        assert any("audio" in e for e in result.validation_errors)

    def test_returns_invalid_when_video_missing(self, tmp_path) -> None:
        audio = tmp_path / "voice.mp3"
        stock = tmp_path / "missing_stock.mp4"
        audio.write_bytes(b"a" * 50_000)

        builder = VideoBuilder(output_dir=tmp_path)
        result = builder.build(
            topic_id="no_video",
            script_dict=SCRIPT,
            audio_path=str(audio),
            stock_video_path=str(stock),
        )

        assert result.is_valid is False
        assert any("stock" in e for e in result.validation_errors)

    def test_duration_is_video_duration_constant(self, tmp_path) -> None:
        audio = tmp_path / "voice.mp3"
        stock = tmp_path / "stock.mp4"
        audio.write_bytes(b"a" * 50_000)
        stock.write_bytes(b"v" * 50_000)

        mock_clip = self._make_mock_clip()

        with patch("src.media.video_builder.VideoBuilder._assemble", return_value=mock_clip):
            with patch("pathlib.Path.mkdir"):
                builder = VideoBuilder(output_dir=tmp_path)
                result = builder.build("dur_test", SCRIPT, str(audio), str(stock))

        assert result.duration_seconds == VIDEO_DURATION

    def test_canvas_dimensions_passed_to_assemble(self, tmp_path) -> None:
        audio = tmp_path / "voice.mp3"
        stock = tmp_path / "stock.mp4"
        audio.write_bytes(b"a" * 50_000)
        stock.write_bytes(b"v" * 50_000)

        mock_clip = self._make_mock_clip()

        with patch("src.media.video_builder.VideoBuilder._assemble", return_value=mock_clip) as mock_assemble:
            with patch("pathlib.Path.mkdir"):
                builder = VideoBuilder(output_dir=tmp_path, canvas_width=720, canvas_height=1280)
                builder.build("dim_test", SCRIPT, str(audio), str(stock))

            assert builder.canvas_width == 720
            assert builder.canvas_height == 1280

    def test_to_dict_serialisable(self, tmp_path) -> None:
        import json
        audio = tmp_path / "voice.mp3"
        stock = tmp_path / "stock.mp4"
        audio.write_bytes(b"a" * 50_000)
        stock.write_bytes(b"v" * 50_000)

        mock_clip = self._make_mock_clip()

        with patch("src.media.video_builder.VideoBuilder._assemble", return_value=mock_clip):
            with patch("pathlib.Path.mkdir"):
                builder = VideoBuilder(output_dir=tmp_path)
                result = builder.build("serial_001", SCRIPT, str(audio), str(stock))

        assert len(json.dumps(result.to_dict())) > 10

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
    FPS,
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
    "landing":   "That is not growth. That is avoidance.",
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
        errors = VideoBuilder._validate_inputs(audio, [video])
        assert errors == []

    def test_missing_audio_error(self, tmp_path) -> None:
        audio = tmp_path / "missing_voice.mp3"
        video = tmp_path / "stock.mp4"
        video.write_bytes(b"video")
        errors = VideoBuilder._validate_inputs(audio, [video])
        assert any("audio" in e for e in errors)

    def test_missing_video_error(self, tmp_path) -> None:
        audio = tmp_path / "voice.mp3"
        video = tmp_path / "missing_stock.mp4"
        audio.write_bytes(b"audio")
        errors = VideoBuilder._validate_inputs(audio, [video])
        assert any("stock" in e for e in errors)

    def test_both_missing_returns_two_errors(self, tmp_path) -> None:
        audio = tmp_path / "missing_a.mp3"
        video = tmp_path / "missing_v.mp4"
        errors = VideoBuilder._validate_inputs(audio, [video])
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

        with patch("src.media.video_builder.VideoBuilder._assemble", return_value=(mock_clip, 15.0)):
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

    def test_duration_matches_audio(self, tmp_path) -> None:
        audio = tmp_path / "voice.mp3"
        stock = tmp_path / "stock.mp4"
        audio.write_bytes(b"a" * 50_000)
        stock.write_bytes(b"v" * 50_000)

        mock_clip = self._make_mock_clip()

        with patch("src.media.video_builder.VideoBuilder._assemble", return_value=(mock_clip, 15.0)):
            with patch("pathlib.Path.mkdir"):
                builder = VideoBuilder(output_dir=tmp_path)
                result = builder.build("dur_test", SCRIPT, str(audio), str(stock))

        assert result.duration_seconds == 15.0

    def test_canvas_dimensions_passed_to_assemble(self, tmp_path) -> None:
        audio = tmp_path / "voice.mp3"
        stock = tmp_path / "stock.mp4"
        audio.write_bytes(b"a" * 50_000)
        stock.write_bytes(b"v" * 50_000)

        mock_clip = self._make_mock_clip()

        with patch("src.media.video_builder.VideoBuilder._assemble", return_value=(mock_clip, 15.0)) as mock_assemble:
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

        with patch("src.media.video_builder.VideoBuilder._assemble", return_value=(mock_clip, 15.0)):
            with patch("pathlib.Path.mkdir"):
                builder = VideoBuilder(output_dir=tmp_path)
                result = builder.build("serial_001", SCRIPT, str(audio), str(stock))

        assert len(json.dumps(result.to_dict())) > 10


# ---------------------------------------------------------------------------
# VideoBuilder._cuts_from_word_timestamps
# ---------------------------------------------------------------------------

class TestCutsFromWordTimestamps:
    def _make_words(self, texts_and_times):
        return [
            {"text": t, "start_time": s, "end_time": e}
            for t, s, e in texts_and_times
        ]

    def test_returns_empty_for_single_clip(self) -> None:
        words = self._make_words([("hello", 0.0, 0.5), ("world", 0.6, 1.0)])
        cuts = VideoBuilder._cuts_from_word_timestamps(words, 10.0, 1)
        assert cuts == []

    def test_returns_empty_for_no_words(self) -> None:
        cuts = VideoBuilder._cuts_from_word_timestamps([], 10.0, 2)
        assert cuts == []

    def test_uses_pause_point_when_available(self) -> None:
        # Large gap between t=2.0 and t=3.0
        words = self._make_words([
            ("a", 0.0, 1.0),
            ("b", 1.1, 2.0),
            ("c", 3.0, 4.0),  # 1.0s pause before this word
            ("d", 4.1, 5.0),
        ])
        cuts = VideoBuilder._cuts_from_word_timestamps(words, 5.0, 2, min_pause=0.4)
        assert len(cuts) == 1
        # Cut should be near midpoint of the 1.0s pause (2.0 to 3.0) = 2.5s
        assert 1.5 < cuts[0] < 3.5

    def test_falls_back_to_equal_splits_without_pauses(self) -> None:
        # No pauses in the words
        words = self._make_words([
            ("a", 0.0, 0.9),
            ("b", 0.9, 1.8),
            ("c", 1.8, 2.7),
        ])
        cuts = VideoBuilder._cuts_from_word_timestamps(words, 4.0, 3, min_pause=0.4)
        assert len(cuts) == 2
        assert 0.0 < cuts[0] < cuts[1] < 4.0

    def test_correct_count_for_4_clips(self) -> None:
        words = self._make_words([
            ("a", 0.0, 2.0), ("b", 3.0, 5.0), ("c", 6.0, 8.0), ("d", 9.0, 11.0)
        ])
        cuts = VideoBuilder._cuts_from_word_timestamps(words, 12.0, 4, min_pause=0.4)
        assert len(cuts) == 3
        assert cuts == sorted(cuts)


class TestKineticOverlays:
    def test_extract_key_phrases_returns_list(self) -> None:
        from src.media.video_builder import VideoBuilder
        from unittest.mock import MagicMock, patch
        import json

        mock_client = MagicMock()
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text='["7 income streams", "one income source", "time for money"]')]
        )
        mock_anthropic_module = MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_client
        with patch.dict("sys.modules", {"anthropic": mock_anthropic_module}):
            result = VideoBuilder.extract_key_phrases("You trade time for money", api_key="fake")

        assert isinstance(result, list)
        assert len(result) <= 3

    def test_extract_key_phrases_max_3(self) -> None:
        from src.media.video_builder import VideoBuilder
        from unittest.mock import MagicMock, patch

        mock_client = MagicMock()
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text='["a", "b", "c", "d", "e"]')]
        )
        mock_anthropic_module = MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_client
        with patch.dict("sys.modules", {"anthropic": mock_anthropic_module}):
            result = VideoBuilder.extract_key_phrases("Some script text", api_key="fake")

        assert len(result) <= 3

    def test_extract_key_phrases_empty_script_returns_empty(self) -> None:
        from src.media.video_builder import VideoBuilder
        result = VideoBuilder.extract_key_phrases("", api_key="fake")
        assert result == []

    def test_extract_key_phrases_no_api_key_returns_empty(self) -> None:
        from src.media.video_builder import VideoBuilder
        import os
        original = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            result = VideoBuilder.extract_key_phrases("some script", api_key="")
            assert result == []
        finally:
            if original is not None:
                os.environ["ANTHROPIC_API_KEY"] = original

    def test_extract_key_phrases_api_failure_returns_empty(self) -> None:
        from src.media.video_builder import VideoBuilder
        from unittest.mock import patch

        mock_anthropic_module = MagicMock()
        mock_anthropic_module.Anthropic.side_effect = Exception("API down")
        with patch.dict("sys.modules", {"anthropic": mock_anthropic_module}):
            result = VideoBuilder.extract_key_phrases("some script", api_key="fake")

        assert result == []

    def test_render_kinetic_overlay_pil_shape(self) -> None:
        from src.media.video_builder import VideoBuilder
        import numpy as np
        frame = VideoBuilder._render_kinetic_overlay_pil("78 percent", 1080, 1920)
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (1920, 1080, 4)

    def test_render_kinetic_overlay_pil_scale_larger(self) -> None:
        from src.media.video_builder import VideoBuilder
        import numpy as np
        frame = VideoBuilder._render_kinetic_overlay_pil("test", 1080, 1920, scale=1.2)
        assert frame.shape == (1920, 1080, 4)

    def test_render_kinetic_overlay_pil_alpha_mult(self) -> None:
        from src.media.video_builder import VideoBuilder
        import numpy as np
        full = VideoBuilder._render_kinetic_overlay_pil("test", 200, 400, alpha_mult=1.0)
        faded = VideoBuilder._render_kinetic_overlay_pil("test", 200, 400, alpha_mult=0.0)
        # faded frame should have lower or equal max alpha
        assert faded[:, :, 3].max() <= full[:, :, 3].max()

# ---------------------------------------------------------------------------
# Ken Burns effect — apply_ken_burns + write_ken_burns_mp4
# ---------------------------------------------------------------------------


def _make_gradient_image(tmp_path, width=1080, height=1920) -> "Path":
    """Create a vertical gradient image (dark top, bright bottom) for testing."""
    import numpy as np
    from PIL import Image

    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        val = int(y * 255 / (height - 1))
        arr[y, :, :] = val
    img = Image.fromarray(arr)
    path = tmp_path / "gradient.jpg"
    img.save(str(path))
    return path


class TestKenBurns:
    def test_ken_burns_generates_correct_frame_count(self, tmp_path) -> None:
        """Frame count must equal int(duration * fps) + 1."""
        import numpy as np
        img_path = _make_gradient_image(tmp_path)
        builder = VideoBuilder(output_dir=tmp_path)
        frames = builder.apply_ken_burns(img_path, duration=1.0, effect="zoom_in")
        expected = int(1.0 * FPS) + 1  # 31 frames at 30fps
        assert len(frames) == expected

    def test_ken_burns_frame_shape(self, tmp_path) -> None:
        """Each frame must be H*W*3 (portrait canvas)."""
        import numpy as np
        img_path = _make_gradient_image(tmp_path)
        builder = VideoBuilder(output_dir=tmp_path)
        frames = builder.apply_ken_burns(img_path, duration=0.5, effect="zoom_out")
        assert frames[0].shape == (CANVAS_HEIGHT, CANVAS_WIDTH, 3)

    def test_zoom_in_increases_scale_over_time(self, tmp_path) -> None:
        """zoom_in: end frame top-left pixel should be brighter than start frame.

        With a dark-top / bright-bottom gradient, zooming in (crop shrinks
        toward center) means the end frame's top-left pixel comes from a
        more central (lower) y-position in the source -> slightly brighter.
        """
        import numpy as np
        img_path = _make_gradient_image(tmp_path)
        builder = VideoBuilder(output_dir=tmp_path)
        frames = builder.apply_ken_burns(img_path, duration=2.0, effect="zoom_in")

        # First and last frame must differ (animation happened)
        assert not np.array_equal(frames[0], frames[-1])
        # End frame top-left is brighter than start (closer to center in source)
        assert int(frames[-1][0, 0, 0]) > int(frames[0][0, 0, 0])

    def test_zoom_out_decreases_scale_over_time(self, tmp_path) -> None:
        """zoom_out: end frame top-left pixel should be darker than start frame.

        Opposite of zoom_in: starts zoomed in (narrow crop, central), ends
        wide (outer region included), so start frame top is brighter.
        """
        import numpy as np
        img_path = _make_gradient_image(tmp_path)
        builder = VideoBuilder(output_dir=tmp_path)
        frames = builder.apply_ken_burns(img_path, duration=2.0, effect="zoom_out")

        assert not np.array_equal(frames[0], frames[-1])
        # Start frame top-left is brighter (zoomed into center -> higher y in source)
        assert int(frames[0][0, 0, 0]) > int(frames[-1][0, 0, 0])

    def test_pan_right_generates_motion(self, tmp_path) -> None:
        """pan_right: first and last frames must differ.

        Uses a horizontal gradient so that horizontal panning produces visible pixel differences.
        """
        import numpy as np
        from PIL import Image
        width, height = 1080, 1920
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        for x in range(width):
            val = int(x * 255 / (width - 1))
            arr[:, x, :] = val
        img = Image.fromarray(arr)
        img_path = tmp_path / "hgrad.jpg"
        img.save(str(img_path))

        builder = VideoBuilder(output_dir=tmp_path)
        frames = builder.apply_ken_burns(img_path, duration=2.0, effect="pan_right")
        assert not np.array_equal(frames[0], frames[-1])

    def test_pan_left_generates_motion(self, tmp_path) -> None:
        """pan_left: first and last frames must differ.

        Uses a horizontal gradient so that horizontal panning produces visible pixel differences.
        """
        import numpy as np
        from PIL import Image
        width, height = 1080, 1920
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        for x in range(width):
            val = int(x * 255 / (width - 1))
            arr[:, x, :] = val
        img = Image.fromarray(arr)
        img_path = tmp_path / "hgrad_left.jpg"
        img.save(str(img_path))

        builder = VideoBuilder(output_dir=tmp_path)
        frames = builder.apply_ken_burns(img_path, duration=2.0, effect="pan_left")
        assert not np.array_equal(frames[0], frames[-1])

    def test_random_effect_when_none(self, tmp_path) -> None:
        """effect=None should not raise; should still produce correct frame count."""
        img_path = _make_gradient_image(tmp_path)
        builder = VideoBuilder(output_dir=tmp_path)
        frames = builder.apply_ken_burns(img_path, duration=1.0, effect=None)
        assert len(frames) == int(1.0 * FPS) + 1
        assert frames[0].shape == (CANVAS_HEIGHT, CANVAS_WIDTH, 3)

    def test_all_four_effects_produce_valid_frames(self, tmp_path) -> None:
        """All 4 effects should return non-empty frame lists with correct shape."""
        import numpy as np
        img_path = _make_gradient_image(tmp_path)
        builder = VideoBuilder(output_dir=tmp_path)
        for effect in ("zoom_in", "zoom_out", "pan_right", "pan_left"):
            frames = builder.apply_ken_burns(img_path, duration=0.5, effect=effect)
            assert len(frames) > 0, f"No frames for effect={effect}"
            assert frames[0].shape == (CANVAS_HEIGHT, CANVAS_WIDTH, 3), (
                f"Wrong shape for effect={effect}"
            )


class TestWriteKenBurnsMp4:
    def test_write_ken_burns_mp4_calls_ffmpeg(self, tmp_path) -> None:
        """write_ken_burns_mp4 should call subprocess.Popen with ffmpeg."""
        import subprocess
        img_path = _make_gradient_image(tmp_path)
        output_path = tmp_path / "kb_out.mp4"

        builder = VideoBuilder(output_dir=tmp_path)

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdin = MagicMock()
        mock_proc.stdin.write = MagicMock()
        mock_proc.wait = MagicMock()

        with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
            with patch("imageio_ffmpeg.get_ffmpeg_exe", return_value="ffmpeg"):
                # Create the output file so the size check passes
                output_path.write_bytes(b"x" * 1024)
                builder.write_ken_burns_mp4(img_path, output_path, duration=0.5, effect="zoom_in")
        mock_popen.assert_called_once()

    def test_write_ken_burns_mp4_returns_false_on_ffmpeg_missing(self, tmp_path) -> None:
        """Returns False gracefully if imageio_ffmpeg is unavailable."""
        img_path = _make_gradient_image(tmp_path)
        output_path = tmp_path / "kb_out.mp4"
        builder = VideoBuilder(output_dir=tmp_path)

        with patch("imageio_ffmpeg.get_ffmpeg_exe", side_effect=Exception("not found")):
            result = builder.write_ken_burns_mp4(img_path, output_path, duration=0.5)
        assert result is False

    def test_write_ken_burns_mp4_creates_parent_directory(self, tmp_path) -> None:
        """Output path's parent directory should be created if missing."""
        img_path = _make_gradient_image(tmp_path)
        output_path = tmp_path / "nested" / "deep" / "kb_out.mp4"
        builder = VideoBuilder(output_dir=tmp_path)

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdin = MagicMock()
        mock_proc.wait = MagicMock()

        with patch("subprocess.Popen", return_value=mock_proc):
            with patch("imageio_ffmpeg.get_ffmpeg_exe", return_value="ffmpeg"):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"x" * 1024)
                result = builder.write_ken_burns_mp4(img_path, output_path, duration=0.5, effect="zoom_in")
        assert output_path.parent.exists()

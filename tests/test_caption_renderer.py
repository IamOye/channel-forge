"""
Tests for src/media/caption_renderer.py

CaptionRenderer.build_specs() is tested without any moviepy dependency.
CaptionRenderer.render() is tested with moviepy mocked at the import level.
"""

from unittest.mock import MagicMock, patch

from src.media.caption_renderer import (
    CAPTION_TIMINGS,
    CAPTION_Y_RATIO,
    CaptionClipSpec,
    CaptionRenderer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_SCRIPT = {
    "hook":      "Most people ignore this ancient secret.",
    "statement": "Stoics knew you control your reactions.",
    "twist":     "But modern life trains you to react.",
    "question":  "What would change if you chose differently today?",
}


# ---------------------------------------------------------------------------
# CaptionClipSpec
# ---------------------------------------------------------------------------

class TestCaptionClipSpec:
    def test_duration_calculated_from_start_end(self) -> None:
        spec = CaptionClipSpec(section="hook", text="test", start=0.0, end=2.0,
                               x="center", y=1248)
        assert spec.duration == 2.0

    def test_duration_for_statement(self) -> None:
        spec = CaptionClipSpec(section="statement", text="test", start=2.0, end=6.0,
                               x="center", y=1248)
        assert spec.duration == 4.0

    def test_to_dict_has_all_keys(self) -> None:
        spec = CaptionClipSpec(section="hook", text="test", start=0.0, end=2.0,
                               x="center", y=1248)
        d = spec.to_dict()
        for key in ("section", "text", "start", "end", "x", "y", "duration"):
            assert key in d


# ---------------------------------------------------------------------------
# CaptionRenderer.build_specs
# ---------------------------------------------------------------------------

class TestBuildSpecs:
    def test_returns_4_specs_for_full_script(self) -> None:
        renderer = CaptionRenderer()
        specs = renderer.build_specs(VALID_SCRIPT)
        assert len(specs) == 4

    def test_sections_in_correct_order(self) -> None:
        renderer = CaptionRenderer()
        specs = renderer.build_specs(VALID_SCRIPT)
        assert specs[0].section == "hook"
        assert specs[1].section == "statement"
        assert specs[2].section == "twist"
        assert specs[3].section == "question"

    def test_timings_match_constants(self) -> None:
        renderer = CaptionRenderer()
        specs = renderer.build_specs(VALID_SCRIPT)
        for spec, (_, start, end) in zip(specs, CAPTION_TIMINGS):
            assert spec.start == start
            assert spec.end == end

    def test_text_matches_script_dict(self) -> None:
        renderer = CaptionRenderer()
        specs = renderer.build_specs(VALID_SCRIPT)
        assert specs[0].text == VALID_SCRIPT["hook"]
        assert specs[3].text == VALID_SCRIPT["question"]

    def test_y_position_at_65_percent(self) -> None:
        renderer = CaptionRenderer(canvas_height=1920)
        specs = renderer.build_specs(VALID_SCRIPT)
        expected_y = int(1920 * CAPTION_Y_RATIO)
        for spec in specs:
            assert spec.y == expected_y

    def test_x_position_is_center(self) -> None:
        renderer = CaptionRenderer()
        specs = renderer.build_specs(VALID_SCRIPT)
        for spec in specs:
            assert spec.x == "center"

    def test_skips_empty_sections(self) -> None:
        script = dict(VALID_SCRIPT)
        script["statement"] = ""
        renderer = CaptionRenderer()
        specs = renderer.build_specs(script)
        assert len(specs) == 3
        assert all(s.section != "statement" for s in specs)

    def test_empty_script_returns_no_specs(self) -> None:
        renderer = CaptionRenderer()
        specs = renderer.build_specs({})
        assert specs == []

    def test_custom_canvas_size(self) -> None:
        renderer = CaptionRenderer(canvas_width=720, canvas_height=1280)
        specs = renderer.build_specs(VALID_SCRIPT)
        expected_y = int(1280 * CAPTION_Y_RATIO)
        for spec in specs:
            assert spec.y == expected_y


# ---------------------------------------------------------------------------
# CaptionRenderer.render (moviepy mocked)
# ---------------------------------------------------------------------------

class TestRender:
    def _make_mock_clip(self) -> MagicMock:
        """Return a MagicMock supporting moviepy v2 fluent chaining."""
        clip = MagicMock()
        clip.with_start.return_value = clip
        clip.with_duration.return_value = clip
        clip.with_position.return_value = clip
        return clip

    def _moviepy_module(self, mock_clip: MagicMock) -> MagicMock:
        """Return a mock moviepy module whose TextClip returns mock_clip."""
        mod = MagicMock()
        mod.TextClip.return_value = mock_clip
        return mod

    def test_render_returns_one_clip_per_script_section(self) -> None:
        mock_clip = self._make_mock_clip()
        moviepy_mod = self._moviepy_module(mock_clip)

        with patch.dict("sys.modules", {"moviepy": moviepy_mod}):
            renderer = CaptionRenderer()
            clips = renderer.render(VALID_SCRIPT)

        # 4 sections → 4 clips
        assert len(clips) == 4

    def test_render_calls_build_specs(self) -> None:
        mock_clip = self._make_mock_clip()
        moviepy_mod = self._moviepy_module(mock_clip)

        with patch.dict("sys.modules", {"moviepy": moviepy_mod}):
            with patch.object(CaptionRenderer, "build_specs", wraps=CaptionRenderer().build_specs) as mock_bs:
                renderer = CaptionRenderer()
                renderer.render(VALID_SCRIPT)
                assert mock_bs.call_count == 1

    def test_render_skips_empty_sections(self) -> None:
        script = dict(VALID_SCRIPT)
        script["statement"] = ""   # missing → no clip

        mock_clip = self._make_mock_clip()
        moviepy_mod = self._moviepy_module(mock_clip)

        with patch.dict("sys.modules", {"moviepy": moviepy_mod}):
            renderer = CaptionRenderer()
            clips = renderer.render(script)

        assert len(clips) == 3   # only 3 sections with text

    def test_render_returns_empty_list_for_empty_script(self) -> None:
        mock_clip = self._make_mock_clip()
        moviepy_mod = self._moviepy_module(mock_clip)

        with patch.dict("sys.modules", {"moviepy": moviepy_mod}):
            renderer = CaptionRenderer()
            clips = renderer.render({})

        assert clips == []

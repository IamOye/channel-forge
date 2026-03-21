"""
Tests for src/media/caption_renderer.py

CaptionRenderer.build_specs() is tested without any moviepy dependency.
CaptionRenderer.render() is tested with moviepy mocked at the import level.
"""

from unittest.mock import MagicMock, patch

from src.media.caption_renderer import (
    CAPTION_TIMINGS,
    CAPTION_Y_RATIO,
    CTA_OVERLAY_END,
    CTA_OVERLAY_START,
    CTA_Y_RATIO,
    HIGHLIGHT_TEXT_COLOR,
    MIN_CAPTION_FONT_SIZE,
    WORD_CAPTION_Y_RATIO,
    WORD_TEXT_COLOR,
    CaptionClipSpec,
    CaptionRenderer,
    _group_words,
    _load_word_font,
    _render_word_frame,
    _visible_at,
    _word_font_size,
    _word_stroke_width,
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

    def test_y_position_uses_caption_y_ratio(self) -> None:
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
# CaptionRenderer.build_cta_spec
# ---------------------------------------------------------------------------

class TestBuildCtaSpec:
    def test_returns_spec_with_correct_fields(self) -> None:
        renderer = CaptionRenderer(canvas_height=1920)
        spec = renderer.build_cta_spec("FREE GUIDE — Link in Description")
        assert spec is not None
        assert spec.section == "cta_overlay"
        assert spec.text == "FREE GUIDE — Link in Description"
        assert spec.start == CTA_OVERLAY_START
        assert spec.end == CTA_OVERLAY_END
        assert spec.x == "center"
        assert spec.y == int(1920 * CTA_Y_RATIO)

    def test_returns_none_for_empty_string(self) -> None:
        renderer = CaptionRenderer()
        assert renderer.build_cta_spec("") is None

    def test_returns_none_for_whitespace_only(self) -> None:
        renderer = CaptionRenderer()
        assert renderer.build_cta_spec("   ") is None


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


# ---------------------------------------------------------------------------
# Module-level helper functions
# ---------------------------------------------------------------------------

class TestGroupWords:
    def test_groups_into_lines_of_3(self) -> None:
        words = [{"text": str(i), "start_time": float(i), "end_time": float(i)+0.5} for i in range(7)]
        grouped = _group_words(words)
        assert grouped[0]["line_idx"] == 0
        assert grouped[2]["line_idx"] == 0
        assert grouped[3]["line_idx"] == 1
        assert grouped[6]["line_idx"] == 2

    def test_word_idx_matches_input_order(self) -> None:
        words = [{"text": "a", "start_time": 0.0, "end_time": 0.5},
                 {"text": "b", "start_time": 0.5, "end_time": 1.0}]
        grouped = _group_words(words)
        assert grouped[0]["word_idx"] == 0
        assert grouped[1]["word_idx"] == 1


class TestVisibleAt:
    def _make_grouped(self):
        words = [
            {"text": "hello", "start_time": 0.0, "end_time": 0.5},
            {"text": "world", "start_time": 0.5, "end_time": 1.0},
            {"text": "foo",   "start_time": 1.0, "end_time": 1.5},
            {"text": "bar",   "start_time": 1.5, "end_time": 2.0},
        ]
        return _group_words(words)

    def test_no_words_before_start(self) -> None:
        grouped = self._make_grouped()
        idx, visible = _visible_at(-0.1, grouped)
        assert idx is None
        assert visible == []

    def test_first_word_visible_at_start(self) -> None:
        grouped = self._make_grouped()
        idx, visible = _visible_at(0.0, grouped)
        assert idx == 0
        assert len(visible) == 1
        assert visible[0]["text"] == "hello"

    def test_two_words_visible_in_same_group(self) -> None:
        grouped = self._make_grouped()
        idx, visible = _visible_at(0.6, grouped)
        assert idx == 1
        assert len(visible) == 2

    def test_new_group_clears_previous(self) -> None:
        grouped = self._make_grouped()
        # "bar" (word_idx=3, line_idx=1) starts at t=1.5 — only "bar" visible in line 1
        idx, visible = _visible_at(1.5, grouped)
        assert idx == 3
        assert len(visible) == 1
        assert visible[0]["text"] == "bar"


class TestRenderWordByWord:
    def _make_word_timestamps(self) -> list[dict]:
        words = "most people ignore this ancient secret stoics knew you control reactions".split()
        return [
            {"text": w, "start_time": i * 0.3, "end_time": (i + 1) * 0.3}
            for i, w in enumerate(words)
        ]

    def test_render_with_word_timestamps_returns_nonempty_list(self) -> None:
        """render() with word_timestamps returns a single VideoClip (binary search) + no CTA."""
        words = self._make_word_timestamps()

        mock_video_clip = MagicMock()
        mock_video_clip.with_mask.return_value = mock_video_clip

        mock_moviepy = MagicMock()
        mock_moviepy.VideoClip.return_value = mock_video_clip

        # Font mock returns concrete integer bbox so PIL arithmetic works
        mock_font = MagicMock()
        mock_font.getbbox.return_value = (0, 0, 100, 68)

        with patch.dict("sys.modules", {"moviepy": mock_moviepy}):
            with patch("src.media.caption_renderer._load_word_font", return_value=mock_font):
                with patch("src.media.caption_renderer._render_word_frame") as mock_rwf:
                    import numpy as _np
                    mock_rwf.return_value = _np.zeros((1920, 1080, 4), dtype=_np.uint8)
                    renderer = CaptionRenderer()
                    clips = renderer.render(
                        script_dict=VALID_SCRIPT,
                        word_timestamps=words,
                        video_duration=3.3,
                    )

        assert isinstance(clips, list)
        # Single VideoClip (binary search) with no CTA in this call
        assert len(clips) == 1

    def test_render_without_word_timestamps_uses_text_clips(self) -> None:
        """Fallback path: no word_timestamps → same count as sections."""
        mock_clip = MagicMock()
        mock_clip.with_start.return_value = mock_clip
        mock_clip.with_duration.return_value = mock_clip
        mock_clip.with_position.return_value = mock_clip

        moviepy_mod = MagicMock()
        moviepy_mod.TextClip.return_value = mock_clip

        with patch.dict("sys.modules", {"moviepy": moviepy_mod}):
            renderer = CaptionRenderer()
            clips = renderer.render(VALID_SCRIPT)

        assert len(clips) == 4


# ---------------------------------------------------------------------------
# Caption style tests (BUG 1 — no pill background, stroke, font size)
# ---------------------------------------------------------------------------

class TestCaptionStyle:
    """Verify VIZIONTIA-style caption rendering: no pills, stroked text, gold highlight."""

    def test_no_pill_background_constants_removed(self) -> None:
        """Verify that pill/badge background constants no longer exist in module.

        The old pill-based rendering used PILL_BG_COLOR, PILL_CORNER_RADIUS,
        HIGHLIGHT_PAD_X, HIGHLIGHT_PAD_Y. These must be removed.
        """
        import src.media.caption_renderer as mod
        assert not hasattr(mod, "PILL_BG_COLOR"), "PILL_BG_COLOR still exists — remove pill background"
        assert not hasattr(mod, "PILL_CORNER_RADIUS"), "PILL_CORNER_RADIUS still exists"
        assert not hasattr(mod, "HIGHLIGHT_PAD_X"), "HIGHLIGHT_PAD_X still exists"
        assert not hasattr(mod, "HIGHLIGHT_PAD_Y"), "HIGHLIGHT_PAD_Y still exists"
        assert not hasattr(mod, "_draw_rounded_rect"), "_draw_rounded_rect still exists"

    def test_font_size_at_360_uses_minimum(self) -> None:
        """At 360px canvas, raw size (34px) is below minimum so clamps to 40px."""
        size = _word_font_size(360)
        assert size >= MIN_CAPTION_FONT_SIZE, f"Font size {size} < {MIN_CAPTION_FONT_SIZE} at 360px"

    def test_font_size_at_1080_is_about_100(self) -> None:
        """At 1080px canvas: round(1080 * 0.093) = 100px."""
        size = _word_font_size(1080)
        assert 95 <= size <= 105, f"Font size {size} not in expected range at 1080px"

    def test_font_size_scales_proportionally(self) -> None:
        """At 1080px canvas, font size should be larger than at 360px."""
        size_360 = _word_font_size(360)
        size_1080 = _word_font_size(1080)
        assert size_1080 > size_360

    def test_font_size_never_below_minimum(self) -> None:
        """Even at very small canvas, font size stays >= MIN_CAPTION_FONT_SIZE."""
        size = _word_font_size(100)
        assert size >= MIN_CAPTION_FONT_SIZE

    def test_stroke_width_scales_with_canvas(self) -> None:
        """Stroke width must be >= 2 at 360px canvas."""
        assert _word_stroke_width(360) >= 2
        # At 1080px should be larger
        assert _word_stroke_width(1080) >= _word_stroke_width(360)

    def test_highlight_color_is_gold(self) -> None:
        """Highlight text colour must be #FFD700 (gold), not a background."""
        assert HIGHLIGHT_TEXT_COLOR == (255, 215, 0)

    def test_text_color_is_white(self) -> None:
        """Non-highlighted word text colour must be white."""
        assert WORD_TEXT_COLOR == (255, 255, 255)

    def test_caption_position_at_75_to_80_percent(self) -> None:
        """Captions should be positioned at 75-80% from top of frame."""
        assert 0.75 <= WORD_CAPTION_Y_RATIO <= 0.80

    def test_get_caption_config_returns_font_size(self) -> None:
        """get_caption_config must include font_size for quality gate."""
        renderer = CaptionRenderer(canvas_width=1080, canvas_height=1920)
        config = renderer.get_caption_config()
        assert "font_size" in config
        assert config["font_size"] >= MIN_CAPTION_FONT_SIZE

    def test_font_loads_via_truetype_not_default(self) -> None:
        """Font must be loaded via truetype (scalable), never load_default (bitmap)."""
        try:
            from PIL import ImageFont
        except ImportError:
            return  # PIL not installed — skip

        size = _word_font_size(360)
        font = _load_word_font(size=size, canvas_w=360)
        assert font is not None, "No font loaded at all"

        # Verify font is a FreeTypeFont (truetype), not a bitmap default
        assert hasattr(font, "getbbox"), "Font does not support getbbox — likely bitmap"
        bb = font.getbbox("TEST")
        height = bb[3] - bb[1]
        # Rendered glyph height is always smaller than requested font size
        # (size is em-square, not glyph height). At 40px request, expect ~28px.
        # A bitmap load_default() would render at ~8px regardless.
        assert height >= 20, (
            f"Font renders at {height}px — too small. "
            "Likely using load_default() bitmap font instead of truetype."
        )

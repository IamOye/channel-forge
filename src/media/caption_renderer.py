"""
caption_renderer.py — CaptionRenderer

Builds timed caption TextClip objects for each section of a YouTube Shorts script.

Usage:
    renderer = CaptionRenderer(canvas_width=1080, canvas_height=1920)
    clips = renderer.render(script_dict)   # list of moviepy TextClip
"""

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Caption timing constants
# ---------------------------------------------------------------------------

# (section_key, start_seconds, end_seconds)
CAPTION_TIMINGS: list[tuple[str, float, float]] = [
    ("hook",      0.0,   2.0),
    ("statement", 2.0,   6.0),
    ("twist",     6.0,  10.0),
    ("question",  10.0, 13.5),
]

CAPTION_FONT_CANDIDATES = ["Impact", "Arial-Bold", "Arial", None]
CAPTION_FONT_SIZE = 72
CAPTION_COLOR     = "white"
CAPTION_STROKE_COLOR = "black"
CAPTION_STROKE_WIDTH = 3

# CTA overlay (gold banner, last 3 seconds of video)
CTA_FONT_SIZE    = 52
CTA_TEXT_COLOR   = "black"
CTA_BG_COLOR     = (201, 168, 76)   # gold #C9A84C
CTA_OVERLAY_START = 10.5
CTA_OVERLAY_END   = 13.5
# Vertical position of CTA banner — 85% from top (near bottom)
CTA_Y_RATIO = 0.85


def _resolve_font() -> str | None:
    """Return the first font from CAPTION_FONT_CANDIDATES that Pillow can open.

    Tries each candidate by rendering a tiny probe TextClip; falls back to
    None (moviepy's built-in default) if nothing else works.
    """
    try:
        from moviepy import TextClip  # lazy — avoid import at module load
    except Exception:
        return None

    for font in CAPTION_FONT_CANDIDATES:
        try:
            TextClip(text="A", font=font, font_size=12, color="white")
            logger.debug("Caption font resolved: %s", font)
            return font
        except Exception:
            logger.debug("Caption font unavailable: %s", font)
            continue

    return None

# Vertical position — 65% from the top of the frame
CAPTION_Y_RATIO = 0.65


# ---------------------------------------------------------------------------
# Word-by-word caption rendering constants
# ---------------------------------------------------------------------------

# Font size for word captions
WORD_FONT_SIZE = 68

# Search paths for bold font (tried in order)
WORD_FONT_SEARCH_PATHS: list[str | None] = [
    "C:/Windows/Fonts/arialbd.ttf",    # Arial Bold — Windows
    "C:/Windows/Fonts/ariblk.ttf",     # Arial Black — Windows
    "C:/Windows/Fonts/arial.ttf",      # Arial — Windows fallback
    "/System/Library/Fonts/Arial Bold.ttf",                         # macOS
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",        # Linux alt
    None,                               # Pillow default
]

HIGHLIGHT_COLOR      = (201, 168, 76)   # Gold #C9A84C
HIGHLIGHT_TEXT_COLOR = (0, 0, 0)        # Black text on gold
WORD_TEXT_COLOR      = (255, 255, 255)  # White text for previous words
WORD_SHADOW_COLOR    = (0, 0, 0, 178)   # Black, ~70% opacity
WORD_SHADOW_OFFSET   = 3               # px
PILL_BG_COLOR        = (0, 0, 0, 140)   # Semi-transparent black ~55% opacity
PILL_CORNER_RADIUS   = 12
HIGHLIGHT_PAD_X      = 16              # left/right padding in gold pill
HIGHLIGHT_PAD_Y      = 14              # top/bottom padding in gold pill
WORD_GAP             = 12              # px gap between word pills
WORD_MAX_PER_LINE    = 3
WORD_CAPTION_Y_START = 0.68            # top of caption area (fraction of height)
WORD_CAPTION_Y_END   = 0.85            # bottom of caption area
WORD_ENTRANCE_DRIFT  = 4               # px upward drift on word entrance
WORD_ENTRANCE_DUR    = 0.08            # seconds for entrance animation


# ---------------------------------------------------------------------------
# Word-caption PIL helpers
# ---------------------------------------------------------------------------

def _load_word_font(size: int = WORD_FONT_SIZE):
    """Load a bold font for word captions, falling back through WORD_FONT_SEARCH_PATHS."""
    try:
        from PIL import ImageFont
    except ImportError:
        return None
    for path in WORD_FONT_SEARCH_PATHS:
        if path is None:
            return ImageFont.load_default()
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return None


def _group_words(words: list[dict], max_per_line: int = WORD_MAX_PER_LINE) -> list[dict]:
    """Assign line_idx and idx_in_line to each word dict."""
    return [
        {**w, "word_idx": i, "line_idx": i // max_per_line, "idx_in_line": i % max_per_line}
        for i, w in enumerate(words)
    ]


def _visible_at(t: float, grouped: list[dict]) -> tuple[int | None, list[dict]]:
    """Return (current_word_idx, visible_words_in_line) at time t."""
    current_idx: int | None = None
    for w in grouped:
        if w["start_time"] <= t:
            current_idx = w["word_idx"]

    if current_idx is None:
        return None, []

    current_line = grouped[current_idx]["line_idx"]
    visible = [w for w in grouped
               if w["line_idx"] == current_line and w["start_time"] <= t]
    return current_idx, visible


def _draw_rounded_rect(draw, bbox: tuple, radius: int, fill: tuple) -> None:
    """Draw a filled rounded rectangle; uses Pillow's built-in when available."""
    try:
        draw.rounded_rectangle(bbox, radius=radius, fill=fill)
    except AttributeError:
        x1, y1, x2, y2 = bbox
        r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
        draw.rectangle([x1 + r, y1, x2 - r, y2], fill=fill)
        draw.rectangle([x1, y1 + r, x2, y2 - r], fill=fill)
        for cx, cy in [(x1, y1), (x2 - 2*r, y1), (x1, y2 - 2*r), (x2 - 2*r, y2 - 2*r)]:
            draw.ellipse([cx, cy, cx + 2*r, cy + 2*r], fill=fill)


def _render_word_frame(
    t: float,
    grouped: list[dict],
    canvas_w: int,
    canvas_h: int,
    font,
) -> "Image.Image":
    """Render a single word-caption frame as an RGBA PIL Image.

    Previous words: white text, no individual background.
    Current word: black text, gold rounded-rect background.
    All previous words share one semi-transparent black pill.
    New word: subtle 4 px upward drift during first WORD_ENTRANCE_DUR seconds.
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    current_idx, visible = _visible_at(t, grouped)
    if not visible:
        return img

    draw = ImageDraw.Draw(img)

    # Measure each visible word
    entries = []
    for w in visible:
        try:
            bb = font.getbbox(w["text"])
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
        except Exception:
            tw, th = 60, WORD_FONT_SIZE

        is_cur = (w["word_idx"] == current_idx)
        elapsed = t - w["start_time"]
        progress = min(1.0, elapsed / WORD_ENTRANCE_DUR) if is_cur else 1.0
        y_drift = -int(WORD_ENTRANCE_DRIFT * (1.0 - progress))

        entries.append({
            "text":    w["text"],
            "tw":      tw,
            "th":      th,
            "pill_w":  tw + 2 * HIGHLIGHT_PAD_X,
            "pill_h":  th + 2 * HIGHLIGHT_PAD_Y,
            "y_drift": y_drift,
            "is_cur":  is_cur,
        })

    total_w = sum(e["pill_w"] for e in entries) + WORD_GAP * (len(entries) - 1)
    max_pill_h = max(e["pill_h"] for e in entries)

    # Vertical center of caption area
    y_mid = int(canvas_h * (WORD_CAPTION_Y_START + WORD_CAPTION_Y_END) / 2)
    x = (canvas_w - total_w) // 2

    # Collect pill regions
    prev_regions: list[tuple] = []
    cur_region: tuple | None = None

    for e in entries:
        y_top = y_mid - e["pill_h"] // 2 + e["y_drift"]
        region = (x, y_top, x + e["pill_w"], y_top + e["pill_h"], e)
        if e["is_cur"]:
            cur_region = region
        else:
            prev_regions.append(region)
        x += e["pill_w"] + WORD_GAP

    # Draw combined black pill behind all previous words
    if prev_regions:
        px1 = min(r[0] for r in prev_regions)
        py1 = min(r[1] for r in prev_regions)
        px2 = max(r[2] for r in prev_regions)
        py2 = max(r[3] for r in prev_regions)
        _draw_rounded_rect(draw, (px1, py1, px2, py2), PILL_CORNER_RADIUS, PILL_BG_COLOR)

    # Draw gold pill behind current word
    if cur_region:
        rx1, ry1, rx2, ry2, _ = cur_region
        _draw_rounded_rect(draw, (rx1, ry1, rx2, ry2), PILL_CORNER_RADIUS,
                           (*HIGHLIGHT_COLOR, 255))

    # Draw text for each word
    x = (canvas_w - total_w) // 2
    for e in entries:
        y_top = y_mid - e["pill_h"] // 2 + e["y_drift"]
        tx = x + HIGHLIGHT_PAD_X
        ty = y_top + HIGHLIGHT_PAD_Y

        if e["is_cur"]:
            draw.text((tx, ty), e["text"], font=font,
                      fill=(*HIGHLIGHT_TEXT_COLOR, 255))
        else:
            # Shadow
            draw.text((tx + WORD_SHADOW_OFFSET, ty + WORD_SHADOW_OFFSET),
                      e["text"], font=font, fill=WORD_SHADOW_COLOR)
            draw.text((tx, ty), e["text"], font=font,
                      fill=(*WORD_TEXT_COLOR, 255))

        x += e["pill_w"] + WORD_GAP

    return img


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CaptionClipSpec:
    """
    Specification for a single caption clip (avoids importing moviepy at definition time).

    Attributes:
        section:   Script section key (hook/statement/twist/question).
        text:      The caption text to display.
        start:     Start time in seconds.
        end:       End time in seconds.
        x:         Horizontal position ('center' or pixel int).
        y:         Vertical position in pixels.
    """

    section: str
    text: str
    start: float
    end: float
    x: Any   # 'center' or int
    y: int

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        return {
            "section":  self.section,
            "text":     self.text,
            "start":    self.start,
            "end":      self.end,
            "x":        self.x,
            "y":        self.y,
            "duration": self.duration,
        }


# ---------------------------------------------------------------------------
# CaptionRenderer
# ---------------------------------------------------------------------------

class CaptionRenderer:
    """
    Generates timed TextClip objects for each section of a 15-second Shorts script.

    Args:
        canvas_width:  Video canvas width in pixels (default 1080).
        canvas_height: Video canvas height in pixels (default 1920).
        font:          Font name passed to moviepy TextClip. Defaults to None,
                       which triggers auto-detection (Impact → Arial-Bold → Arial → None).
        font_size:     Font size in pixels.
        y_ratio:       Vertical position of captions as fraction of canvas height (0–1).
    """

    def __init__(
        self,
        canvas_width: int = 1080,
        canvas_height: int = 1920,
        font: str | None = None,
        font_size: int = CAPTION_FONT_SIZE,
        y_ratio: float = CAPTION_Y_RATIO,
    ) -> None:
        self.canvas_width  = canvas_width
        self.canvas_height = canvas_height
        self.font          = font if font is not None else _resolve_font()
        self.font_size     = font_size
        self.y_ratio       = y_ratio

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_specs(self, script_dict: dict[str, str]) -> list[CaptionClipSpec]:
        """
        Build CaptionClipSpec objects for each script section.

        These specs describe every caption (position, timing, text) without
        requiring moviepy at this point.  Call render() to get actual TextClips.

        Args:
            script_dict: Dict with keys hook, statement, twist, question.

        Returns:
            List of CaptionClipSpec, one per script section that has text.
        """
        y_pos = int(self.canvas_height * self.y_ratio)
        specs: list[CaptionClipSpec] = []

        for section, start, end in CAPTION_TIMINGS:
            text = script_dict.get(section, "").strip()
            if not text:
                logger.debug("Skipping empty caption section: %s", section)
                continue
            specs.append(CaptionClipSpec(
                section=section,
                text=text,
                start=start,
                end=end,
                x="center",
                y=y_pos,
            ))

        logger.info("Built %d caption specs", len(specs))
        return specs

    def build_cta_spec(
        self,
        cta_overlay: str,
        video_duration: float | None = None,
    ) -> CaptionClipSpec | None:
        """
        Build a CaptionClipSpec for the gold CTA overlay banner.

        The banner appears during the last 3 seconds of the video.
        When video_duration is supplied the timing is calculated dynamically
        so the banner always lands at the end of the actual video length.

        Args:
            cta_overlay:    Text to display (e.g. "FREE GUIDE — Link in Description").
            video_duration: Actual video length in seconds. If None, uses the
                            default constants CTA_OVERLAY_START / CTA_OVERLAY_END.

        Returns:
            A CaptionClipSpec with section="cta_overlay", or None if text is empty.
        """
        if not cta_overlay.strip():
            return None
        if video_duration is not None:
            start = max(0.0, video_duration - 3.0)
            end = video_duration
        else:
            start = CTA_OVERLAY_START
            end = CTA_OVERLAY_END
        y_pos = int(self.canvas_height * CTA_Y_RATIO)
        return CaptionClipSpec(
            section="cta_overlay",
            text=cta_overlay,
            start=start,
            end=end,
            x="center",
            y=y_pos,
        )

    def _render_word_by_word(
        self,
        word_timestamps: list[dict],
        cta_overlay: str = "",
        video_duration: float = 13.5,
    ) -> list:
        """Pre-render word-by-word captions as a single VideoClip with binary search.

        Renders one PIL frame per word (stable state after entrance animation),
        then wraps them in a single VideoClip whose make_frame uses bisect to look
        up the correct pre-baked frame at time t — keeping the compositor layer
        count constant regardless of word count.

        Args:
            word_timestamps: List of {text, start_time, end_time} from ElevenLabs.
            cta_overlay:     Optional CTA banner text (rendered via TextClip as before).
            video_duration:  Total video duration in seconds.

        Returns:
            List with one VideoClip (word captions) plus an optional CTA TextClip.
        """
        import bisect
        import numpy as np
        from moviepy import TextClip, VideoClip

        grouped = _group_words(word_timestamps)
        font = _load_word_font(WORD_FONT_SIZE)
        W, H = self.canvas_width, self.canvas_height

        # Pre-render one stable frame per word state
        states: list[tuple] = []   # (rgb_array uint8, alpha_array float64)
        start_times: list[float] = []

        for word in grouped:
            t_stable = word["start_time"] + WORD_ENTRANCE_DUR + 0.001
            img = _render_word_frame(t_stable, grouped, W, H, font)
            arr = np.array(img)
            rgb   = arr[:, :, :3].astype(np.uint8)
            alpha = (arr[:, :, 3] / 255.0).astype(np.float64)
            states.append((rgb, alpha))
            start_times.append(word["start_time"])

        # Blank frames returned before the first word starts
        blank_rgb   = np.zeros((H, W, 3), dtype=np.uint8)
        blank_alpha = np.zeros((H, W),    dtype=np.float64)

        def make_frame(t: float) -> np.ndarray:
            idx = bisect.bisect_right(start_times, t) - 1
            if idx < 0 or idx >= len(states):
                return blank_rgb
            return states[idx][0]

        def make_mask(t: float) -> np.ndarray:
            idx = bisect.bisect_right(start_times, t) - 1
            if idx < 0 or idx >= len(states):
                return blank_alpha
            return states[idx][1]

        caption_clip = (
            VideoClip(make_frame, duration=video_duration)
            .with_mask(VideoClip(make_mask, duration=video_duration, is_mask=True))
        )

        logger.info(
            "Pre-rendered %d word states into single VideoClip with binary search (%.1fs video)",
            len(states), video_duration,
        )

        clips: list = [caption_clip]

        # CTA overlay (gold TextClip banner, same as before)
        cta_spec = self.build_cta_spec(cta_overlay, video_duration=video_duration)
        if cta_spec is not None:
            cta_clip = (
                TextClip(
                    text=cta_spec.text,
                    font=self.font,
                    font_size=CTA_FONT_SIZE,
                    color=CTA_TEXT_COLOR,
                    bg_color=CTA_BG_COLOR,
                    method="caption",
                    size=(self.canvas_width - 80, None),
                    text_align="center",
                )
                .with_start(cta_spec.start)
                .with_duration(cta_spec.duration)
                .with_position((cta_spec.x, cta_spec.y))
            )
            clips.append(cta_clip)
            logger.debug("CTA overlay added at %.1fs-%.1fs", cta_spec.start, cta_spec.end)

        return clips

    def render(
        self,
        script_dict: dict[str, str],
        word_timestamps: list[dict] | None = None,
        cta_overlay: str = "",
        video_duration: float | None = None,
    ) -> list:
        """
        Build and return a list of positioned, timed moviepy TextClip objects.

        Args:
            script_dict:    Dict with keys hook, statement, twist, question.
            word_timestamps: Optional list of {text, start_time, end_time} dicts from
                             ElevenLabs. When supplied, uses word-by-word PIL rendering
                             with gold highlight on the current word. Falls back to
                             section-level TextClips when None.
            cta_overlay:    Optional CTA banner text for the last 3 seconds.
                            Rendered with a gold background and black bold text.
            video_duration: Actual video length in seconds. When supplied, caption
                            timings are scaled proportionally so captions stay in sync
                            with a longer voiceover, and the CTA banner lands in the
                            final 3 seconds of the real video.

        Returns:
            List of moviepy TextClip objects ready to be composited.
        """
        # Word-by-word rendering when timestamps are available
        if word_timestamps:
            dur = video_duration or (word_timestamps[-1]["end_time"] if word_timestamps else 13.5)
            return self._render_word_by_word(
                word_timestamps=word_timestamps,
                cta_overlay=cta_overlay,
                video_duration=dur,
            )

        from moviepy import TextClip  # lazy import — moviepy v2

        specs = self.build_specs(script_dict)

        # Scale caption timings proportionally when video is longer than the default
        _default_dur = CAPTION_TIMINGS[-1][2]  # 13.5 s
        if video_duration is not None and video_duration > _default_dur:
            scale = video_duration / _default_dur
            scaled: list[CaptionClipSpec] = []
            for spec in specs:
                scaled.append(CaptionClipSpec(
                    section=spec.section,
                    text=spec.text,
                    start=round(spec.start * scale, 3),
                    end=round(spec.end * scale, 3),
                    x=spec.x,
                    y=spec.y,
                ))
            specs = scaled

        clips = []

        for spec in specs:
            clip = (
                TextClip(
                    text=spec.text,
                    font=self.font,
                    font_size=self.font_size,
                    color=CAPTION_COLOR,
                    stroke_color=CAPTION_STROKE_COLOR,
                    stroke_width=CAPTION_STROKE_WIDTH,
                    method="caption",
                    size=(self.canvas_width - 80, None),
                    text_align="center",
                )
                .with_start(spec.start)
                .with_duration(spec.duration)
                .with_position((spec.x, spec.y))
            )
            clips.append(clip)

        # CTA overlay banner (gold background, black text)
        cta_spec = self.build_cta_spec(cta_overlay, video_duration=video_duration)
        if cta_spec is not None:
            cta_clip = (
                TextClip(
                    text=cta_spec.text,
                    font=self.font,
                    font_size=CTA_FONT_SIZE,
                    color=CTA_TEXT_COLOR,
                    bg_color=CTA_BG_COLOR,
                    method="caption",
                    size=(self.canvas_width - 80, None),
                    text_align="center",
                )
                .with_start(cta_spec.start)
                .with_duration(cta_spec.duration)
                .with_position((cta_spec.x, cta_spec.y))
            )
            clips.append(cta_clip)
            logger.debug("CTA overlay added at %.1fs–%.1fs", cta_spec.start, cta_spec.end)

        logger.info("Rendered %d caption clips", len(clips))
        return clips

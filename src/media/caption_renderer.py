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

    def render(self, script_dict: dict[str, str]) -> list:
        """
        Build and return a list of positioned, timed moviepy TextClip objects.

        Args:
            script_dict: Dict with keys hook, statement, twist, question.

        Returns:
            List of moviepy TextClip objects ready to be composited.
        """
        from moviepy import TextClip  # lazy import — moviepy v2

        specs = self.build_specs(script_dict)
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

        logger.info("Rendered %d caption clips", len(clips))
        return clips

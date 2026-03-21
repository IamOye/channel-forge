"""
caption_renderer.py — CaptionRenderer

Builds timed caption TextClip objects for each section of a YouTube Shorts script.

Style reference: VIZIONTIA YouTube Shorts captions
  - ALL CAPS, heavy font (Impact / Arial Black)
  - White text with 2-3px black stroke outline
  - Current word highlighted in gold (#FFD700) text colour
  - NO background pill, box, or badge
  - Positioned 75-80% from top of frame

Usage:
    renderer = CaptionRenderer(canvas_width=1080, canvas_height=1920)
    clips = renderer.render(script_dict)   # list of moviepy TextClip
"""

import logging
import os
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

CAPTION_FONT_CANDIDATES = ["Impact", "Arial-Black", "Arial-Bold", "Arial", None]
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

# Vertical position — 77% from the top of the frame
CAPTION_Y_RATIO = 0.77


# ---------------------------------------------------------------------------
# Word-by-word caption rendering constants
# ---------------------------------------------------------------------------

# Minimum acceptable font size (quality gate threshold)
MIN_CAPTION_FONT_SIZE = 40

# Bundled font — committed to repo, guaranteed available on all environments
_BUNDLED_FONT_PATHS: list[str] = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 '..', '..', 'assets', 'fonts', 'Roboto-Bold.ttf'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 '..', '..', 'assets', 'fonts', 'DejaVuSans-Bold.ttf'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 '..', '..', 'assets', 'fonts', 'LiberationSans-Bold.ttf'),
]

# System font candidates — fallback if bundled font is missing
WORD_FONT_SEARCH_PATHS: list[str] = [
    # Railway Linux (fonts-dejavu-core / fonts-liberation)
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
    # Downloaded/cached path
    "/app/fonts/DejaVuSans-Bold.ttf",
    # Windows dev paths
    "C:/Windows/Fonts/impact.ttf",
    "C:/Windows/Fonts/ariblk.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    # macOS
    "/System/Library/Fonts/Impact.ttf",
    "/System/Library/Fonts/Supplemental/Impact.ttf",
]

# Font download URLs (tried in order) and cache directory
_FONT_DOWNLOAD_URLS = [
    "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans-Bold.ttf",
    "https://github.com/liberationfonts/liberation-fonts/raw/main/src/LiberationSans-Bold.ttf",
    "https://github.com/matomo-org/travis-scripts/raw/master/fonts/Impact.ttf",
]
_FONT_CACHE_PATH = "/app/fonts/DejaVuSans-Bold.ttf"

HIGHLIGHT_TEXT_COLOR = (255, 215, 0)    # Gold #FFD700 — highlighted word text colour only
WORD_TEXT_COLOR      = (255, 255, 255)  # White — non-highlighted word text colour
WORD_STROKE_COLOR    = (0, 0, 0)        # Black outline on ALL text
WORD_GAP             = 14               # px gap between words
WORD_MAX_PER_LINE    = 3
WORD_CAPTION_Y_RATIO = 0.77            # 77% from top of frame, centred
WORD_ENTRANCE_DRIFT  = 4               # px upward drift on word entrance
WORD_ENTRANCE_DUR    = 0.08            # seconds for entrance animation


# ---------------------------------------------------------------------------
# Word-caption PIL helpers
# ---------------------------------------------------------------------------

def _word_font_size(canvas_w: int) -> int:
    """Compute word caption font size scaled to canvas width.

    Formula: round(canvas_w * 0.155)
    Gives ~56px at 360px canvas, ~167px at 1080px canvas.
    Never below MIN_CAPTION_FONT_SIZE (40px).
    """
    return max(MIN_CAPTION_FONT_SIZE, round(canvas_w * 0.155))


def _word_stroke_width(canvas_w: int) -> int:
    """Compute stroke width scaled to canvas. 2-3px at 360px, scales up."""
    return max(2, canvas_w // 160)


def _download_font(dest_path: str = _FONT_CACHE_PATH) -> bool:
    """Download a scalable font from GitHub CDN. Tries DejaVu, Liberation, Impact.

    Returns True if a valid truetype font was downloaded to dest_path.
    """
    from PIL import ImageFont
    import urllib.request

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    # Return immediately if already cached and valid
    if os.path.exists(dest_path):
        try:
            ImageFont.truetype(dest_path, size=20)
            return True
        except Exception:
            pass  # cached file is corrupt — re-download

    for url in _FONT_DOWNLOAD_URLS:
        try:
            urllib.request.urlretrieve(url, dest_path)
            # Verify it is a valid truetype font
            ImageFont.truetype(dest_path, size=20)
            logger.info("[caption] Downloaded font from %s to %s", url, dest_path)
            return True
        except Exception as exc:
            logger.warning("[caption] Font download failed from %s: %s", url, exc)
            continue

    return False


def _try_install_system_fonts() -> str | None:
    """Last resort: install fonts-dejavu-core via apt-get on Railway. Returns path or None."""
    target = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    try:
        import subprocess
        subprocess.run(
            ["apt-get", "install", "-y", "fonts-dejavu-core"],
            capture_output=True, check=True, timeout=30,
        )
        if os.path.exists(target):
            logger.info("[caption] Installed fonts-dejavu-core via apt-get")
            return target
    except Exception as exc:
        logger.debug("[caption] apt-get font install failed: %s", exc)
    return None


def _load_word_font(size: int | None = None, canvas_w: int = 1080):
    """Load a scalable truetype font for word captions.

    NEVER uses ImageFont.load_default() — that returns a fixed bitmap font
    that ignores size and renders at ~8px regardless. All paths here use
    ImageFont.truetype() which respects the requested size.

    Fallback chain:
      1. System truetype fonts (WORD_FONT_SEARCH_PATHS)
      2. Downloaded font from CDN (DejaVu → Liberation → Impact)
      3. Runtime apt-get install fonts-dejavu-core
      4. RuntimeError if nothing works

    If size is None, computes it from canvas_w.
    """
    if size is None:
        size = _word_font_size(canvas_w)
    try:
        from PIL import ImageFont
    except ImportError:
        return None

    # 0. Try bundled font first (committed to repo — guaranteed available)
    for path in _BUNDLED_FONT_PATHS:
        abspath = os.path.abspath(path)
        if os.path.exists(abspath) and os.path.getsize(abspath) > 10000:
            try:
                font = ImageFont.truetype(abspath, size)
                logger.info("[caption] Using bundled font: %s at %dpx", abspath, size)
                return font
            except (IOError, OSError):
                continue

    # 1. Try each system font path (all are truetype)
    for path in WORD_FONT_SEARCH_PATHS:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size)
                logger.info("[caption] Font loaded: %s at %dpx", path, size)
                return font
            except (IOError, OSError):
                continue

    # 2. Try downloading font from CDN (DejaVu → Liberation → Impact)
    if os.path.exists(_FONT_CACHE_PATH) or _download_font(_FONT_CACHE_PATH):
        try:
            font = ImageFont.truetype(_FONT_CACHE_PATH, size)
            logger.info("[caption] Font loaded from cache/download at %dpx", size)
            return font
        except (IOError, OSError) as exc:
            logger.error("[caption] Cache font failed: %s", exc)

    # 3. Try runtime apt-get install as last resort
    installed = _try_install_system_fonts()
    if installed:
        try:
            font = ImageFont.truetype(installed, size)
            logger.info("[caption] Font loaded: DejaVu Bold (installed) at %dpx", size)
            return font
        except (IOError, OSError):
            pass

    # 4. FATAL — no scalable font found. Do NOT fall back to load_default().
    logger.error(
        "[caption] FATAL: No scalable truetype font found. "
        "Cannot render captions at %dpx. Install fonts-dejavu-core on Railway.", size,
    )
    raise RuntimeError(
        f"[caption] No scalable font found. Cannot render captions at {size}px. "
        f"Install fonts-dejavu-core on Railway."
    )


def validate_font_rendering(canvas_w: int = 360) -> None:
    """Validate that the loaded font actually renders at the correct size.

    Creates a tiny test image, renders "TEST" and checks the glyph height.
    Called at startup to catch font issues early.

    Raises RuntimeError if rendered height < MIN_CAPTION_FONT_SIZE.
    """
    size = _word_font_size(canvas_w)
    font = _load_word_font(size=size, canvas_w=canvas_w)
    if font is None:
        return  # PIL not installed — skip validation

    try:
        bb = font.getbbox("TEST")
        height = bb[3] - bb[1]
        logger.info(
            "[caption] Font size validation: %dpx rendered at %dpx requested ✅",
            height, size,
        )
        if height < MIN_CAPTION_FONT_SIZE:
            raise RuntimeError(
                f"[caption] Font renders at {height}px but minimum is "
                f"{MIN_CAPTION_FONT_SIZE}px. The font is not scalable."
            )
    except RuntimeError:
        raise
    except Exception as exc:
        logger.warning("[caption] Font validation could not measure: %s", exc)


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


def _render_word_frame(
    t: float,
    grouped: list[dict],
    canvas_w: int,
    canvas_h: int,
    font,
) -> "Image.Image":
    """Render a single word-caption frame as an RGBA PIL Image.

    ALL CAPS text with black stroke outline — no background pill or badge.
    Current word: gold #FFD700 text colour.
    Previous words: white text colour.
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    current_idx, visible = _visible_at(t, grouped)
    if not visible:
        return img

    draw = ImageDraw.Draw(img)
    font_size = _word_font_size(canvas_w)
    stroke_w = _word_stroke_width(canvas_w)

    # Measure each visible word (ALL CAPS)
    entries: list[dict] = []
    for w in visible:
        text = w["text"].upper()
        try:
            bb = font.getbbox(text)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
        except Exception:
            tw, th = 60, font_size

        is_cur = (w["word_idx"] == current_idx)
        elapsed = t - w["start_time"]
        progress = min(1.0, elapsed / WORD_ENTRANCE_DUR) if is_cur else 1.0
        y_drift = -int(WORD_ENTRANCE_DRIFT * (1.0 - progress))

        entries.append({
            "text":    text,
            "tw":      tw,
            "th":      th,
            "y_drift": y_drift,
            "is_cur":  is_cur,
        })

    total_w = sum(e["tw"] for e in entries) + WORD_GAP * (len(entries) - 1)
    max_th = max(e["th"] for e in entries)

    # Vertical position: centred at WORD_CAPTION_Y_RATIO from top
    y_mid = int(canvas_h * WORD_CAPTION_Y_RATIO)
    x = (canvas_w - total_w) // 2

    # Draw text with stroke — NO background pill/badge of any kind
    for e in entries:
        y_top = y_mid - max_th // 2 + e["y_drift"]

        if e["is_cur"]:
            fill = (*HIGHLIGHT_TEXT_COLOR, 255)   # Gold #FFD700
        else:
            fill = (*WORD_TEXT_COLOR, 255)         # White

        # Text with black stroke outline (scaled to canvas)
        draw.text(
            (x, y_top), e["text"], font=font, fill=fill,
            stroke_width=stroke_w,
            stroke_fill=(*WORD_STROKE_COLOR, 255),
        )

        x += e["tw"] + WORD_GAP

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

    def get_caption_config(self) -> dict[str, Any]:
        """Return caption configuration for quality gate inspection."""
        return {
            "font_size": _word_font_size(self.canvas_width),
            "stroke_width": _word_stroke_width(self.canvas_width),
            "highlight_color": HIGHLIGHT_TEXT_COLOR,
            "text_color": WORD_TEXT_COLOR,
            "y_ratio": WORD_CAPTION_Y_RATIO,
        }

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
        font_size = _word_font_size(self.canvas_width)
        font = _load_word_font(size=font_size, canvas_w=self.canvas_width)
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

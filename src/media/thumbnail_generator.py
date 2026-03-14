"""
thumbnail_generator.py — ThumbnailGenerator

Generates a 1280x720 YouTube thumbnail using PIL.

Usage:
    gen = ThumbnailGenerator()
    path = gen.generate(hook="Working harder keeps you poor", topic="passive income", category="money")
    print(path)  # data/output/<topic_id>_thumb.jpg
"""

import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

THUMBNAIL_WIDTH  = 1280
THUMBNAIL_HEIGHT = 720
OUTPUT_DIR = Path("data/output")

# Brand colours
BLACK_BG   = (10, 10, 10)       # #0A0A0A
GOLD       = (201, 168, 76)     # #C9A84C
WHITE      = (255, 255, 255)
BLACK_TEXT = (0, 0, 0)

# Channel name shown in bottom strip
CHANNEL_NAME = "MONEY HERESY"

# Font search paths (Arial Black → Impact → fallback)
_FONT_PATHS = [
    "C:/Windows/Fonts/ariblk.ttf",   # Arial Black
    "C:/Windows/Fonts/arialbd.ttf",  # Arial Bold
    "/usr/share/fonts/truetype/msttcorefonts/Arial_Black.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    None,  # PIL default
]


class ThumbnailGenerator:
    """
    Generates 1280x720 JPEG thumbnails for YouTube Shorts.

    Args:
        output_dir: Directory to write thumbnail files.
    """

    def __init__(self, output_dir: str | Path = OUTPUT_DIR) -> None:
        self.output_dir = Path(output_dir)

    def generate(self, hook: str, topic: str, category: str = "money") -> str:
        """
        Generate a thumbnail and return its file path.

        Args:
            hook:     Hook text (first 6 words used in thumbnail).
            topic:    Topic identifier used for the output filename.
            category: Content category — controls right-side symbol.

        Returns:
            Absolute path to the saved JPEG file.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        from PIL import Image, ImageDraw

        img = Image.new("RGB", (THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT), BLACK_BG)
        draw = ImageDraw.Draw(img)

        # ── Right-side symbol ─────────────────────────────────────────
        symbol = self._pick_symbol(hook, category)
        self._draw_right_symbol(draw, symbol)

        # ── Left-side hook text ───────────────────────────────────────
        display_text = self._truncate_hook(hook, max_words=6)
        self._draw_hook_text(draw, display_text)

        # ── Bottom gold strip ─────────────────────────────────────────
        self._draw_bottom_strip(draw, img)

        # ── Vignette ─────────────────────────────────────────────────
        img = self._apply_vignette(img)

        # ── Save ──────────────────────────────────────────────────────
        safe_topic = re.sub(r"[^\w\-]", "_", str(topic))[:40]
        out_path = self.output_dir / f"{safe_topic}_thumb.jpg"
        img.save(str(out_path), "JPEG", quality=92)
        logger.info("[thumbnail] Saved: %s", out_path)
        return str(out_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _pick_symbol(self, hook: str, category: str) -> str:
        """Return ?, !, or $ based on hook content and category."""
        hook_lower = hook.lower()
        if category == "money" or any(
            w in hook_lower for w in ("dollar", "income", "salary", "earn", "money", "cost", "pay")
        ):
            return "$"
        if "?" in hook or any(
            w in hook_lower for w in ("why", "how", "what", "secret", "told", "never", "reason")
        ):
            return "?"
        return "!"

    def _truncate_hook(self, hook: str, max_words: int = 6) -> str:
        """Return the first max_words words of hook."""
        words = hook.split()
        if len(words) <= max_words:
            return hook
        return " ".join(words[:max_words])

    def _load_font(self, size: int):
        """Load the best available bold font at the requested size."""
        from PIL import ImageFont
        for path in _FONT_PATHS:
            if path is None:
                return ImageFont.load_default()
            try:
                return ImageFont.truetype(path, size)
            except (IOError, OSError):
                continue
        from PIL import ImageFont
        return ImageFont.load_default()

    def _find_provocative_word(self, text: str) -> str:
        """Return the most provocative word in text (first noun/adjective heuristic)."""
        # Prefer strong emotive words; fall back to longest word
        strong = ["broke", "poor", "rich", "never", "always", "lie", "secret",
                  "truth", "wrong", "mistake", "lose", "cost", "steal", "trap",
                  "fool", "myth", "scam", "hack", "cheat", "fail", "win"]
        lower = text.lower()
        for w in strong:
            if w in lower.split():
                # Find original-case version
                for word in text.split():
                    if word.lower().rstrip(",.!?") == w:
                        return word.rstrip(",.!?")
        # Fall back: longest word
        words = [w.rstrip(",.!?") for w in text.split() if len(w) > 3]
        return max(words, key=len) if words else (text.split()[0] if text else "")

    def _draw_hook_text(self, draw, text: str) -> None:
        """Draw bold hook text on the left 65% of canvas."""
        from PIL import ImageFont

        left_width = int(THUMBNAIL_WIDTH * 0.65)
        margin = 80
        max_text_w = left_width - 2 * margin

        # Find font size that fits
        font_size = 80
        font = self._load_font(font_size)
        while font_size > 28:
            try:
                bb = font.getbbox(text)
                tw = bb[2] - bb[0]
            except Exception:
                tw = font_size * len(text) * 0.6
            if tw <= max_text_w:
                break
            font_size -= 4
            font = self._load_font(font_size)

        # Wrap into lines if still too wide
        lines = self._wrap_text(text, font, max_text_w)

        # Measure total height
        line_height = font_size + 12
        total_h = len(lines) * line_height
        y = (THUMBNAIL_HEIGHT - 60 - total_h) // 2  # centre vertically above bottom strip

        provocative = self._find_provocative_word(text)

        for line in lines:
            try:
                bb = font.getbbox(line)
                tw = bb[2] - bb[0]
            except Exception:
                tw = font_size * len(line) * 0.6
            x = margin

            # Draw white text
            draw.text((x, y), line, font=font, fill=WHITE)

            # Gold underline under provocative word if it's in this line
            if provocative and provocative.lower() in line.lower():
                # Find position of provocative word within the line
                idx = line.lower().find(provocative.lower())
                before = line[:idx]
                try:
                    bef_bb = font.getbbox(before) if before else (0, 0, 0, 0)
                    prov_bb = font.getbbox(provocative)
                    x_start = x + (bef_bb[2] - bef_bb[0])
                    x_end   = x_start + (prov_bb[2] - prov_bb[0])
                except Exception:
                    x_start = x
                    x_end   = x + tw
                underline_y = y + line_height - 8
                draw.rectangle(
                    [x_start, underline_y, x_end, underline_y + 4],
                    fill=GOLD,
                )

            y += line_height

    def _wrap_text(self, text: str, font, max_width: int) -> list[str]:
        """Wrap text into lines that fit within max_width pixels."""
        words = text.split()
        lines: list[str] = []
        current = ""
        for word in words:
            test = (current + " " + word).strip()
            try:
                bb = font.getbbox(test)
                tw = bb[2] - bb[0]
            except Exception:
                tw = len(test) * 14
            if tw <= max_width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines or [text]

    def _draw_right_symbol(self, draw, symbol: str) -> None:
        """Draw a large gold symbol on the right 35% of canvas."""
        from PIL import ImageFont

        font = self._load_font(280)
        right_start = int(THUMBNAIL_WIDTH * 0.65)
        right_w = THUMBNAIL_WIDTH - right_start
        center_x = right_start + right_w // 2
        center_y = (THUMBNAIL_HEIGHT - 60) // 2

        try:
            bb = font.getbbox(symbol)
            sw = bb[2] - bb[0]
            sh = bb[3] - bb[1]
        except Exception:
            sw, sh = 200, 250

        x = center_x - sw // 2
        y = center_y - sh // 2
        draw.text((x, y), symbol, font=font, fill=GOLD)

    def _draw_bottom_strip(self, draw, img) -> None:
        """Draw a gold bottom strip with channel name."""
        strip_h = 60
        y0 = THUMBNAIL_HEIGHT - strip_h
        draw.rectangle([0, y0, THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT], fill=GOLD)

        font = self._load_font(24)
        margin = 30
        try:
            bb = font.getbbox(CHANNEL_NAME)
            tw = bb[2] - bb[0]
            th = bb[3] - bb[1]
        except Exception:
            tw, th = 200, 24

        x = THUMBNAIL_WIDTH - margin - tw
        y = y0 + (strip_h - th) // 2
        draw.text((x, y), CHANNEL_NAME, font=font, fill=BLACK_TEXT)

    def _apply_vignette(self, img):
        """Apply a subtle dark gradient vignette on all 4 edges (40% opacity)."""
        from PIL import Image, ImageDraw
        import math

        vignette = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(vignette)

        w, h = img.size
        max_dim = max(w, h)
        steps = 80
        for i in range(steps):
            alpha = int(100 * (1 - i / steps))  # 40% max
            inset = int(max_dim * 0.35 * i / steps)
            # Guard: rectangle is invalid if inset crosses the midpoint
            if inset * 2 >= w or inset * 2 >= h:
                break
            draw.rectangle(
                [inset, inset, w - inset, h - inset],
                outline=(0, 0, 0, alpha),
            )

        img_rgba = img.convert("RGBA")
        result = Image.alpha_composite(img_rgba, vignette)
        return result.convert("RGB")

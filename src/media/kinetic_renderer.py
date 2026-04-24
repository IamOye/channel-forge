"""
kinetic_renderer.py — KineticRenderer
Produces a pure kinetic typography YouTube Short (1080x1920) from script,
voiceover audio, and word timestamps. No b-roll required.

Architecture:
  1. Script parser    — classifies words: HERO / STAT / BODY / CTA
  2. Hook selector    — picks opening animation A / B / C from script
  3. Background engine — drifting grid + vignette + ghost echo layer
  4. Animation scheduler — assigns animation type + timing per word
  5. Frame renderer   — PIL frames piped to ffmpeg stdin at 30fps
  6. SFX scheduler    — mixes voice + SFX via ffmpeg filter_complex

Output matches VideoBuilder.BuildResult interface exactly.
"""
from __future__ import annotations

import logging
import math
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CANVAS_W      = 1080
CANVAS_H      = 1920
FPS           = 30
OUTPUT_DIR    = Path("data/output")

# Colours
C_BLACK       = (0,   0,   0,   255)
C_GOLD        = (255, 184,   0, 255)
C_WHITE       = (245, 245, 245, 255)
C_RED         = (255,  51,  51, 255)
C_GREEN       = (0,  255, 136, 255)
C_GREY        = (51,  51,  51, 255)
C_GRID        = (13,  32,  16, 255)

ASSETS        = Path("assets")
FONTS_DIR     = ASSETS / "fonts"
SFX_DIR       = ASSETS / "sfx"

# Font paths
F_BEBAS       = str(FONTS_DIR / "BebasNeue-Regular.ttf")
F_MONTSERRAT  = str(FONTS_DIR / "Montserrat-ExtraBold.ttf")
F_OSWALD      = str(FONTS_DIR / "Oswald-Bold.ttf")
F_ROBOTO      = str(FONTS_DIR / "Roboto-Bold.ttf")

# SFX paths
SFX_IMPACT    = str(SFX_DIR / "impact.mp3")
SFX_WHOOSH    = str(SFX_DIR / "whoosh.mp3")
SFX_WHOOSH1   = str(SFX_DIR / "whoosh1.mp3")
SFX_CASH      = str(SFX_DIR / "cash.mp3")
SFX_CASH1     = str(SFX_DIR / "cash1.mp3")
SFX_RISER     = str(SFX_DIR / "riser.mp3")

# Word classification triggers
HERO_WORDS    = {
    "never", "always", "stop", "start", "truth", "lie", "myth",
    "secret", "system", "rich", "poor", "broke", "wealth", "money",
    "salary", "income", "freedom", "trap", "wrong", "real",
}
STAT_PATTERN  = re.compile(r"^\d[\d,\.%kKmMbB]*$")
MONEY_WORDS   = {"money", "cash", "salary", "income", "wealth", "pay",
                 "wage", "invest", "profit", "debt", "bank"}


# ---------------------------------------------------------------------------
# BuildResult — identical interface to VideoBuilder.BuildResult
# ---------------------------------------------------------------------------
@dataclass
class BuildResult:
    topic_id:          str
    output_path:       str
    duration_seconds:  float
    is_valid:          bool
    validation_errors: list[str] = field(default_factory=list)
    built_at:          str = ""

    def __post_init__(self) -> None:
        if not self.built_at:
            self.built_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic_id":          self.topic_id,
            "output_path":       self.output_path,
            "duration_seconds":  self.duration_seconds,
            "is_valid":          self.is_valid,
            "validation_errors": self.validation_errors,
            "built_at":          self.built_at,
        }


# ---------------------------------------------------------------------------
# Word event — one animated word unit
# ---------------------------------------------------------------------------
@dataclass
class WordEvent:
    text:       str
    start:      float       # seconds
    end:        float
    role:       str         # HERO / STAT / BODY / CTA
    anim:       str         # SLAM / PUNCH / SLIDE_L / SLIDE_R / TYPEWRITER / FADE_RISE
    colour:     tuple
    font_path:  str
    font_size:  int
    sfx:        str | None  # path to SFX file or None
    x_align:    str         # LEFT / CENTRE / RIGHT
    y_frac:     float       # 0.0–1.0 vertical position


# ---------------------------------------------------------------------------
# KineticRenderer
# ---------------------------------------------------------------------------
class KineticRenderer:
    """
    Renders a kinetic typography Short from script + voiceover.
    Drop-in replacement for VideoBuilder — same build() interface.
    """

    def build(
        self,
        topic_id:          str,
        script_dict:       dict[str, str],
        audio_path:        str | Path,
        word_timestamps:   list[dict] | None = None,
        cta_overlay:       str = "",
        anthropic_api_key: str = "",
        # VideoBuilder compat — ignored
        stock_video_path:  Any = None,
    ) -> BuildResult:

        start_ts = time.time()
        audio_path = Path(audio_path)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"{topic_id}_kinetic.mp4"

        errors: list[str] = []

        # Determine video duration from audio
        duration = self._audio_duration(audio_path)
        if duration <= 0:
            duration = 50.0

        # Parse script into word events
        events = self._build_events(script_dict, word_timestamps, duration, cta_overlay)

        # Detect hook type
        hook_type = self._detect_hook_type(script_dict.get("hook", ""))

        # Schedule SFX
        sfx_schedule = self._build_sfx_schedule(events, duration)

        # Render frames + assemble
        try:
            self._render(
                topic_id=topic_id,
                output_path=output_path,
                audio_path=audio_path,
                duration=duration,
                events=events,
                hook_type=hook_type,
                sfx_schedule=sfx_schedule,
            )
            elapsed = time.time() - start_ts
            logger.info("[kinetic] Built %s in %.1fs", output_path.name, elapsed)
            return BuildResult(
                topic_id=topic_id,
                output_path=str(output_path),
                duration_seconds=duration,
                is_valid=True,
            )
        except Exception as exc:
            logger.error("[kinetic] Render failed: %s", exc)
            errors.append(str(exc))
            return BuildResult(
                topic_id=topic_id,
                output_path="",
                duration_seconds=0.0,
                is_valid=False,
                validation_errors=errors,
            )

    # ------------------------------------------------------------------
    # Script parsing → WordEvent list
    # ------------------------------------------------------------------

    def _build_events(
        self,
        script_dict:     dict[str, str],
        word_timestamps: list[dict] | None,
        duration:        float,
        cta_overlay:     str,
    ) -> list[WordEvent]:

        # Merge script parts in order
        parts_order = ["hook", "statement", "twist", "landing", "question", "cta"]
        full_text = " ".join(
            script_dict.get(p, "") for p in parts_order
            if script_dict.get(p, "").strip()
        )

        if word_timestamps:
            return self._events_from_timestamps(word_timestamps, script_dict, duration)
        else:
            return self._events_evenly_spaced(full_text, duration)

    def _events_from_timestamps(
        self,
        wts:         list[dict],
        script_dict: dict[str, str],
        duration:    float,
    ) -> list[WordEvent]:

        # Identify CTA start time — last 8 seconds
        cta_start = duration - 8.0
        events: list[WordEvent] = []
        anim_cycle = ["SLAM", "SLIDE_L", "PUNCH", "SLIDE_R", "FADE_RISE"]
        cycle_i = 0

        next_starts = []
        for i in range(len(wts)):
            nxt = float(wts[i+1].get("start_time", duration)) if i+1 < len(wts) else duration
            next_starts.append(nxt)

        for i, wt in enumerate(wts):
            text  = wt.get("text", "").strip().strip(".,!?;:")
            start = float(wt.get("start_time", 0.0))
            end   = float(wt.get("end_time",   start + 0.4))
            if not text:
                continue

            end = min(end, next_starts[i] - 0.03)
            end = max(end, start + 0.15)    # guarantee ≥0.15s visibility

            role, colour, font_path, font_size, sfx = self._classify_word(
                text, start, cta_start
            )

            # Animation type
            if role == "HERO":
                anim = "SPLIT"
            elif role == "STAT":
                anim = "PUNCH"
            elif role == "CTA":
                anim = "FADE_RISE"
            else:
                anim = anim_cycle[cycle_i % len(anim_cycle)]
                cycle_i += 1

            events.append(WordEvent(
                text=text, start=start, end=end,
                role=role, anim=anim, colour=colour,
                font_path=font_path, font_size=font_size,
                sfx=sfx,
                x_align="CENTRE",
                y_frac=0.5,
            ))

        return events

    def _events_evenly_spaced(self, full_text: str, duration: float) -> list[WordEvent]:
        words = full_text.split()
        if not words:
            return []
        interval = duration / len(words)
        cta_start = duration - 8.0
        events: list[WordEvent] = []
        anim_cycle = ["SLAM", "SLIDE_L", "PUNCH", "SLIDE_R", "FADE_RISE"]
        cycle_i = 0

        for i, word in enumerate(words):
            text  = word.strip(".,!?;:")
            start = i * interval
            end   = start + interval * 0.85
            role, colour, font_path, font_size, sfx = self._classify_word(
                text, start, cta_start
            )
            anim = "SPLIT" if role == "HERO" else "PUNCH" if role == "STAT" \
                else "FADE_RISE" if role == "CTA" \
                else anim_cycle[cycle_i % len(anim_cycle)]
            if role not in ("HERO", "STAT", "CTA"):
                cycle_i += 1

            events.append(WordEvent(
                text=text, start=start, end=end,
                role=role, anim=anim, colour=colour,
                font_path=font_path, font_size=font_size,
                sfx=sfx, x_align="CENTRE",
                y_frac=0.5,
            ))
        return events

    def _classify_word(
        self, text: str, start: float, cta_start: float
    ) -> tuple[str, tuple, str, int, str | None]:

        lower = text.lower()

        if start >= cta_start:
            return ("CTA", C_GREEN, F_MONTSERRAT, 170, None)

        if STAT_PATTERN.match(text):
            return ("STAT", C_GOLD, F_MONTSERRAT, 300, SFX_IMPACT)

        if lower in HERO_WORDS:
            return ("HERO", C_GOLD, F_MONTSERRAT, 280, SFX_IMPACT)

        if lower in MONEY_WORDS:
            return ("BODY", C_WHITE, F_OSWALD, 150, SFX_CASH)

        return ("BODY", C_WHITE, F_ROBOTO, 130, None)

    # ------------------------------------------------------------------
    # Hook type detection
    # ------------------------------------------------------------------

    def _detect_hook_type(self, hook: str) -> str:
        if STAT_PATTERN.search(hook.split()[0]) if hook.split() else False:
            return "B"
        first = hook.strip().lower()
        if any(first.startswith(q) for q in ("why", "what", "how", "do ", "are ", "is ")):
            return "C"
        return "A"

    # ------------------------------------------------------------------
    # SFX schedule
    # ------------------------------------------------------------------

    def _build_sfx_schedule(
        self, events: list[WordEvent], duration: float
    ) -> list[tuple[float, str]]:
        schedule: list[tuple[float, str]] = []
        # Riser at CTA start
        cta_start = duration - 8.0
        if Path(SFX_RISER).exists():
            schedule.append((max(0, cta_start - 0.8), SFX_RISER))

        whoosh_toggle = True
        for ev in events:
            if ev.sfx and Path(ev.sfx).exists():
                schedule.append((ev.start, ev.sfx))
            elif ev.role == "BODY":
                sfx = SFX_WHOOSH if whoosh_toggle else SFX_WHOOSH1
                whoosh_toggle = not whoosh_toggle
                if Path(sfx).exists():
                    schedule.append((ev.start, sfx))

        return sorted(schedule, key=lambda x: x[0])

    # ------------------------------------------------------------------
    # Frame rendering + ffmpeg assembly
    # ------------------------------------------------------------------

    def _render(
        self,
        topic_id:     str,
        output_path:  Path,
        audio_path:   Path,
        duration:     float,
        events:       list[WordEvent],
        hook_type:    str,
        sfx_schedule: list[tuple[float, str]],
    ) -> None:

        total_frames = int(duration * FPS)
        W, H = CANVAS_W, CANVAS_H

        # Pre-load fonts
        fonts = self._preload_fonts()

        # Build ffmpeg command
        cmd = [
            "ffmpeg", "-y",
            "-loglevel", "warning",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{W}x{H}",
            "-pix_fmt", "rgba",
            "-r", str(FPS),
            "-i", "pipe:0",
            "-i", str(audio_path),
        ]

        # Add SFX inputs
        sfx_inputs = []
        for _, sfx_path in sfx_schedule:
            if sfx_path not in sfx_inputs:
                sfx_inputs.append(sfx_path)
                cmd += ["-i", sfx_path]

        # Audio filter_complex
        filter_complex, audio_map = self._build_audio_filter(
            sfx_schedule, sfx_inputs, duration
        )
        if filter_complex:
            cmd += ["-filter_complex", filter_complex, "-map", "0:v", "-map", audio_map]
        else:
            cmd += ["-map", "0:v", "-map", "1:a"]

        cmd += [
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-shortest",
            "-movflags", "+faststart",
            str(output_path),
        ]

        logger.info("[kinetic] Rendering %d frames → %s", total_frames, output_path.name)

        logger.info("[kinetic] ffmpeg cmd: %s", " ".join(cmd))

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

        # Drain stderr continuously so ffmpeg never blocks on a full pipe.
        stderr_chunks = []

        def _drain_stderr():
            try:
                while True:
                    chunk = proc.stderr.read(4096)
                    if not chunk:
                        break
                    stderr_chunks.append(chunk)
            except Exception:
                pass

        stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        stderr_thread.start()

        failed_frame = None
        write_error = None

        try:
            for frame_i in range(total_frames):
                t = frame_i / FPS
                frame = self._make_frame(
                    t=t,
                    frame_i=frame_i,
                    total_frames=total_frames,
                    events=events,
                    hook_type=hook_type,
                    fonts=fonts,
                    W=W, H=H,
                )
                try:
                    proc.stdin.write(frame.tobytes())
                except (BrokenPipeError, OSError, ValueError) as exc:
                    failed_frame = frame_i
                    write_error = exc
                    break

            try:
                proc.stdin.close()
            except (BrokenPipeError, OSError, ValueError):
                pass

            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

            stderr_thread.join(timeout=5)
            stderr_text = b"".join(stderr_chunks).decode("utf-8", errors="replace")
            stderr_tail = stderr_text[-2000:] if stderr_text else "<empty>"

            if write_error is not None:
                logger.error(
                    "[kinetic] ffmpeg died at frame %d/%d (%.1fs). "
                    "write_err=%r. ffmpeg stderr tail:\n%s",
                    failed_frame,
                    total_frames,
                    (failed_frame or 0) / FPS,
                    write_error,
                    stderr_tail,
                )
                raise RuntimeError(
                    f"ffmpeg died at frame {failed_frame}/{total_frames}: "
                    f"write_err={write_error!r}; "
                    f"stderr_tail={stderr_text[-500:] or '<empty>'}"
                )

            if proc.returncode != 0:
                logger.error(
                    "[kinetic] ffmpeg exit %d. stderr tail:\n%s",
                    proc.returncode, stderr_tail,
                )
                raise RuntimeError(
                    f"ffmpeg exit {proc.returncode}: "
                    f"stderr_tail={stderr_text[-500:] or '<empty>'}"
                )

            if stderr_text.strip():
                logger.warning(
                    "[kinetic] ffmpeg warnings:\n%s",
                    stderr_text[-1500:],
                )

        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
            raise

    def _make_frame(
        self,
        t:            float,
        frame_i:      int,
        total_frames: int,
        events:       list[WordEvent],
        hook_type:    str,
        fonts:        dict,
        W:            int,
        H:            int,
    ) -> np.ndarray:

        img = Image.new("RGBA", (W, H), (0, 0, 0, 255))
        draw = ImageDraw.Draw(img)

        # 1. Background grid (drifting)
        self._draw_grid(draw, t, W, H)

        # 2. Vignette
        self._draw_vignette(img, W, H)

        # 3. Segment flash (8-frame black between major scenes)
        if self._is_flash_frame(t):
            return np.zeros((H, W, 4), dtype=np.uint8)

        # 4. Ghost echo layer — previous word at low opacity
        self._draw_ghost_echo(draw, t, events, W, H, fonts)

        # 5. Active word animations
        for ev in events:
            if ev.start - 0.05 <= t <= ev.end + 0.02:
                self._draw_word_event(draw, img, ev, t, W, H, fonts)

        return np.array(img, dtype=np.uint8)

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_grid(self, draw: ImageDraw.Draw, t: float, W: int, H: int) -> None:
        grid_size = 60
        speed = 1.5
        # Direction shifts per time segment
        if t < 7:
            dx, dy = speed, 0
        elif t < 20:
            dx, dy = speed * 0.7, speed * 0.7
        elif t < 35:
            dx, dy = -speed * 0.7, speed * 0.7
        elif t < 43:
            dx, dy = 0, speed
        else:
            dx, dy = 0, -speed

        offset_x = int(t * dx) % grid_size
        offset_y = int(t * dy) % grid_size

        grid_colour = (13, 32, 16, 18)

        for x in range(-grid_size + offset_x, W + grid_size, grid_size):
            draw.line([(x, 0), (x, H)], fill=grid_colour, width=1)
        for y in range(-grid_size + offset_y, H + grid_size, grid_size):
            draw.line([(0, y), (W, y)], fill=grid_colour, width=1)

        # Grid flash on impact — bright pulse for 3 frames
        if self._is_impact_moment(t):
            flash_colour = (26, 58, 26, 30)
            for x in range(-grid_size + offset_x, W + grid_size, grid_size):
                draw.line([(x, 0), (x, H)], fill=flash_colour, width=1)

    def _draw_vignette(self, img: Image.Image, W: int, H: int) -> None:
        vignette = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        v_draw = ImageDraw.Draw(vignette)
        steps = 40
        for i in range(steps):
            alpha = int(80 * (i / steps) ** 2)
            margin = int(i * (min(W, H) / 2) / steps)
            v_draw.rectangle(
                [margin, margin, W - margin, H - margin],
                outline=(0, 0, 0, alpha), width=1
            )
        img.alpha_composite(vignette)

    def _draw_hook_intro(
        self,
        draw:      ImageDraw.Draw,
        t:         float,
        hook_type: str,
        W:         int,
        H:         int,
        fonts:     dict,
    ) -> None:
        # Type A — sweeping line morphs into text
        if hook_type == "A":
            progress = t / 0.8
            x_end = int(W * min(progress, 1.0))
            y = H // 2
            if x_end > 0:
                draw.line([(0, y), (x_end, y)], fill=(245, 245, 245, 200), width=3)
            if t > 0.9:
                fade = min(1.0, (t - 0.9) / 0.3)
                alpha = int(255 * fade)
                font = fonts.get("bebas_180")
                if font:
                    draw.text(
                        (W // 2, H // 2),
                        "...",
                        font=font,
                        fill=(255, 184, 0, alpha),
                        anchor="mm",
                    )

        # Type B — stat appears instantly
        elif hook_type == "B":
            if t > 0.1:
                font = fonts.get("mont_200")
                if font:
                    draw.text(
                        (W // 2, H // 2),
                        "?",
                        font=font,
                        fill=(255, 184, 0, 255),
                        anchor="mm",
                    )

        # Type C — typewriter
        elif hook_type == "C":
            chars_per_sec = 20
            n_chars = int(t * chars_per_sec)
            text = "WHY?"[:n_chars]
            font = fonts.get("bebas_140")
            if font and text:
                draw.text(
                    (W // 2, H // 2),
                    text,
                    font=font,
                    fill=(245, 245, 245, 255),
                    anchor="mm",
                )

    def _draw_ghost_echo(
        self,
        draw:   ImageDraw.Draw,
        t:      float,
        events: list[WordEvent],
        W:      int,
        H:      int,
        fonts:  dict,
    ) -> None:
        for ev in events:
            if ev.end < t <= ev.end + 2.0:
                fade = 1.0 - (t - ev.end) / 2.0
                alpha = int(13 * fade)
                if alpha <= 0:
                    continue
                font = self._get_font(ev.font_path, max(1, ev.font_size // 2), fonts)
                if not font:
                    continue
                x = W // 2 + 12
                y = int(H * ev.y_frac) + 12
                r, g, b = ev.colour[:3]
                draw.text(
                    (x, y), ev.text.upper(),
                    font=font,
                    fill=(r, g, b, alpha),
                    anchor="mm",
                )

    def _draw_word_event(
        self,
        draw:   ImageDraw.Draw,
        img:    Image.Image,
        ev:     WordEvent,
        t:      float,
        W:      int,
        H:      int,
        fonts:  dict,
    ) -> None:
        anim_t = t - ev.start
        total  = max(ev.end - ev.start, 0.1)
        frac   = min(anim_t / 0.25, 1.0)   # 0→1 in first 0.25s

        font = self._get_font(ev.font_path, ev.font_size, fonts)
        if not font:
            return

        r, g, b = ev.colour[:3]
        alpha = 255

        base_x = W // 2
        base_y = int(H * ev.y_frac)

        if ev.anim == "SPLIT":
            # HERO words: negative-space band reveal. Accent bg on top (black text),
            # black bg on bottom (accent text). Band wipes open from center.
            self._draw_negative_split_hero(
                img, ev.text.upper(), font,
                cx=base_x, cy=base_y,
                W=W, H=H,
                anim_frac=frac,
                accent=(r, g, b, 255),
            )
            return

        if ev.anim == "SLAM":
            # Drop from above with overshoot
            if frac < 1.0:
                ease = self._ease_out_bounce(frac)
                y_offset = int((1.0 - ease) * -120)
                scale = 1.0 + (1.0 - frac) * 0.3
            else:
                y_offset = 0
                scale = 1.0
            self._draw_scaled_text(
                draw, img, ev.text.upper(), font, ev.font_size,
                base_x, base_y + y_offset, (r, g, b, alpha), scale
            )

        elif ev.anim == "PUNCH":
            # Scale from 0 → 130% → 100%
            if frac < 0.7:
                scale = frac / 0.7 * 1.3
            else:
                scale = 1.3 - (frac - 0.7) / 0.3 * 0.3
            self._draw_scaled_text(
                draw, img, ev.text.upper(), font, ev.font_size,
                base_x, base_y, (r, g, b, alpha), max(scale, 0.01)
            )

        elif ev.anim == "SLIDE_L":
            x_offset = int((1.0 - self._ease_out(frac)) * -W * 0.6)
            draw.text(
                (base_x + x_offset, base_y),
                ev.text.upper(), font=font,
                fill=(r, g, b, alpha), anchor="mm",
            )

        elif ev.anim == "SLIDE_R":
            x_offset = int((1.0 - self._ease_out(frac)) * W * 0.6)
            draw.text(
                (base_x + x_offset, base_y),
                ev.text.upper(), font=font,
                fill=(r, g, b, alpha), anchor="mm",
            )

        elif ev.anim == "TYPEWRITER":
            n_chars = max(1, int(frac * len(ev.text)))
            draw.text(
                (base_x, base_y),
                ev.text.upper()[:n_chars], font=font,
                fill=(r, g, b, alpha), anchor="mm",
            )

        elif ev.anim == "FADE_RISE":
            alpha = int(255 * self._ease_out(frac))
            y_offset = int((1.0 - self._ease_out(frac)) * 20)
            draw.text(
                (base_x, base_y - y_offset),
                ev.text.upper(), font=font,
                fill=(r, g, b, alpha), anchor="mm",
            )

        # Glow on HERO/STAT words
        if ev.role in ("HERO", "STAT") and frac > 0.8:
            self._draw_glow(draw, ev.text.upper(), font, base_x, base_y, (r, g, b))

    def _draw_scaled_text(
        self,
        draw:      ImageDraw.Draw,
        img:       Image.Image,
        text:      str,
        font:      ImageFont.FreeTypeFont,
        font_size: int,
        cx:        int,
        cy:        int,
        colour:    tuple,
        scale:     float,
    ) -> None:
        if abs(scale - 1.0) < 0.02:
            draw.text((cx, cy), text, font=font, fill=colour, anchor="mm")
            return
        # Render to temp surface then resize
        try:
            bb = font.getbbox(text)
            tw = bb[2] - bb[0] + 40
            th = bb[3] - bb[1] + 40
        except Exception:
            tw, th = font_size * len(text), font_size + 40

        tmp = Image.new("RGBA", (max(tw, 1), max(th, 1)), (0, 0, 0, 0))
        tmp_draw = ImageDraw.Draw(tmp)
        tmp_draw.text((tw // 2, th // 2), text, font=font, fill=colour, anchor="mm")
        new_w = max(1, int(tw * scale))
        new_h = max(1, int(th * scale))
        tmp = tmp.resize((new_w, new_h), Image.LANCZOS)
        x0 = cx - new_w // 2
        y0 = cy - new_h // 2
        img.alpha_composite(tmp, (max(0, x0), max(0, y0)))

    def _draw_glow(
        self,
        draw:   ImageDraw.Draw,
        text:   str,
        font:   ImageFont.FreeTypeFont,
        cx:     int,
        cy:     int,
        colour: tuple,
    ) -> None:
        r, g, b = colour
        for offset, alpha in [(6, 15), (4, 25), (2, 35)]:
            for dx, dy in [(-offset, 0), (offset, 0), (0, -offset), (0, offset)]:
                draw.text(
                    (cx + dx, cy + dy), text,
                    font=font,
                    fill=(r, g, b, alpha),
                    anchor="mm",
                )

    def _draw_negative_split_hero(
        self,
        img:       Image.Image,
        text:      str,
        font:      ImageFont.FreeTypeFont,
        cx:        int,
        cy:        int,
        W:         int,
        H:         int,
        anim_frac: float,
        accent:    tuple,
        band_h_mult: float = 1.8,
    ) -> None:
        """
        Negative-space split band for HERO words.
        Top half: accent background, black text. Bottom half: black background, accent text.
        Band spans full canvas width and wipes open from centre outward.
        """
        bb = font.getbbox(text)
        text_h = bb[3] - bb[1]

        band_h = max(1, int(text_h * band_h_mult))
        band_y0 = cy - band_h // 2
        split_y_local = band_h // 2

        # Ease-out cubic reveal
        ease = 1.0 - (1.0 - min(max(anim_frac, 0.0), 1.0)) ** 3
        reveal_w = max(2, int(W * ease))
        reveal_x0 = (W - reveal_w) // 2

        black = (0, 0, 0, 255)

        # Full-width band panel with split backgrounds
        panel = Image.new("RGBA", (W, band_h), (0, 0, 0, 0))
        pd = ImageDraw.Draw(panel)
        pd.rectangle([0, 0, W, split_y_local], fill=accent)
        pd.rectangle([0, split_y_local, W, band_h], fill=black)

        cx_panel = W // 2

        # Top-half text in BLACK — mask alpha to top half then composite
        tl = Image.new("RGBA", (W, band_h), (0, 0, 0, 0))
        ImageDraw.Draw(tl).text(
            (cx_panel, band_h // 2), text, font=font, fill=black, anchor="mm"
        )
        top_rect = Image.new("L", (W, band_h), 0)
        ImageDraw.Draw(top_rect).rectangle([0, 0, W, split_y_local], fill=255)
        tl.putalpha(ImageChops.multiply(tl.split()[-1], top_rect))
        panel.alpha_composite(tl)

        # Bottom-half text in ACCENT — same pattern
        bl = Image.new("RGBA", (W, band_h), (0, 0, 0, 0))
        ImageDraw.Draw(bl).text(
            (cx_panel, band_h // 2), text, font=font, fill=accent, anchor="mm"
        )
        bot_rect = Image.new("L", (W, band_h), 0)
        ImageDraw.Draw(bot_rect).rectangle([0, split_y_local, W, band_h], fill=255)
        bl.putalpha(ImageChops.multiply(bl.split()[-1], bot_rect))
        panel.alpha_composite(bl)

        # Crop to reveal window and composite onto main canvas
        cropped = panel.crop((reveal_x0, 0, reveal_x0 + reveal_w, band_h))
        paste_y = max(0, band_y0)
        img.alpha_composite(cropped, (reveal_x0, paste_y))

    # ------------------------------------------------------------------
    # Easing functions
    # ------------------------------------------------------------------

    @staticmethod
    def _ease_out(t: float) -> float:
        return 1.0 - (1.0 - min(t, 1.0)) ** 3

    @staticmethod
    def _ease_out_bounce(t: float) -> float:
        t = min(t, 1.0)
        n1, d1 = 7.5625, 2.75
        if t < 1 / d1:
            return n1 * t * t
        elif t < 2 / d1:
            t -= 1.5 / d1
            return n1 * t * t + 0.75
        elif t < 2.5 / d1:
            t -= 2.25 / d1
            return n1 * t * t + 0.9375
        else:
            t -= 2.625 / d1
            return n1 * t * t + 0.984375

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_flash_frame(self, t: float) -> bool:
        flash_times = [7.0, 20.0, 35.0, 43.0]
        flash_duration = 3 / FPS
        return any(abs(t - ft) < flash_duration for ft in flash_times)

    def _is_impact_moment(self, t: float) -> bool:
        impact_times = [1.2, 7.0, 20.0, 35.0, 43.0]
        pulse_duration = 3 / FPS
        return any(abs(t - it) < pulse_duration for it in impact_times)

    def _preload_fonts(self) -> dict:
        fonts: dict = {}
        specs = [
            ("bebas_180", F_BEBAS, 180),
            ("bebas_160", F_BEBAS, 160),
            ("bebas_140", F_BEBAS, 140),
            ("bebas_120", F_BEBAS, 120),
            ("bebas_100", F_BEBAS, 100),
            ("mont_200",  F_MONTSERRAT, 200),
            ("mont_160",  F_MONTSERRAT, 160),
            ("mont_110",  F_MONTSERRAT, 110),
            ("oswald_88", F_OSWALD, 88),
            ("oswald_80", F_OSWALD, 80),
            ("oswald_72", F_OSWALD, 72),
            ("oswald_64", F_OSWALD, 64),
            ("roboto_72", F_ROBOTO, 72),
            ("roboto_64", F_ROBOTO, 64),
            ("roboto_56", F_ROBOTO, 56),
            ("roboto_52", F_ROBOTO, 52),
            ("bebas_280",  F_BEBAS,      280),
            ("mont_320",   F_MONTSERRAT, 320),
            ("mont_300",   F_MONTSERRAT, 300),
            ("mont_280",   F_MONTSERRAT, 280),
            ("mont_170",   F_MONTSERRAT, 170),
            ("oswald_160", F_OSWALD,     160),
            ("oswald_150", F_OSWALD,     150),
            ("roboto_130", F_ROBOTO,     130),
            ("roboto_120", F_ROBOTO,     120),
        ]
        for key, path, size in specs:
            try:
                fonts[key] = ImageFont.truetype(path, size)
            except Exception:
                try:
                    fonts[key] = ImageFont.truetype(F_ROBOTO, size)
                except Exception:
                    fonts[key] = ImageFont.load_default()
        return fonts

    def _get_font(
        self, font_path: str, size: int, fonts: dict
    ) -> ImageFont.FreeTypeFont | None:
        # Find closest preloaded font
        if font_path == F_BEBAS:
            prefix = "bebas"
        elif font_path == F_MONTSERRAT:
            prefix = "mont"
        elif font_path == F_OSWALD:
            prefix = "oswald"
        else:
            prefix = "roboto"

        candidates = {
            k: v for k, v in fonts.items() if k.startswith(prefix)
        }
        if not candidates:
            return fonts.get("roboto_72")

        best_key = min(
            candidates,
            key=lambda k: abs(int(k.split("_")[1]) - size)
        )
        return candidates[best_key]

    def _build_audio_filter(
        self,
        sfx_schedule: list[tuple[float, str]],
        sfx_inputs:   list[str],
        duration:     float,
    ) -> tuple[str, str]:
        loudnorm = "loudnorm=I=-14:TP=-1.5:LRA=7"

        if not sfx_inputs:
            return (f"[1:a]{loudnorm}[aout]", "[aout]")

        # Input 0 = video frames, Input 1 = voice, Inputs 2+ = SFX
        parts: list[str] = []
        mix_inputs: list[str] = "[1:a]"

        for i, (ts, sfx_path) in enumerate(sfx_schedule):
            idx = sfx_inputs.index(sfx_path) + 2
            label = f"[sfx{i}]"
            parts.append(
                f"[{idx}:a]adelay={int(ts * 1000)}|{int(ts * 1000)},"
                f"apad=whole_dur={duration}{label}"
            )
            mix_inputs += label

        n_inputs = 1 + len(sfx_schedule)
        parts.append(
            f"{mix_inputs}amix=inputs={n_inputs}:normalize=0[amix]"
        )
        parts.append(f"[amix]{loudnorm}[aout]")
        return ";".join(parts), "[aout]"

    def _audio_duration(self, audio_path: Path) -> float:
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "quiet",
                    "-show_entries", "format=duration",
                    "-of", "csv=p=0",
                    str(audio_path),
                ],
                capture_output=True, text=True, timeout=10,
            )
            return float(result.stdout.strip())
        except Exception:
            return 50.0
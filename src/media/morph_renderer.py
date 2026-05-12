"""
morph_renderer.py — MorphRenderer

Semantic morph video renderer for ChannelForge YouTube Shorts (1080x1920).
Produces a 5-beat storyboard with cross-fade icon morphs, safe-zone captions,
motion accents, and SFX/VO audio pipeline.
Drop-in interface replacement for KineticRenderer.build().
"""
from __future__ import annotations

import logging
import math
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Layout / path constants — borrowed from kinetic_renderer
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ASSETS       = _PROJECT_ROOT / "assets"
_FONTS_DIR    = _ASSETS / "fonts"
_SFX_DIR      = _ASSETS / "sfx"

CANVAS_W   = 1080
CANVAS_H   = 1920
FPS        = 30
OUTPUT_DIR = Path("data/output")

F_BEBAS      = str(_FONTS_DIR / "BebasNeue-Regular.ttf")
F_MONTSERRAT = str(_FONTS_DIR / "Montserrat-ExtraBold.ttf")
SFX_IMPACT   = str(_SFX_DIR / "impact.mp3")
SFX_WHOOSH   = str(_SFX_DIR / "whoosh.mp3")
SFX_WHOOSH1  = str(_SFX_DIR / "whoosh1.mp3")
SFX_RISER    = str(_SFX_DIR / "riser.mp3")

# ---------------------------------------------------------------------------
# Morph timing / motion constants
# ---------------------------------------------------------------------------
MORPH_DURATION    = 0.40
HOLD_DURATION     = 1.00
ICON_SIZE         = 140       # bounding radius (px) — inside ICON_RADIUS=180 safe zone
CAPTION_UPPER_OFF = -220      # offset from canvas cy
CAPTION_LOWER_OFF = +220
ZOOM_PEAK         = 1.30
ZOOM_RISE_S       = 0.40
ZOOM_FALL_S       = 0.20
PAN_PX            = 86
PAN_DUR_S         = 0.80
PULSE_HZ          = 0.8
PULSE_AMP         = 0.02
GHOST_ALPHA       = 15

COLOUR_GOLD = (245, 197,  24, 255)
COLOUR_BODY = (160, 162, 175, 255)
COLOUR_ICON = (240, 242, 255, 255)

BEAT_ORDER = ("hook", "statement", "twist", "landing", "cta")

# ---------------------------------------------------------------------------
# Morph dictionary
# ---------------------------------------------------------------------------
_MORPH_TABLE: list[tuple[tuple[str, ...], str, str]] = [
    (("coffee", "serve", "barista"),                    "circle",      "coffee_mug"),
    (("retire", "retired", "freedom"),                  "rectangle",   "open_door"),
    (("money", "dollars", "earn", "income", "salary"),  "rectangle",   "dollar_bill"),
    (("spend", "spent", "cost", "pay"),                 "rectangle",   "dollar_bill_down"),
    (("apartment", "home", "house", "live", "lived"),   "rectangle",   "house"),
    (("car", "drive", "drove", "vehicle"),              "rectangle",   "car"),
    (("think", "thought", "mind", "brain", "smart"),    "circle",      "brain"),
    (("time", "waiting", "clock", "hours"),             "circle",      "clock"),
    (("growth", "gain", "grew", "rise", "rising"),      "tri_up",      "bar_chart"),
    (("loss", "drop", "losing", "fell", "down"),        "tri_down",    "arrow_down"),
    (("subscribe", "follow", "join"),                   "rect_tall",   "smartphone"),
    (("scale", "balance", "decision", "compare"),       "hline",       "balance_scale"),
    (("coin", "invest", "asset", "assets"),             "circle",      "coin"),
    (("chain", "habit", "trap", "stuck"),               "two_ovals",   "chain_links"),
    (("key", "unlock", "secret", "system"),             "line_circle", "key"),
    (("lock", "risk", "hidden"),                        "sq_arc",      "padlock"),
    (("eye", "see", "notice", "look"),                  "ellipse",     "eye"),
    (("heart", "want", "desire", "love"),               "two_arcs",    "heart"),
    (("seed", "start", "begin", "born"),                "dot",         "sprout"),
    (("door", "opportunity", "open"),                   "rectangle",   "door_handle"),
]

_ICON_TO_PRIM: dict[str, str] = {
    "coffee_mug": "circle", "open_door": "rectangle", "dollar_bill": "rectangle",
    "dollar_bill_down": "rectangle", "house": "rectangle", "car": "rectangle",
    "brain": "circle", "clock": "circle", "bar_chart": "tri_up",
    "arrow_down": "tri_down", "smartphone": "rect_tall", "balance_scale": "hline",
    "coin": "circle", "chain_links": "two_ovals", "key": "line_circle",
    "padlock": "sq_arc", "eye": "ellipse", "heart": "two_arcs",
    "sprout": "dot", "door_handle": "rectangle",
}

_FINANCIAL = frozenset({
    "coffee", "money", "dollars", "earn", "income", "salary",
    "spend", "spent", "cost", "pay", "retire", "retired",
    "invest", "asset", "assets", "coin", "bank", "debt",
})
_ACTION_VERBS = frozenset({
    "serve", "drive", "drove", "think", "thought",
    "subscribe", "follow", "join", "unlock", "see",
    "start", "begin", "open", "scale", "balance",
})


# ---------------------------------------------------------------------------
# Data classes
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


@dataclass
class BeatPlan:
    beat_name:    str
    text:         str
    anchor:       str
    source_shape: str
    target_icon:  str
    morph_style:  str
    start_t:      float
    end_t:        float
    zoom_t:       float = 0.0
    is_pan_beat:  bool  = False


# ---------------------------------------------------------------------------
# MorphRenderer
# ---------------------------------------------------------------------------
class MorphRenderer:
    """
    Semantic morph renderer. Produces 1080x1920 YouTube Shorts with cross-fade
    icon morphs, safe-zone captions, glow, and motion accents.
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(
        self,
        topic_id:          str,
        script_dict:       dict[str, str],
        audio_path:        str | Path,
        word_timestamps:   list[dict] | None = None,
        cta_overlay:       str = "",
        anthropic_api_key: str = "",
        stock_video_path:  Any = None,
        **kwargs: Any,
    ) -> BuildResult:
        start_ts   = time.time()
        audio_path = Path(audio_path)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"{topic_id}_morph.mp4"

        duration = self._audio_duration(audio_path)
        if duration <= 0:
            duration = 50.0

        beats        = self._plan_storyboard(script_dict, duration)
        sfx_schedule = self._build_sfx_schedule(beats, duration)

        # Log storyboard
        logger.info("[morph] === STORYBOARD ===")
        for i, b in enumerate(beats):
            logger.info(
                "[morph] beat %d %-10s | anchor=%-12s | %-12s -> %-16s | t=%.1f-%.1f | pan=%s",
                i, b.beat_name, b.anchor, b.source_shape, b.target_icon,
                b.start_t, b.end_t, b.is_pan_beat,
            )
        logger.info("[morph] === SFX SCHEDULE (%d cues) ===", len(sfx_schedule))
        for ts, path in sfx_schedule:
            logger.info("[morph]   t=%.2fs  %s", ts, Path(path).name)

        try:
            self._render(output_path, audio_path, duration, beats, sfx_schedule)
            elapsed = time.time() - start_ts
            logger.info("[morph] Built %s in %.1fs", output_path.name, elapsed)
            return BuildResult(
                topic_id=topic_id,
                output_path=str(output_path),
                duration_seconds=duration,
                is_valid=True,
            )
        except Exception as exc:
            logger.error("[morph] Render failed: %s", exc, exc_info=True)
            return BuildResult(
                topic_id=topic_id,
                output_path="",
                duration_seconds=0.0,
                is_valid=False,
                validation_errors=[str(exc)],
            )

    # ------------------------------------------------------------------
    # Storyboard
    # ------------------------------------------------------------------
    def _plan_storyboard(self, script_dict: dict[str, str], duration: float) -> list[BeatPlan]:
        beat_dur = duration / len(BEAT_ORDER)
        beats: list[BeatPlan] = []
        prev_icon = ""

        for i, name in enumerate(BEAT_ORDER):
            text   = script_dict.get(name, "")
            anchor = self._extract_anchor(text)
            src, tgt, style = self._lookup_morph(anchor)

            if prev_icon and prev_icon in _ICON_TO_PRIM:
                src = _ICON_TO_PRIM[prev_icon]
            prev_icon = tgt

            start_t  = i * beat_dur
            end_t    = (i + 1) * beat_dur
            zoom_t   = (beat_dur - MORPH_DURATION) * 0.40
            is_pan   = (i == 2)

            beats.append(BeatPlan(
                beat_name=name, text=text, anchor=anchor,
                source_shape=src, target_icon=tgt, morph_style=style,
                start_t=start_t, end_t=end_t,
                zoom_t=zoom_t, is_pan_beat=is_pan,
            ))
        return beats

    def _extract_anchor(self, text: str) -> str:
        words = [w.lower().strip(".,!?;:") for w in text.split()]
        for priority in (_FINANCIAL, _ACTION_VERBS, None):
            for w in words:
                if priority is not None and w not in priority:
                    continue
                for kws, _, _ in _MORPH_TABLE:
                    if w in kws:
                        return w
        return words[0] if words else "money"

    def _lookup_morph(self, anchor: str) -> tuple[str, str, str]:
        for kws, src, tgt in _MORPH_TABLE:
            if anchor in kws:
                return src, tgt, "cross_fade"
        return "circle", "coin", "cross_fade"

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------
    def _audio_duration(self, audio_path: Path) -> float:
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "csv=p=0", str(audio_path)],
                capture_output=True, text=True, timeout=10,
            )
            return float(r.stdout.strip())
        except Exception:
            return 50.0

    def _build_sfx_schedule(self, beats: list[BeatPlan], duration: float) -> list[tuple[float, str]]:
        schedule: list[tuple[float, str]] = []
        per_sec: dict[int, int] = {}

        def _add(t: float, path: str) -> None:
            if not Path(path).exists():
                return
            if t < 0.2 or t > duration - 0.2:
                return
            bkt = int(t)
            if per_sec.get(bkt, 0) >= 3:
                return
            schedule.append((t, path))
            per_sec[bkt] = per_sec.get(bkt, 0) + 1

        cta = next((b for b in beats if b.beat_name == "cta"), None)
        if cta:
            _add(max(0.2, cta.start_t - 0.8), SFX_RISER)

        for beat in beats:
            _add(beat.start_t + 0.05, SFX_WHOOSH1)
            _add(beat.start_t + MORPH_DURATION + beat.zoom_t, SFX_IMPACT)

        return sorted(schedule, key=lambda x: x[0])

    def _build_audio_filter(
        self,
        sfx_schedule: list[tuple[float, str]],
        sfx_inputs:   list[str],
        duration:     float,
    ) -> tuple[str, str]:
        if not sfx_schedule:
            return "", "[1:a]"
        loudnorm  = "loudnorm=I=-14:TP=-1.5:LRA=7"
        parts: list[str] = []
        mix = "[1:a]"
        for i, (ts, sfx_path) in enumerate(sfx_schedule):
            idx   = sfx_inputs.index(sfx_path) + 2
            label = f"[sfx{i}]"
            parts.append(
                f"[{idx}:a]adelay={int(ts*1000)}|{int(ts*1000)},"
                f"apad=whole_dur={duration},volume=0.4{label}"
            )
            mix += label
        n = 1 + len(sfx_schedule)
        parts.append(f"{mix}amix=inputs={n}:normalize=0[amix]")
        parts.append(f"[amix]{loudnorm}[aout]")
        return ";".join(parts), "[aout]"

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------
    def _render(
        self,
        output_path:  Path,
        audio_path:   Path,
        duration:     float,
        beats:        list[BeatPlan],
        sfx_schedule: list[tuple[float, str]],
    ) -> None:
        total_frames = int(duration * FPS)
        W, H = CANVAS_W, CANVAS_H
        fonts = self._load_fonts()

        cmd = [
            "ffmpeg", "-y", "-loglevel", "warning",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{W}x{H}", "-pix_fmt", "rgba",
            "-r", str(FPS), "-i", "pipe:0",
            "-i", str(audio_path),
        ]
        sfx_inputs: list[str] = []
        for _, sp in sfx_schedule:
            if sp not in sfx_inputs:
                sfx_inputs.append(sp)
                cmd += ["-i", sp]

        fc, amap = self._build_audio_filter(sfx_schedule, sfx_inputs, duration)
        if fc:
            cmd += ["-filter_complex", fc, "-map", "0:v", "-map", amap]
        else:
            cmd += ["-map", "0:v", "-map", "1:a"]

        cmd += [
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-shortest", "-movflags", "+faststart",
            str(output_path),
        ]
        logger.info("[morph] Rendering %d frames -> %s", total_frames, output_path.name)

        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        chunks: list[bytes] = []

        def _drain() -> None:
            try:
                while True:
                    c = proc.stderr.read(4096)
                    if not c:
                        break
                    chunks.append(c)
            except Exception:
                pass

        dt = threading.Thread(target=_drain, daemon=True)
        dt.start()

        write_err = None
        failed_f  = None
        try:
            for fi in range(total_frames):
                img = self._make_frame(fi / FPS, beats, fonts, W, H)
                try:
                    proc.stdin.write(img.tobytes())
                except (BrokenPipeError, OSError, ValueError) as exc:
                    failed_f  = fi
                    write_err = exc
                    break
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            dt.join(timeout=5)

        stderr = b"".join(chunks).decode("utf-8", errors="replace")
        if stderr.strip():
            logger.warning("[morph] ffmpeg warnings:\n%s", stderr[-600:])

        if write_err is not None:
            raise RuntimeError(
                f"ffmpeg died at frame {failed_f} ({(failed_f or 0)/FPS:.1f}s): {write_err}\n"
                f"stderr: {stderr[-600:]}"
            )
        if proc.returncode not in (0, None):
            raise RuntimeError(f"ffmpeg exit {proc.returncode}. stderr: {stderr[-600:]}")

    # ------------------------------------------------------------------
    # Frame construction
    # ------------------------------------------------------------------
    def _make_frame(
        self, t: float, beats: list[BeatPlan], fonts: dict, W: int, H: int
    ) -> Image.Image:
        img  = Image.new("RGBA", (W, H), (0, 0, 0, 255))
        draw = ImageDraw.Draw(img)
        cx, cy = W // 2, H // 2

        beat_idx, beat, beat_local = self._current_beat(t, beats)
        beat_dur = beat.end_t - beat.start_t

        # Ghost: previous beat icon drifts upward
        if beat_idx > 0:
            prev_icon = beats[beat_idx - 1].target_icon
            drift_y   = int(beat_local * 1.5)
            r, g, b   = COLOUR_ICON[:3]
            self._icon_raw(draw, prev_icon, cx, cy - drift_y, ICON_SIZE, (r, g, b, GHOST_ALPHA))

        # Morph phase or hold phase
        if beat_local <= MORPH_DURATION:
            frac   = beat_local / MORPH_DURATION
            ease   = frac * frac * (3 - 2 * frac)
            sa     = int(255 * (1.0 - ease))
            ta     = int(255 * ease)
            tsz    = int(ICON_SIZE * (0.7 + 0.3 * ease))
            r, g, b = COLOUR_ICON[:3]
            if sa > 5:
                self._draw_prim(draw, beat.source_shape, cx, cy, ICON_SIZE, (r, g, b, sa))
            if ta > 5:
                if ta > 100:
                    self._icon_glow(draw, beat.target_icon, cx, cy, tsz, r, g, b, ta)
                else:
                    self._icon_raw(draw, beat.target_icon, cx, cy, tsz, (r, g, b, ta))
            if beat_local < 0.04:
                logger.info("[morph] beat %d: %s -> %s", beat_idx, beat.source_shape, beat.target_icon)
        else:
            hl     = beat_local - MORPH_DURATION
            hd     = beat_dur - MORPH_DURATION
            pulse  = 1.0 + PULSE_AMP * math.sin(2 * math.pi * PULSE_HZ * t)
            zoom   = self._zoom_at(beat, hl)
            pan_x  = self._pan_x(beat, hl, hd)
            sz     = int(ICON_SIZE * pulse * zoom)
            r, g, b = COLOUR_ICON[:3]
            self._icon_glow(draw, beat.target_icon, cx + pan_x, cy, sz, r, g, b, 255)

        self._draw_captions(draw, fonts, beat, beat_local, beat_dur, cx, cy)
        return img

    def _current_beat(self, t: float, beats: list[BeatPlan]) -> tuple[int, BeatPlan, float]:
        for i, b in enumerate(beats):
            if t < b.end_t:
                return i, b, max(0.0, t - b.start_t)
        last = beats[-1]
        return len(beats) - 1, last, max(0.0, t - last.start_t)

    def _zoom_at(self, beat: BeatPlan, hl: float) -> float:
        z0 = beat.zoom_t
        re = z0 + ZOOM_RISE_S
        fe = re + ZOOM_FALL_S
        if hl < z0 or hl > fe:
            return 1.0
        if hl <= re:
            f = (hl - z0) / ZOOM_RISE_S
            e = f * f * (3 - 2 * f)
            return 1.0 + (ZOOM_PEAK - 1.0) * e
        f = (hl - re) / ZOOM_FALL_S
        e = f * f * (3 - 2 * f)
        return ZOOM_PEAK + (1.0 - ZOOM_PEAK) * e

    def _pan_x(self, beat: BeatPlan, hl: float, hd: float) -> int:
        if not beat.is_pan_beat:
            return 0
        ps = hd * 0.20
        pe = ps + PAN_DUR_S
        half = PAN_PX // 2
        if hl < ps:
            return -half
        if hl > pe:
            return half
        f = (hl - ps) / PAN_DUR_S
        e = f * f * (3 - 2 * f)
        return int(-half + PAN_PX * e)

    # ------------------------------------------------------------------
    # Captions (safe zones — never overlap the icon)
    # ------------------------------------------------------------------
    def _draw_captions(
        self,
        draw: ImageDraw.ImageDraw,
        fonts: dict,
        beat: BeatPlan,
        beat_local: float,
        beat_dur: float,
        cx: int,
        cy: int,
    ) -> None:
        fi_end = 0.20
        fo_st  = max(fi_end + 0.1, beat_dur - 0.15)
        if beat_local < fi_end:
            alpha = int(255 * beat_local / fi_end)
        elif beat_local > fo_st:
            alpha = int(255 * max(0.0, (beat_dur - beat_local) / 0.15))
        else:
            alpha = 255
        alpha = max(0, min(255, alpha))
        if alpha == 0:
            return

        uy = cy + CAPTION_UPPER_OFF
        ly = cy + CAPTION_LOWER_OFF
        r_g, g_g, b_g = COLOUR_GOLD[:3]
        r_b, g_b, b_b = COLOUR_BODY[:3]

        f_anc = fonts.get("bebas_72", fonts["fallback"])
        f_sec = fonts.get("mont_48",  fonts["fallback"])

        draw.text((cx, uy), beat.anchor.upper(), font=f_anc,
                  fill=(r_g, g_g, b_g, alpha), anchor="mm")

        sec_words = [w for w in beat.text.split()
                     if w.lower().strip(".,!?;:") != beat.anchor]
        if sec_words:
            lines  = self._wrap((" ".join(sec_words)), f_sec, 900)
            lh     = 58
            y0     = ly - (len(lines) * lh) // 2
            for j, line in enumerate(lines):
                draw.text((cx, y0 + j * lh), line, font=f_sec,
                          fill=(r_b, g_b, b_b, alpha), anchor="mm")

    def _wrap(self, text: str, font: Any, max_w: int) -> list[str]:
        words   = text.split()
        lines:  list[str] = []
        current = ""
        for word in words:
            test = (current + " " + word).strip()
            try:
                w = font.getbbox(test)[2] - font.getbbox(test)[0]
            except Exception:
                w = len(test) * 20
            if w <= max_w:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines or [text]

    # ------------------------------------------------------------------
    # Font loading
    # ------------------------------------------------------------------
    def _load_fonts(self) -> dict:
        fmap: dict = {}
        for key, path, size in [
            ("bebas_72", F_BEBAS,      72),
            ("bebas_96", F_BEBAS,      96),
            ("mont_48",  F_MONTSERRAT, 48),
            ("mont_36",  F_MONTSERRAT, 36),
        ]:
            try:
                fmap[key] = ImageFont.truetype(path, size)
            except Exception:
                fmap[key] = ImageFont.load_default()
        fmap["fallback"] = fmap.get("bebas_72") or ImageFont.load_default()
        return fmap

    # ------------------------------------------------------------------
    # Primitive shapes
    # ------------------------------------------------------------------
    def _draw_prim(
        self, draw: ImageDraw.ImageDraw, shape: str,
        cx: int, cy: int, size: int, color: tuple,
    ) -> None:
        lw = max(2, size // 50)
        s  = size
        c  = color

        if shape == "circle":
            draw.ellipse((cx-s, cy-s, cx+s, cy+s), outline=c, width=lw)
        elif shape == "rectangle":
            h = max(4, s // 2)
            draw.rectangle((cx-s, cy-h, cx+s, cy+h), outline=c, width=lw)
        elif shape == "rect_tall":
            rw = max(4, int(s * 0.55))
            draw.rectangle((cx-rw, cy-s, cx+rw, cy+s), outline=c, width=lw)
        elif shape == "tri_up":
            draw.polygon([(cx, cy-s), (cx-s, cy+s), (cx+s, cy+s)], outline=c)
        elif shape == "tri_down":
            draw.polygon([(cx, cy+s), (cx-s, cy-s), (cx+s, cy-s)], outline=c)
        elif shape == "hline":
            draw.line((cx-s, cy, cx+s, cy), fill=c, width=lw*2)
        elif shape == "ellipse":
            draw.ellipse((cx-s, cy-s//2, cx+s, cy+s//2), outline=c, width=lw)
        elif shape == "dot":
            dr = max(6, s // 8)
            draw.ellipse((cx-dr, cy-dr, cx+dr, cy+dr), fill=c)
        elif shape == "two_ovals":
            ow = max(4, int(s * 0.6))
            oh = max(4, int(s * 0.35))
            draw.ellipse((cx-s,    cy-oh, cx-s+ow*2,    cy+oh), outline=c, width=lw)
            draw.ellipse((cx-ow,   cy-oh, cx-ow+ow*2,   cy+oh), outline=c, width=lw)
        elif shape == "line_circle":
            cr = max(4, int(s * 0.30))
            draw.ellipse((cx-cr, cy-s, cx+cr, cy-s+cr*2), outline=c, width=lw)
            draw.line((cx, cy-s+cr*2, cx, cy+s), fill=c, width=lw)
        elif shape == "sq_arc":
            draw.rectangle((cx-s, cy-s//4, cx+s, cy+s//2), outline=c, width=lw)
            draw.arc((cx-s//2, cy-s, cx+s//2, cy-s//4), start=0, end=180, fill=c, width=lw)
        elif shape == "two_arcs":
            hw = max(4, int(s * 0.7))
            draw.arc((cx-hw*2+hw//2, cy-s, cx+hw//2,       cy), start=0, end=180, fill=c, width=lw)
            draw.arc((cx-hw//2,      cy-s, cx+hw*2-hw//2,  cy), start=0, end=180, fill=c, width=lw)
        else:
            draw.ellipse((cx-s, cy-s, cx+s, cy+s), outline=c, width=lw)

    # ------------------------------------------------------------------
    # Icon glow + raw dispatch
    # ------------------------------------------------------------------
    def _icon_glow(
        self, draw: ImageDraw.ImageDraw, name: str,
        cx: int, cy: int, size: int,
        r: int, g: int, b: int, alpha: int,
    ) -> None:
        dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]
        for offpx, ga in ((6, 35), (3, 65), (1, 120)):
            ga2 = max(0, int(ga * alpha / 255))
            for dx, dy in dirs:
                self._icon_raw(draw, name, cx+dx*offpx, cy+dy*offpx, size, (r, g, b, ga2))
        self._icon_raw(draw, name, cx, cy, size, (r, g, b, alpha))

    def _icon_raw(
        self, draw: ImageDraw.ImageDraw, name: str,
        cx: int, cy: int, size: int, color: tuple,
    ) -> None:
        fn = getattr(self, f"_icon_{name}", None)
        if fn is None:
            lw = max(2, size // 50)
            draw.ellipse((cx-size, cy-size, cx+size, cy+size), outline=color, width=lw)
            return
        fn(draw, cx, cy, size, color)

    # ------------------------------------------------------------------
    # Icon drawing methods
    # ------------------------------------------------------------------
    def _icon_coffee_mug(self, draw: ImageDraw.ImageDraw, cx: int, cy: int, s: int, c: tuple) -> None:
        lw = max(2, s // 50)
        # Mug body (squat ellipse)
        draw.ellipse((cx-s, cy-s//2, cx+s, cy+s//2), outline=c, width=lw)
        # Handle arc on the right
        hr = max(4, int(s * 0.45))
        draw.arc((cx+s-hr, cy-hr, cx+s+hr, cy+hr), start=270, end=90, fill=c, width=lw)
        # Two steam lines above
        ox = max(4, s // 4)
        for x_off in (-ox, ox):
            yb = cy - s // 2 - 4
            yt = yb - max(8, s // 3)
            mid = (yb + yt) // 2
            draw.line([(cx+x_off, yb), (cx+x_off+6, mid), (cx+x_off, yt)], fill=c, width=lw)

    def _icon_open_door(self, draw: ImageDraw.ImageDraw, cx: int, cy: int, s: int, c: tuple) -> None:
        lw = max(2, s // 50)
        rw = max(4, int(s * 0.7))
        rh = s
        ajar = max(2, s // 9)
        pts = [
            (cx-rw+ajar, cy-rh), (cx+rw, cy-rh),
            (cx+rw, cy+rh), (cx-rw, cy+rh),
            (cx-rw, cy-rh+ajar*2),
        ]
        draw.line(pts, fill=c, width=lw)
        hr = max(5, s // 18)
        hx = cx + rw - hr * 3
        draw.ellipse((hx-hr, cy-hr, hx+hr, cy+hr), outline=c, width=lw)

    def _icon_dollar_bill(self, draw: ImageDraw.ImageDraw, cx: int, cy: int, s: int, c: tuple) -> None:
        lw = max(2, s // 50)
        h  = max(4, s // 2)
        draw.rectangle((cx-s, cy-h, cx+s, cy+h), outline=c, width=lw)
        pad = max(3, s // 7)
        draw.line((cx-s+pad, cy-h+pad, cx+s-pad, cy-h+pad), fill=c, width=lw)
        draw.line((cx-s+pad, cy+h-pad, cx+s-pad, cy+h-pad), fill=c, width=lw)
        # $ as S-arcs + vertical stem
        ss = max(8, s // 3)
        draw.arc((cx-ss, cy-ss, cx+ss, cy),     start=90,  end=315, fill=c, width=lw)
        draw.arc((cx-ss, cy,    cx+ss, cy+ss),  start=270, end=135, fill=c, width=lw)
        draw.line((cx, cy-ss-6, cx, cy+ss+6), fill=c, width=lw)

    def _icon_dollar_bill_down(self, draw: ImageDraw.ImageDraw, cx: int, cy: int, s: int, c: tuple) -> None:
        lw = max(2, s // 50)
        cy2 = cy - s // 5
        self._icon_dollar_bill(draw, cx, cy2, int(s * 0.75), c)
        ay  = cy + s * 3 // 4
        asw = max(6, s // 3)
        stem_top = cy2 + int(s * 0.75 * 0.55)
        draw.line((cx, stem_top, cx, ay - asw), fill=c, width=lw)
        draw.polygon([(cx, ay), (cx-asw, ay-asw), (cx+asw, ay-asw)], outline=c)

    def _icon_house(self, draw: ImageDraw.ImageDraw, cx: int, cy: int, s: int, c: tuple) -> None:
        lw = max(2, s // 50)
        bh = max(4, int(s * 0.55))
        by = cy + s - bh * 2
        draw.rectangle((cx-s, by, cx+s, cy+s), outline=c, width=lw)
        draw.polygon([(cx, cy-s), (cx-s-4, by+4), (cx+s+4, by+4)], outline=c)

    def _icon_car(self, draw: ImageDraw.ImageDraw, cx: int, cy: int, s: int, c: tuple) -> None:
        lw = max(2, s // 50)
        h  = max(4, int(s * 0.40))
        r  = max(2, int(s * 0.12))
        draw.rounded_rectangle((cx-s, cy-h, cx+s, cy+h//2), radius=r, outline=c, width=lw)
        cw = max(4, int(s * 0.65))
        ch = max(4, int(h * 0.8))
        rc = max(2, int(s * 0.10))
        draw.rounded_rectangle((cx-cw, cy-h-ch, cx+cw, cy-h+lw), radius=rc, outline=c, width=lw)
        wr = max(4, int(s * 0.28))
        wy = cy + h // 2
        for wx in (-int(s * 0.55), int(s * 0.55)):
            draw.ellipse((cx+wx-wr, wy, cx+wx+wr, wy+wr*2), outline=c, width=lw)

    def _icon_brain(self, draw: ImageDraw.ImageDraw, cx: int, cy: int, s: int, c: tuple) -> None:
        lw = max(2, s // 50)
        draw.ellipse((cx-s, cy-s, cx+s, cy+s), outline=c, width=lw)
        step = max(4, s // 4)
        pts  = [
            (cx - step, cy - s + 4),
            (cx + step, cy - s // 2),
            (cx - step, cy),
            (cx + step, cy + s // 2),
            (cx - step, cy + s - 4),
        ]
        draw.line(pts, fill=c, width=lw)

    def _icon_clock(self, draw: ImageDraw.ImageDraw, cx: int, cy: int, s: int, c: tuple) -> None:
        lw = max(2, s // 50)
        draw.ellipse((cx-s, cy-s, cx+s, cy+s), outline=c, width=lw)
        for angle_deg, length in ((-60, 0.55), (60, 0.70)):
            a  = math.radians(angle_deg)
            ex = int(cx + s * length * math.sin(a))
            ey = int(cy - s * length * math.cos(a))
            draw.line((cx, cy, ex, ey), fill=c, width=lw + 1)

    def _icon_bar_chart(self, draw: ImageDraw.ImageDraw, cx: int, cy: int, s: int, c: tuple) -> None:
        lw  = max(2, s // 50)
        bw  = max(4, s // 3)
        gap = s // 3
        for i, h_frac in enumerate((0.45, 0.70, 1.00)):
            bx = cx + (i - 1) * (bw * 2 + gap // 2)
            h  = int(s * h_frac * 2)
            draw.rectangle((bx - bw, cy + s - h, bx + bw, cy + s), outline=c, width=lw)

    def _icon_arrow_down(self, draw: ImageDraw.ImageDraw, cx: int, cy: int, s: int, c: tuple) -> None:
        lw  = max(2, s // 50)
        sh  = max(4, s // 2)
        tw  = max(6, int(s * 0.55))
        stem_bot = cy - s + sh * 2
        draw.line((cx, cy - s, cx, stem_bot), fill=c, width=lw * 2)
        tip = cy + s // 2
        draw.polygon([(cx, tip), (cx-tw, stem_bot), (cx+tw, stem_bot)], outline=c)

    def _icon_smartphone(self, draw: ImageDraw.ImageDraw, cx: int, cy: int, s: int, c: tuple) -> None:
        lw = max(2, s // 50)
        rw = max(4, int(s * 0.55))
        r  = max(2, int(s * 0.10))
        draw.rounded_rectangle((cx-rw, cy-s, cx+rw, cy+s), radius=r, outline=c, width=lw)
        hr = max(5, int(s * 0.10))
        by = cy + s - hr * 3
        draw.ellipse((cx-hr, by, cx+hr, by+hr*2), outline=c, width=lw)

    def _icon_balance_scale(self, draw: ImageDraw.ImageDraw, cx: int, cy: int, s: int, c: tuple) -> None:
        lw = max(2, s // 50)
        draw.line((cx, cy - s // 4, cx, cy + s), fill=c, width=lw)
        draw.line((cx - s, cy - s // 4, cx + s, cy - s // 4), fill=c, width=lw + 1)
        ph = max(4, s // 3)
        pw = max(4, s // 3)
        for side in (-1, 1):
            bx = cx + side * s
            draw.line((bx, cy - s // 4, bx, cy - s // 4 + ph), fill=c, width=lw)
            draw.line((bx - pw, cy - s // 4 + ph, bx + pw, cy - s // 4 + ph), fill=c, width=lw + 1)

    def _icon_coin(self, draw: ImageDraw.ImageDraw, cx: int, cy: int, s: int, c: tuple) -> None:
        lw = max(2, s // 50)
        draw.ellipse((cx-s, cy-s, cx+s, cy+s), outline=c, width=lw)
        inner = max(4, int(s * 0.78))
        draw.ellipse((cx-inner, cy-inner, cx+inner, cy+inner), outline=c, width=lw)

    def _icon_chain_links(self, draw: ImageDraw.ImageDraw, cx: int, cy: int, s: int, c: tuple) -> None:
        lw = max(2, s // 50)
        ow = max(4, int(s * 0.55))
        oh = max(4, int(s * 0.30))
        for ox in (-ow, 0, ow):
            draw.ellipse((cx+ox-ow, cy-oh, cx+ox+ow, cy+oh), outline=c, width=lw)

    def _icon_key(self, draw: ImageDraw.ImageDraw, cx: int, cy: int, s: int, c: tuple) -> None:
        lw = max(2, s // 50)
        cr = max(4, int(s * 0.32))
        draw.ellipse((cx-cr, cy-s, cx+cr, cy-s+cr*2), outline=c, width=lw)
        draw.line((cx, cy-s+cr*2, cx, cy+s), fill=c, width=lw+1)
        nl = max(4, int(s * 0.30))
        for ny in (cy + int(s * 0.40), cy + int(s * 0.65)):
            draw.line((cx, ny, cx+nl, ny), fill=c, width=lw)

    def _icon_padlock(self, draw: ImageDraw.ImageDraw, cx: int, cy: int, s: int, c: tuple) -> None:
        lw = max(2, s // 50)
        bh = max(4, int(s * 0.60))
        draw.arc((cx-s//2, cy-s, cx+s//2, cy-s//4), start=0, end=180, fill=c, width=lw)
        draw.rectangle((cx-s, cy-s//4, cx+s, cy+bh), outline=c, width=lw)

    def _icon_eye(self, draw: ImageDraw.ImageDraw, cx: int, cy: int, s: int, c: tuple) -> None:
        lw = max(2, s // 50)
        eh = max(4, int(s * 0.45))
        draw.ellipse((cx-s, cy-eh, cx+s, cy+eh), outline=c, width=lw)
        pr = max(5, int(s * 0.22))
        draw.ellipse((cx-pr, cy-pr, cx+pr, cy+pr), fill=c)

    def _icon_heart(self, draw: ImageDraw.ImageDraw, cx: int, cy: int, s: int, c: tuple) -> None:
        lw = max(2, s // 50)
        hw = max(4, int(s * 0.65))
        # Two half-circles on top
        draw.arc((cx-hw*2+hw//2, cy-s, cx+hw//2,      cy), start=0, end=180, fill=c, width=lw)
        draw.arc((cx-hw//2,      cy-s, cx+hw*2-hw//2, cy), start=0, end=180, fill=c, width=lw)
        # Lines converging to bottom point
        draw.line([(cx-hw*2+hw//2, cy), (cx, cy+s)], fill=c, width=lw)
        draw.line([(cx+hw*2-hw//2, cy), (cx, cy+s)], fill=c, width=lw)

    def _icon_sprout(self, draw: ImageDraw.ImageDraw, cx: int, cy: int, s: int, c: tuple) -> None:
        lw = max(2, s // 50)
        dr = max(6, int(s * 0.18))
        seed_y = cy + s
        draw.ellipse((cx-dr, seed_y-dr*2, cx+dr, seed_y), fill=c)
        stem_top = cy - s // 3
        draw.line((cx, seed_y-dr*2, cx, stem_top), fill=c, width=lw)
        lw2 = max(4, int(s * 0.45))
        lh  = max(4, int(s * 0.35))
        draw.arc((cx-lw2, stem_top-lh, cx, stem_top+lh), start=90,  end=270, fill=c, width=lw)
        draw.arc((cx, stem_top-lh, cx+lw2, stem_top+lh), start=270, end=90,  fill=c, width=lw)

    def _icon_door_handle(self, draw: ImageDraw.ImageDraw, cx: int, cy: int, s: int, c: tuple) -> None:
        lw = max(2, s // 50)
        rw = max(4, int(s * 0.70))
        draw.rectangle((cx-rw, cy-s, cx+rw, cy+s), outline=c, width=lw)
        hr = max(5, int(s * 0.10))
        hx = cx + rw - hr * 4
        draw.ellipse((hx-hr, cy-hr, hx+hr, cy+hr), outline=c, width=lw)

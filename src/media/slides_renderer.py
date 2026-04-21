"""
slides_renderer.py — SlidesRenderer
Produces a branded slides YouTube Short (1080x1920) using:
  HTML/CSS/JS animation template → Playwright frame capture → ffmpeg assembly

Architecture:
  1. Script parser    — maps script_dict to scene/word data
  2. HTML generator   — injects scene data into dark_cinematic.html template
  3. Playwright capture — renders frames at 30fps via window.renderFrame(t)
  4. SFX scheduler    — maps SFX to timestamps
  5. ffmpeg assembly  — stitches frames + voice + SFX → final MP4

Output matches VideoBuilder.BuildResult interface exactly.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CANVAS_W     = 1080
CANVAS_H     = 1920
FPS          = 30
OUTPUT_DIR   = Path("data/output")
TEMPLATES    = Path("assets/templates")
SFX_DIR      = Path("assets/sfx")

SFX_IMPACT   = str(SFX_DIR / "impact.mp3")
SFX_WHOOSH   = str(SFX_DIR / "whoosh.mp3")
SFX_WHOOSH1  = str(SFX_DIR / "whoosh1.mp3")
SFX_CASH     = str(SFX_DIR / "cash.mp3")
SFX_CASH1    = str(SFX_DIR / "cash1.mp3")
SFX_RISER    = str(SFX_DIR / "riser.mp3")

HERO_WORDS   = {
    "never", "always", "stop", "start", "truth", "lie", "myth",
    "secret", "system", "rich", "poor", "broke", "wealth", "money",
    "salary", "income", "freedom", "trap", "wrong", "real",
}
STAT_PATTERN = re.compile(r"^\d[\d,\.%kKmMbB]*$")
MONEY_WORDS  = {
    "money", "cash", "salary", "income", "wealth", "pay",
    "wage", "invest", "profit", "debt", "bank",
}
DANGER_WORDS = {
    "never", "wrong", "lie", "myth", "trap", "broke", "poor",
}


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
# SlidesRenderer
# ---------------------------------------------------------------------------
class SlidesRenderer:
    """
    Renders a branded slides Short from script + voiceover.
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
        output_path = OUTPUT_DIR / f"{topic_id}_slides.mp4"
        errors: list[str] = []

        # Duration from audio
        duration = self._audio_duration(audio_path)
        if duration <= 0:
            duration = 50.0

        # Build scene data
        scenes = self._build_scenes(script_dict, word_timestamps, duration)
        hook_type = self._detect_hook_type(script_dict.get("hook", ""))

        # SFX schedule
        sfx_schedule = self._build_sfx_schedule(scenes, duration)

        try:
            self._render(
                topic_id=topic_id,
                output_path=output_path,
                audio_path=audio_path,
                duration=duration,
                scenes=scenes,
                hook_type=hook_type,
                sfx_schedule=sfx_schedule,
            )
            elapsed = time.time() - start_ts
            logger.info("[slides] Built %s in %.1fs", output_path.name, elapsed)
            return BuildResult(
                topic_id=topic_id,
                output_path=str(output_path),
                duration_seconds=duration,
                is_valid=True,
            )
        except Exception as exc:
            logger.error("[slides] Render failed: %s", exc)
            errors.append(str(exc))
            return BuildResult(
                topic_id=topic_id,
                output_path="",
                duration_seconds=0.0,
                is_valid=False,
                validation_errors=errors,
            )

    # ------------------------------------------------------------------
    # Scene builder
    # ------------------------------------------------------------------

    def _build_scenes(
        self,
        script_dict:     dict[str, str],
        word_timestamps: list[dict] | None,
        duration:        float,
    ) -> list[dict]:

        parts_order = ["hook", "statement", "twist", "landing", "question", "cta"]
        cta_start = duration - 8.0

        scenes: list[dict] = []

        # Scene boundary times
        scene_times = {
            "hook":      (1.2,  7.0),
            "statement": (7.0,  20.0),
            "twist":     (20.0, 35.0),
            "landing":   (35.0, 43.0),
            "cta":       (43.0, duration),
        }

        for part in parts_order:
            text = script_dict.get(part, "").strip()
            if not text:
                continue

            key = part if part in scene_times else "cta"
            start, end = scene_times[key]

            # Build word list for this scene
            words = self._build_words(
                text=text,
                scene_start=start,
                scene_end=end,
                word_timestamps=word_timestamps,
                cta_start=cta_start,
            )

            scene_type = "CTA" if part == "cta" else part.upper()
            layout = "left" if part in ("statement", "landing") else "centre"
            show_sweep = part in ("twist", "cta")
            ghost = words[0]["text"] if words else ""

            scenes.append({
                "type":      scene_type,
                "start":     start,
                "end":       end,
                "words":     words,
                "layout":    layout,
                "showSweep": show_sweep,
                "ghost":     ghost,
            })

        return scenes

    def _build_words(
        self,
        text:            str,
        scene_start:     float,
        scene_end:       float,
        word_timestamps: list[dict] | None,
        cta_start:       float,
    ) -> list[dict]:

        raw_words = [w.strip(".,!?;:") for w in text.split() if w.strip()]
        if not raw_words:
            return []

        interval = (scene_end - scene_start) / len(raw_words)
        anim_cycle = ["SLAM", "SLIDE_L", "PUNCH", "SLIDE_R", "FADE_RISE", "TYPEWRITER"]
        cycle_i = 0
        words: list[dict] = []

        for i, word in enumerate(raw_words):
            # Try to match word_timestamps
            t_start = scene_start + i * interval
            t_end = t_start + interval * 0.9

            if word_timestamps:
                for wt in word_timestamps:
                    if wt.get("text", "").strip(".,!?;:").lower() == word.lower():
                        t_start = float(wt.get("start_time", t_start))
                        t_end   = float(wt.get("end_time",   t_end))
                        break

            lower = word.lower()
            is_cta = t_start >= cta_start

            if is_cta:
                role, anim, danger = "CTA", "FADE_RISE", False
            elif STAT_PATTERN.match(word):
                role, anim, danger = "STAT", "PUNCH", False
            elif lower in HERO_WORDS:
                role, anim, danger = "HERO", "SLAM", lower in DANGER_WORDS
            else:
                role  = "BODY"
                anim  = anim_cycle[cycle_i % len(anim_cycle)]
                danger = False
                cycle_i += 1

            words.append({
                "text":   word,
                "start":  t_start,
                "end":    t_end,
                "role":   role,
                "anim":   anim,
                "danger": danger,
            })

        return words

    # ------------------------------------------------------------------
    # Hook type detection
    # ------------------------------------------------------------------

    def _detect_hook_type(self, hook: str) -> str:
        words = hook.strip().split()
        if words and STAT_PATTERN.match(words[0]):
            return "B"
        first = hook.strip().lower()
        if any(first.startswith(q) for q in (
            "why", "what", "how", "do ", "are ", "is ", "did "
        )):
            return "C"
        return "A"

    # ------------------------------------------------------------------
    # SFX schedule
    # ------------------------------------------------------------------

    def _build_sfx_schedule(
        self,
        scenes:   list[dict],
        duration: float,
    ) -> list[tuple[float, str]]:

        schedule: list[tuple[float, str]] = []
        cta_start = duration - 8.0

        if Path(SFX_RISER).exists():
            schedule.append((max(0, cta_start - 0.8), SFX_RISER))

        whoosh_toggle = True
        for scene in scenes:
            for word in scene.get("words", []):
                t = word["start"]
                role = word["role"]
                if role == "HERO" and Path(SFX_IMPACT).exists():
                    schedule.append((t, SFX_IMPACT))
                elif role == "STAT":
                    if Path(SFX_IMPACT).exists():
                        schedule.append((t, SFX_IMPACT))
                    if Path(SFX_CASH1).exists():
                        schedule.append((t + 0.05, SFX_CASH1))
                elif role == "BODY":
                    lower = word["text"].lower()
                    if lower in MONEY_WORDS and Path(SFX_CASH).exists():
                        schedule.append((t, SFX_CASH))
                    else:
                        sfx = SFX_WHOOSH if whoosh_toggle else SFX_WHOOSH1
                        whoosh_toggle = not whoosh_toggle
                        if Path(sfx).exists():
                            schedule.append((t, sfx))

        return sorted(schedule, key=lambda x: x[0])

    # ------------------------------------------------------------------
    # Playwright render
    # ------------------------------------------------------------------

    def _render(
        self,
        topic_id:     str,
        output_path:  Path,
        audio_path:   Path,
        duration:     float,
        scenes:       list[dict],
        hook_type:    str,
        sfx_schedule: list[tuple[float, str]],
    ) -> None:

        # Load HTML template
        template_path = TEMPLATES / "dark_cinematic.html"
        html = template_path.read_text(encoding="utf-8")

        # Inject scene data
        scenes_json   = json.dumps(scenes,    ensure_ascii=False)
        duration_json = json.dumps(duration)
        hook_json     = json.dumps(hook_type)

        html = html.replace(
            "const SCENES = window.__SCENES__ || [];",
            f"const SCENES = {scenes_json};"
        ).replace(
            "const TOTAL_DURATION = window.__DURATION__ || 50;",
            f"const TOTAL_DURATION = {duration_json};"
        ).replace(
            "const hookType = window.__HOOK_TYPE__ || 'A';",
            f"const hookType = {hook_json};"
        )

        # Also set window globals for safety
        inject = f"""
window.__SCENES__    = {scenes_json};
window.__DURATION__  = {duration_json};
window.__HOOK_TYPE__ = {hook_json};
"""
        html = html.replace("<script>", f"<script>\n{inject}")

        # Write temp HTML file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False,
            encoding="utf-8"
        ) as f:
            f.write(html)
            tmp_html = f.name

        # Frame output directory
        frames_dir = Path(tempfile.mkdtemp(prefix=f"{topic_id}_slides_"))
        total_frames = int(duration * FPS)

        try:
            logger.info(
                "[slides] Capturing %d frames via Playwright → %s",
                total_frames, frames_dir
            )
            self._playwright_capture(
                html_path=tmp_html,
                frames_dir=frames_dir,
                total_frames=total_frames,
                duration=duration,
            )
            logger.info("[slides] Frame capture complete, assembling video")
            self._ffmpeg_assemble(
                frames_dir=frames_dir,
                audio_path=audio_path,
                sfx_schedule=sfx_schedule,
                duration=duration,
                output_path=output_path,
            )
        finally:
            Path(tmp_html).unlink(missing_ok=True)
            # Clean up frames
            import shutil
            shutil.rmtree(frames_dir, ignore_errors=True)

    def _playwright_capture(
        self,
        html_path:    str,
        frames_dir:   Path,
        total_frames: int,
        duration:     float,
    ) -> None:

        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--disable-web-security",
                ],
            )
            page = browser.new_page(
                viewport={"width": CANVAS_W, "height": CANVAS_H},
            )
            page.goto(f"file://{html_path}")
            page.wait_for_load_state("networkidle")

            for frame_i in range(total_frames):
                t = frame_i / FPS
                # Advance animation
                page.evaluate(f"window.renderFrame({t})")
                # Screenshot
                frame_path = frames_dir / f"frame_{frame_i:06d}.png"
                page.screenshot(
                    path=str(frame_path),
                    clip={"x": 0, "y": 0, "width": CANVAS_W, "height": CANVAS_H},
                )

                if frame_i % 30 == 0:
                    logger.info(
                        "[slides] Frame %d/%d (t=%.1fs)",
                        frame_i, total_frames, t
                    )

            browser.close()

    def _ffmpeg_assemble(
        self,
        frames_dir:   Path,
        audio_path:   Path,
        sfx_schedule: list[tuple[float, str]],
        duration:     float,
        output_path:  Path,
    ) -> None:

        # Build unique SFX input list
        sfx_inputs: list[str] = []
        for _, sfx_path in sfx_schedule:
            if sfx_path not in sfx_inputs:
                sfx_inputs.append(sfx_path)

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(FPS),
            "-i", str(frames_dir / "frame_%06d.png"),
            "-i", str(audio_path),
        ]

        for sfx_path in sfx_inputs:
            cmd += ["-i", sfx_path]

        # Audio filter
        filter_complex, audio_map = self._build_audio_filter(
            sfx_schedule, sfx_inputs, duration
        )

        if filter_complex:
            cmd += [
                "-filter_complex", filter_complex,
                "-map", "0:v",
                "-map", audio_map,
            ]
        else:
            cmd += ["-map", "0:v", "-map", "1:a"]

        cmd += [
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
            "-shortest",
            "-movflags", "+faststart",
            str(output_path),
        ]

        logger.info("[slides] Running ffmpeg assembly")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg assembly failed: {result.stderr[-500:]}"
            )

    def _build_audio_filter(
        self,
        sfx_schedule: list[tuple[float, str]],
        sfx_inputs:   list[str],
        duration:     float,
    ) -> tuple[str, str]:

        if not sfx_inputs:
            return "", "[1:a]"

        parts: list[str] = []
        mix_labels = "[1:a]"

        for i, (ts, sfx_path) in enumerate(sfx_schedule):
            idx = sfx_inputs.index(sfx_path) + 2
            label = f"[sfx{i}]"
            parts.append(
                f"[{idx}:a]adelay={int(ts * 1000)}|{int(ts * 1000)},"
                f"apad=whole_dur={duration}{label}"
            )
            mix_labels += label

        n = 1 + len(sfx_schedule)
        parts.append(
            f"{mix_labels}amix=inputs={n}:normalize=0[aout]"
        )
        return ";".join(parts), "[aout]"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
"""
anime_renderer.py — AnimeRenderer

Generates high-energy HyperFrames HTML compositions driven by anime.js v4
and renders to MP4 via the HyperFrames CLI. Designed for Slot 3 (high-energy
kinetic style): text slams in from edges, gold accent punches, fast beat pacing.

Interface matches HyperFramesRenderer.build() exactly.

Design choices vs Slot 1 (HyperFramesRenderer):
  - 4 s beats (vs 6 s) for higher pace
  - Slam-in / slam-out transitions (vs smooth fade)
  - Text-first layout — no SVG icon library
  - easeOutBack / easeInBack / easeOutElastic for aggressive motion
  - anime.js v4 createTimeline() + window.__hfAnime adapter
"""
from __future__ import annotations

import html as _html_mod
import json
import logging
import os
import shutil
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
OUTPUT_DIR       = Path("data/output")
BEAT_ORDER       = ("hook", "statement", "twist", "landing", "cta")
BEAT_DUR_S       = 4.0       # seconds per beat
BEAT_DUR_MS      = 4000      # ms per beat (for anime.js timeline)
IN_DUR           = 500       # slam-in duration (ms)
OUT_DUR          = 350       # slam-out duration (ms)
BAR_WIDTHS       = (160, 200, 140, 160, 200)
DEFAULT_DURATION = 20.0      # fallback total duration (5 beats × 4 s)

# Gold accent colour
GOLD = "#F5C518"

# Each beat gets a short "eyebrow" label shown above the headline
_BEAT_EYEBROWS = {
    "hook":      "Real Talk",
    "statement": "The Numbers",
    "twist":     "The Secret",
    "landing":   "The Truth",
    "cta":       "Don't Miss Out",
}

# Words that should be rendered in gold per beat slot.
# The match is case-insensitive and uses simple substring search.
_GOLD_WORDS: dict[str, list[str]] = {
    "hook":      ["retired", "retire", "free", "35", "early", "quit"],
    "statement": ["thousand", "k", "$", "percent", "%", "year"],
    "twist":     ["free", "freedom", "escape", "spent", "money"],
    "landing":   ["out-thought", "outthought", "thought", "secret", "truth"],
    "cta":       ["subscribe", "never", "hear", "money", "moves"],
}


# ---------------------------------------------------------------------------
# BuildResult — same interface as HyperFramesRenderer
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
# AnimeRenderer
# ---------------------------------------------------------------------------
class AnimeRenderer:
    """
    High-energy kinetic video renderer using HyperFrames + anime.js v4.
    Produces 1080×1920 YouTube Shorts with slam-in / slam-out beat transitions.

    Drop-in replacement for KineticRenderer / HyperFramesRenderer (same
    build() interface).

    Visual style:
      - Deep black background with gold (#F5C518) accent elements
      - Text slams in from the right (easeOutBack) and out to the left (easeInBack)
      - Key words highlighted in gold with elastic scale punch
      - Corner bracket accents flash on each beat hit
      - Gold bar indicator under each headline
    """

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
        """
        Generate a high-energy anime.js video for one topic.

        Args:
            topic_id:    Unique identifier (used for output filename).
            script_dict: Parts dict with keys hook/statement/twist/landing/cta
                         (also accepts 'question' for the CTA beat).
            audio_path:  Path to voiceover MP3.
            word_timestamps, cta_overlay, anthropic_api_key, stock_video_path:
                         Accepted for interface compatibility; not used.

        Returns:
            BuildResult with output_path and is_valid flag.
        """
        start_ts   = time.time()
        audio_path = Path(audio_path)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"{topic_id}_anime.mp4"

        total_secs = len(BEAT_ORDER) * BEAT_DUR_S  # always 20 s for 5 beats

        beats = self._plan_beats(script_dict)

        logger.info("[an] === ANIME STORYBOARD ===")
        for i, b in enumerate(beats):
            logger.info("[an] beat %d %-10s | eyebrow=%-16s | text=%.40s",
                        i, b["name"], b["eyebrow"], b["text"])

        try:
            html = self._generate_html(beats, total_secs)
            self._render_html(html, output_path, audio_path)
            elapsed = time.time() - start_ts
            logger.info("[an] Built %s in %.1fs", output_path.name, elapsed)
            return BuildResult(
                topic_id=topic_id,
                output_path=str(output_path),
                duration_seconds=total_secs,
                is_valid=True,
            )
        except Exception as exc:
            logger.error("[an] Render failed: %s", exc, exc_info=True)
            return BuildResult(
                topic_id=topic_id,
                output_path="",
                duration_seconds=0.0,
                is_valid=False,
                validation_errors=[str(exc)],
            )

    # ------------------------------------------------------------------
    # Beat planning
    # ------------------------------------------------------------------

    def _plan_beats(self, script_dict: dict[str, str]) -> list[dict]:
        """Map each BEAT_ORDER slot to text, eyebrow label, and gold-word markup."""
        beats: list[dict] = []
        for name in BEAT_ORDER:
            if name == "cta":
                text = script_dict.get("cta", "") or script_dict.get("question", "")
            else:
                text = script_dict.get(name, "")
            eyebrow = _BEAT_EYEBROWS.get(name, name.upper())
            beats.append({
                "name":    name,
                "text":    text,
                "eyebrow": eyebrow,
            })
        return beats

    # ------------------------------------------------------------------
    # HTML generation
    # ------------------------------------------------------------------

    def _mark_gold_words(self, text: str, beat_name: str) -> str:
        """
        Return HTML-escaped text with gold words wrapped in <span class="gold">.
        Matches case-insensitively; preserves original casing in output.
        """
        gold_list = _GOLD_WORDS.get(beat_name, [])
        if not gold_list:
            return _html_mod.escape(text)

        import re
        pattern = r'\b(' + '|'.join(re.escape(w) for w in gold_list) + r')\b'

        def replacer(m: re.Match) -> str:
            return f'<span class="gold" data-accent="1">{_html_mod.escape(m.group(0))}</span>'

        # HTML-escape first, then wrap gold words (safe since gold words are plain ASCII)
        escaped = _html_mod.escape(text)
        result = re.sub(pattern, replacer, escaped, flags=re.IGNORECASE)
        return result

    def _beat_html(self, beats: list[dict]) -> str:
        """Build the HTML block for all 5 beat containers."""
        blocks: list[str] = []
        for i, beat in enumerate(beats):
            eyebrow = _html_mod.escape(beat["eyebrow"])
            body    = self._mark_gold_words(beat["text"], beat["name"])
            bar_bg  = "background:#fff" if beat["name"] == "cta" else ""

            cta_block = ""
            if beat["name"] == "cta":
                cta_block = '\n      <div id="cta-sub">&#x2193; SUBSCRIBE &#x2193;</div>'

            blocks.append(
                f'      <!-- Beat {i}: {beat["name"]} -->\n'
                f'      <div id="beat-{i}" class="beat">\n'
                f'        <div class="eyebrow" id="ey-{i}">{eyebrow}</div>\n'
                f'        <div class="headline" id="hl-{i}">{body}</div>\n'
                f'        <div class="gold-bar" id="bar-{i}"'
                + (f' style="{bar_bg}"' if bar_bg else '') + '></div>'
                + cta_block + '\n'
                f'      </div>'
            )
        return "\n\n".join(blocks)

    def _gsap_script(self, n_beats: int) -> str:
        """
        Build the anime.js v4 animation script.

        Uses:
          anime.createTimeline({ autoplay: false })
          tl.add(targets, params, timeOffsetMs)   ← v4 signature
          window.__hfAnime.push(tl)               ← anime.js adapter
          window.__timelines['main'] shim          ← satisfies HyperFrames linter
        """
        lines: list[str] = [
            "      const BEAT_DUR  = " + str(BEAT_DUR_MS) + ";",
            "      const N_BEATS   = " + str(n_beats) + ";",
            "      const IN_DUR    = " + str(IN_DUR) + ";",
            "      const OUT_DUR   = " + str(OUT_DUR) + ";",
            "      const OUT_START = BEAT_DUR - OUT_DUR;",
            "",
            "      const BAR_WIDTHS = " + json.dumps(list(BAR_WIDTHS[:n_beats])) + ";",
            "",
            "      const tl = anime.createTimeline({ autoplay: false });",
            "",
            "      // ── Background: grid + diagonal lines appear ──────────",
            "      tl.add('#gl1, #gl2, #gl3, #hr1',",
            "             { opacity: [0, 1], duration: 1200, easing: 'easeOutExpo' }, 0);",
            "      tl.add('#dl1, #dl2',",
            "             { opacity: [0, 1], duration: 1000, easing: 'easeOutExpo' }, 200);",
            "",
            "      // ── Corner brackets appear ────────────────────────────",
            "      tl.add('.ca', { opacity: [0, 0.75], duration: 800, easing: 'easeOutExpo' }, 0);",
            "",
            "      // ── Per-beat animations ───────────────────────────────",
            "      var accentSelectors = [",
        ]
        # Pre-build accent selectors per beat
        for i in range(n_beats):
            sel = f'#beat-{i} [data-accent]'
            lines.append(f"        '{sel}'" + ("," if i < n_beats - 1 else ""))
        lines += [
            "      ];",
            "",
            "      for (var i = 0; i < N_BEATS; i++) {",
            "        var o        = i * BEAT_DUR;",
            "        var beatSel  = '#beat-' + i;",
            "        var eyeSel   = '#ey-'   + i;",
            "        var barSel   = '#bar-'  + i;",
            "",
            "        // SLAM IN from right",
            "        tl.add(beatSel,",
            "               { translateX: [1200, 0], opacity: [0, 1],",
            "                 duration: IN_DUR, easing: 'easeOutBack(1.6)' }, o);",
            "",
            "        // Corner brackets flash on beat hit",
            "        tl.add('.ca', { opacity: [0.75, 1],   duration: 120, easing: 'easeOutSine' }, o);",
            "        tl.add('.ca', { opacity: [1, 0.55],   duration: 380, easing: 'easeOutSine' }, o + 120);",
            "",
            "        // Eyebrow punches in",
            "        tl.add(eyeSel,",
            "               { translateY: [-20, 0], opacity: [0, 1],",
            "                 duration: 320, easing: 'easeOutExpo' }, o + 140);",
            "",
            "        // Gold bar slams in",
            "        tl.add(barSel,",
            "               { width: [0, BAR_WIDTHS[i]],",
            "                 duration: 450, easing: 'easeOutExpo' }, o + 220);",
            "",
            "        // Accent words elastic punch",
            "        tl.add(accentSelectors[i],",
            "               { scale: [1.7, 1.0], duration: 550,",
            "                 easing: 'easeOutElastic(1, 0.5)' }, o + 360);",
            "",
            "        // CTA: subscribe block punches in",
            "        if (i === 4) {",
            "          tl.add('#cta-sub',",
            "                 { scale: [0.6, 1.0], opacity: [0, 1],",
            "                   duration: 600, easing: 'easeOutBack(1.5)' }, o + 750);",
            "        }",
            "",
            "        // SLAM OUT to left (skip final beat)",
            "        if (i < N_BEATS - 1) {",
            "          tl.add(beatSel,",
            "                 { translateX: [0, -1200], opacity: [1, 0],",
            "                   duration: OUT_DUR, easing: 'easeInBack(1.6)' },",
            "                 o + OUT_START);",
            "        }",
            "      }",
            "",
            "      // Register with HyperFrames anime.js adapter",
            "      window.__hfAnime = window.__hfAnime || [];",
            "      window.__hfAnime.push(tl);",
            "",
            "      // GSAP adapter shim: HyperFrames calls seek(seconds); convert → ms",
            "      window.__timelines = window.__timelines || {};",
            "      window.__timelines['main'] = {",
            "        seek:  function(s) { tl.seek(s * 1000); },",
            "        pause: function()  { tl.pause(); },",
            "        play:  function()  { tl.play();  },",
            "      };",
        ]
        return "\n".join("      " + ln if ln else "" for ln in lines)

    def _generate_html(self, beats: list[dict], total_secs: float) -> str:
        """Build the complete HyperFrames HTML composition string."""
        duration_int = int(total_secs)
        beats_html   = self._beat_html(beats)
        script       = self._gsap_script(len(beats))

        return (
            "<!doctype html>\n"
            '<html lang="en">\n'
            "  <head>\n"
            '    <meta charset="UTF-8" />\n'
            '    <meta name="viewport" content="width=1080, height=1920" />\n'
            '    <link href="https://fonts.googleapis.com/css2?family=Bebas+Neue'
            '&family=Montserrat:wght@700;900&display=block" rel="stylesheet" />\n'
            '    <script src="https://cdn.jsdelivr.net/npm/animejs@4.0.2/lib/anime.iife.min.js"></script>\n'
            "    <style>\n"
            "      * { margin: 0; padding: 0; box-sizing: border-box; }\n"
            "      html, body {\n"
            "        margin: 0; width: 1080px; height: 1920px;\n"
            "        overflow: hidden; background: #000;\n"
            "      }\n"
            "      #root {\n"
            "        position: relative; width: 1080px; height: 1920px;\n"
            "        overflow: hidden; background: #000;\n"
            "      }\n"
            "      #bg { position: absolute; inset: 0; pointer-events: none; }\n"
            "      .ca {\n"
            "        position: absolute; width: 64px; height: 64px;\n"
            "        border: 3px solid " + GOLD + "; opacity: 0;\n"
            "      }\n"
            "      #ca-tl { top: 64px;    left: 64px;  border-right: none; border-bottom: none; }\n"
            "      #ca-tr { top: 64px;    right: 64px; border-left:  none; border-bottom: none; }\n"
            "      #ca-bl { bottom: 64px; left: 64px;  border-right: none; border-top:    none; }\n"
            "      #ca-br { bottom: 64px; right: 64px; border-left:  none; border-top:    none; }\n"
            "      .beat {\n"
            "        position: absolute; inset: 0;\n"
            "        display: flex; flex-direction: column;\n"
            "        justify-content: center; align-items: center;\n"
            "        padding: 0 80px;\n"
            "        opacity: 0; transform: translateX(1200px);\n"
            "      }\n"
            "      .eyebrow {\n"
            "        font-family: 'Montserrat', sans-serif; font-weight: 700;\n"
            "        font-size: 26px; color: " + GOLD + ";\n"
            "        letter-spacing: 7px; text-transform: uppercase;\n"
            "        margin-bottom: 28px;\n"
            "        opacity: 0; transform: translateY(-20px);\n"
            "      }\n"
            "      .headline {\n"
            "        font-family: 'Bebas Neue', Impact, sans-serif;\n"
            "        font-size: 104px; color: #fff;\n"
            "        text-align: center; line-height: 1.0; letter-spacing: 2px;\n"
            "      }\n"
            "      .gold   { color: " + GOLD + "; display: inline-block; }\n"
            "      .xl     { font-size: 148px; line-height: 1.0; }\n"
            "      .sub    {\n"
            "        font-size: 58px; color: #888;\n"
            "        font-family: 'Montserrat', sans-serif; font-weight: 700;\n"
            "        display: block; margin-top: 16px;\n"
            "      }\n"
            "      .gold-bar {\n"
            "        width: 0; height: 5px; background: " + GOLD + "; margin-top: 44px;\n"
            "      }\n"
            "      #cta-sub {\n"
            "        margin-top: 60px;\n"
            "        font-family: 'Bebas Neue', sans-serif; font-size: 72px;\n"
            "        color: #000; background: " + GOLD + ";\n"
            "        padding: 16px 64px; letter-spacing: 6px;\n"
            "        opacity: 0; transform: scale(0.6);\n"
            "      }\n"
            "    </style>\n"
            "  </head>\n"
            "  <body>\n"
            "    <div\n"
            '      id="root"\n'
            '      data-composition-id="main"\n'
            '      data-start="0"\n'
            f'      data-duration="{duration_int}"\n'
            '      data-width="1080"\n'
            '      data-height="1920"\n'
            "    >\n\n"
            '      <!-- Geometric SVG background -->\n'
            '      <svg id="bg" viewBox="0 0 1080 1920" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">\n'
            '        <line id="gl1" x1="270"  y1="0"    x2="270"  y2="1920" stroke="#F5C51812" stroke-width="1"   opacity="0"/>\n'
            '        <line id="gl2" x1="540"  y1="0"    x2="540"  y2="1920" stroke="#F5C51812" stroke-width="1"   opacity="0"/>\n'
            '        <line id="gl3" x1="810"  y1="0"    x2="810"  y2="1920" stroke="#F5C51812" stroke-width="1"   opacity="0"/>\n'
            '        <line id="dl1" x1="0"   y1="480"   x2="480"  y2="0"    stroke="#F5C51828" stroke-width="1.5" opacity="0"/>\n'
            '        <line id="dl2" x1="600" y1="1920"  x2="1080" y2="1440" stroke="#F5C51828" stroke-width="1.5" opacity="0"/>\n'
            '        <line id="hr1" x1="0"   y1="960"   x2="1080" y2="960"  stroke="#ffffff08" stroke-width="1"   opacity="0"/>\n'
            '      </svg>\n\n'
            '      <!-- Corner bracket accents -->\n'
            '      <div class="ca" id="ca-tl"></div>\n'
            '      <div class="ca" id="ca-tr"></div>\n'
            '      <div class="ca" id="ca-bl"></div>\n'
            '      <div class="ca" id="ca-br"></div>\n\n'
            f"{beats_html}\n\n"
            "    </div>\n\n"
            "    <script>\n"
            f"{script}\n"
            "    </script>\n"
            "  </body>\n"
            "</html>\n"
        )

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _render_html(
        self,
        html: str,
        output_path: Path,
        audio_path: Path | None = None,
    ) -> None:
        """Write HTML to a temp HyperFrames project and invoke the CLI renderer."""
        tmpdir = Path(tempfile.mkdtemp(prefix="an_slot3_"))
        try:
            self._write_project_files(tmpdir, html, output_path.stem)

            env = os.environ.copy()
            ffmpeg_dir = self._ensure_ffmpeg()
            if ffmpeg_dir:
                sep = ";" if os.name == "nt" else ":"
                env["PATH"] = str(ffmpeg_dir) + sep + env.get("PATH", "")

            abs_out = str(output_path.resolve())
            cmd = f'npx --yes hyperframes@0.6.0 render --output "{abs_out}"'
            logger.info("[an] %s (cwd=%s)", cmd, tmpdir)

            result = subprocess.run(
                cmd, cwd=str(tmpdir), env=env,
                shell=True, capture_output=True, text=True, timeout=600,
            )
            stdout = result.stdout[-800:] if result.stdout else ""
            stderr = result.stderr[-400:] if result.stderr else ""
            if stdout.strip():
                logger.info("[an] stdout: %s", stdout)
            if result.returncode != 0:
                raise RuntimeError(
                    f"hyperframes render exit {result.returncode}.\n"
                    f"stdout: {stdout}\nstderr: {stderr}"
                )
            if not output_path.exists() or output_path.stat().st_size == 0:
                raise RuntimeError(
                    f"hyperframes render returned 0 but {output_path} not found or empty."
                )

            # Mix voiceover audio into the visual-only MP4
            if audio_path is not None and audio_path.exists() and audio_path.stat().st_size > 0:
                ffmpeg_exe: Path | str | None = None
                if ffmpeg_dir:
                    candidate = ffmpeg_dir / "ffmpeg.exe"
                    if candidate.exists():
                        ffmpeg_exe = candidate
                if ffmpeg_exe is None:
                    found = shutil.which("ffmpeg")
                    if found:
                        ffmpeg_exe = found
                if ffmpeg_exe:
                    mix_start  = time.time()
                    tmp_mixed  = output_path.with_suffix(".mixed.mp4")
                    mix_result = subprocess.run(
                        [
                            str(ffmpeg_exe),
                            "-i",  str(output_path),
                            "-i",  str(audio_path),
                            "-c:v", "copy", "-c:a", "aac",
                            "-shortest",
                            "-map", "0:v:0", "-map", "1:a:0",
                            "-y", str(tmp_mixed),
                        ],
                        capture_output=True, text=True, timeout=120,
                    )
                    if mix_result.returncode == 0 and tmp_mixed.exists() and tmp_mixed.stat().st_size > 0:
                        tmp_mixed.replace(output_path)
                        logger.info("[an] Mixed voiceover (%.1fs)", time.time() - mix_start)
                    else:
                        logger.warning(
                            "[an] ffmpeg voiceover mix failed (rc=%d) — visual-only kept",
                            mix_result.returncode,
                        )
                        if tmp_mixed.exists():
                            tmp_mixed.unlink(missing_ok=True)
                else:
                    logger.warning("[an] ffmpeg not available — voiceover not mixed")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _write_project_files(self, tmpdir: Path, html: str, stem: str) -> None:
        """Write the minimal HyperFrames project structure to tmpdir."""
        (tmpdir / "index.html").write_text(html, encoding="utf-8")
        (tmpdir / "hyperframes.json").write_text(
            json.dumps({
                "$schema": "https://hyperframes.heygen.com/schema/hyperframes.json",
                "registry": "https://raw.githubusercontent.com/heygen-com/hyperframes/main/registry",
                "paths": {
                    "blocks":     "compositions",
                    "components": "compositions/components",
                    "assets":     "assets",
                },
            }, indent=2),
            encoding="utf-8",
        )
        (tmpdir / "meta.json").write_text(
            json.dumps({
                "id":        f"an-{stem}",
                "name":      f"an-{stem}",
                "createdAt": datetime.now(timezone.utc).isoformat(),
            }, indent=2),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _audio_duration(self, audio_path: Path) -> float:
        """Return audio duration in seconds (ffprobe → ffmpeg -i → mutagen)."""
        ffprobe = self._find_ffprobe()
        if ffprobe:
            try:
                r = subprocess.run(
                    [ffprobe, "-v", "quiet", "-show_entries", "format=duration",
                     "-of", "csv=p=0", str(audio_path)],
                    capture_output=True, text=True, timeout=10,
                )
                val = float(r.stdout.strip())
                if val > 0:
                    return val
            except Exception:
                pass

        ffmpeg_dir = self._ensure_ffmpeg()
        if ffmpeg_dir:
            ffmpeg_exe = ffmpeg_dir / "ffmpeg.exe"
            if ffmpeg_exe.exists():
                try:
                    import re as _re
                    r = subprocess.run(
                        [str(ffmpeg_exe), "-i", str(audio_path)],
                        capture_output=True, text=True, timeout=10,
                    )
                    m = _re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", r.stderr)
                    if m:
                        h, mi, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
                        return h * 3600 + mi * 60 + s
                except Exception:
                    pass

        try:
            from mutagen.mp3 import MP3
            audio = MP3(str(audio_path))
            if audio.info.length > 0:
                return audio.info.length
        except Exception:
            pass

        return 0.0

    def _ensure_ffmpeg(self) -> Path | None:
        """Stage imageio_ffmpeg binary as ffmpeg.exe in a temp dir."""
        try:
            import imageio_ffmpeg
            src      = Path(imageio_ffmpeg.get_ffmpeg_exe())
            dest_dir = Path(tempfile.gettempdir()) / "hf_ffmpeg_bin"
            dest_dir.mkdir(exist_ok=True)
            dest = dest_dir / "ffmpeg.exe"
            if not dest.exists() or dest.stat().st_size != src.stat().st_size:
                shutil.copy2(src, dest)
            return dest_dir
        except Exception as exc:
            logger.debug("[an] Could not stage imageio_ffmpeg: %s", exc)
            return None

    def _find_ffprobe(self) -> str | None:
        """Locate ffprobe: PATH → imageio_ffmpeg dir."""
        if shutil.which("ffprobe"):
            return "ffprobe"
        try:
            import imageio_ffmpeg
            exe_dir = Path(imageio_ffmpeg.get_ffmpeg_exe()).parent
            for name in ("ffprobe.exe", "ffprobe"):
                candidate = exe_dir / name
                if candidate.exists():
                    return str(candidate)
        except Exception:
            pass
        return None

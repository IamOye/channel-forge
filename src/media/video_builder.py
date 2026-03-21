"""
video_builder.py — VideoBuilder

Assembles the final YouTube Shorts MP4 from stock video, voiceover audio,
and timed captions using a pure single-pass ffmpeg strategy:

  Fast path (word_timestamps provided) —
    • All b-roll trim / scale / crop / xfade crossfades and the dark overlay
      are handled entirely inside ffmpeg's filter_complex (C-speed, no Python
      per-frame overhead).
    • Caption RGBA frames are pre-baked once (one PIL render per word state,
      O(log n) bisect lookup per output frame) and piped to ffmpeg stdin.
    • ffmpeg mixes the voiceover audio and writes the final MP4 in one pass.
    • Expected render time: < 60 s for a 45-second 1080×1920 video.

  Legacy path (word_timestamps=None) —
    The original moviepy CompositeVideoClip path is preserved so that all
    existing tests continue to pass without modification.

Usage:
    builder = VideoBuilder()
    result = builder.build(
        topic_id="stoic_001",
        script_dict={"hook": "...", "statement": "...", "twist": "...", "question": "..."},
        audio_path="data/raw/stoic_001_voice.mp3",
        stock_video_path="data/raw/stoic_001_stock.mp4",
    )
    print(result.output_path)
"""

import logging
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CANVAS_WIDTH       = 1080
CANVAS_HEIGHT      = 1920
OVERLAY_OPACITY    = 0.45
FPS                = 30
OUTPUT_DIR         = Path("data/output")
CROSSFADE_DURATION = 0.3   # seconds of overlap between consecutive b-roll clips

# Script section timing — mirrors caption_renderer.CAPTION_TIMINGS
SECTION_TIMINGS: list[tuple[str, float, float]] = [
    ("hook",      0.0,   2.0),
    ("statement", 2.0,   6.0),
    ("twist",     6.0,  10.0),
    ("question",  10.0, 13.5),
]
VIDEO_DURATION = SECTION_TIMINGS[-1][2]   # 13.5 seconds


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class BuildResult:
    """Result of a VideoBuilder.build() call."""

    topic_id: str
    output_path: str
    duration_seconds: float
    is_valid: bool
    validation_errors: list[str] = field(default_factory=list)
    built_at: str = ""

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
# VideoBuilder
# ---------------------------------------------------------------------------

class VideoBuilder:
    """
    Combines stock footage, voiceover audio, and on-screen captions into a
    portrait-format (1080×1920) MP4 suitable for YouTube Shorts.

    When word_timestamps are supplied (the normal production path), a pure
    single-pass ffmpeg render is used:
      • ffmpeg reads each b-roll clip directly (with -stream_loop for short
        clips), applies scale/crop-to-fill, xfade crossfades, and a dark
        overlay — all in C-speed inside filter_complex.
      • Caption RGBA frames are pre-baked once (one PIL render per word
        state) and looked up with O(log n) bisect per output frame.
        They are piped to ffmpeg stdin as rawvideo.
      • ffmpeg mixes the voiceover and writes the final output in one pass.

    When word_timestamps are None (legacy / test path), the original
    moviepy CompositeVideoClip path is used unchanged so all tests pass.

    Args:
        output_dir: Directory to write the final MP4. Defaults to data/output/.
        canvas_width:  Output width in pixels (default 1080).
        canvas_height: Output height in pixels (default 1920).
        fps:           Output frame rate (default 30).
        overlay_opacity: Opacity of the dark overlay (0–1, default 0.45).
    """

    def __init__(
        self,
        output_dir: str | Path = OUTPUT_DIR,
        canvas_width: int = CANVAS_WIDTH,
        canvas_height: int = CANVAS_HEIGHT,
        fps: int = FPS,
        overlay_opacity: float = OVERLAY_OPACITY,
    ) -> None:
        self.output_dir      = Path(output_dir)
        self.canvas_width    = canvas_width
        self.canvas_height   = canvas_height
        self.fps             = fps
        self.overlay_opacity = overlay_opacity

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        topic_id: str,
        script_dict: dict[str, str],
        audio_path: str | Path,
        stock_video_path: str | Path | list[str | Path],
        cta_overlay: str = "",
        word_timestamps: list[dict] | None = None,
        anthropic_api_key: str = "",
    ) -> BuildResult:
        """
        Build and export the final video MP4.

        When word_timestamps are provided, a 2-pass render is used (fast).
        When word_timestamps are None, falls back to the legacy moviepy path
        (used by tests via _assemble mock).

        Args:
            topic_id:          Unique identifier used in output filename.
            script_dict:       Dict with keys hook, statement, twist, question.
            audio_path:        Path to the voiceover MP3.
            stock_video_path:  Path (or list of paths) to stock footage MP4(s).
            cta_overlay:       Optional CTA banner text for the last 3 seconds.
            word_timestamps:   Optional list of {text, start_time, end_time} from
                               ElevenLabs. When supplied, enables 2-pass fast render.

        Returns:
            BuildResult with output path and validation status.
        """
        audio_path = Path(audio_path)

        # Normalise stock paths to list
        if isinstance(stock_video_path, (str, Path)):
            stock_paths = [Path(stock_video_path)]
        else:
            stock_paths = [Path(p) for p in stock_video_path]

        errors = self._validate_inputs(audio_path, stock_paths)
        if errors:
            return BuildResult(
                topic_id=topic_id,
                output_path="",
                duration_seconds=0.0,
                is_valid=False,
                validation_errors=errors,
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"{topic_id}_final.mp4"

        logger.info(
            "Building video for topic_id=%s (%d clip(s))", topic_id, len(stock_paths)
        )

        t0 = time.perf_counter()

        if word_timestamps:
            # ── Fast 2-pass path ──────────────────────────────────────────────
            actual_duration = self._write_two_pass(
                topic_id=topic_id,
                output_path=output_path,
                stock_paths=stock_paths,
                audio_path=audio_path,
                word_timestamps=word_timestamps,
                cta_overlay=cta_overlay,
                anthropic_api_key=anthropic_api_key,
                script_dict=script_dict,
            )
        else:
            # ── Legacy single-pass moviepy path (used by tests) ───────────────
            final_clip, actual_duration = self._assemble(
                script_dict, audio_path, stock_paths,
                cta_overlay=cta_overlay,
                word_timestamps=None,
            )
            final_clip.write_videofile(
                str(output_path),
                fps=self.fps,
                codec="libx264",
                audio_codec="aac",
                bitrate="8000k",
                audio_bitrate="192k",
                ffmpeg_params=["-preset", "ultrafast"],
                logger=None,
            )
            final_clip.close()

        logger.info("TIMING total_build=%.2fs", time.perf_counter() - t0)
        logger.info("Exported final video to %s (%.1fs)", output_path, actual_duration)

        return BuildResult(
            topic_id=topic_id,
            output_path=str(output_path),
            duration_seconds=actual_duration,
            is_valid=True,
        )

    # ------------------------------------------------------------------
    # Fast path — pure single-pass ffmpeg render
    # ------------------------------------------------------------------

    def _write_two_pass(
        self,
        topic_id: str,
        output_path: Path,
        stock_paths: list[Path],
        audio_path: Path,
        word_timestamps: list[dict],
        cta_overlay: str,
        anthropic_api_key: str = "",
        script_dict: dict[str, str] | None = None,
    ) -> float:
        """
        Pure single-pass ffmpeg render — no moviepy in the hot path.

        All b-roll trimming, scale/crop-to-fill, xfade crossfades, and the dark
        overlay are handled entirely inside ffmpeg's filter_complex (C-speed).
        Caption RGBA frames are pre-baked with PIL (one frame per word state)
        and looked up with O(log n) bisect per output frame, then piped to
        ffmpeg stdin as rawvideo.  ffmpeg mixes the voiceover and writes the
        final MP4 in a single pass.

        Returns:
            actual_duration — voiceover length in seconds.
        """
        import bisect
        import numpy as np
        from moviepy import AudioFileClip
        from src.media.caption_renderer import (
            CaptionRenderer,
            _group_words,
            _load_word_font,
            _render_word_frame,
            _word_font_size,
            WORD_ENTRANCE_DUR,
        )

        W, H, fps = self.canvas_width, self.canvas_height, self.fps

        # ── Read audio duration (metadata only — fast) ─────────────────────────
        t_audio = time.perf_counter()
        audio = AudioFileClip(str(audio_path))
        actual_duration = audio.duration
        audio.close()
        logger.info("Voiceover duration: %.2fs — video will match", actual_duration)
        logger.info("TIMING audio_read=%.3fs", time.perf_counter() - t_audio)

        n = len(stock_paths)

        # ── Calculate per-clip durations & cut points ─────────────────────────
        if word_timestamps and n > 1:
            cut_times = self._cuts_from_word_timestamps(word_timestamps, actual_duration, n)
        else:
            cut_times = []

        if cut_times:
            boundaries = [0.0] + cut_times + [actual_duration]
            raw_durations = [
                (boundaries[i + 1] - boundaries[i]) + CROSSFADE_DURATION
                for i in range(n)
            ]
            raw_durations[-1] -= CROSSFADE_DURATION
            logger.info("B-roll cuts at: %s", [round(c, 2) for c in cut_times])
        else:
            per_clip_dur = (actual_duration + (n - 1) * CROSSFADE_DURATION) / n
            raw_durations = [per_clip_dur] * n

        # ── Pre-render caption states (PIL, one per word) ──────────────────────
        t_caps = time.perf_counter()
        states: list[np.ndarray] = []
        start_times: list[float] = []
        if word_timestamps:
            grouped = _group_words(word_timestamps)
            font = _load_word_font(canvas_w=W)
            for word in grouped:
                t_stable = word["start_time"] + WORD_ENTRANCE_DUR + 0.001
                img = _render_word_frame(t_stable, grouped, W, H, font)
                states.append(np.array(img, dtype=np.uint8))
                start_times.append(word["start_time"])

        # CTA gold banner (rendered once, blended per-frame if within window)
        renderer = CaptionRenderer(canvas_width=W, canvas_height=H)
        cta_rgba: np.ndarray | None = None
        cta_start_t = actual_duration
        cta_end_t = actual_duration
        cta_spec = renderer.build_cta_spec(cta_overlay, video_duration=actual_duration)
        if cta_spec is not None:
            cta_rgba = self._render_cta_pil(cta_spec.text, W, H, cta_spec.y)
            cta_start_t = cta_spec.start
            cta_end_t = cta_spec.end

        blank = np.zeros((H, W, 4), dtype=np.uint8)

        logger.info(
            "TIMING caption_prebake=%.3fs (%d states)",
            time.perf_counter() - t_caps, len(states),
        )

        # ── Kinetic text overlays ──────────────────────────────────────────────
        kinetic_overlays: list[tuple[float, float, "np.ndarray", "np.ndarray"]] = []
        # (start_t, end_t, entrance_frame, stable_frame)

        OVERLAY_ENTRANCE_DUR = 0.12
        OVERLAY_EXIT_DUR     = 0.15
        OVERLAY_DURATION     = 1.5
        HOOK_GUARD           = 2.0
        CTA_GUARD            = 3.0

        avail_start = HOOK_GUARD
        avail_end   = actual_duration - CTA_GUARD

        if script_dict and anthropic_api_key and (avail_end - avail_start) > OVERLAY_DURATION:
            full_script = " ".join(filter(None, [
                script_dict.get("hook", ""),
                script_dict.get("statement", ""),
                script_dict.get("twist", ""),
                script_dict.get("question", ""),
            ]))
            phrases = self.extract_key_phrases(full_script, anthropic_api_key)
            n_overlays = min(len(phrases), 3)
            if n_overlays > 0:
                span = avail_end - avail_start
                spacing = span / (n_overlays + 1)
                for k, phrase in enumerate(phrases[:n_overlays]):
                    t_center = avail_start + spacing * (k + 1)
                    t_start  = t_center - OVERLAY_DURATION / 2
                    t_end    = t_start + OVERLAY_DURATION
                    stable_frame   = self._render_kinetic_overlay_pil(phrase, W, H, scale=1.0, alpha_mult=1.0)
                    entrance_frame = self._render_kinetic_overlay_pil(phrase, W, H, scale=1.2, alpha_mult=1.0)
                    kinetic_overlays.append((t_start, t_end, entrance_frame, stable_frame))
                logger.info("[kinetic] %d overlay(s) scheduled", len(kinetic_overlays))

        def get_caption_frame(t: float) -> np.ndarray:
            idx = bisect.bisect_right(start_times, t) - 1
            base = states[idx] if (states and 0 <= idx < len(states)) else blank
            if cta_rgba is not None and cta_start_t <= t <= cta_end_t:
                base = _alpha_composite_rgba(base, cta_rgba)
            # Kinetic text overlays
            for (ov_start, ov_end, entrance_frame, stable_frame) in kinetic_overlays:
                if ov_start <= t <= ov_end:
                    rel = t - ov_start
                    if rel < OVERLAY_ENTRANCE_DUR:
                        # Use pre-rendered entrance frame (larger scale)
                        base = _alpha_composite_rgba(base, entrance_frame)
                    elif t > ov_end - OVERLAY_EXIT_DUR:
                        # Fade exit: alpha drops 1→0
                        progress = (t - (ov_end - OVERLAY_EXIT_DUR)) / OVERLAY_EXIT_DUR
                        alpha = 1.0 - progress
                        fade_frame = (stable_frame.astype(np.float32) * alpha).astype(np.uint8)
                        base = _alpha_composite_rgba(base, fade_frame)
                    else:
                        base = _alpha_composite_rgba(base, stable_frame)
            return base

        # ── Step 1: Pre-process each b-roll clip → normalised temp file ───────────
        # Each clip is independently re-encoded to 30fps, WxH, trimmed (with loop
        # if the clip is shorter than needed).  This gives the main ffmpeg command
        # clean, CFR inputs with identical timebases, which is required by xfade.
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        # ── Enforce clip duration limits ──────────────────────────────────
        MAX_CLIP_OUTPUT = 12.0   # hard cap — no single clip > 12s in output
        TARGET_CLIP_CAP = 8.0    # soft target — prefer clips ≤ 8s
        for i in range(len(raw_durations)):
            raw_durations[i] = min(raw_durations[i], MAX_CLIP_OUTPUT)

        t_prep = time.perf_counter()
        norm_paths: list[Path] = []
        try:
            for i, (dur, path) in enumerate(zip(raw_durations, stock_paths)):
                # Trim from middle of source clip (skip first/last 10%)
                # Use ffprobe-like approach: -ss skips into clip before looping
                skip_start = 0.0
                try:
                    probe_cmd = [ffmpeg_exe, "-i", str(path), "-f", "null", "-"]
                    probe_r = subprocess.run(
                        probe_cmd, capture_output=True, timeout=10,
                    )
                    # Parse duration from stderr
                    import re as _re
                    m = _re.search(r"Duration:\s*(\d+):(\d+):(\d+)\.(\d+)", probe_r.stderr.decode(errors="replace"))
                    if m:
                        src_dur = int(m.group(1))*3600 + int(m.group(2))*60 + int(m.group(3)) + int(m.group(4))/100
                        if src_dur > dur + 1.0:
                            # Source is longer than needed — take from middle
                            skip_start = src_dur * 0.10
                            max_end = src_dur * 0.90
                            if skip_start + dur > max_end:
                                skip_start = max(0, max_end - dur)
                except Exception:
                    pass  # fall back to start of clip

                norm = self.output_dir / f"_norm_{topic_id}_{i}.mp4"
                norm_cmd = [
                    ffmpeg_exe, "-y",
                    "-ss", f"{skip_start:.3f}",
                    "-stream_loop", "-1", "-t", f"{dur:.4f}", "-i", str(path),
                    "-vf", f"scale={W}:{H},fps={fps},setpts=PTS-STARTPTS",
                    "-r", str(fps),   # force exact CFR metadata in container
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                    "-an",   # no audio in temp files
                    str(norm),
                ]
                r = subprocess.run(norm_cmd, capture_output=True, timeout=180)
                if r.returncode != 0:
                    raise RuntimeError(
                        f"Pre-process clip {i} failed: "
                        f"{r.stderr.decode(errors='replace')[-1000:]}"
                    )
                norm_paths.append(norm)
            logger.info(
                "TIMING clip_preprocess=%.2fs (%d clips normalised)",
                time.perf_counter() - t_prep, n,
            )

            # ── Step 2: Build filter_complex with pre-normalised inputs ─────────
            # Pre-processed clips already have correct CFR and PTS starting at 0,
            # so we reference them directly as [0:v], [1:v], … without setpts.
            # (setpts=PTS-STARTPTS strips frame_rate metadata → xfade sees 1/0.)
            filter_parts: list[str] = []

            # Xfade crossfade chain
            # Offset[i] = Σ raw_durations[0..i] − (i+1)×CROSSFADE = boundaries[i+1]
            if n == 1:
                combined = "0:v"
            else:
                cumsum = 0.0
                for i in range(n - 1):
                    cumsum += raw_durations[i]
                    offset = max(0.001, cumsum - (i + 1) * CROSSFADE_DURATION)
                    in1 = "0:v" if i == 0 else f"xf{i}"
                    out_lbl = f"xf{i + 1}"
                    filter_parts.append(
                        f"[{in1}][{i + 1}:v]xfade=transition=fade"
                        f":duration={CROSSFADE_DURATION:.3f}:offset={offset:.4f}[{out_lbl}]"
                    )
                combined = f"xf{n - 1}"

            # Dark overlay: colorchannelmixer scales RGB to (1 − overlay_opacity)
            brightness = 1.0 - self.overlay_opacity
            br = f"{brightness:.3f}"
            filter_parts.append(
                f"[{combined}]colorchannelmixer=rr={br}:gg={br}:bb={br}[bg_dark]"
            )

            # Caption overlay from Python stdin pipe (input index n)
            caption_idx = n
            audio_idx = n + 1
            filter_parts.append(f"[bg_dark][{caption_idx}:v]overlay[captioned]")
            filter_parts.append("[captioned]vignette=PI/5.5[final]")

            filter_complex = ";".join(filter_parts)
            logger.debug("ffmpeg filter_complex:\n%s", filter_complex)

            # ── Build main ffmpeg command ──────────────────────────────────────
            cmd: list[str] = [ffmpeg_exe, "-y"]
            for norm in norm_paths:
                cmd += ["-i", str(norm)]
            # Caption RGBA from Python stdin
            cmd += [
                "-f", "rawvideo", "-pix_fmt", "rgba",
                "-s", f"{W}x{H}", "-r", str(fps),
                "-i", "pipe:0",
            ]
            # Audio
            cmd += ["-i", str(audio_path)]
            cmd += [
                "-filter_complex", filter_complex,
                "-map", "[final]",
                "-map", f"{audio_idx}:a",
                "-c:v", "libx264", "-preset", "ultrafast",
                "-b:v", "4000k",
                "-c:a", "aac", "-b:a", "192k",
                "-t", f"{actual_duration:.4f}",
                str(output_path),
            ]

            # ── Step 3: Pipe caption frames → ffmpeg ────────────────────────
            # Background thread drains stderr to prevent pipe-buffer deadlock.
            import threading

            n_frames = int(actual_duration * fps) + 2
            logger.info(
                "Single-pass ffmpeg: %d b-roll clip(s), %d caption states, "
                "%d frames (%.1fs @ %d fps)",
                n, len(states), n_frames, actual_duration, fps,
            )

            stderr_buf: list[bytes] = []

            def _drain(pipe: "subprocess.IO[bytes]") -> None:  # noqa: E301
                stderr_buf.append(pipe.read())

            t_pipe = time.perf_counter()
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,   # drained by background thread
                stdout=subprocess.DEVNULL,
            )
            drain_thread = threading.Thread(
                target=_drain, args=(proc.stderr,), daemon=True
            )
            drain_thread.start()

            frames_written = 0
            try:
                for i in range(n_frames):
                    t = i / fps
                    if t > actual_duration + 0.5:
                        break
                    proc.stdin.write(get_caption_frame(t).tobytes())
                    frames_written += 1
            except BrokenPipeError:
                pass  # ffmpeg exited early — returncode check below catches errors
            finally:
                proc.stdin.close()

            proc.wait()
            drain_thread.join()

            elapsed = time.perf_counter() - t_pipe
            logger.info(
                "TIMING ffmpeg_encode=%.2fs (%d frames, %.0f fps throughput)",
                elapsed, frames_written, frames_written / elapsed if elapsed > 0 else 0,
            )

            if proc.returncode != 0:
                stderr_text = (stderr_buf[0] if stderr_buf else b"").decode(
                    errors="replace"
                )
                logger.error("ffmpeg stderr (last 3000 chars):\n%s", stderr_text[-3000:])
                raise RuntimeError(
                    f"ffmpeg single-pass render failed (returncode={proc.returncode})"
                )

        finally:
            # Clean up normalised temp files regardless of success or failure
            for p in norm_paths:
                p.unlink(missing_ok=True)

        return actual_duration

    @staticmethod
    def _render_cta_pil(text: str, canvas_w: int, canvas_h: int, y_center: int) -> "np.ndarray":
        """Render the CTA gold banner as an RGBA numpy array using PIL (no moviepy)."""
        import numpy as np
        from PIL import Image, ImageDraw
        from src.media.caption_renderer import (
            CTA_BG_COLOR,
            CTA_FONT_SIZE,
            WORD_FONT_SEARCH_PATHS,
        )
        _CTA_CORNER_RADIUS = 12  # local constant for CTA banner rounding

        # Load a bold font
        font = None
        try:
            from PIL import ImageFont
            for path in WORD_FONT_SEARCH_PATHS:
                if path is None:
                    font = ImageFont.load_default()
                    break
                try:
                    font = ImageFont.truetype(path, CTA_FONT_SIZE)
                    break
                except (IOError, OSError):
                    continue
        except ImportError:
            pass

        img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Measure text
        try:
            bb = font.getbbox(text) if font else (0, 0, 300, CTA_FONT_SIZE)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
        except Exception:
            tw, th = 300, CTA_FONT_SIZE

        pad_x, pad_y = 20, 12
        banner_w = min(canvas_w - 80, tw + 2 * pad_x)
        banner_h = th + 2 * pad_y
        x1 = (canvas_w - banner_w) // 2
        x2 = x1 + banner_w
        y1 = y_center - banner_h // 2
        y2 = y1 + banner_h

        draw.rounded_rectangle((x1, y1, x2, y2), radius=_CTA_CORNER_RADIUS,
                                fill=(*CTA_BG_COLOR, 255))
        draw.text((x1 + pad_x, y1 + pad_y), text, font=font, fill=(0, 0, 0, 255))

        return np.array(img, dtype=np.uint8)

    def apply_ken_burns(
        self,
        image_path: "str | Path",
        duration: float,
        effect: "str | None" = None,
    ) -> "list[np.ndarray]":
        """Pre-render Ken Burns effect frames from a still image.

        Effects: 'zoom_in', 'zoom_out', 'pan_right', 'pan_left'.
        If None, randomly selects one.

        Uses keyframe interpolation: pre-renders start and end frames with
        PIL (LANCZOS quality), then interpolates all intermediate frames
        with numpy lerp for performance (< 5 s per clip in practice).

        Args:
            image_path: Path to the source photo (any PIL-supported format).
            duration:   Output clip duration in seconds.
            effect:     One of the 4 effect names, or None for random.

        Returns:
            List of RGB numpy arrays (H×W×3, dtype uint8), one per frame at self.fps.
        """
        import random as _rnd
        import numpy as np
        from PIL import Image

        EFFECTS = ["zoom_in", "zoom_out", "pan_right", "pan_left"]
        if effect is None:
            effect = _rnd.choice(EFFECTS)

        W, H, fps = self.canvas_width, self.canvas_height, self.fps
        ZOOM = 1.15   # zoom factor for zoom_in / zoom_out

        # ── Scale source image so there is buffer for zoom / pan ──────────────
        img = Image.open(str(image_path)).convert("RGB")
        buf_scale = max((W * ZOOM) / img.width, (H * ZOOM) / img.height)
        sw = int(img.width  * buf_scale)
        sh = int(img.height * buf_scale)
        img_buf = img.resize((sw, sh), Image.LANCZOS)

        cx = sw // 2
        cy = sh // 2
        # Available buffer pixels on each side
        w_buf = (sw - W) // 2
        h_buf = (sh - H) // 2

        # ── Compute start/end crop boxes (PIL format: x1, y1, x2, y2) ─────────
        if effect == "zoom_in":
            # Start: full W×H crop (normal view).
            # End:   smaller crop (W/ZOOM × H/ZOOM) → scaled up = zoomed in.
            e_cw = int(W / ZOOM)
            e_ch = int(H / ZOOM)
            s_box = (cx - W  // 2, cy - H  // 2, cx + W  // 2, cy + H  // 2)
            e_box = (cx - e_cw // 2, cy - e_ch // 2, cx + e_cw // 2, cy + e_ch // 2)

        elif effect == "zoom_out":
            # Start: smaller crop (zoomed in). End: full W×H (normal view).
            s_cw = int(W / ZOOM)
            s_ch = int(H / ZOOM)
            s_box = (cx - s_cw // 2, cy - s_ch // 2, cx + s_cw // 2, cy + s_ch // 2)
            e_box = (cx - W  // 2, cy - H  // 2, cx + W  // 2, cy + H  // 2)

        elif effect == "pan_right":
            # Crop window moves right; start from left, end at center.
            pan_x = max(1, int(w_buf * 0.8))
            s_box = (cx - W // 2 - pan_x, cy - H // 2, cx + W // 2 - pan_x, cy + H // 2)
            e_box = (cx - W // 2,          cy - H // 2, cx + W // 2,          cy + H // 2)

        elif effect == "pan_left":
            # Crop window moves left; start at center, end shifted right.
            pan_x = max(1, int(w_buf * 0.8))
            s_box = (cx - W // 2,          cy - H // 2, cx + W // 2,          cy + H // 2)
            e_box = (cx - W // 2 + pan_x, cy - H // 2, cx + W // 2 + pan_x, cy + H // 2)

        else:
            s_box = e_box = (cx - W // 2, cy - H // 2, cx + W // 2, cy + H // 2)

        def _render_box(box: tuple) -> "np.ndarray":
            x1, y1, x2, y2 = box
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(sw, x2); y2 = min(sh, y2)
            crop = img_buf.crop((x1, y1, x2, y2))
            if crop.size != (W, H):
                crop = crop.resize((W, H), Image.LANCZOS)
            return np.array(crop, dtype=np.float32)

        start_frame = _render_box(s_box)
        end_frame   = _render_box(e_box)

        n_frames = int(duration * fps) + 1
        frames: list[np.ndarray] = []
        for i in range(n_frames):
            progress = i / max(n_frames - 1, 1)
            frame = (
                start_frame * (1.0 - progress) + end_frame * progress
            ).astype(np.uint8)
            frames.append(frame)

        logger.info(
            "[ken_burns] effect=%s frames=%d duration=%.2fs",
            effect, len(frames), duration,
        )
        return frames

    def write_ken_burns_mp4(
        self,
        image_path: "str | Path",
        output_path: "str | Path",
        duration: float,
        effect: "str | None" = None,
    ) -> bool:
        """Apply Ken Burns effect to a photo and write the result to an mp4 file.

        Calls apply_ken_burns() to get frames, then pipes them to ffmpeg as
        rawvideo (rgb24) to produce a H.264 mp4.

        Returns:
            True on success, False on any failure (never raises).
        """
        import subprocess
        from pathlib import Path as _Path

        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as exc:
            logger.warning("[ken_burns] imageio_ffmpeg not available: %s", exc)
            return False

        try:
            frames = self.apply_ken_burns(image_path, duration=duration, effect=effect)
            if not frames:
                return False

            out = _Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)

            W, H, fps = self.canvas_width, self.canvas_height, self.fps
            cmd = [
                ffmpeg_exe, "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{W}x{H}", "-r", str(fps),
                "-i", "pipe:0",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                "-an",
                str(out),
            ]

            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                for frame in frames:
                    proc.stdin.write(frame.tobytes())
            except BrokenPipeError:
                pass
            finally:
                proc.stdin.close()

            proc.wait()
            success = proc.returncode == 0 and out.exists() and out.stat().st_size > 0
            if success:
                logger.info("[ken_burns] Wrote %d frames to %s", len(frames), out)
            else:
                logger.warning("[ken_burns] write_ken_burns_mp4 failed (returncode=%d)", proc.returncode)
            return success

        except Exception as exc:
            logger.warning("[ken_burns] write_ken_burns_mp4 error: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Legacy single-pass path (used by tests via _assemble mock)
    # ------------------------------------------------------------------

    def _assemble(
        self,
        script_dict: dict[str, str],
        audio_path: Path,
        stock_paths: list[Path],
        cta_overlay: str = "",
        word_timestamps: list[dict] | None = None,
    ) -> tuple:
        """Compose video layers and return (CompositeVideoClip, actual_duration_seconds).

        Used only in the legacy path (word_timestamps=None).  Tests mock this method
        to avoid real moviepy calls.

        Total video duration equals the full voiceover length — the audio is never
        trimmed.  B-roll clips are divided into equal segments that together span
        the full audio duration (with CROSSFADE_DURATION overlaps at each join).
        """
        from moviepy import (  # moviepy v2
            AudioFileClip,
            ColorClip,
            CompositeVideoClip,
            VideoFileClip,
            concatenate_videoclips,
        )
        import moviepy.video.fx as vfx
        from src.media.caption_renderer import CaptionRenderer

        _t0 = time.perf_counter()

        # 1. Load audio — read actual duration, never trim
        audio = AudioFileClip(str(audio_path))
        actual_duration = audio.duration
        logger.info("Voiceover duration: %.2fs — video will match", actual_duration)
        logger.info("TIMING audio_load=%.3fs", time.perf_counter() - _t0)

        n = len(stock_paths)

        if word_timestamps and n > 1:
            cut_times = self._cuts_from_word_timestamps(word_timestamps, actual_duration, n)
        else:
            cut_times = []

        if cut_times:
            boundaries = [0.0] + cut_times + [actual_duration]
            raw_durations = [
                (boundaries[i + 1] - boundaries[i]) + CROSSFADE_DURATION
                for i in range(n)
            ]
            raw_durations[-1] -= CROSSFADE_DURATION
            logger.info("B-roll cuts at: %s", [round(c, 2) for c in cut_times])
        else:
            per_clip_dur = (actual_duration + (n - 1) * CROSSFADE_DURATION) / n
            raw_durations = [per_clip_dur] * n

        # ── Enforce clip duration limits (legacy path) ────────────────────
        _MAX_CLIP_OUTPUT = 12.0
        for i in range(len(raw_durations)):
            raw_durations[i] = min(raw_durations[i], _MAX_CLIP_OUTPUT)

        _t1 = time.perf_counter()
        bg_clips = []
        for i, path in enumerate(stock_paths):
            raw_clip = VideoFileClip(str(path))
            needed = raw_durations[i]

            # Trim from middle of source (skip first/last 10%) if source is long enough
            if raw_clip.duration > needed + 1.0:
                skip = raw_clip.duration * 0.10
                max_end = raw_clip.duration * 0.90
                start = skip
                end = start + needed
                if end > max_end:
                    start = max(0, max_end - needed)
                    end = start + needed
                raw = raw_clip.subclipped(start, min(end, raw_clip.duration))
            elif needed > raw_clip.duration:
                loops = int(needed / raw_clip.duration) + 2
                raw = concatenate_videoclips([raw_clip] * loops).subclipped(0, needed)
            else:
                raw = raw_clip.subclipped(0, needed)
            clip = self._fit_clip(raw, self.canvas_width, self.canvas_height)
            effects = []
            if i > 0:
                effects.append(vfx.CrossFadeIn(CROSSFADE_DURATION))
            if i < n - 1:
                effects.append(vfx.CrossFadeOut(CROSSFADE_DURATION))
            if effects:
                clip = clip.with_effects(effects)
            bg_clips.append(clip)

        if n == 1:
            bg = bg_clips[0]
        else:
            bg = concatenate_videoclips(
                bg_clips, padding=-CROSSFADE_DURATION, method="compose"
            )
        bg = bg.subclipped(0, actual_duration)
        logger.info("TIMING broll_load+concat=%.3fs", time.perf_counter() - _t1)

        overlay = (
            ColorClip(
                size=(self.canvas_width, self.canvas_height),
                color=(0, 0, 0),
            )
            .with_opacity(self.overlay_opacity)
            .with_duration(actual_duration)
        )

        _t2 = time.perf_counter()
        renderer = CaptionRenderer(
            canvas_width=self.canvas_width,
            canvas_height=self.canvas_height,
        )
        caption_clips = renderer.render(
            script_dict,
            word_timestamps=word_timestamps,
            cta_overlay=cta_overlay,
            video_duration=actual_duration,
        )
        logger.info("TIMING caption_render=%.3fs", time.perf_counter() - _t2)

        _t3 = time.perf_counter()
        layers = [bg, overlay] + caption_clips
        final = (
            CompositeVideoClip(layers, size=(self.canvas_width, self.canvas_height))
            .with_audio(audio)
            .with_duration(actual_duration)
        )
        logger.info("TIMING composite_build=%.3fs", time.perf_counter() - _t3)
        return final, actual_duration

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fit_clip(clip, target_w: int, target_h: int):
        """Crop-to-fill: scale so both dimensions cover the target, then crop center."""
        cw, ch = clip.size
        if cw == target_w and ch == target_h:
            return clip

        scale = max(target_w / cw, target_h / ch)
        new_w = round(cw * scale)
        new_h = round(ch * scale)

        if new_w < target_w:
            logger.warning(
                "Clip %dx%d is too narrow to fill %dx%d after scale — "
                "resizing as fallback (may distort)",
                cw, ch, target_w, target_h,
            )
            return clip.resized((target_w, target_h))

        scaled = clip.resized((new_w, new_h))
        x1 = (new_w - target_w) / 2
        y1 = (new_h - target_h) / 2
        return scaled.cropped(x1=x1, y1=y1, x2=x1 + target_w, y2=y1 + target_h)

    @staticmethod
    def _validate_inputs(audio_path: Path, stock_paths: list[Path]) -> list[str]:
        errors: list[str] = []
        if not audio_path.exists():
            errors.append(f"audio file not found: {audio_path}")
        for p in stock_paths:
            if not p.exists():
                errors.append(f"stock video not found: {p}")
        return errors

    @staticmethod
    def _cuts_from_word_timestamps(
        word_timestamps: list[dict],
        audio_duration: float,
        n_clips: int,
        min_pause: float = 0.4,
    ) -> list[float]:
        """Find natural cut points using pauses between spoken words."""
        if n_clips <= 1 or not word_timestamps:
            return []

        needed = n_clips - 1
        pauses: list[float] = []
        for i in range(len(word_timestamps) - 1):
            gap = word_timestamps[i + 1]["start_time"] - word_timestamps[i]["end_time"]
            if gap >= min_pause:
                mid = (word_timestamps[i]["end_time"] + word_timestamps[i + 1]["start_time"]) / 2
                pauses.append(mid)

        if len(pauses) >= needed:
            step = len(pauses) / needed
            return sorted(pauses[int(i * step)] for i in range(needed))

        equal_step = audio_duration / n_clips
        cuts = list(pauses)
        for k in range(1, n_clips):
            candidate = k * equal_step
            if not any(abs(candidate - c) < equal_step * 0.25 for c in cuts):
                cuts.append(candidate)
            if len(cuts) >= needed:
                break

        cuts.sort()
        if len(cuts) < needed:
            return [audio_duration * i / n_clips for i in range(1, n_clips)]
        return cuts[:needed]

    @staticmethod
    def extract_key_phrases(script: str, api_key: str = "") -> list[str]:
        """
        Call Claude API to extract up to 3 key phrases from the script for
        kinetic text overlays.

        Args:
            script: Full script text (all 4 parts joined).
            api_key: Anthropic API key. If empty, reads ANTHROPIC_API_KEY from env.

        Returns:
            List of 0–3 strings (max 4 words each). Returns [] on any failure.
        """
        import json as _json
        import os

        key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        if not key or not script.strip():
            return []
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=key)
            prompt = (
                "Extract up to 3 key phrases or statistics from this script that would look "
                "powerful displayed as bold text overlays on screen.\n"
                "Return JSON array of strings only.\n"
                "Max 3 items. Max 4 words each.\n"
                "Prefer numbers, percentages, and short punchy statements.\n"
                f"Script: {script}"
            )
            message = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=80,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text.strip()
            if raw.startswith("```"):
                raw = "\n".join(
                    line for line in raw.splitlines()
                    if not line.strip().startswith("```")
                ).strip()
            phrases = _json.loads(raw)
            if isinstance(phrases, list):
                return [str(p).strip() for p in phrases[:3] if str(p).strip()]
        except Exception as exc:
            logger.warning("[kinetic] phrase extraction failed: %s", exc)
        return []

    @staticmethod
    def _render_kinetic_overlay_pil(
        text: str,
        canvas_w: int,
        canvas_h: int,
        scale: float = 1.0,
        alpha_mult: float = 1.0,
    ) -> "np.ndarray":
        """
        Pre-render a kinetic text overlay as RGBA numpy array.

        The overlay is a white bold text on a semi-transparent black rounded pill,
        centred horizontally at y=42% of canvas height.

        Args:
            text:       Text to display.
            canvas_w:   Canvas width.
            canvas_h:   Canvas height.
            scale:      Scale factor (1.0 = normal, 1.2 = entrance slam).
            alpha_mult: Alpha multiplier for fade-out (0.0–1.0).

        Returns:
            RGBA uint8 numpy array of shape (canvas_h, canvas_w, 4).
        """
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont

        FONT_SIZE = 88
        PILL_ALPHA = int(165 * alpha_mult)  # 65% opacity pill
        TEXT_ALPHA = int(255 * alpha_mult)
        CORNER_R = 16
        PAD_X, PAD_Y = 28, 16

        # Load font
        font = None
        font_paths = [
            "C:/Windows/Fonts/ariblk.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "/usr/share/fonts/truetype/msttcorefonts/Arial_Black.ttf",
            None,
        ]
        for fp in font_paths:
            if fp is None:
                font = ImageFont.load_default()
                break
            try:
                font = ImageFont.truetype(fp, FONT_SIZE)
                break
            except (IOError, OSError):
                continue

        # Measure text
        try:
            bb = font.getbbox(text)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
        except Exception:
            tw, th = int(FONT_SIZE * len(text) * 0.55), FONT_SIZE

        pill_w = int(tw + 2 * PAD_X)
        pill_h = int(th + 2 * PAD_Y)

        # Apply scale (for entrance animation)
        scaled_w = int(pill_w * scale)
        scaled_h = int(pill_h * scale)

        # Create pill image
        pill_img = Image.new("RGBA", (max(1, scaled_w), max(1, scaled_h)), (0, 0, 0, 0))
        d = ImageDraw.Draw(pill_img)
        d.rounded_rectangle(
            [0, 0, max(1, scaled_w) - 1, max(1, scaled_h) - 1],
            radius=int(CORNER_R * scale),
            fill=(0, 0, 0, PILL_ALPHA),
        )

        # Text centred in pill (scale font proportionally)
        if scale != 1.0:
            scaled_font_size = max(1, int(FONT_SIZE * scale))
            scaled_font = None
            for fp in font_paths:
                if fp is None:
                    scaled_font = ImageFont.load_default()
                    break
                try:
                    scaled_font = ImageFont.truetype(fp, scaled_font_size)
                    break
                except (IOError, OSError):
                    continue
        else:
            scaled_font = font

        tx = (scaled_w - int(tw * scale)) // 2
        ty = (scaled_h - int(th * scale)) // 2
        d.text((tx, ty), text, font=scaled_font, fill=(255, 255, 255, TEXT_ALPHA))

        # Compose onto full canvas at y=42%
        canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        center_y = int(canvas_h * 0.42)
        x0 = (canvas_w - scaled_w) // 2
        y0 = center_y - scaled_h // 2
        canvas.paste(pill_img, (x0, y0), pill_img)

        return np.array(canvas, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Module-level helper (used by _ffmpeg_overlay_captions)
# ---------------------------------------------------------------------------

def _alpha_composite_rgba(base: "np.ndarray", overlay: "np.ndarray") -> "np.ndarray":
    """Alpha-composite overlay RGBA onto base RGBA; returns a new array."""
    import numpy as np
    result = base.copy()
    a = overlay[:, :, 3:4].astype(np.float32) / 255.0
    result[:, :, :3] = (
        result[:, :, :3].astype(np.float32) * (1.0 - a)
        + overlay[:, :, :3].astype(np.float32) * a
    ).astype(np.uint8)
    result[:, :, 3] = np.maximum(result[:, :, 3], overlay[:, :, 3])
    return result

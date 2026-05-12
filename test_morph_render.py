"""
test_morph_render.py — Render a test video using MorphRenderer.

Produces data/output/test_morph.mp4 without any API calls, DB writes,
YouTube uploads, or Telegram messages.

Usage:
    python test_morph_render.py
"""
from __future__ import annotations

import atexit
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure ffmpeg is reachable (Windows: copy imageio_ffmpeg binary as ffmpeg.exe)
# ---------------------------------------------------------------------------
_ffmpeg_tmpdir: str | None = None
try:
    import imageio_ffmpeg as _iff
    _src_exe = Path(_iff.get_ffmpeg_exe())
    _ffmpeg_tmpdir = tempfile.mkdtemp(prefix="cf_ffmpeg_")
    _dest_exe = Path(_ffmpeg_tmpdir) / "ffmpeg.exe"
    shutil.copy2(_src_exe, _dest_exe)
    # Also copy ffprobe if present in the same directory
    _src_probe = _src_exe.parent / (_src_exe.name.replace("ffmpeg", "ffprobe"))
    if _src_probe.exists():
        shutil.copy2(_src_probe, Path(_ffmpeg_tmpdir) / "ffprobe.exe")
    os.environ["PATH"] = _ffmpeg_tmpdir + os.pathsep + os.environ.get("PATH", "")
    _ffmpeg_exe = str(_dest_exe)
    atexit.register(shutil.rmtree, _ffmpeg_tmpdir, ignore_errors=True)
except ImportError:
    _ffmpeg_exe = "ffmpeg"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    encoding="utf-8",
)
log = logging.getLogger("test_morph")

# ---------------------------------------------------------------------------
# Script content
# ---------------------------------------------------------------------------
SCRIPT_DICT: dict[str, str] = {
    "hook":      "The guy who serves you coffee just retired at 35",
    "statement": "He was making twenty five thousand a year while you were making sixty thousand",
    "twist":     "He spent money to become free while you upgraded apartments and car payments",
    "landing":   "He did not out-earn you he out-thought you",
    "cta":       "Subscribe we say the money things your bank hopes you never hear",
}

OUTPUT_PATH = Path("data/output/test_morph.mp4")
TARGET_DURATION = 15.0   # seconds — short enough for a fast test render


def _make_silent_audio(duration: float, path: Path) -> None:
    """Generate a silent AAC audio file using ffmpeg."""
    log.info("Generating %.1fs silent audio -> %s", duration, path)
    cmd = [
        _ffmpeg_exe, "-y",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
        "-t", str(duration),
        "-c:a", "aac", "-b:a", "128k",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("ffmpeg silent audio failed:\n%s", result.stderr[-800:])
        raise RuntimeError("Could not generate silent audio")
    log.info("Silent audio ready: %s (%.0f bytes)", path, path.stat().st_size)


def main() -> None:
    log.info("=" * 60)
    log.info("ChannelForge -- MorphRenderer test render")
    log.info("=" * 60)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    log.info("Importing MorphRenderer ...")
    try:
        from src.media.morph_renderer import MorphRenderer
    except ImportError as exc:
        log.error("Import failed: %s", exc)
        log.error("Run from repo root: python test_morph_render.py")
        sys.exit(1)

    log.info("Script parts: %s", list(SCRIPT_DICT.keys()))

    with tempfile.TemporaryDirectory() as tmp:
        audio_path = Path(tmp) / "silent.aac"
        _make_silent_audio(TARGET_DURATION, audio_path)

        log.info("Instantiating MorphRenderer ...")
        renderer = MorphRenderer()

        log.info("Starting render -> %s", OUTPUT_PATH)
        t_start = __import__("time").time()

        result = renderer.build(
            topic_id="test_morph",
            script_dict=SCRIPT_DICT,
            audio_path=audio_path,
            word_timestamps=None,
            cta_overlay="",
        )

    elapsed = __import__("time").time() - t_start
    log.info("-" * 60)

    if result.is_valid:
        rendered = Path(result.output_path)
        if rendered.exists() and rendered != OUTPUT_PATH:
            rendered.replace(OUTPUT_PATH)
            log.info("Moved %s -> %s", rendered.name, OUTPUT_PATH.name)
        out = OUTPUT_PATH
        size_mb = out.stat().st_size / 1_048_576
        log.info("SUCCESS -- %s  (%.2f MB)  render_time=%.1fs", out, size_mb, elapsed)
        print(f"\nRender complete: {out}  [{size_mb:.2f} MB]  ({elapsed:.1f}s)")
    else:
        log.error("FAILED -- errors: %s", result.validation_errors)
        sys.exit(1)


if __name__ == "__main__":
    main()

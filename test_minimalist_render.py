"""
test_minimalist_render.py — Render a test video using the minimalist_kinetic style.

Produces data/output/test_minimalist.mp4 without any API calls, DB writes,
YouTube uploads, or Telegram messages.

Usage:
    python test_minimalist_render.py
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
# Ensure ffmpeg is reachable.
# imageio_ffmpeg bundles the binary as "ffmpeg-win-x86_64-vX.Y.exe" on Windows,
# so we copy it to a temp dir as "ffmpeg.exe" and prepend that dir to PATH.
# ---------------------------------------------------------------------------
_ffmpeg_tmpdir: str | None = None

try:
    import imageio_ffmpeg as _iff
    _src_exe = Path(_iff.get_ffmpeg_exe())
    _ffmpeg_tmpdir = tempfile.mkdtemp(prefix="cf_ffmpeg_")
    _dest_exe = Path(_ffmpeg_tmpdir) / "ffmpeg.exe"
    shutil.copy2(_src_exe, _dest_exe)
    os.environ["PATH"] = _ffmpeg_tmpdir + os.pathsep + os.environ.get("PATH", "")
    _ffmpeg_exe = str(_dest_exe)
    # Cleanup on exit
    atexit.register(shutil.rmtree, _ffmpeg_tmpdir, ignore_errors=True)
except ImportError:
    _ffmpeg_exe = "ffmpeg"   # assume it is on system PATH

# ---------------------------------------------------------------------------
# Logging — surface progress on console
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    encoding="utf-8",
)
log = logging.getLogger("test_minimalist")

# ---------------------------------------------------------------------------
# Script content
# ---------------------------------------------------------------------------
SCRIPT_DICT: dict[str, str] = {
    "hook":      "The guy who served you coffee just retired at 35",
    "statement": "He was making twenty five thousand a year while you were making sixty thousand",
    "twist":     "He spent money to become free while you upgraded apartments and car payments",
    "landing":   "He did not out-earn you he out-thought you",
    "cta":       "Subscribe we say the money things your bank hopes you never hear",
}

OUTPUT_PATH = Path("data/output/test_minimalist.mp4")


def _make_silent_audio(duration: float, path: Path) -> None:
    """Generate a silent AAC audio file of the given duration using ffmpeg."""
    log.info("Generating %.1fs silent audio track -> %s", duration, path)
    cmd = [
        _ffmpeg_exe, "-y",
        "-f", "lavfi",
        "-i", "anullsrc=r=44100:cl=stereo",
        "-t", str(duration),
        "-c:a", "aac",
        "-b:a", "128k",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("ffmpeg silent audio failed:\n%s", result.stderr[-1000:])
        raise RuntimeError("Could not generate silent audio — is ffmpeg in PATH?")
    log.info("Silent audio ready: %s (%.0f bytes)", path, path.stat().st_size)


def main() -> None:
    log.info("=" * 60)
    log.info("ChannelForge — minimalist_kinetic render test")
    log.info("=" * 60)

    # Ensure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Import renderer after confirming project structure is accessible
    log.info("Importing KineticRenderer and MINIMALIST_STYLE …")
    try:
        from src.media.kinetic_renderer import KineticRenderer, MINIMALIST_STYLE
    except ImportError as exc:
        log.error("Import failed: %s", exc)
        log.error("Run this script from the repo root: python test_minimalist_render.py")
        sys.exit(1)

    log.info("Style preset: %s (mode=%s)", MINIMALIST_STYLE.name, MINIMALIST_STYLE.mode)
    log.info("Script parts: %s", list(SCRIPT_DICT.keys()))

    # Build a temporary silent audio file sized to target duration
    TARGET_DURATION = 15.0  # seconds — short enough for a fast test render

    with tempfile.TemporaryDirectory() as tmp:
        audio_path = Path(tmp) / "silent.aac"
        _make_silent_audio(TARGET_DURATION, audio_path)

        log.info("Instantiating KineticRenderer …")
        renderer = KineticRenderer(style_preset=MINIMALIST_STYLE)

        log.info("Starting render -> %s", OUTPUT_PATH)
        log.info("(No word timestamps supplied — evenly-spaced layout will be used)")

        result = renderer.build(
            topic_id="test_minimalist",
            script_dict=SCRIPT_DICT,
            audio_path=audio_path,
            word_timestamps=None,
            cta_overlay="",
        )

    # Report result
    log.info("-" * 60)
    if result.is_valid:
        rendered = Path(result.output_path)
        # Renderer always appends _kinetic; rename to the requested filename
        if rendered.exists() and rendered != OUTPUT_PATH:
            rendered.rename(OUTPUT_PATH)
            log.info("Renamed %s -> %s", rendered.name, OUTPUT_PATH.name)
        out = OUTPUT_PATH
        size_mb = out.stat().st_size / 1_048_576
        log.info("SUCCESS  -- %s  (%.2f MB)", out, size_mb)
        print(f"\nRender complete: {out}  [{size_mb:.2f} MB]")
    else:
        log.error("FAILED -- errors: %s", result.validation_errors)
        sys.exit(1)


if __name__ == "__main__":
    main()

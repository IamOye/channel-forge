"""
test_hyperframes_pipeline.py

Integration smoke-test for HyperFramesRenderer.
Creates a 30s silent voiceover, runs the renderer with the same test script
used in morph-test, and confirms data/output/test_pipeline.mp4 is produced.

Run:
    .venv/Scripts/python.exe test_hyperframes_pipeline.py
"""
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Test script — same as morph-test/index.html
# ---------------------------------------------------------------------------
TEST_SCRIPT = {
    "hook":      "The guy who serves you coffee just retired at 35",
    "statement": "He was making twenty five thousand a year while you were making sixty thousand",
    "twist":     "He spent money to become free while you upgraded apartments and car payments",
    "landing":   "He did not out-earn you he out-thought you",
    "cta":       "Subscribe we say the money things your bank hopes you never hear",
}

OUTPUT_PATH = Path("data/output/test_pipeline.mp4")


def make_silent_audio(duration: float = 30.0) -> Path:
    """Generate a silent MP3 of the given duration using imageio_ffmpeg."""
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg_exe = shutil.which("ffmpeg") or "ffmpeg"

    out = Path(tempfile.gettempdir()) / "test_pipeline_silence.mp3"
    cmd = [
        ffmpeg_exe, "-y",
        "-f", "lavfi",
        "-i", f"anullsrc=r=44100:cl=mono",
        "-t", str(duration),
        "-q:a", "9",
        "-acodec", "libmp3lame",
        str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg silence gen failed: {result.stderr[-300:]}")
    log.info("Silent audio: %s (%.0fs)", out, duration)
    return out


def main() -> int:
    log.info("=== HyperFramesRenderer pipeline test ===")

    # Ensure output dir exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Remove stale output
    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    # Create silent 30s audio
    try:
        audio_path = make_silent_audio(30.0)
    except Exception as exc:
        log.error("Could not create silent audio: %s", exc)
        return 1

    # Run renderer
    from src.media.hyperframes_renderer import HyperFramesRenderer

    renderer = HyperFramesRenderer()
    log.info("Calling HyperFramesRenderer.build() ...")
    result = renderer.build(
        topic_id="test_pipeline",
        script_dict=TEST_SCRIPT,
        audio_path=audio_path,
    )

    if not result.is_valid:
        log.error("FAILED: %s", result.validation_errors)
        return 1

    out = Path(result.output_path)
    if not out.exists():
        log.error("FAILED: output file not found at %s", out)
        return 1

    size_kb = out.stat().st_size / 1024
    log.info("SUCCESS: %s (%.1f KB, %.1fs)", out.name, size_kb, result.duration_seconds)

    # Copy to canonical test path if different
    if out.resolve() != OUTPUT_PATH.resolve():
        shutil.copy2(out, OUTPUT_PATH)
        log.info("Copied to %s", OUTPUT_PATH)

    return 0


if __name__ == "__main__":
    sys.exit(main())

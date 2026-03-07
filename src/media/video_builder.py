"""
video_builder.py — VideoBuilder

Assembles the final YouTube Shorts MP4 from stock video, voiceover audio,
and timed captions using moviepy.

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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CANVAS_WIDTH   = 1080
CANVAS_HEIGHT  = 1920
OVERLAY_OPACITY = 0.45
FPS            = 30
OUTPUT_DIR     = Path("data/output")

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
        stock_video_path: str | Path,
    ) -> BuildResult:
        """
        Build and export the final video MP4.

        Args:
            topic_id:          Unique identifier used in output filename.
            script_dict:       Dict with keys hook, statement, twist, question.
            audio_path:        Path to the voiceover MP3.
            stock_video_path:  Path to the stock footage MP4.

        Returns:
            BuildResult with output path and validation status.
        """
        audio_path       = Path(audio_path)
        stock_video_path = Path(stock_video_path)

        errors = self._validate_inputs(audio_path, stock_video_path)
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

        logger.info("Building video for topic_id=%s", topic_id)

        final_clip = self._assemble(script_dict, audio_path, stock_video_path)

        final_clip.write_videofile(
            str(output_path),
            fps=self.fps,
            codec="libx264",
            audio_codec="aac",
            logger=None,   # suppress moviepy progress bar
        )
        final_clip.close()

        logger.info("Exported final video to %s", output_path)

        return BuildResult(
            topic_id=topic_id,
            output_path=str(output_path),
            duration_seconds=VIDEO_DURATION,
            is_valid=True,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _assemble(
        self,
        script_dict: dict[str, str],
        audio_path: Path,
        stock_video_path: Path,
    ):
        """Compose video layers and return a CompositeVideoClip."""
        from moviepy import (  # moviepy v2
            AudioFileClip,
            ColorClip,
            CompositeVideoClip,
            VideoFileClip,
        )
        from src.media.caption_renderer import CaptionRenderer

        # 1. Load and trim stock video to VIDEO_DURATION
        stock = (
            VideoFileClip(str(stock_video_path))
            .subclipped(0, VIDEO_DURATION)
            .resized((self.canvas_width, self.canvas_height))
        )

        # 2. Dark overlay
        overlay = (
            ColorClip(
                size=(self.canvas_width, self.canvas_height),
                color=(0, 0, 0),
            )
            .with_opacity(self.overlay_opacity)
            .with_duration(VIDEO_DURATION)
        )

        # 3. Captions
        renderer = CaptionRenderer(
            canvas_width=self.canvas_width,
            canvas_height=self.canvas_height,
        )
        caption_clips = renderer.render(script_dict)

        # 4. Audio
        audio = AudioFileClip(str(audio_path)).subclipped(0, VIDEO_DURATION)

        # 5. Compose everything
        layers = [stock, overlay] + caption_clips
        final = (
            CompositeVideoClip(layers, size=(self.canvas_width, self.canvas_height))
            .with_audio(audio)
            .with_duration(VIDEO_DURATION)
        )
        return final

    @staticmethod
    def _validate_inputs(audio_path: Path, stock_video_path: Path) -> list[str]:
        errors: list[str] = []
        if not audio_path.exists():
            errors.append(f"audio file not found: {audio_path}")
        if not stock_video_path.exists():
            errors.append(f"stock video not found: {stock_video_path}")
        return errors

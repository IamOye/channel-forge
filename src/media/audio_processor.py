"""
audio_processor.py — AudioProcessor

Validates and inspects MP3 audio files.

Usage:
    proc = AudioProcessor()
    info = proc.inspect("data/raw/topic_001_voice.mp3")
    print(info.duration_seconds, info.is_valid)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_FILE_SIZE_BYTES = 10_240   # 10 KB — below this is likely a corrupt/empty file
MIN_DURATION_SECONDS = 10.0
MAX_DURATION_SECONDS = 16.0


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class AudioInfo:
    """
    Inspection result for an MP3 file.

    Attributes:
        file_path:         Absolute path to the MP3 file.
        duration_seconds:  Audio duration in seconds (0 on read failure).
        file_size_bytes:   File size in bytes (0 if file missing).
        is_valid:          True when all validation checks pass.
        validation_errors: List of human-readable error strings.
    """

    file_path: str
    duration_seconds: float
    file_size_bytes: int
    is_valid: bool
    validation_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path":         self.file_path,
            "duration_seconds":  self.duration_seconds,
            "file_size_bytes":   self.file_size_bytes,
            "is_valid":          self.is_valid,
            "validation_errors": self.validation_errors,
        }


# ---------------------------------------------------------------------------
# AudioProcessor
# ---------------------------------------------------------------------------

class AudioProcessor:
    """
    Reads and validates MP3 audio files using mutagen.

    All methods are designed to fail gracefully — they return an AudioInfo
    with is_valid=False rather than raising exceptions.
    """

    def inspect(self, file_path: str | Path) -> AudioInfo:
        """
        Inspect an MP3 file and return its metadata plus validation status.

        Args:
            file_path: Path to the MP3 file.

        Returns:
            AudioInfo dataclass with duration, size, and validation result.
        """
        path = Path(file_path)
        errors: list[str] = []

        # --- File existence ---
        if not path.exists():
            return AudioInfo(
                file_path=str(path),
                duration_seconds=0.0,
                file_size_bytes=0,
                is_valid=False,
                validation_errors=[f"file not found: {path}"],
            )

        # --- File size ---
        file_size = path.stat().st_size
        if file_size < MIN_FILE_SIZE_BYTES:
            errors.append(
                f"file size {file_size} bytes is below minimum {MIN_FILE_SIZE_BYTES} bytes"
            )

        # --- Duration via mutagen ---
        duration = self._read_duration(path)
        duration_errors = self._validate_duration(duration)
        errors.extend(duration_errors)

        info = AudioInfo(
            file_path=str(path),
            duration_seconds=duration,
            file_size_bytes=file_size,
            is_valid=len(errors) == 0,
            validation_errors=errors,
        )
        if errors:
            logger.warning("Audio validation failed for %s: %s", path.name, errors)
        return info

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_duration(path: Path) -> float:
        """Return MP3 duration in seconds, or 0.0 on error."""
        try:
            from mutagen.mp3 import MP3
            audio = MP3(str(path))
            return float(audio.info.length)
        except Exception as exc:
            logger.warning("Failed to read duration from %s: %s", path.name, exc)
            return 0.0

    @staticmethod
    def _validate_duration(duration: float) -> list[str]:
        errors: list[str] = []
        if duration <= 0:
            errors.append("could not read audio duration")
        elif duration < MIN_DURATION_SECONDS:
            errors.append(
                f"duration {duration:.1f}s is below minimum {MIN_DURATION_SECONDS}s"
            )
        elif duration > MAX_DURATION_SECONDS:
            errors.append(
                f"duration {duration:.1f}s exceeds maximum {MAX_DURATION_SECONDS}s"
            )
        return errors

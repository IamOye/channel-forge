"""
voiceover.py — VoiceoverGenerator

Generates MP3 voiceovers from script text using the ElevenLabs API.

Usage:
    gen = VoiceoverGenerator()
    result = gen.generate(script_dict, topic_id="stoic_001", category="success")
    print(result.audio_path)
    print(result.duration_seconds)
"""

import logging
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
_ELEVENLABS_API_URL_WITH_TIMESTAMPS = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/with-timestamps"
_MODEL_ID = "eleven_turbo_v2"

# ElevenLabs voice IDs for named voices
VOICE_MAP: dict[str, tuple[str, str]] = {
    "money":   ("Adam",   "pNInz6obpgDQGcFmaJgB"),
    "career":  ("Josh",   "TxGEqnHWrfWFTfGW9XjX"),
    "success": ("Josh",   "TxGEqnHWrfWFTfGW9XjX"),
}
DEFAULT_VOICE: tuple[str, str] = ("Adam", "pNInz6obpgDQGcFmaJgB")

VOICE_SETTINGS = {
    "stability":        0.35,   # lower = more expressive delivery
    "similarity_boost": 0.85,
    "style":            0.40,   # higher = more natural style variation
    "use_speaker_boost": True,
}

MIN_DURATION_SECONDS = 10.0

# -14 LUFS is the YouTube recommended integrated loudness level
TARGET_LUFS = -14.0

OUTPUT_DIR = Path("data/raw")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class VoiceoverResult:
    """Result of a voiceover generation request."""

    topic_id: str
    audio_path: str
    voice_name: str
    voice_id: str
    duration_seconds: float
    is_valid: bool
    validation_errors: list[str] = field(default_factory=list)
    generated_at: str = ""
    words_path: str = ""

    def __post_init__(self) -> None:
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic_id":          self.topic_id,
            "audio_path":        self.audio_path,
            "voice_name":        self.voice_name,
            "voice_id":          self.voice_id,
            "duration_seconds":  self.duration_seconds,
            "is_valid":          self.is_valid,
            "validation_errors": self.validation_errors,
            "generated_at":      self.generated_at,
            "words_path":        self.words_path,
        }


# ---------------------------------------------------------------------------
# VoiceoverGenerator
# ---------------------------------------------------------------------------

class VoiceoverGenerator:
    """
    Generates voiceover MP3 files via ElevenLabs TTS API.

    Args:
        api_key: ElevenLabs API key. If None, reads ELEVENLABS_API_KEY from env.
        output_dir: Directory to save MP3 files. Defaults to data/raw/.
    """

    def __init__(
        self,
        api_key: str | None = None,
        output_dir: str | Path = OUTPUT_DIR,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("ELEVENLABS_API_KEY", "")
        self.output_dir = Path(output_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        script_dict: dict[str, str],
        topic_id: str,
        category: str = "default",
    ) -> VoiceoverResult:
        """
        Generate a voiceover MP3 for the given script.

        Args:
            script_dict: Dict with keys hook, statement, twist, question
                         (or any 'full_script' key). Text is joined in order.
            topic_id: Unique identifier for the topic (used in filename).
            category: Topic category for voice selection (money/career/success).

        Returns:
            VoiceoverResult with path, duration, and validation status.

        Raises:
            ValueError: If ELEVENLABS_API_KEY is not configured.
        """
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY not set")

        voice_name, voice_id = self._select_voice(category)
        text = self._build_text(script_dict)
        output_path = self.output_dir / f"{topic_id}_voice.mp3"

        logger.info(
            "Generating voiceover: topic_id=%s, voice=%s, chars=%d",
            topic_id, voice_name, len(text),
        )

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Call ElevenLabs API
        audio_bytes, word_timestamps = self._call_api(voice_id, text)
        output_path.write_bytes(audio_bytes)
        logger.info("Saved voiceover to %s (%d bytes)", output_path, len(audio_bytes))

        # Save word timestamps JSON alongside the audio
        import json as _json
        words_path = self.output_dir / f"{topic_id}_words.json"
        words_path.write_text(_json.dumps(word_timestamps, indent=2), encoding="utf-8")
        logger.debug("Saved %d word timestamps to %s", len(word_timestamps), words_path)

        # Normalize audio loudness with ffmpeg
        self._normalize_audio(output_path)

        # Validate duration
        duration = self._get_duration(output_path)
        errors = self._validate_duration(duration)

        result = VoiceoverResult(
            topic_id=topic_id,
            audio_path=str(output_path),
            voice_name=voice_name,
            voice_id=voice_id,
            duration_seconds=duration,
            is_valid=len(errors) == 0,
            validation_errors=errors,
            words_path=str(words_path),
        )
        if errors:
            logger.warning("Voiceover validation errors: %s", errors)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _select_voice(category: str) -> tuple[str, str]:
        """Return (voice_name, voice_id) for the given category."""
        return VOICE_MAP.get(category.lower(), DEFAULT_VOICE)

    @staticmethod
    def _build_text(script_dict: dict[str, str]) -> str:
        """Concatenate script parts into a single spoken text string."""
        if "full_script" in script_dict:
            return script_dict["full_script"].strip()
        parts = [
            script_dict.get("hook", ""),
            script_dict.get("statement", ""),
            script_dict.get("twist", ""),
            script_dict.get("question", ""),
        ]
        return " ".join(p.strip() for p in parts if p.strip())

    def _call_api(self, voice_id: str, text: str) -> tuple[bytes, list[dict]]:
        """POST to ElevenLabs TTS with-timestamps endpoint; return (audio_bytes, word_timestamps)."""
        import base64
        url = _ELEVENLABS_API_URL_WITH_TIMESTAMPS.format(voice_id=voice_id)
        payload = {
            "text":           text,
            "model_id":       _MODEL_ID,
            "voice_settings": VOICE_SETTINGS,
        }
        headers = {
            "xi-api-key":   self.api_key,
            "Content-Type": "application/json",
            "Accept":       "application/json",
        }
        response = httpx.post(url, json=payload, headers=headers, timeout=30.0)
        response.raise_for_status()
        data = response.json()
        audio_bytes = base64.b64decode(data["audio_base64"])
        alignment = data.get("alignment", {})
        word_timestamps = self._extract_word_timestamps(alignment)
        return audio_bytes, word_timestamps

    @staticmethod
    def _extract_word_timestamps(alignment: dict) -> list[dict]:
        """Extract word-level timestamps from ElevenLabs alignment data.

        Args:
            alignment: Dict with 'characters', 'character_start_times_seconds',
                       'character_end_times_seconds' lists.

        Returns:
            List of dicts: [{text, start_time, end_time}, ...]
        """
        chars = alignment.get("characters", [])
        starts = alignment.get("character_start_times_seconds", [])
        ends = alignment.get("character_end_times_seconds", [])

        if not chars:
            return []

        words: list[dict] = []
        buf_chars: list[str] = []
        buf_starts: list[float] = []
        buf_ends: list[float] = []

        def _flush():
            if buf_chars:
                words.append({
                    "text":       "".join(buf_chars),
                    "start_time": buf_starts[0],
                    "end_time":   buf_ends[-1],
                })
                buf_chars.clear()
                buf_starts.clear()
                buf_ends.clear()

        for i, ch in enumerate(chars):
            s = starts[i] if i < len(starts) else 0.0
            e = ends[i] if i < len(ends) else 0.0
            if ch in (" ", "\n", "\t"):
                _flush()
            else:
                buf_chars.append(ch)
                buf_starts.append(s)
                buf_ends.append(e)

        _flush()
        return words

    def _normalize_audio(self, audio_path: Path) -> None:
        """
        Normalize audio to TARGET_LUFS using ffmpeg loudnorm filter.
        Overwrites the file in place via a temp file.
        Skipped silently if ffmpeg is not installed.
        """
        import shutil
        if not shutil.which("ffmpeg"):
            logger.debug("ffmpeg not found — skipping loudness normalization")
            return

        temp_path = audio_path.with_suffix(".norm.mp3")
        cmd = [
            "ffmpeg", "-y",
            "-i", str(audio_path),
            "-af", f"loudnorm=I={TARGET_LUFS}:TP=-1.5:LRA=11",
            str(temp_path),
        ]
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                timeout=60,
            )
            temp_path.replace(audio_path)
            logger.info("Normalized audio to %0.1f LUFS: %s", TARGET_LUFS, audio_path)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
            logger.warning("ffmpeg normalization failed (continuing): %s", exc)
            if temp_path.exists():
                temp_path.unlink()

    def _get_duration(self, audio_path: Path) -> float:
        """Get MP3 duration in seconds using mutagen."""
        try:
            from mutagen.mp3 import MP3
            audio = MP3(str(audio_path))
            return float(audio.info.length)
        except Exception as exc:
            logger.warning("Could not read audio duration: %s", exc)
            return 0.0

    @staticmethod
    def _validate_duration(duration: float) -> list[str]:
        """Validate duration — only enforces a minimum; no upper bound.

        The video length will extend to match the full voiceover duration.
        """
        if duration < MIN_DURATION_SECONDS:
            return [f"duration {duration:.1f}s is below minimum {MIN_DURATION_SECONDS}s"]
        return []

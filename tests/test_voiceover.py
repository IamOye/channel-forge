"""
Tests for src/media/voiceover.py

All external calls (httpx, mutagen, subprocess, filesystem) are mocked.
"""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

from src.media.voiceover import (
    DEFAULT_VOICE,
    MIN_DURATION_SECONDS,
    VOICE_MAP,
    VOICE_SETTINGS,
    VoiceoverGenerator,
    VoiceoverResult,
)

import base64 as _base64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_SCRIPT = {
    "hook":      "Most people ignore this secret.",
    "statement": "Stoics mastered their reactions.",
    "twist":     "Modern life rewired your brain.",
    "question":  "What would you change today?",
}


def _make_gen(api_key: str = "fake") -> VoiceoverGenerator:
    return VoiceoverGenerator(api_key=api_key)


# ---------------------------------------------------------------------------
# VoiceoverResult
# ---------------------------------------------------------------------------

class TestVoiceoverResult:
    def _make(self, **kw) -> VoiceoverResult:
        defaults = dict(
            topic_id="test_001",
            audio_path="data/raw/test_001_voice.mp3",
            voice_name="Adam",
            voice_id="pNInz6obpgDQGcFmaJgB",
            duration_seconds=13.0,
            is_valid=True,
            validation_errors=[],
        )
        defaults.update(kw)
        return VoiceoverResult(**defaults)

    def test_generated_at_auto_set(self) -> None:
        r = self._make()
        assert r.generated_at != ""

    def test_to_dict_has_all_keys(self) -> None:
        r = self._make()
        d = r.to_dict()
        for key in ("topic_id", "audio_path", "voice_name", "voice_id",
                    "duration_seconds", "is_valid", "validation_errors", "generated_at"):
            assert key in d


# ---------------------------------------------------------------------------
# VoiceoverGenerator._select_voice
# ---------------------------------------------------------------------------

class TestSelectVoice:
    def test_money_returns_adam(self) -> None:
        name, vid = VoiceoverGenerator._select_voice("money")
        assert name == "Adam"
        assert vid == VOICE_MAP["money"][1]

    def test_career_returns_josh(self) -> None:
        name, _ = VoiceoverGenerator._select_voice("career")
        assert name == "Josh"

    def test_success_returns_josh(self) -> None:
        name, _ = VoiceoverGenerator._select_voice("success")
        assert name == "Josh"

    def test_unknown_returns_default(self) -> None:
        name, vid = VoiceoverGenerator._select_voice("unknown_category")
        assert (name, vid) == DEFAULT_VOICE

    def test_case_insensitive(self) -> None:
        name, _ = VoiceoverGenerator._select_voice("MONEY")
        assert name == "Adam"


# ---------------------------------------------------------------------------
# VoiceoverGenerator._build_text
# ---------------------------------------------------------------------------

class TestBuildText:
    def test_joins_parts_in_order(self) -> None:
        text = VoiceoverGenerator._build_text(SAMPLE_SCRIPT)
        assert "Most people ignore this secret." in text
        assert "Stoics mastered their reactions." in text
        assert "What would you change today?" in text

    def test_uses_full_script_key_if_present(self) -> None:
        d = {"full_script": "This is the full text.", "hook": "ignored"}
        text = VoiceoverGenerator._build_text(d)
        assert text == "This is the full text."

    def test_skips_empty_parts(self) -> None:
        d = {"hook": "Hook text.", "statement": "", "twist": "Twist text.", "question": "Why?"}
        text = VoiceoverGenerator._build_text(d)
        assert "Hook text." in text
        assert "Twist text." in text
        # Double space would indicate empty part wasn't skipped
        assert "  " not in text


# ---------------------------------------------------------------------------
# VoiceoverGenerator._validate_duration
# ---------------------------------------------------------------------------

class TestValidateDuration:
    def test_valid_duration_no_errors(self) -> None:
        errors = VoiceoverGenerator._validate_duration(13.0)
        assert errors == []

    def test_exactly_at_min_passes(self) -> None:
        errors = VoiceoverGenerator._validate_duration(MIN_DURATION_SECONDS)
        assert errors == []

    def test_below_min_error(self) -> None:
        errors = VoiceoverGenerator._validate_duration(5.0)
        assert any("minimum" in e for e in errors)

    def test_long_duration_passes(self) -> None:
        # No upper bound — video extends to match full voiceover length
        errors = VoiceoverGenerator._validate_duration(30.0)
        assert errors == []


# ---------------------------------------------------------------------------
# VoiceoverGenerator.generate (fully mocked)
# ---------------------------------------------------------------------------

class TestVoiceoverGeneratorGenerate:
    def _mock_httpx_response(self, content: bytes = b"fake_mp3_audio") -> MagicMock:
        import base64
        resp = MagicMock()
        resp.json.return_value = {
            "audio_base64": base64.b64encode(content).decode(),
            "alignment": {
                "characters": list("hello world"),
                "character_start_times_seconds": [i * 0.05 for i in range(11)],
                "character_end_times_seconds": [(i + 1) * 0.05 for i in range(11)],
            },
        }
        resp.raise_for_status = MagicMock()
        return resp

    def _mock_mp3(self, duration: float = 13.0) -> MagicMock:
        info = MagicMock()
        info.length = duration
        mp3 = MagicMock()
        mp3.info = info
        return mp3

    @patch("src.media.voiceover.subprocess.run")
    @patch("src.media.voiceover.httpx.post")
    def test_returns_valid_result(self, mock_post, mock_subprocess) -> None:
        mock_post.return_value = self._mock_httpx_response()
        mock_subprocess.return_value = MagicMock(returncode=0)

        with patch("src.media.voiceover.VoiceoverGenerator._get_duration", return_value=13.0):
            with patch("pathlib.Path.mkdir"), patch("pathlib.Path.write_bytes"), patch("pathlib.Path.write_text"):
                gen = _make_gen()
                result = gen.generate(SAMPLE_SCRIPT, topic_id="test_001", category="success")

        assert isinstance(result, VoiceoverResult)
        assert result.is_valid is True
        assert result.voice_name == "Josh"
        assert result.topic_id == "test_001"
        assert result.duration_seconds == 13.0

    @patch("src.media.voiceover.subprocess.run")
    @patch("src.media.voiceover.httpx.post")
    def test_invalid_when_duration_too_short(self, mock_post, mock_subprocess) -> None:
        mock_post.return_value = self._mock_httpx_response()
        mock_subprocess.return_value = MagicMock(returncode=0)

        with patch("src.media.voiceover.VoiceoverGenerator._get_duration", return_value=5.0):
            with patch("pathlib.Path.mkdir"), patch("pathlib.Path.write_bytes"), patch("pathlib.Path.write_text"):
                gen = _make_gen()
                result = gen.generate(SAMPLE_SCRIPT, topic_id="short_001", category="money")

        assert result.is_valid is False
        assert any("minimum" in e for e in result.validation_errors)

    @patch("src.media.voiceover.subprocess.run")
    @patch("src.media.voiceover.httpx.post")
    def test_valid_when_duration_long(self, mock_post, mock_subprocess) -> None:
        # No upper bound — long voiceovers are valid; video extends to match
        mock_post.return_value = self._mock_httpx_response()
        mock_subprocess.return_value = MagicMock(returncode=0)

        with patch("src.media.voiceover.VoiceoverGenerator._get_duration", return_value=25.0):
            with patch("pathlib.Path.mkdir"), patch("pathlib.Path.write_bytes"), patch("pathlib.Path.write_text"):
                gen = _make_gen()
                result = gen.generate(SAMPLE_SCRIPT, topic_id="long_001", category="career")

        assert result.is_valid is True
        assert result.duration_seconds == 25.0

    def test_raises_without_api_key(self) -> None:
        gen = VoiceoverGenerator(api_key="")
        with pytest.raises(ValueError, match="ELEVENLABS_API_KEY not set"):
            gen.generate(SAMPLE_SCRIPT, topic_id="test", category="default")

    @patch("src.media.voiceover.subprocess.run")
    @patch("src.media.voiceover.httpx.post")
    def test_uses_default_voice_for_unknown_category(self, mock_post, mock_subprocess) -> None:
        mock_post.return_value = self._mock_httpx_response()
        mock_subprocess.return_value = MagicMock(returncode=0)

        with patch("src.media.voiceover.VoiceoverGenerator._get_duration", return_value=13.0):
            with patch("pathlib.Path.mkdir"), patch("pathlib.Path.write_bytes"), patch("pathlib.Path.write_text"):
                gen = _make_gen()
                result = gen.generate(SAMPLE_SCRIPT, topic_id="def_001", category="unknown")

        assert result.voice_name == DEFAULT_VOICE[0]

    @patch("src.media.voiceover.subprocess.run")
    @patch("src.media.voiceover.httpx.post")
    def test_output_path_contains_topic_id(self, mock_post, mock_subprocess) -> None:
        mock_post.return_value = self._mock_httpx_response()
        mock_subprocess.return_value = MagicMock(returncode=0)

        with patch("src.media.voiceover.VoiceoverGenerator._get_duration", return_value=13.0):
            with patch("pathlib.Path.mkdir"), patch("pathlib.Path.write_bytes"), patch("pathlib.Path.write_text"):
                gen = _make_gen()
                result = gen.generate(SAMPLE_SCRIPT, topic_id="mytopic_042")

        assert "mytopic_042" in result.audio_path
        assert result.audio_path.endswith("_voice.mp3")

    @patch("src.media.voiceover.subprocess.run")
    @patch("src.media.voiceover.httpx.post")
    def test_ffmpeg_failure_does_not_raise(self, mock_post, mock_subprocess) -> None:
        mock_post.return_value = self._mock_httpx_response()
        # Simulate ffmpeg not found
        mock_subprocess.side_effect = FileNotFoundError("ffmpeg not found")

        with patch("src.media.voiceover.VoiceoverGenerator._get_duration", return_value=13.0):
            with patch("pathlib.Path.mkdir"), patch("pathlib.Path.write_bytes"), patch("pathlib.Path.write_text"):
                gen = _make_gen()
                # Should not raise — ffmpeg errors are caught and logged
                result = gen.generate(SAMPLE_SCRIPT, topic_id="noffmpeg_001")

        assert isinstance(result, VoiceoverResult)

    @patch("src.media.voiceover.subprocess.run")
    @patch("src.media.voiceover.httpx.post")
    def test_to_dict_is_serialisable(self, mock_post, mock_subprocess) -> None:
        import json
        mock_post.return_value = self._mock_httpx_response()
        mock_subprocess.return_value = MagicMock(returncode=0)

        with patch("src.media.voiceover.VoiceoverGenerator._get_duration", return_value=12.0):
            with patch("pathlib.Path.mkdir"), patch("pathlib.Path.write_bytes"), patch("pathlib.Path.write_text"):
                gen = _make_gen()
                result = gen.generate(SAMPLE_SCRIPT, topic_id="serial_001")

        assert len(json.dumps(result.to_dict())) > 10

    @patch("src.media.voiceover.subprocess.run")
    @patch("src.media.voiceover.httpx.post")
    def test_words_path_set_in_result(self, mock_post, mock_subprocess) -> None:
        mock_post.return_value = self._mock_httpx_response()
        mock_subprocess.return_value = MagicMock(returncode=0)

        with patch("src.media.voiceover.VoiceoverGenerator._get_duration", return_value=13.0):
            with patch("pathlib.Path.mkdir"), patch("pathlib.Path.write_bytes"), patch("pathlib.Path.write_text"):
                gen = _make_gen()
                result = gen.generate(SAMPLE_SCRIPT, topic_id="words_001", category="money")

        assert result.words_path.endswith("_words.json")
        assert "words_001" in result.words_path


# ---------------------------------------------------------------------------
# VoiceoverGenerator._extract_word_timestamps
# ---------------------------------------------------------------------------

class TestExtractWordTimestamps:
    def test_extracts_words_from_alignment(self) -> None:
        alignment = {
            "characters": list("hi there"),
            "character_start_times_seconds": [0.0, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "character_end_times_seconds":   [0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        }
        words = VoiceoverGenerator._extract_word_timestamps(alignment)
        assert len(words) == 2
        assert words[0]["text"] == "hi"
        assert words[1]["text"] == "there"
        assert words[0]["start_time"] == 0.0
        assert words[1]["start_time"] == 0.4

    def test_returns_empty_for_empty_alignment(self) -> None:
        words = VoiceoverGenerator._extract_word_timestamps({})
        assert words == []

    def test_single_word(self) -> None:
        alignment = {
            "characters": list("word"),
            "character_start_times_seconds": [0.0, 0.1, 0.2, 0.3],
            "character_end_times_seconds":   [0.1, 0.2, 0.3, 0.4],
        }
        words = VoiceoverGenerator._extract_word_timestamps(alignment)
        assert len(words) == 1
        assert words[0]["text"] == "word"
        assert words[0]["end_time"] == 0.4

"""
Tests for src/content/script_generator.py

All Claude API calls are mocked — no real API calls during tests.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.content.script_generator import (
    MAX_WORDS,
    REQUIRED_PARTS,
    ScriptGenerator,
    ScriptPart,
    ScriptResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_PARTS = {
    "hook":      "Most people ignore this ancient secret every single day.",
    "statement": "Stoics believed that controlling your reaction is the only real power you have.",
    "twist":     "But modern life has completely rewired your brain to avoid that discomfort entirely.",
    "question":  "So what would change if you chose discomfort on purpose today?",
}


def _mock_message(payload: dict) -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock(text=json.dumps(payload))]
    return msg


def _make_generator(api_key: str = "fake") -> ScriptGenerator:
    return ScriptGenerator(api_key=api_key)


# ---------------------------------------------------------------------------
# ScriptPart
# ---------------------------------------------------------------------------

class TestScriptPart:
    def test_word_count(self) -> None:
        part = ScriptPart("hook", "this is four words", 2)
        assert part.word_count == 4

    def test_word_count_single(self) -> None:
        part = ScriptPart("question", "Why?", 3)
        assert part.word_count == 1


# ---------------------------------------------------------------------------
# ScriptResult
# ---------------------------------------------------------------------------

class TestScriptResult:
    def _make_result(self, **overrides) -> ScriptResult:
        defaults = dict(
            topic="stoic quotes",
            hook=VALID_PARTS["hook"],
            statement=VALID_PARTS["statement"],
            twist=VALID_PARTS["twist"],
            question=VALID_PARTS["question"],
            full_script=" ".join(VALID_PARTS.values()),
            word_count=50,
            is_valid=True,
            validation_errors=[],
        )
        defaults.update(overrides)
        return ScriptResult(**defaults)

    def test_generated_at_auto_set(self) -> None:
        r = self._make_result()
        assert r.generated_at != ""

    def test_to_dict_has_all_keys(self) -> None:
        r = self._make_result()
        d = r.to_dict()
        for key in ("topic", "hook", "statement", "twist", "question",
                    "full_script", "word_count", "is_valid", "validation_errors"):
            assert key in d

    def test_parts_property_returns_4_parts(self) -> None:
        r = self._make_result()
        parts = r.parts
        assert len(parts) == 4
        assert parts[0].name == "hook"
        assert parts[1].name == "statement"
        assert parts[2].name == "twist"
        assert parts[3].name == "question"

    def test_parts_durations(self) -> None:
        r = self._make_result()
        durations = [p.duration_seconds for p in r.parts]
        assert durations == [2, 4, 4, 3]


# ---------------------------------------------------------------------------
# ScriptGenerator._validate
# ---------------------------------------------------------------------------

class TestValidate:
    def test_valid_script_no_errors(self) -> None:
        errors = ScriptGenerator._validate(
            hook=VALID_PARTS["hook"],
            statement=VALID_PARTS["statement"],
            twist=VALID_PARTS["twist"],
            question=VALID_PARTS["question"],
            word_count=50,
        )
        assert errors == []

    def test_empty_hook_error(self) -> None:
        errors = ScriptGenerator._validate(
            hook="",
            statement=VALID_PARTS["statement"],
            twist=VALID_PARTS["twist"],
            question=VALID_PARTS["question"],
            word_count=40,
        )
        assert any("hook" in e for e in errors)

    def test_empty_question_error(self) -> None:
        errors = ScriptGenerator._validate(
            hook=VALID_PARTS["hook"],
            statement=VALID_PARTS["statement"],
            twist=VALID_PARTS["twist"],
            question="",
            word_count=40,
        )
        assert any("question" in e for e in errors)

    def test_over_word_limit_error(self) -> None:
        errors = ScriptGenerator._validate(
            hook=VALID_PARTS["hook"],
            statement=VALID_PARTS["statement"],
            twist=VALID_PARTS["twist"],
            question=VALID_PARTS["question"],
            word_count=MAX_WORDS,
        )
        assert any("word_count" in e for e in errors)

    def test_exactly_at_limit_fails(self) -> None:
        errors = ScriptGenerator._validate(
            hook=VALID_PARTS["hook"],
            statement=VALID_PARTS["statement"],
            twist=VALID_PARTS["twist"],
            question=VALID_PARTS["question"],
            word_count=MAX_WORDS,
        )
        assert any("word_count" in e for e in errors)

    def test_under_limit_passes(self) -> None:
        errors = ScriptGenerator._validate(
            hook=VALID_PARTS["hook"],
            statement=VALID_PARTS["statement"],
            twist=VALID_PARTS["twist"],
            question=VALID_PARTS["question"],
            word_count=74,
        )
        assert not any("word_count" in e for e in errors)

    def test_missing_question_mark_error(self) -> None:
        errors = ScriptGenerator._validate(
            hook=VALID_PARTS["hook"],
            statement=VALID_PARTS["statement"],
            twist=VALID_PARTS["twist"],
            question="Think about it.",   # no question mark
            word_count=40,
        )
        assert any("question mark" in e for e in errors)

    def test_question_part_without_mark_is_caught(self) -> None:
        errors = ScriptGenerator._validate(
            hook=VALID_PARTS["hook"],
            statement=VALID_PARTS["statement"],
            twist=VALID_PARTS["twist"],
            question="No question mark here",
            word_count=40,
        )
        assert len(errors) >= 1
        assert any("question mark" in e for e in errors)


# ---------------------------------------------------------------------------
# ScriptGenerator._parse_parts
# ---------------------------------------------------------------------------

class TestParseParts:
    def setup_method(self) -> None:
        self.gen = _make_generator()

    def test_parses_valid_json(self) -> None:
        parts = self.gen._parse_parts(json.dumps(VALID_PARTS))
        assert parts["hook"] == VALID_PARTS["hook"]
        assert parts["statement"] == VALID_PARTS["statement"]

    def test_strips_markdown_fences(self) -> None:
        raw = f"```json\n{json.dumps(VALID_PARTS)}\n```"
        parts = self.gen._parse_parts(raw)
        assert parts["hook"] == VALID_PARTS["hook"]

    def test_returns_empty_strings_on_bad_json(self) -> None:
        parts = self.gen._parse_parts("not json")
        for key in REQUIRED_PARTS:
            assert parts[key] == ""

    def test_missing_keys_return_empty_string(self) -> None:
        parts = self.gen._parse_parts(json.dumps({"hook": "a hook"}))
        assert parts["statement"] == ""
        assert parts["twist"] == ""
        assert parts["question"] == ""


# ---------------------------------------------------------------------------
# ScriptGenerator.generate (fully mocked)
# ---------------------------------------------------------------------------

class TestScriptGeneratorGenerate:
    @patch("src.content.script_generator.anthropic.Anthropic")
    def test_returns_valid_result(self, mock_cls) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_message(VALID_PARTS)
        mock_cls.return_value = mock_client

        gen = _make_generator()
        result = gen.generate(topic="stoic quotes", hook=VALID_PARTS["hook"])

        assert isinstance(result, ScriptResult)
        assert result.topic == "stoic quotes"
        assert result.hook == VALID_PARTS["hook"]
        assert result.statement == VALID_PARTS["statement"]
        assert result.twist == VALID_PARTS["twist"]
        assert result.question == VALID_PARTS["question"]
        assert result.is_valid is True
        assert result.word_count > 0

    @patch("src.content.script_generator.anthropic.Anthropic")
    def test_full_script_contains_all_parts(self, mock_cls) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_message(VALID_PARTS)
        mock_cls.return_value = mock_client

        gen = _make_generator()
        result = gen.generate(topic="test", hook=VALID_PARTS["hook"])

        for part_text in VALID_PARTS.values():
            assert part_text in result.full_script

    @patch("src.content.script_generator.anthropic.Anthropic")
    def test_invalid_when_no_question_mark(self, mock_cls) -> None:
        bad_parts = dict(VALID_PARTS)
        bad_parts["question"] = "Think about that statement."
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_message(bad_parts)
        mock_cls.return_value = mock_client

        gen = _make_generator()
        result = gen.generate(topic="test", hook=bad_parts["hook"])

        assert result.is_valid is False
        assert any("question mark" in e for e in result.validation_errors)

    @patch("src.content.script_generator.anthropic.Anthropic")
    def test_invalid_when_word_count_over_limit(self, mock_cls) -> None:
        # 50+50+50+9 = 159 words — clearly over the 140-word limit
        long_parts = {
            "hook":      " ".join(["word"] * 50),
            "statement": " ".join(["word"] * 50),
            "twist":     " ".join(["word"] * 50),
            "question":  "Is this too long and over the limit?",
        }
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_message(long_parts)
        mock_cls.return_value = mock_client

        gen = _make_generator()
        result = gen.generate(topic="test", hook=long_parts["hook"])

        assert result.is_valid is False
        assert any("word_count" in e for e in result.validation_errors)

    @patch("src.content.script_generator.anthropic.Anthropic")
    def test_handles_malformed_api_response(self, mock_cls) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="not json at all")]
        )
        mock_cls.return_value = mock_client

        gen = _make_generator()
        result = gen.generate(topic="test", hook="some hook")

        # Should not raise; all parts empty → is_valid=False
        assert result.is_valid is False
        assert result.word_count == 0

    def test_raises_without_api_key(self) -> None:
        gen = ScriptGenerator(api_key="")
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY not set"):
            gen.generate(topic="test", hook="test hook")

    @patch("src.content.script_generator.anthropic.Anthropic")
    def test_word_count_under_75_passes(self, mock_cls) -> None:
        # Exactly 74 words across 4 parts
        short_parts = {
            "hook":      "This ancient secret changes everything you know now.",     # 9
            "statement": "Stoics said that your mind alone determines your happiness not circumstances.",  # 12
            "twist":     "Yet you spend every day letting outside events dictate your entire emotional state.",  # 15
            "question":  "What would your life look like if you finally took back full control today?",  # 15
        }
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_message(short_parts)
        mock_cls.return_value = mock_client

        gen = _make_generator()
        result = gen.generate(topic="stoicism", hook=short_parts["hook"])

        assert result.word_count < MAX_WORDS
        assert result.is_valid is True

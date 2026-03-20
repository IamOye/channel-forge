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


# ---------------------------------------------------------------------------
# ScriptGenerator word-count retry
# ---------------------------------------------------------------------------


class TestWordCountRetry:
    @patch("src.content.script_generator.anthropic.Anthropic")
    def test_retry_triggered_when_over_120(self, mock_cls) -> None:
        """generate() must call messages.create twice when first response > 120 words."""
        long_parts = {
            "hook":      " ".join(["word"] * 35),
            "statement": " ".join(["word"] * 35),
            "twist":     " ".join(["word"] * 35),
            "question":  "Is this still over the limit?",  # ~7 words → total ~112+7=119... use more
        }
        # Make first response exceed 120 words
        long_parts["question"] = " ".join(["word"] * 20) + " Is this over limit?"

        short_parts = {
            "hook":      "This changes everything you know.",
            "statement": "Most people spend years chasing the wrong thing.",
            "twist":     "The data proves it does not work.",
            "question":  "So what are you actually working for?",
        }

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [
            _mock_message(long_parts),   # first call → too long
            _mock_message(short_parts),  # retry → short enough
        ]
        mock_cls.return_value = mock_client

        gen = _make_generator()
        result = gen.generate(topic="test", hook=long_parts["hook"])

        assert mock_client.messages.create.call_count == 2
        assert result.word_count < MAX_WORDS

    @patch("src.content.script_generator.anthropic.Anthropic")
    def test_no_retry_when_under_limit(self, mock_cls) -> None:
        """generate() must call messages.create once when first response ≤ 120 words."""
        short_parts = {
            "hook":      "This ancient secret changes everything you know now.",
            "statement": "Stoics said your mind determines your happiness not circumstances.",
            "twist":     "Yet you let outside events dictate your emotional state every day.",
            "question":  "What would your life look like if you took back control today?",
        }
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_message(short_parts)
        mock_cls.return_value = mock_client

        gen = _make_generator()
        gen.generate(topic="test", hook=short_parts["hook"])

        assert mock_client.messages.create.call_count == 1

    @patch("src.content.script_generator.anthropic.Anthropic")
    def test_retry_api_failure_uses_original(self, mock_cls) -> None:
        """If the retry API call fails, the original (long) result is returned."""
        long_parts = {
            "hook":      " ".join(["word"] * 35),
            "statement": " ".join(["word"] * 35),
            "twist":     " ".join(["word"] * 35),
            "question":  " ".join(["word"] * 20) + " Is this over limit?",
        }
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [
            _mock_message(long_parts),
            Exception("API timeout"),
        ]
        mock_cls.return_value = mock_client

        gen = _make_generator()
        result = gen.generate(topic="test", hook=long_parts["hook"])

        # Should not raise; still returns a result
        assert result.word_count > 0
        assert result.is_valid is False  # still over limit

    def test_parts_too_long_true_when_over_limit(self) -> None:
        from src.content.script_generator import RETRY_WORD_LIMIT
        gen = _make_generator()
        long_parts = {p: " ".join(["word"] * 35) for p in ("hook", "statement", "twist", "question")}
        assert gen._parts_too_long(long_parts) is True

    def test_parts_too_long_false_when_under_limit(self) -> None:
        gen = _make_generator()
        short_parts = {
            "hook": "Short hook here.",
            "statement": "Brief statement.",
            "twist": "Quick twist.",
            "question": "What now?",
        }
        assert gen._parts_too_long(short_parts) is False

    def test_parts_too_long_false_when_any_part_empty(self) -> None:
        gen = _make_generator()
        parts = {
            "hook": " ".join(["word"] * 40),
            "statement": " ".join(["word"] * 40),
            "twist": " ".join(["word"] * 40),
            "question": "",  # empty → no retry
        }
        assert gen._parts_too_long(parts) is False


# ---------------------------------------------------------------------------
# ScriptGenerator._cta_matches
# ---------------------------------------------------------------------------

class TestCtaMatches:
    def test_subscribe_and_trigger_keyword_match(self) -> None:
        """CTA with 'subscribe' and trigger keyword 'SYSTEM' passes."""
        gen = _make_generator()
        assert gen._cta_matches(
            "Subscribe now. We expose this daily. Comment SYSTEM below for the free guide.",
            "If this hit different, subscribe. Comment SYSTEM below and I will send you the 5-day money reset free.",
        )

    def test_case_insensitive_match(self) -> None:
        gen = _make_generator()
        assert gen._cta_matches(
            "subscribe if nobody told you this. comment automate below.",
            "Subscribe if nobody told you this. Comment AUTOMATE below.",
        )

    def test_subscribe_missing_returns_false(self) -> None:
        """Missing 'subscribe' should fail even with trigger keyword."""
        gen = _make_generator()
        assert not gen._cta_matches(
            "Comment SYSTEM below and I will send you the guide.",
            "Subscribe. Comment SYSTEM below.",
        )

    def test_trigger_keyword_missing_returns_false(self) -> None:
        """Missing trigger keyword should fail even with 'subscribe'."""
        gen = _make_generator()
        assert not gen._cta_matches(
            "Subscribe now. Drop a comment below.",
            "Subscribe. Comment SYSTEM below.",
        )

    def test_empty_question_returns_false(self) -> None:
        gen = _make_generator()
        assert not gen._cta_matches("", "Subscribe. Comment BLUEPRINT below.")

    def test_legacy_cta_without_trigger_uses_verbatim(self) -> None:
        """CTAs without SYSTEM/AUTOMATE/BLUEPRINT fall back to verbatim check."""
        gen = _make_generator()
        assert gen._cta_matches(
            "Comment FREE below and I'll send you the guide.",
            "Comment FREE below and I'll send you the guide.",
        )

    def test_blueprint_trigger_detected(self) -> None:
        gen = _make_generator()
        assert gen._cta_matches(
            "Subscribe. We drop truths daily. Comment BLUEPRINT below for the AI guide.",
            "Subscribe. Comment BLUEPRINT below.",
        )


# ---------------------------------------------------------------------------
# ScriptGenerator._enforce_cta
# ---------------------------------------------------------------------------

CTA = "Subscribe if nobody told you this before. We post daily. Comment AUTOMATE below and I will send you the salary playbook free."

PARTS_WITH_CTA = {
    "hook":      "The system is rigged so your nine to five never builds wealth",
    "statement": "Picture this. You work forty hours a week and have nothing left.",
    "twist":     "Your salary is capped. Prices keep climbing. The game is rigged.",
    "question":  "So ask yourself... are you working to live? Subscribe if nobody told you this. Comment AUTOMATE below for the salary playbook free.",
}

PARTS_WITHOUT_CTA = {
    "hook":      PARTS_WITH_CTA["hook"],
    "statement": PARTS_WITH_CTA["statement"],
    "twist":     PARTS_WITH_CTA["twist"],
    "question":  "Drop a comment and I will share some resources with you.",
}


class TestEnforceCta:
    @patch("src.content.script_generator.anthropic.Anthropic")
    def test_cta_matched_verbatim_no_retry(self, mock_cls) -> None:
        """When CTA matches first time, no retry call is made."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_message(PARTS_WITH_CTA)
        mock_cls.return_value = mock_client

        gen = _make_generator()
        result = gen.generate(topic="salary", hook=PARTS_WITH_CTA["hook"], cta_script=CTA)

        # Only one API call — no retry
        assert mock_client.messages.create.call_count == 1
        assert "subscribe" in result.question.lower()
        assert "automate" in result.question.lower()

    @patch("src.content.script_generator.anthropic.Anthropic")
    def test_cta_drift_triggers_regeneration(self, mock_cls) -> None:
        """When first response misses CTA, retry is called and succeeds."""
        mock_client = MagicMock()
        # First call: CTA missing. Second call (retry): CTA present.
        mock_client.messages.create.side_effect = [
            _mock_message(PARTS_WITHOUT_CTA),
            _mock_message(PARTS_WITH_CTA),
        ]
        mock_cls.return_value = mock_client

        gen = _make_generator()
        result = gen.generate(topic="salary", hook=PARTS_WITH_CTA["hook"], cta_script=CTA)

        assert mock_client.messages.create.call_count == 2
        assert "subscribe" in result.question.lower()
        assert "automate" in result.question.lower()

    @patch("src.content.script_generator.anthropic.Anthropic")
    def test_cta_force_replaced_when_retry_also_drifts(self, mock_cls) -> None:
        """When both attempts miss the CTA, question is force-replaced with cta_script."""
        mock_client = MagicMock()
        # Both calls return question without CTA
        mock_client.messages.create.side_effect = [
            _mock_message(PARTS_WITHOUT_CTA),
            _mock_message(PARTS_WITHOUT_CTA),
        ]
        mock_cls.return_value = mock_client

        gen = _make_generator()
        result = gen.generate(topic="salary", hook=PARTS_WITH_CTA["hook"], cta_script=CTA)

        assert mock_client.messages.create.call_count == 2
        # question must now be exactly the CTA text
        assert result.question == CTA

    @patch("src.content.script_generator.anthropic.Anthropic")
    def test_cta_force_replaced_when_retry_api_fails(self, mock_cls) -> None:
        """When first call misses CTA and retry raises, question is force-replaced."""
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [
            _mock_message(PARTS_WITHOUT_CTA),
            Exception("network timeout"),
        ]
        mock_cls.return_value = mock_client

        gen = _make_generator()
        result = gen.generate(topic="salary", hook=PARTS_WITH_CTA["hook"], cta_script=CTA)

        assert result.question == CTA

    @patch("src.content.script_generator.anthropic.Anthropic")
    def test_no_cta_enforcement_when_cta_script_empty(self, mock_cls) -> None:
        """When cta_script='', _enforce_cta is not called — only one API call."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_message(VALID_PARTS)
        mock_cls.return_value = mock_client

        gen = _make_generator()
        result = gen.generate(topic="test", hook=VALID_PARTS["hook"], cta_script="")

        assert mock_client.messages.create.call_count == 1
        assert result.question == VALID_PARTS["question"]

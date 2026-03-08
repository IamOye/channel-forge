"""
Tests for src/content/metadata_generator.py

All Claude API calls are mocked — no real API calls during tests.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.content.metadata_generator import (
    DESCRIPTION_MAX_CHARS,
    DESCRIPTION_SUFFIX,
    REQUIRED_HASHTAG,
    REQUIRED_HASHTAG_COUNT,
    TITLE_MAX_CHARS,
    MetadataGenerator,
    MetadataResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_TITLE = "The Stoic Secret That Changes Everything"   # 41 chars
VALID_DESC  = "Stoics knew the one truth modern life hides. Master your reactions, master your life. Comment below 👇"
VALID_TAGS  = [
    "#Shorts", "#Stoicism", "#StoicWisdom", "#Motivation", "#MindsetShift",
    "#SelfImprovement", "#Mindfulness", "#PersonalGrowth", "#Philosophy",
    "#AncientWisdom", "#Productivity", "#MentalHealth", "#DailyMotivation",
    "#LifeLessons", "#Success",
]

assert len(VALID_TAGS) == 15
assert VALID_DESC.endswith(DESCRIPTION_SUFFIX)
assert len(VALID_TITLE) <= TITLE_MAX_CHARS
assert len(VALID_DESC) <= DESCRIPTION_MAX_CHARS


def _mock_message(payload: dict) -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock(text=json.dumps(payload))]
    return msg


def _valid_payload() -> dict:
    return {
        "title": VALID_TITLE,
        "description": VALID_DESC,
        "hashtags": VALID_TAGS,
        "cta_product": "Wealth Systems Blueprint PDF",
    }


# ---------------------------------------------------------------------------
# MetadataResult
# ---------------------------------------------------------------------------

class TestMetadataResult:
    def _make_result(self, **overrides) -> MetadataResult:
        defaults = dict(
            topic="stoic quotes",
            title=VALID_TITLE,
            description=VALID_DESC,
            hashtags=VALID_TAGS,
            is_valid=True,
            validation_errors=[],
        )
        defaults.update(overrides)
        return MetadataResult(**defaults)

    def test_generated_at_auto_set(self) -> None:
        r = self._make_result()
        assert r.generated_at != ""

    def test_to_dict_has_all_keys(self) -> None:
        r = self._make_result()
        d = r.to_dict()
        for key in ("topic", "title", "description", "hashtags",
                    "hashtags_string", "cta_product", "is_valid", "validation_errors"):
            assert key in d

    def test_to_dict_hashtags_string_space_joined(self) -> None:
        r = self._make_result()
        d = r.to_dict()
        assert d["hashtags_string"] == " ".join(VALID_TAGS)


# ---------------------------------------------------------------------------
# MetadataGenerator._validate
# ---------------------------------------------------------------------------

class TestValidate:
    def test_valid_metadata_no_errors(self) -> None:
        errors = MetadataGenerator._validate(VALID_TITLE, VALID_DESC, VALID_TAGS)
        assert errors == []

    # --- Title ---
    def test_empty_title_error(self) -> None:
        errors = MetadataGenerator._validate("", VALID_DESC, VALID_TAGS)
        assert any("title" in e for e in errors)

    def test_title_over_60_chars_error(self) -> None:
        long_title = "A" * 61
        errors = MetadataGenerator._validate(long_title, VALID_DESC, VALID_TAGS)
        assert any("title" in e and "60" in e for e in errors)

    def test_title_exactly_60_chars_passes(self) -> None:
        title_60 = "A" * 60
        errors = MetadataGenerator._validate(title_60, VALID_DESC, VALID_TAGS)
        assert not any("title" in e for e in errors)

    # --- Description ---
    def test_empty_description_error(self) -> None:
        errors = MetadataGenerator._validate(VALID_TITLE, "", VALID_TAGS)
        assert any("description" in e for e in errors)

    def test_description_over_200_chars_error(self) -> None:
        long_desc = "x" * 190 + " Comment below 👇"  # > 200 chars
        errors = MetadataGenerator._validate(VALID_TITLE, long_desc, VALID_TAGS)
        assert any("description" in e and "200" in e for e in errors)

    def test_description_missing_suffix_error(self) -> None:
        bad_desc = "Great stoic wisdom for your day. Learn and grow."
        errors = MetadataGenerator._validate(VALID_TITLE, bad_desc, VALID_TAGS)
        assert any("Comment below" in e for e in errors)

    def test_description_ends_with_suffix_passes(self) -> None:
        good_desc = "Short intro. Comment below 👇"
        errors = MetadataGenerator._validate(VALID_TITLE, good_desc, VALID_TAGS)
        assert not any("Comment below" in e for e in errors)

    # --- Hashtags ---
    def test_wrong_hashtag_count_error(self) -> None:
        errors = MetadataGenerator._validate(VALID_TITLE, VALID_DESC, VALID_TAGS[:10])
        assert any(str(REQUIRED_HASHTAG_COUNT) in e for e in errors)

    def test_missing_shorts_error(self) -> None:
        tags_no_shorts = [f"#tag{i}" for i in range(15)]
        errors = MetadataGenerator._validate(VALID_TITLE, VALID_DESC, tags_no_shorts)
        assert any("#Shorts" in e for e in errors)

    def test_hashtag_without_hash_error(self) -> None:
        bad_tags = list(VALID_TAGS)
        bad_tags[3] = "NoHash"   # missing #
        errors = MetadataGenerator._validate(VALID_TITLE, VALID_DESC, bad_tags)
        assert any("prefix" in e for e in errors)

    def test_hashtag_with_space_error(self) -> None:
        bad_tags = list(VALID_TAGS)
        bad_tags[4] = "#has space"
        errors = MetadataGenerator._validate(VALID_TITLE, VALID_DESC, bad_tags)
        assert any("space" in e for e in errors)

    def test_exactly_15_tags_with_shorts_passes(self) -> None:
        errors = MetadataGenerator._validate(VALID_TITLE, VALID_DESC, VALID_TAGS)
        assert errors == []


# ---------------------------------------------------------------------------
# MetadataGenerator._parse_response
# ---------------------------------------------------------------------------

class TestParseResponse:
    def setup_method(self) -> None:
        self.gen = MetadataGenerator(api_key="")

    def test_parses_valid_json(self) -> None:
        parsed = self.gen._parse_response(json.dumps(_valid_payload()))
        assert parsed["title"] == VALID_TITLE
        assert parsed["description"] == VALID_DESC
        assert parsed["hashtags"] == VALID_TAGS

    def test_strips_markdown_fences(self) -> None:
        raw = f"```json\n{json.dumps(_valid_payload())}\n```"
        parsed = self.gen._parse_response(raw)
        assert parsed["title"] == VALID_TITLE

    def test_returns_empty_on_bad_json(self) -> None:
        parsed = self.gen._parse_response("!!not json!!")
        assert parsed["title"] == ""
        assert parsed["description"] == ""
        assert parsed["hashtags"] == []
        assert parsed["cta_product"] == ""

    def test_filters_empty_hashtag_strings(self) -> None:
        payload = _valid_payload()
        payload["hashtags"] = ["#Shorts", "", "#tag2"] + [f"#t{i}" for i in range(12)]
        parsed = self.gen._parse_response(json.dumps(payload))
        assert "" not in parsed["hashtags"]


# ---------------------------------------------------------------------------
# MetadataGenerator.generate (fully mocked)
# ---------------------------------------------------------------------------

class TestMetadataGeneratorGenerate:
    @patch("src.content.metadata_generator.anthropic.Anthropic")
    def test_returns_valid_result(self, mock_cls) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_message(_valid_payload())
        mock_cls.return_value = mock_client

        gen = MetadataGenerator(api_key="fake")
        result = gen.generate(
            topic="stoic quotes",
            script="Most people ignore this ancient secret.",
        )

        assert isinstance(result, MetadataResult)
        assert result.title == VALID_TITLE
        assert result.description == VALID_DESC
        assert result.hashtags == VALID_TAGS
        assert result.is_valid is True
        assert result.validation_errors == []

    @patch("src.content.metadata_generator.anthropic.Anthropic")
    def test_invalid_when_title_too_long(self, mock_cls) -> None:
        payload = _valid_payload()
        payload["title"] = "A" * 61
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_message(payload)
        mock_cls.return_value = mock_client

        gen = MetadataGenerator(api_key="fake")
        result = gen.generate(topic="test", script="test script")

        assert result.is_valid is False
        assert any("title" in e for e in result.validation_errors)

    @patch("src.content.metadata_generator.anthropic.Anthropic")
    def test_invalid_when_description_missing_suffix(self, mock_cls) -> None:
        payload = _valid_payload()
        payload["description"] = "Great content but no suffix here."
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_message(payload)
        mock_cls.return_value = mock_client

        gen = MetadataGenerator(api_key="fake")
        result = gen.generate(topic="test", script="test script")

        assert result.is_valid is False
        assert any("Comment below" in e for e in result.validation_errors)

    @patch("src.content.metadata_generator.anthropic.Anthropic")
    def test_invalid_when_wrong_hashtag_count(self, mock_cls) -> None:
        payload = _valid_payload()
        payload["hashtags"] = payload["hashtags"][:10]  # only 10
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_message(payload)
        mock_cls.return_value = mock_client

        gen = MetadataGenerator(api_key="fake")
        result = gen.generate(topic="test", script="test script")

        assert result.is_valid is False

    @patch("src.content.metadata_generator.anthropic.Anthropic")
    def test_invalid_when_shorts_missing(self, mock_cls) -> None:
        payload = _valid_payload()
        payload["hashtags"] = [f"#tag{i}" for i in range(15)]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_message(payload)
        mock_cls.return_value = mock_client

        gen = MetadataGenerator(api_key="fake")
        result = gen.generate(topic="test", script="test script")

        assert result.is_valid is False
        assert any("#Shorts" in e for e in result.validation_errors)

    @patch("src.content.metadata_generator.anthropic.Anthropic")
    def test_handles_malformed_api_response(self, mock_cls) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="totally not json")]
        )
        mock_cls.return_value = mock_client

        gen = MetadataGenerator(api_key="fake")
        result = gen.generate(topic="test", script="test script")

        assert result.is_valid is False
        assert result.title == ""

    def test_raises_without_api_key(self) -> None:
        gen = MetadataGenerator(api_key="")
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY not set"):
            gen.generate(topic="test", script="test script")

    @patch("src.content.metadata_generator.anthropic.Anthropic")
    def test_to_dict_is_serialisable(self, mock_cls) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_message(_valid_payload())
        mock_cls.return_value = mock_client

        gen = MetadataGenerator(api_key="fake")
        result = gen.generate(topic="stoic quotes", script="hook statement twist question?")
        d = result.to_dict()

        import json as _json
        serialised = _json.dumps(d)
        assert len(serialised) > 10

    @patch("src.content.metadata_generator.anthropic.Anthropic")
    def test_result_has_topic(self, mock_cls) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_message(_valid_payload())
        mock_cls.return_value = mock_client

        gen = MetadataGenerator(api_key="fake")
        result = gen.generate(topic="mindfulness tips", script="test")

        assert result.topic == "mindfulness tips"

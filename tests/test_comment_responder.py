"""
Tests for src/publisher/comment_responder.py

All Claude API calls are mocked — no real API calls during tests.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.publisher.comment_responder import (
    MAX_REPLY_CHARS,
    TRIGGER_COMPLIMENT,
    TRIGGER_GENERAL,
    TRIGGER_HOT_LEAD,
    TRIGGER_NEGATIVE,
    TRIGGER_QUESTION,
    TRIGGER_YES,
    VALID_TRIGGERS,
    CommentResponder,
    ReplyResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_message(text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    return msg


def _make_responder(api_key: str = "fake") -> CommentResponder:
    return CommentResponder(api_key=api_key)


SAMPLE_REPLY = (
    "That is the move right there. Most people treat a raise as permission "
    "to spend more. You broke that cycle early. Real generational thinking."
)
assert len(SAMPLE_REPLY) <= MAX_REPLY_CHARS


# ---------------------------------------------------------------------------
# ReplyResult
# ---------------------------------------------------------------------------

class TestReplyResult:
    def _make(self, **kw) -> ReplyResult:
        defaults = dict(
            comment_text="Yes",
            commenter_name="Jordan",
            category="money",
            trigger_type=TRIGGER_YES,
            text=SAMPLE_REPLY,
            is_valid=True,
        )
        defaults.update(kw)
        return ReplyResult(**defaults)

    def test_generated_at_auto_set(self) -> None:
        r = self._make()
        assert r.generated_at != ""

    def test_char_count_set_from_text(self) -> None:
        r = self._make(text="Hello there.")
        assert r.char_count == len("Hello there.")

    def test_to_dict_has_required_keys(self) -> None:
        d = self._make().to_dict()
        for key in ("comment_text", "commenter_name", "category", "trigger_type",
                    "text", "char_count", "was_truncated", "is_valid",
                    "validation_errors", "generated_at"):
            assert key in d

    def test_to_dict_char_count_matches_text(self) -> None:
        r = self._make(text="Short reply.")
        assert r.to_dict()["char_count"] == len("Short reply.")


# ---------------------------------------------------------------------------
# _enforce_length
# ---------------------------------------------------------------------------

class TestEnforceLength:
    def test_short_text_unchanged(self) -> None:
        text = "This is short."
        result, truncated = CommentResponder._enforce_length(text)
        assert result == text
        assert truncated is False

    def test_exactly_at_limit_unchanged(self) -> None:
        text = "x" * MAX_REPLY_CHARS
        result, truncated = CommentResponder._enforce_length(text)
        assert result == text
        assert truncated is False

    def test_truncates_at_sentence_boundary(self) -> None:
        # Build a string where a sentence ends well before the limit,
        # followed by extra content that pushes it over.
        short_sentence = "This is the first sentence."   # 27 chars
        filler = " " + "word " * 100                     # pushes well over 500
        text = short_sentence + filler
        result, truncated = CommentResponder._enforce_length(text)
        assert truncated is True
        assert len(result) <= MAX_REPLY_CHARS
        assert result.endswith(".")

    def test_falls_back_to_hard_truncation_when_no_sentence_boundary(self) -> None:
        # A single word-run with no punctuation
        text = "word " * 110   # ~550 chars, no punctuation
        result, truncated = CommentResponder._enforce_length(text)
        assert truncated is True
        assert len(result) <= MAX_REPLY_CHARS

    def test_truncation_ends_at_last_sentence_in_window(self) -> None:
        # Two sentences within limit, then overflow
        text = "First sentence. Second sentence. " + "overflow " * 60
        result, truncated = CommentResponder._enforce_length(text)
        assert truncated is True
        # Must end at a sentence boundary
        assert result[-1] in ".!?"

    def test_exclamation_mark_is_valid_boundary(self) -> None:
        text = "Wow, that is amazing! " + "overflow " * 60
        result, truncated = CommentResponder._enforce_length(text)
        assert truncated is True
        assert result.endswith("!")

    def test_question_mark_is_valid_boundary(self) -> None:
        text = "Are you sure? " + "overflow " * 60
        result, truncated = CommentResponder._enforce_length(text)
        assert truncated is True
        assert result.endswith("?")


# ---------------------------------------------------------------------------
# _validate
# ---------------------------------------------------------------------------

class TestValidate:
    def test_valid_reply_no_errors(self) -> None:
        errors = CommentResponder._validate(SAMPLE_REPLY, TRIGGER_YES)
        assert errors == []

    def test_empty_text_error(self) -> None:
        errors = CommentResponder._validate("", TRIGGER_YES)
        assert any("empty" in e for e in errors)

    def test_over_limit_error(self) -> None:
        long_text = "word " * 120  # well over 500
        errors = CommentResponder._validate(long_text, TRIGGER_YES)
        assert any(str(MAX_REPLY_CHARS) in e for e in errors)

    def test_em_dash_error(self) -> None:
        text = "Good point \u2014 here is why."
        errors = CommentResponder._validate(text, TRIGGER_YES)
        assert any("dash" in e for e in errors)

    def test_en_dash_error(self) -> None:
        text = "Good point \u2013 here is why."
        errors = CommentResponder._validate(text, TRIGGER_YES)
        assert any("dash" in e for e in errors)

    def test_unknown_trigger_error(self) -> None:
        errors = CommentResponder._validate(SAMPLE_REPLY, "UNKNOWN_TRIGGER")
        assert any("trigger_type" in e for e in errors)

    def test_all_valid_triggers_pass(self) -> None:
        for trigger in VALID_TRIGGERS:
            errors = CommentResponder._validate(SAMPLE_REPLY, trigger)
            assert not any("trigger_type" in e for e in errors), f"Failed for trigger={trigger}"


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_contains_comment_text(self) -> None:
        prompt = CommentResponder._build_prompt("Yes!", "Jordan", "money", TRIGGER_YES, "https://gumroad.com/x")
        assert "Yes!" in prompt

    def test_contains_commenter_name(self) -> None:
        prompt = CommentResponder._build_prompt("Nice", "Alex", "money", TRIGGER_YES, "")
        assert "Alex" in prompt

    def test_contains_trigger_type(self) -> None:
        prompt = CommentResponder._build_prompt("Nice", "Alex", "money", TRIGGER_HOT_LEAD, "")
        assert TRIGGER_HOT_LEAD in prompt

    def test_contains_gumroad_url_when_provided(self) -> None:
        url = "https://gumroad.com/l/test"
        prompt = CommentResponder._build_prompt("Yes", "", "money", TRIGGER_YES, url)
        assert url in prompt

    def test_gumroad_url_omitted_when_empty(self) -> None:
        prompt = CommentResponder._build_prompt("Nice", "", "money", TRIGGER_GENERAL, "")
        assert "gumroad" not in prompt.lower()

    def test_unknown_name_label_when_empty(self) -> None:
        prompt = CommentResponder._build_prompt("Nice", "", "money", TRIGGER_GENERAL, "")
        assert "(unknown)" in prompt


# ---------------------------------------------------------------------------
# _get_gumroad_url
# ---------------------------------------------------------------------------

class TestGetGumroadUrl:
    @patch("config.constants.PRODUCTS", {
        "money": {"gumroad_url": "https://gumroad.com/l/money-test"},
        "career": {"gumroad_url": "https://gumroad.com/l/career-test"},
        "success": {"gumroad_url": "https://gumroad.com/l/success-test"},
    })
    def test_returns_money_url(self) -> None:
        assert CommentResponder._get_gumroad_url("money") == "https://gumroad.com/l/money-test"

    @patch("config.constants.PRODUCTS", {
        "money": {"gumroad_url": "https://gumroad.com/l/money-test"},
        "career": {"gumroad_url": "https://gumroad.com/l/career-test"},
        "success": {"gumroad_url": "https://gumroad.com/l/success-test"},
    })
    def test_returns_career_url(self) -> None:
        assert CommentResponder._get_gumroad_url("career") == "https://gumroad.com/l/career-test"

    @patch("config.constants.PRODUCTS", {
        "money": {"gumroad_url": "https://gumroad.com/l/money-test"},
        "career": {"gumroad_url": "https://gumroad.com/l/career-test"},
        "success": {"gumroad_url": "https://gumroad.com/l/success-test"},
    })
    def test_returns_success_url(self) -> None:
        assert CommentResponder._get_gumroad_url("success") == "https://gumroad.com/l/success-test"

    @patch("config.constants.PRODUCTS", {})
    def test_returns_empty_for_unknown_category(self) -> None:
        assert CommentResponder._get_gumroad_url("unknown") == ""


# ---------------------------------------------------------------------------
# CommentResponder.generate_reply (fully mocked)
# ---------------------------------------------------------------------------

class TestGenerateReply:
    @patch("src.publisher.comment_responder.anthropic.Anthropic")
    def test_returns_reply_result(self, mock_cls) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_message(SAMPLE_REPLY)
        mock_cls.return_value = mock_client

        responder = _make_responder()
        result = responder.generate_reply(
            comment_text="Yes",
            commenter_name="Jordan",
            category="money",
            trigger_type=TRIGGER_YES,
        )

        assert isinstance(result, ReplyResult)
        assert result.text == SAMPLE_REPLY
        assert result.is_valid is True
        assert result.char_count == len(SAMPLE_REPLY)

    @patch("src.publisher.comment_responder.anthropic.Anthropic")
    def test_reply_not_truncated_when_within_limit(self, mock_cls) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_message(SAMPLE_REPLY)
        mock_cls.return_value = mock_client

        result = _make_responder().generate_reply("Yes", trigger_type=TRIGGER_YES)
        assert result.was_truncated is False

    @patch("src.publisher.comment_responder.anthropic.Anthropic")
    def test_reply_truncated_when_over_limit(self, mock_cls) -> None:
        long_reply = "This is a sentence. " * 30   # ~600 chars
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_message(long_reply)
        mock_cls.return_value = mock_client

        result = _make_responder().generate_reply("Nice video", trigger_type=TRIGGER_COMPLIMENT)
        assert result.was_truncated is True
        assert result.char_count <= MAX_REPLY_CHARS

    @patch("src.publisher.comment_responder.anthropic.Anthropic")
    def test_em_dash_in_response_flagged_invalid(self, mock_cls) -> None:
        bad_reply = "Good point \u2014 here is why it works."
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_message(bad_reply)
        mock_cls.return_value = mock_client

        result = _make_responder().generate_reply("Great video", trigger_type=TRIGGER_COMPLIMENT)
        assert result.is_valid is False
        assert any("dash" in e for e in result.validation_errors)

    @patch("src.publisher.comment_responder.anthropic.Anthropic")
    def test_commenter_name_and_category_stored(self, mock_cls) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_message(SAMPLE_REPLY)
        mock_cls.return_value = mock_client

        result = _make_responder().generate_reply(
            "Yes", commenter_name="Alex", category="career", trigger_type=TRIGGER_YES
        )
        assert result.commenter_name == "Alex"
        assert result.category == "career"

    @patch("src.publisher.comment_responder.anthropic.Anthropic")
    def test_trigger_type_stored(self, mock_cls) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_message(SAMPLE_REPLY)
        mock_cls.return_value = mock_client

        result = _make_responder().generate_reply("Build it for me", trigger_type=TRIGGER_HOT_LEAD)
        assert result.trigger_type == TRIGGER_HOT_LEAD

    @patch("src.publisher.comment_responder.anthropic.Anthropic")
    def test_to_dict_is_serialisable(self, mock_cls) -> None:
        import json
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_message(SAMPLE_REPLY)
        mock_cls.return_value = mock_client

        result = _make_responder().generate_reply("Yes", trigger_type=TRIGGER_YES)
        serialised = json.dumps(result.to_dict())
        assert len(serialised) > 10

    def test_raises_without_api_key(self) -> None:
        responder = CommentResponder(api_key="")
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY not set"):
            responder.generate_reply("Yes", trigger_type=TRIGGER_YES)

    @patch("src.publisher.comment_responder.anthropic.Anthropic")
    def test_all_trigger_types_accepted(self, mock_cls) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_message(SAMPLE_REPLY)
        mock_cls.return_value = mock_client

        responder = _make_responder()
        for trigger in VALID_TRIGGERS:
            result = responder.generate_reply("test", trigger_type=trigger)
            assert not any("trigger_type" in e for e in result.validation_errors), \
                f"trigger {trigger} flagged as invalid"

    @patch("src.publisher.comment_responder.anthropic.Anthropic")
    def test_hot_lead_reply_stored_correctly(self, mock_cls) -> None:
        hot_lead_reply = (
            "That makes a lot of sense. "
            "What kind of income stream are you trying to build?"
        )
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_message(hot_lead_reply)
        mock_cls.return_value = mock_client

        result = _make_responder().generate_reply(
            "I want the whole system done for me",
            trigger_type=TRIGGER_HOT_LEAD,
        )
        assert result.text == hot_lead_reply
        assert result.is_valid is True

    @patch("src.publisher.comment_responder.anthropic.Anthropic")
    def test_negative_comment_handled_without_error(self, mock_cls) -> None:
        negative_reply = "Fair point and you are probably further ahead than most. Appreciate you watching."
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_message(negative_reply)
        mock_cls.return_value = mock_client

        result = _make_responder().generate_reply(
            "This is obvious stuff anyone already knows",
            trigger_type=TRIGGER_NEGATIVE,
        )
        assert result.is_valid is True

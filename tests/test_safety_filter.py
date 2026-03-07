"""
Tests for src/filter/safety_filter.py

All Claude API calls are mocked so tests run without API keys.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.filter.safety_filter import (
    ClaudeSafetyClassifier,
    SafetyFilter,
    SafetyResult,
    _blocklist_check,
)


# ---------------------------------------------------------------------------
# _blocklist_check (unit tests for internal function)
# ---------------------------------------------------------------------------

class TestBlocklistCheck:
    def test_blocked_token_exact(self) -> None:
        blocked, reason = _blocklist_check("how to make a bomb")
        assert blocked is True
        assert "bomb" in reason

    def test_blocked_token_substring(self) -> None:
        blocked, reason = _blocklist_check("free porn sites 2024")
        assert blocked is True

    def test_blocked_pattern_how_to_kill(self) -> None:
        blocked, reason = _blocklist_check("how to kill a process in Python")
        # "how to kill" matches the violence pattern — should be blocked
        assert blocked is True

    def test_safe_keyword(self) -> None:
        blocked, reason = _blocklist_check("best budget microphone 2024")
        assert blocked is False
        assert reason is None

    def test_safe_technology_keyword(self) -> None:
        blocked, reason = _blocklist_check("Python tutorial for beginners")
        assert blocked is False

    def test_safe_finance_keyword(self) -> None:
        blocked, reason = _blocklist_check("how to invest in index funds")
        assert blocked is False

    def test_case_insensitive(self) -> None:
        blocked, _ = _blocklist_check("FREE PORN DOWNLOAD")
        assert blocked is True

    def test_empty_string(self) -> None:
        blocked, reason = _blocklist_check("")
        assert blocked is False  # blocklist doesn't block empty; SafetyFilter does


# ---------------------------------------------------------------------------
# SafetyResult
# ---------------------------------------------------------------------------

class TestSafetyResult:
    def test_defaults(self) -> None:
        result = SafetyResult(keyword="test", is_safe=True, reason=None, method="allowed")
        assert result.confidence == 1.0
        assert result.checked_at != ""

    def test_to_dict(self) -> None:
        result = SafetyResult(
            keyword="hack account",
            is_safe=False,
            reason="blocked token: 'hack account'",
            method="blocklist",
            confidence=1.0,
        )
        d = result.to_dict()
        assert d["keyword"] == "hack account"
        assert d["is_safe"] == 0
        assert d["block_reason"] is not None
        assert d["method"] == "blocklist"

    def test_to_dict_safe(self) -> None:
        result = SafetyResult(
            keyword="best headphones",
            is_safe=True,
            reason=None,
            method="allowed",
        )
        d = result.to_dict()
        assert d["is_safe"] == 1
        assert d["block_reason"] is None


# ---------------------------------------------------------------------------
# ClaudeSafetyClassifier (mocked)
# ---------------------------------------------------------------------------

class TestClaudeSafetyClassifier:
    def _make_mock_message(self, payload: dict) -> MagicMock:
        msg = MagicMock()
        msg.content = [MagicMock(text=json.dumps(payload))]
        return msg

    @patch("src.filter.safety_filter.anthropic.Anthropic")
    def test_classify_safe(self, mock_anthropic_cls) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._make_mock_message(
            {"safe": True, "confidence": 0.95, "reason": None}
        )
        mock_anthropic_cls.return_value = mock_client

        classifier = ClaudeSafetyClassifier(api_key="fake")
        result = classifier.classify("best microphone for streaming")

        assert result.is_safe is True
        assert result.confidence == 0.95
        assert result.method == "claude_api"

    @patch("src.filter.safety_filter.anthropic.Anthropic")
    def test_classify_unsafe(self, mock_anthropic_cls) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._make_mock_message(
            {"safe": False, "confidence": 0.90, "reason": "promotes illegal activity"}
        )
        mock_anthropic_cls.return_value = mock_client

        classifier = ClaudeSafetyClassifier(api_key="fake")
        result = classifier.classify("how to pick a lock illegally")

        assert result.is_safe is False
        assert result.reason == "promotes illegal activity"

    @patch("src.filter.safety_filter.anthropic.Anthropic")
    def test_classify_handles_malformed_json(self, mock_anthropic_cls) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="not valid JSON!!!")]
        )
        mock_anthropic_cls.return_value = mock_client

        classifier = ClaudeSafetyClassifier(api_key="fake")
        # Should fail open (return safe) on parse error
        result = classifier.classify("some keyword")
        assert result.is_safe is True
        assert result.confidence == 0.5

    @patch("src.filter.safety_filter.anthropic.Anthropic")
    def test_classify_handles_api_exception(self, mock_anthropic_cls) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API down")
        mock_anthropic_cls.return_value = mock_client

        classifier = ClaudeSafetyClassifier(api_key="fake")
        result = classifier.classify("test keyword")
        assert result.is_safe is True  # fail open
        assert result.confidence == 0.5

    def test_raises_without_api_key(self) -> None:
        classifier = ClaudeSafetyClassifier(api_key="")
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY not set"):
            classifier._get_client()

    @patch("src.filter.safety_filter.anthropic.Anthropic")
    def test_strips_markdown_fences(self, mock_anthropic_cls) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text='```json\n{"safe": true, "confidence": 0.9}\n```')]
        )
        mock_anthropic_cls.return_value = mock_client

        classifier = ClaudeSafetyClassifier(api_key="fake")
        result = classifier.classify("test")
        assert result.is_safe is True


# ---------------------------------------------------------------------------
# SafetyFilter (integration-level with mocked Claude)
# ---------------------------------------------------------------------------

class TestSafetyFilter:
    def test_blocks_empty_keyword(self) -> None:
        filt = SafetyFilter(use_claude=False)
        result = filt.check("")
        assert result.is_safe is False
        assert "empty" in result.reason

    def test_blocks_blocklist_keyword(self) -> None:
        filt = SafetyFilter(use_claude=False)
        result = filt.check("buy cocaine online")
        assert result.is_safe is False
        assert result.method == "blocklist"

    def test_passes_safe_keyword_without_claude(self) -> None:
        filt = SafetyFilter(use_claude=False)
        result = filt.check("best Python courses 2024")
        assert result.is_safe is True
        assert result.method == "allowed"

    def test_extra_blocked_tokens(self) -> None:
        filt = SafetyFilter(use_claude=False, extra_blocked_tokens=["MLM", "pyramid"])
        result = filt.check("join my MLM business")
        assert result.is_safe is False

    def test_extra_tokens_case_insensitive(self) -> None:
        filt = SafetyFilter(use_claude=False, extra_blocked_tokens=["SPAM"])
        result = filt.check("avoid spam emails tutorial")
        assert result.is_safe is False  # 'spam' matches 'SPAM' (case-insensitive)

    @patch("src.filter.safety_filter.anthropic.Anthropic")
    def test_uses_claude_when_enabled(self, mock_anthropic_cls) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text='{"safe": true, "confidence": 0.88}')]
        )
        mock_anthropic_cls.return_value = mock_client

        filt = SafetyFilter(use_claude=True, claude_api_key="fake")
        result = filt.check("how to start a podcast")

        assert result.is_safe is True
        assert result.method == "claude_api"

    @patch("src.filter.safety_filter.anthropic.Anthropic")
    def test_blocklist_takes_priority_over_claude(self, mock_anthropic_cls) -> None:
        """Blocked keywords should never reach Claude."""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        filt = SafetyFilter(use_claude=True, claude_api_key="fake")
        result = filt.check("buy heroin now")

        assert result.is_safe is False
        assert result.method == "blocklist"
        # Claude should NOT have been called
        mock_client.messages.create.assert_not_called()

    def test_check_batch(self) -> None:
        filt = SafetyFilter(use_claude=False)
        results = filt.check_batch(
            ["Python tutorial", "buy cocaine", "best headphones"]
        )
        assert len(results) == 3
        safe_flags = [r.is_safe for r in results]
        assert safe_flags == [True, False, True]

    def test_filter_safe_returns_only_safe(self) -> None:
        filt = SafetyFilter(use_claude=False)
        keywords = ["Python tutorial", "buy heroin", "make money online", "bomb making"]
        safe = filt.filter_safe(keywords)
        assert "buy heroin" not in safe
        assert "bomb making" not in safe
        assert "Python tutorial" in safe
        assert "make money online" in safe

    def test_whitespace_only_keyword(self) -> None:
        filt = SafetyFilter(use_claude=False)
        result = filt.check("   ")
        # strip() makes it empty → blocked
        assert result.is_safe is False

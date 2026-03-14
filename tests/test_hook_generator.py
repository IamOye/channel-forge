"""
Tests for src/content/hook_generator.py

All Claude API calls are mocked — no real API calls during tests.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.content.hook_generator import (
    HookGenerator,
    HookResult,
    HookVariant,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_variants_payload(n: int = 5) -> list[dict]:
    """Build a valid JSON payload of n hook variants."""
    return [
        {
            "text": f"Hook variant {i} that grabs attention right now",
            "curiosity_score": 7 + (i % 3),
            "clarity_score": 6 + (i % 4),
        }
        for i in range(n)
    ]


def _mock_message(payload) -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock(text=json.dumps(payload))]
    return msg


# ---------------------------------------------------------------------------
# HookVariant
# ---------------------------------------------------------------------------

class TestHookVariant:
    def test_combined_score_is_average(self) -> None:
        v = HookVariant(text="test", curiosity_score=8.0, clarity_score=6.0)
        assert v.combined_score == 7.0

    def test_combined_score_equal_weights(self) -> None:
        v = HookVariant(text="test", curiosity_score=10.0, clarity_score=10.0)
        assert v.combined_score == 10.0

    def test_combined_score_zeros(self) -> None:
        v = HookVariant(text="test", curiosity_score=0.0, clarity_score=0.0)
        assert v.combined_score == 0.0


# ---------------------------------------------------------------------------
# HookResult
# ---------------------------------------------------------------------------

class TestHookResult:
    def _make_result(self) -> HookResult:
        variants = [HookVariant(text=f"hook {i}", curiosity_score=float(i), clarity_score=float(i)) for i in range(5)]
        best = variants[-1]
        return HookResult(topic="stoic quotes", emotion="curiosity", input_score=75.0, best=best, variants=variants)

    def test_generated_at_auto_set(self) -> None:
        r = self._make_result()
        assert r.generated_at != ""

    def test_to_dict_shape(self) -> None:
        r = self._make_result()
        d = r.to_dict()
        assert "topic" in d
        assert "best_hook" in d
        assert "best_curiosity" in d
        assert "best_clarity" in d
        assert "best_combined" in d
        assert "all_variants" in d
        assert len(d["all_variants"]) == 5

    def test_to_dict_all_variants_have_combined(self) -> None:
        r = self._make_result()
        for v in r.to_dict()["all_variants"]:
            assert "combined_score" in v


# ---------------------------------------------------------------------------
# HookGenerator._parse_variants
# ---------------------------------------------------------------------------

class TestParseVariants:
    def setup_method(self) -> None:
        self.gen = HookGenerator(api_key="")

    def test_parses_valid_json(self) -> None:
        payload = _make_variants_payload(5)
        variants = self.gen._parse_variants(json.dumps(payload))
        assert len(variants) == 5
        assert variants[0].text == payload[0]["text"]

    def test_clamps_scores_above_10(self) -> None:
        payload = [{"text": "hook", "curiosity_score": 15, "clarity_score": 20}]
        variants = self.gen._parse_variants(json.dumps(payload))
        assert variants[0].curiosity_score == 10.0
        assert variants[0].clarity_score == 10.0

    def test_clamps_scores_below_0(self) -> None:
        payload = [{"text": "hook", "curiosity_score": -5, "clarity_score": -3}]
        variants = self.gen._parse_variants(json.dumps(payload))
        assert variants[0].curiosity_score == 0.0
        assert variants[0].clarity_score == 0.0

    def test_strips_markdown_fences(self) -> None:
        payload = _make_variants_payload(5)
        raw = f"```json\n{json.dumps(payload)}\n```"
        variants = self.gen._parse_variants(raw)
        assert len(variants) == 5

    def test_returns_fallback_on_bad_json(self) -> None:
        variants = self.gen._parse_variants("not valid JSON!!!")
        assert len(variants) == 5
        assert all(v.text == "Hook generation failed" for v in variants)

    def test_pads_to_5_if_fewer_returned(self) -> None:
        payload = _make_variants_payload(2)
        variants = self.gen._parse_variants(json.dumps(payload))
        assert len(variants) == 5
        # First 2 are real, rest are padding
        assert variants[0].text != "Hook unavailable"
        assert variants[2].text == "Hook unavailable"

    def test_returns_fallback_for_empty_array(self) -> None:
        variants = self.gen._parse_variants("[]")
        assert len(variants) == 5


# ---------------------------------------------------------------------------
# HookGenerator._select_best
# ---------------------------------------------------------------------------

class TestSelectBest:
    def test_selects_highest_combined(self) -> None:
        variants = [
            HookVariant(text="low",  curiosity_score=3.0, clarity_score=3.0),
            HookVariant(text="high", curiosity_score=9.0, clarity_score=8.0),
            HookVariant(text="mid",  curiosity_score=6.0, clarity_score=5.0),
        ]
        best = HookGenerator._select_best(variants)
        assert best.text == "high"

    def test_tie_returns_first_max(self) -> None:
        variants = [
            HookVariant(text="first",  curiosity_score=8.0, clarity_score=8.0),
            HookVariant(text="second", curiosity_score=8.0, clarity_score=8.0),
        ]
        best = HookGenerator._select_best(variants)
        assert best.text == "first"


# ---------------------------------------------------------------------------
# HookGenerator.generate (fully mocked)
# ---------------------------------------------------------------------------

class TestHookGeneratorGenerate:
    @patch("src.content.hook_generator.anthropic.Anthropic")
    def test_returns_hook_result(self, mock_cls) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_message(
            _make_variants_payload(5)
        )
        mock_cls.return_value = mock_client

        gen = HookGenerator(api_key="fake")
        result = gen.generate(topic="stoic quotes", score=80.0, emotion="curiosity")

        assert isinstance(result, HookResult)
        assert result.topic == "stoic quotes"
        assert result.emotion == "curiosity"
        assert result.input_score == 80.0
        assert len(result.variants) == 5
        assert isinstance(result.best, HookVariant)

    @patch("src.content.hook_generator.anthropic.Anthropic")
    def test_best_is_highest_combined(self, mock_cls) -> None:
        payload = [
            {"text": "low hook", "curiosity_score": 2, "clarity_score": 2},
            {"text": "best hook", "curiosity_score": 9, "clarity_score": 9},
            {"text": "mid hook", "curiosity_score": 5, "clarity_score": 5},
            {"text": "ok hook",  "curiosity_score": 6, "clarity_score": 4},
            {"text": "meh hook", "curiosity_score": 3, "clarity_score": 7},
        ]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_message(payload)
        mock_cls.return_value = mock_client

        gen = HookGenerator(api_key="fake")
        result = gen.generate(topic="test", score=50.0, emotion="shock")

        assert result.best.text == "best hook"

    @patch("src.content.hook_generator.anthropic.Anthropic")
    def test_result_has_generated_at(self, mock_cls) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_message(_make_variants_payload())
        mock_cls.return_value = mock_client

        gen = HookGenerator(api_key="fake")
        result = gen.generate(topic="test", score=60.0, emotion="fear")

        assert result.generated_at != ""

    @patch("src.content.hook_generator.anthropic.Anthropic")
    def test_handles_malformed_api_response(self, mock_cls) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="not json at all")]
        )
        mock_cls.return_value = mock_client

        gen = HookGenerator(api_key="fake")
        result = gen.generate(topic="test", score=50.0, emotion="curiosity")

        # Should not raise — returns fallback variants
        assert len(result.variants) == 5
        assert result.best.text == "Hook generation failed"

    def test_raises_without_api_key(self) -> None:
        gen = HookGenerator(api_key="")
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY not set"):
            gen.generate(topic="test", score=50.0, emotion="curiosity")

    @patch("src.content.hook_generator.anthropic.Anthropic")
    def test_to_dict_serialisable(self, mock_cls) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_message(_make_variants_payload())
        mock_cls.return_value = mock_client

        gen = HookGenerator(api_key="fake")
        result = gen.generate(topic="test", score=70.0, emotion="inspiration")
        d = result.to_dict()

        # Should be JSON-serialisable
        import json as _json
        serialised = _json.dumps(d)
        assert len(serialised) > 10


# ---------------------------------------------------------------------------
# Hook formula scoring (new Phase 3 fields)
# ---------------------------------------------------------------------------


class TestHookVariantNewScores:
    def test_new_scores_default_zero(self) -> None:
        v = HookVariant(text="test hook", curiosity_score=5.0, clarity_score=5.0)
        assert v.open_loop_score == 0.0
        assert v.personal_relevance_score == 0.0
        assert v.contradiction_score == 0.0

    def test_combined_uses_new_scores_when_set(self) -> None:
        v = HookVariant(
            text="test",
            curiosity_score=5.0,
            clarity_score=5.0,
            open_loop_score=9.0,
            personal_relevance_score=7.0,
            contradiction_score=8.0,
        )
        assert abs(v.combined_score - (9.0 + 7.0 + 8.0) / 3.0) < 0.001

    def test_combined_falls_back_to_old_scores_when_new_are_zero(self) -> None:
        v = HookVariant(text="test", curiosity_score=8.0, clarity_score=6.0)
        assert v.combined_score == 7.0  # (8 + 6) / 2

    def test_formula_field_defaults_zero(self) -> None:
        v = HookVariant(text="test hook", curiosity_score=5.0, clarity_score=5.0)
        assert v.formula == 0

    def test_parse_variants_reads_new_fields(self) -> None:
        import json
        gen = HookGenerator(api_key="")
        payload = [
            {
                "text": "hook text",
                "formula": 2,
                "open_loop": 8,
                "personal_relevance": 7,
                "contradiction": 9,
            }
        ] + [
            {"text": f"h{i}", "formula": i + 1, "open_loop": 5, "personal_relevance": 5, "contradiction": 5}
            for i in range(4)
        ]
        variants = gen._parse_variants(json.dumps(payload))
        assert variants[0].open_loop_score == 8.0
        assert variants[0].personal_relevance_score == 7.0
        assert variants[0].contradiction_score == 9.0

    def test_five_formulas_produce_different_texts(self) -> None:
        """Claude prompt should produce 5 different formula-based hooks."""
        import json
        from unittest.mock import MagicMock, patch

        payload = [
            {"text": f"Formula {i+1} hook text example here", "formula": i+1,
             "open_loop": 7, "personal_relevance": 6, "contradiction": 8}
            for i in range(5)
        ]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text=json.dumps(payload))]
        )
        with patch("src.content.hook_generator.anthropic.Anthropic", return_value=mock_client):
            gen = HookGenerator(api_key="fake")
            result = gen.generate(topic="passive income", score=80.0, emotion="shock")

        texts = {v.text for v in result.variants}
        assert len(texts) == 5  # all unique

    def test_system_prompt_contains_formula_names(self) -> None:
        from src.content.hook_generator import _SYSTEM_PROMPT
        assert "CONTRADICTION" in _SYSTEM_PROMPT
        assert "PERSONAL ACCUSATION" in _SYSTEM_PROMPT
        assert "INSIDER SECRET" in _SYSTEM_PROMPT
        assert "UNCOMFORTABLE TRUTH" in _SYSTEM_PROMPT
        assert "PATTERN INTERRUPT" in _SYSTEM_PROMPT

    def test_system_prompt_no_yes_no_questions_rule(self) -> None:
        from src.content.hook_generator import _SYSTEM_PROMPT
        assert "yes/no" in _SYSTEM_PROMPT.lower() or "yes-no" in _SYSTEM_PROMPT.lower()

    def test_to_dict_includes_new_score_fields(self) -> None:
        import json
        from unittest.mock import MagicMock, patch

        payload = [
            {"text": f"h{i}", "formula": i+1,
             "open_loop": 7, "personal_relevance": 6, "contradiction": 8}
            for i in range(5)
        ]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text=json.dumps(payload))]
        )
        with patch("src.content.hook_generator.anthropic.Anthropic", return_value=mock_client):
            gen = HookGenerator(api_key="fake")
            result = gen.generate(topic="test", score=70.0, emotion="curiosity")

        d = result.to_dict()
        assert "best_open_loop" in d
        assert "best_personal_relevance" in d
        assert "best_contradiction" in d

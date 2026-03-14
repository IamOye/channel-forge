"""Tests for src/media/thumbnail_generator.py"""

import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestThumbnailGenerator:
    def test_generate_returns_path(self, tmp_path) -> None:
        from src.media.thumbnail_generator import ThumbnailGenerator
        gen = ThumbnailGenerator(output_dir=tmp_path)
        path = gen.generate(hook="Working harder keeps you poor", topic="test_topic", category="money")
        assert Path(path).exists()

    def test_thumbnail_is_1280x720(self, tmp_path) -> None:
        from src.media.thumbnail_generator import ThumbnailGenerator
        from PIL import Image
        gen = ThumbnailGenerator(output_dir=tmp_path)
        path = gen.generate(hook="Your savings lose value every year", topic="t1", category="money")
        img = Image.open(path)
        assert img.size == (1280, 720)

    def test_thumbnail_is_jpeg(self, tmp_path) -> None:
        from src.media.thumbnail_generator import ThumbnailGenerator
        gen = ThumbnailGenerator(output_dir=tmp_path)
        path = gen.generate(hook="Save money lose money", topic="t2", category="money")
        assert path.endswith(".jpg")

    def test_output_dir_created(self, tmp_path) -> None:
        from src.media.thumbnail_generator import ThumbnailGenerator
        nested = tmp_path / "deep" / "dir"
        gen = ThumbnailGenerator(output_dir=nested)
        gen.generate(hook="Short punchy hook here", topic="t3", category="money")
        assert nested.exists()

    def test_pick_symbol_money_category(self) -> None:
        from src.media.thumbnail_generator import ThumbnailGenerator
        gen = ThumbnailGenerator()
        assert gen._pick_symbol("any hook", "money") == "$"

    def test_pick_symbol_question_word(self) -> None:
        from src.media.thumbnail_generator import ThumbnailGenerator
        gen = ThumbnailGenerator()
        assert gen._pick_symbol("Why the rich never tell you", "success") == "?"

    def test_pick_symbol_shocking_fact(self) -> None:
        from src.media.thumbnail_generator import ThumbnailGenerator
        gen = ThumbnailGenerator()
        assert gen._pick_symbol("Banks are lying to your face", "success") == "!"

    def test_truncate_hook_short(self) -> None:
        from src.media.thumbnail_generator import ThumbnailGenerator
        gen = ThumbnailGenerator()
        result = gen._truncate_hook("Short hook", max_words=6)
        assert result == "Short hook"

    def test_truncate_hook_long(self) -> None:
        from src.media.thumbnail_generator import ThumbnailGenerator
        gen = ThumbnailGenerator()
        result = gen._truncate_hook("One two three four five six seven eight", max_words=6)
        assert result == "One two three four five six"

    def test_find_provocative_word_strong(self) -> None:
        from src.media.thumbnail_generator import ThumbnailGenerator
        gen = ThumbnailGenerator()
        result = gen._find_provocative_word("You are broke and tired")
        assert result == "broke"

    def test_find_provocative_word_fallback_longest(self) -> None:
        from src.media.thumbnail_generator import ThumbnailGenerator
        gen = ThumbnailGenerator()
        result = gen._find_provocative_word("The quick fox")
        assert result == "quick"  # longest of 4+ char words

    def test_filename_uses_topic(self, tmp_path) -> None:
        from src.media.thumbnail_generator import ThumbnailGenerator
        gen = ThumbnailGenerator(output_dir=tmp_path)
        path = gen.generate(hook="You are losing money", topic="my_topic_123", category="money")
        assert "my_topic_123" in Path(path).name

    def test_wrap_text_short_fits_one_line(self, tmp_path) -> None:
        from src.media.thumbnail_generator import ThumbnailGenerator
        from PIL import ImageFont
        gen = ThumbnailGenerator(output_dir=tmp_path)
        font = gen._load_font(30)
        lines = gen._wrap_text("Hi", font, 500)
        assert len(lines) == 1

    def test_different_categories_generate_file(self, tmp_path) -> None:
        from src.media.thumbnail_generator import ThumbnailGenerator
        gen = ThumbnailGenerator(output_dir=tmp_path)
        for cat in ["money", "career", "success"]:
            path = gen.generate(hook=f"Hook for {cat}", topic=f"topic_{cat}", category=cat)
            assert Path(path).exists()

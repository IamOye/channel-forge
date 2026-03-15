"""
Tests for src/media/pixabay_fetcher.py

All httpx calls are mocked — no real network requests during tests.
"""

import json
from unittest.mock import MagicMock, patch, call

import pytest

from src.media.pixabay_fetcher import (
    FALLBACK_CLIP_IDS,
    KEYWORD_MAP,
    FetchResult,
    PixabayFetcher,
    PixabayVideo,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hit(
    vid_id: int = 1,
    duration: int = 15,
    url: str = "https://cdn.pixabay.com/video/sample.mp4",
    tags: str = "nature landscape",
) -> dict:
    """Build a minimal Pixabay API hit dict with portrait (1080x1920) dimensions."""
    return {
        "id":       vid_id,
        "duration": duration,
        "pageURL":  f"https://pixabay.com/videos/{vid_id}/",
        "tags":     tags,
        "videos": {
            "large":  {"url": url, "width": 1080, "height": 1920},
            "medium": {"url": url, "width": 720,  "height": 1280},
            "small":  {"url": url, "width": 360,  "height": 640},
            "tiny":   {"url": url, "width": 180,  "height": 320},
        },
    }


def _mock_api_response(hits: list) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = {"hits": hits, "total": len(hits)}
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# PixabayVideo
# ---------------------------------------------------------------------------

class TestPixabayVideo:
    def test_to_dict_has_all_keys(self) -> None:
        v = PixabayVideo(
            pixabay_id=42,
            duration=15,
            width=1080,
            height=1920,
            download_url="https://example.com/vid.mp4",
            page_url="https://pixabay.com/videos/42/",
            tags="nature",
        )
        d = v.to_dict()
        for key in ("pixabay_id", "duration", "width", "height",
                    "download_url", "page_url", "tags"):
            assert key in d


# ---------------------------------------------------------------------------
# FetchResult
# ---------------------------------------------------------------------------

class TestFetchResult:
    def test_fetched_at_auto_set(self) -> None:
        r = FetchResult(topic_id="t1", video_path="", video_meta=None, is_valid=False)
        assert r.fetched_at != ""

    def test_to_dict_has_all_keys(self) -> None:
        r = FetchResult(topic_id="t1", video_path="", video_meta=None, is_valid=False)
        d = r.to_dict()
        for key in ("topic_id", "video_path", "video_meta", "is_valid",
                    "validation_errors", "fetched_at"):
            assert key in d

    def test_to_dict_with_video_meta(self) -> None:
        meta = PixabayVideo(1, 15, 1080, 1920, "url", "page", "tags")
        r = FetchResult(topic_id="t1", video_path="/data/raw/t1_stock.mp4",
                        video_meta=meta, is_valid=True)
        d = r.to_dict()
        assert d["video_meta"] is not None
        assert d["video_meta"]["pixabay_id"] == 1


# ---------------------------------------------------------------------------
# PixabayFetcher._best_url
# ---------------------------------------------------------------------------

class TestBestUrl:
    def test_prefers_large(self) -> None:
        videos = {
            "large":  {"url": "large_url", "width": 1080, "height": 1920},
            "medium": {"url": "medium_url", "width": 720, "height": 1280},
        }
        url, w, h = PixabayFetcher._best_url(videos)
        assert url == "large_url"
        assert w == 1080
        assert h == 1920

    def test_falls_back_to_medium(self) -> None:
        videos = {"medium": {"url": "medium_url", "width": 720, "height": 1280}}
        url, w, h = PixabayFetcher._best_url(videos)
        assert url == "medium_url"

    def test_returns_empty_when_no_url(self) -> None:
        url, w, h = PixabayFetcher._best_url({})
        assert url == ""
        assert w == 0
        assert h == 0


# ---------------------------------------------------------------------------
# PixabayFetcher.fetch (fully mocked)
# ---------------------------------------------------------------------------

class TestPixabayFetcherFetch:
    @patch("src.media.pixabay_fetcher.httpx.get")
    @patch("src.media.pixabay_fetcher.PixabayFetcher._download_verified", return_value=True)
    def test_returns_valid_result(self, mock_dl, mock_get) -> None:
        mock_get.return_value = _mock_api_response([_make_hit()])
        with patch("pathlib.Path.mkdir"):
            fetcher = PixabayFetcher(api_key="fake")
            result = fetcher.fetch(topic_id="test_001", category="success")

        assert isinstance(result, FetchResult)
        assert result.is_valid is True
        assert "test_001" in result.video_path
        assert result.video_path.endswith("_stock.mp4")
        assert result.video_meta is not None

    @patch("src.media.pixabay_fetcher.httpx.get")
    def test_returns_invalid_when_no_hits(self, mock_get) -> None:
        mock_get.return_value = _mock_api_response([])

        fetcher = PixabayFetcher(api_key="fake")
        result = fetcher.fetch(topic_id="no_hit", category="default")

        assert result.is_valid is False
        assert any("no suitable" in e for e in result.validation_errors)

    @patch("src.media.pixabay_fetcher.httpx.get")
    def test_skips_videos_shorter_than_min_duration(self, mock_get) -> None:
        hit_short = _make_hit(vid_id=1, duration=3)   # too short
        hit_ok    = _make_hit(vid_id=2, duration=20)  # ok
        mock_get.return_value = _mock_api_response([hit_short, hit_ok])

        with patch("src.media.pixabay_fetcher.PixabayFetcher._download_verified", return_value=True):
            with patch("pathlib.Path.mkdir"):
                fetcher = PixabayFetcher(api_key="fake", min_duration=10)
                result = fetcher.fetch(topic_id="dur_test", category="default")

        assert result.is_valid is True
        assert result.video_meta.pixabay_id == 2

    def test_raises_without_api_key(self) -> None:
        fetcher = PixabayFetcher(api_key="")
        with pytest.raises(ValueError, match="PIXABAY_API_KEY not set"):
            fetcher.fetch(topic_id="test")

    @patch("src.media.pixabay_fetcher.httpx.get")
    def test_api_error_returns_invalid(self, mock_get) -> None:
        import httpx
        mock_get.side_effect = httpx.RequestError("network error")

        fetcher = PixabayFetcher(api_key="fake")
        result = fetcher.fetch(topic_id="err_test")

        assert result.is_valid is False
        assert any("no suitable" in e for e in result.validation_errors)

    @patch("src.media.pixabay_fetcher.httpx.get")
    @patch("src.media.pixabay_fetcher.PixabayFetcher._download_verified", return_value=True)
    def test_uses_keyword_map_for_category(self, mock_dl, mock_get) -> None:
        mock_get.return_value = _mock_api_response([_make_hit()])
        with patch("pathlib.Path.mkdir"):
            fetcher = PixabayFetcher(api_key="fake")
            fetcher.fetch(topic_id="kw_test", category="money")

        # The query param should contain one of the money keywords
        call_kwargs = mock_get.call_args_list[0][1]
        query_used = call_kwargs["params"]["q"]
        expected_keywords = KEYWORD_MAP["money"]
        assert any(phrase == query_used for phrase in expected_keywords)

    @patch("src.media.pixabay_fetcher.httpx.get")
    @patch("src.media.pixabay_fetcher.PixabayFetcher._download_verified", return_value=True)
    def test_unknown_category_falls_back_to_default(self, mock_dl, mock_get) -> None:
        mock_get.return_value = _mock_api_response([_make_hit()])
        with patch("pathlib.Path.mkdir"):
            fetcher = PixabayFetcher(api_key="fake")
            result = fetcher.fetch(topic_id="def_test", category="nonexistent")

        assert result.is_valid is True

    @patch("src.media.pixabay_fetcher.httpx.get")
    @patch("src.media.pixabay_fetcher.PixabayFetcher._download_verified", return_value=True)
    def test_to_dict_is_serialisable(self, mock_dl, mock_get) -> None:
        import json as _json
        mock_get.return_value = _mock_api_response([_make_hit()])
        with patch("pathlib.Path.mkdir"):
            fetcher = PixabayFetcher(api_key="fake")
            result = fetcher.fetch(topic_id="serial_001")

        assert len(_json.dumps(result.to_dict())) > 10


# ---------------------------------------------------------------------------
# FIX 1 — portrait ratio filter
# ---------------------------------------------------------------------------


def _make_hit_dims(vid_id: int, width: int, height: int, duration: int = 15) -> dict:
    """Build a Pixabay hit with explicit dimensions."""
    url = f"https://cdn.pixabay.com/video/s{vid_id}.mp4"
    return {
        "id": vid_id,
        "duration": duration,
        "pageURL": f"https://pixabay.com/videos/{vid_id}/",
        "tags": f"tag_{vid_id}",
        "videos": {
            "large": {"url": url, "width": width, "height": height},
        },
    }


class TestRatioFilter:
    @patch("src.media.pixabay_fetcher.httpx.get")
    def test_accepts_ideal_portrait_ratio(self, mock_get) -> None:
        """1080×1920 = ratio 0.5625, inside [0.50, 0.62] → accepted."""
        mock_get.return_value = _mock_api_response([_make_hit(vid_id=1)])
        fetcher = PixabayFetcher(api_key="fake")
        candidates = fetcher._query_api("test")
        assert len(candidates) == 1
        assert candidates[0].pixabay_id == 1

    @patch("src.media.pixabay_fetcher.httpx.get")
    def test_rejects_clip_ratio_above_0_62(self, mock_get) -> None:
        """1080×1500 = ratio 0.72 > 0.62 → rejected (too wide for portrait frame)."""
        hit = _make_hit_dims(vid_id=2, width=1080, height=1500)
        mock_get.return_value = _mock_api_response([hit])
        fetcher = PixabayFetcher(api_key="fake")
        candidates = fetcher._query_api("test")
        assert candidates == []

    @patch("src.media.pixabay_fetcher.httpx.get")
    def test_rejects_clip_ratio_below_0_50(self, mock_get) -> None:
        """1080×2500 = ratio 0.432 < 0.50 → rejected (too narrow)."""
        hit = _make_hit_dims(vid_id=3, width=1080, height=2500)
        mock_get.return_value = _mock_api_response([hit])
        fetcher = PixabayFetcher(api_key="fake")
        candidates = fetcher._query_api("test")
        assert candidates == []

    @patch("src.media.pixabay_fetcher.httpx.get")
    def test_accepts_clip_at_boundary_0_62(self, mock_get) -> None:
        """width/height = 0.62 exactly → accepted (boundary is inclusive)."""
        # 1116×1800 → ratio = 0.62
        hit = _make_hit_dims(vid_id=4, width=1116, height=1800)
        mock_get.return_value = _mock_api_response([hit])
        fetcher = PixabayFetcher(api_key="fake")
        candidates = fetcher._query_api("test")
        assert len(candidates) == 1

    @patch("src.media.pixabay_fetcher.httpx.get")
    def test_accepts_clip_at_boundary_0_50(self, mock_get) -> None:
        """width/height = 0.50 exactly → accepted (boundary is inclusive)."""
        # 1080×2160 → ratio = 0.50
        hit = _make_hit_dims(vid_id=5, width=1080, height=2160)
        mock_get.return_value = _mock_api_response([hit])
        fetcher = PixabayFetcher(api_key="fake")
        candidates = fetcher._query_api("test")
        assert len(candidates) == 1

    @patch("src.media.pixabay_fetcher.httpx.get")
    def test_mixed_clips_only_valid_ratio_returned(self, mock_get) -> None:
        """Two clips: one valid (0.5625), one too wide (0.72) → only valid returned."""
        valid_hit = _make_hit(vid_id=10)          # 1080×1920 = 0.5625 ✓
        wide_hit  = _make_hit_dims(20, 1080, 1500)  # ratio = 0.72 ✗
        mock_get.return_value = _mock_api_response([valid_hit, wide_hit])
        fetcher = PixabayFetcher(api_key="fake")
        candidates = fetcher._query_api("test")
        assert len(candidates) == 1
        assert candidates[0].pixabay_id == 10


# ---------------------------------------------------------------------------
# FIX 2 — relevance scoring
# ---------------------------------------------------------------------------


class TestScoreClipRelevance:
    def _make_clips(self, count: int = 2) -> list[PixabayVideo]:
        return [
            PixabayVideo(i, 15, 1080, 1920, f"url{i}", f"page{i}", f"finance money tag{i}")
            for i in range(1, count + 1)
        ]

    def test_returns_all_clips_without_anthropic_key(self) -> None:
        """No Anthropic key → clips returned unchanged."""
        fetcher = PixabayFetcher(api_key="fake", anthropic_api_key="")
        clips = self._make_clips(2)
        result = fetcher.score_clip_relevance(clips, "money habits", "script preview")
        assert result == clips

    def test_returns_all_clips_on_empty_clips(self) -> None:
        fetcher = PixabayFetcher(api_key="fake", anthropic_api_key="key")
        result = fetcher.score_clip_relevance([], "topic", "preview")
        assert result == []

    def test_returns_all_clips_on_api_failure(self) -> None:
        """API error → falls back to returning all clips unchanged."""
        fetcher = PixabayFetcher(api_key="fake", anthropic_api_key="test-key")
        clips = self._make_clips(2)
        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.create.side_effect = Exception("API error")
            result = fetcher.score_clip_relevance(clips, "topic", "preview")
        assert result == clips

    def test_filters_clips_below_score_6(self) -> None:
        """Clips with score < 6 are removed from the result."""
        fetcher = PixabayFetcher(api_key="fake", anthropic_api_key="test-key")
        clips = self._make_clips(3)
        scores_data = [
            {"clip_id": 1, "score": 8, "reason": "relevant"},
            {"clip_id": 2, "score": 3, "reason": "animals — not relevant"},
            {"clip_id": 3, "score": 7, "reason": "financial scene"},
        ]
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text=json.dumps(scores_data))]

        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.create.return_value = mock_msg
            result = fetcher.score_clip_relevance(clips, "money habits", "script preview")

        assert len(result) == 2
        assert {v.pixabay_id for v in result} == {1, 3}

    def test_accepts_all_clips_above_threshold(self) -> None:
        """All clips >= 6 are kept."""
        fetcher = PixabayFetcher(api_key="fake", anthropic_api_key="test-key")
        clips = self._make_clips(2)
        scores_data = [
            {"clip_id": 1, "score": 7, "reason": "relevant"},
            {"clip_id": 2, "score": 9, "reason": "highly relevant"},
        ]
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text=json.dumps(scores_data))]

        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.create.return_value = mock_msg
            result = fetcher.score_clip_relevance(clips, "topic", "script")

        assert len(result) == 2

    def test_unscored_clips_treated_as_zero(self) -> None:
        """Clips not returned by the API are treated as score 0 (rejected)."""
        fetcher = PixabayFetcher(api_key="fake", anthropic_api_key="test-key")
        clips = self._make_clips(2)  # clip_id 1 and 2
        scores_data = [
            {"clip_id": 1, "score": 8, "reason": "relevant"},
            # clip_id 2 missing from response
        ]
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text=json.dumps(scores_data))]

        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.create.return_value = mock_msg
            result = fetcher.score_clip_relevance(clips, "topic", "script")

        assert len(result) == 1
        assert result[0].pixabay_id == 1

    def test_handles_markdown_fenced_json_response(self) -> None:
        """Claude sometimes returns JSON wrapped in ```json ... ``` fences."""
        fetcher = PixabayFetcher(api_key="fake", anthropic_api_key="test-key")
        clips = self._make_clips(1)
        scores_data = [{"clip_id": 1, "score": 8, "reason": "relevant"}]
        raw = f"```json\n{json.dumps(scores_data)}\n```"
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text=raw)]

        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.create.return_value = mock_msg
            result = fetcher.score_clip_relevance(clips, "topic", "script")

        assert len(result) == 1


# ---------------------------------------------------------------------------
# FIX 3 — fallback clip library
# ---------------------------------------------------------------------------


class TestFallbackClipIds:
    def test_fallback_clip_ids_is_list(self) -> None:
        assert isinstance(FALLBACK_CLIP_IDS, list)

    def test_fallback_clip_ids_has_enough_entries(self) -> None:
        assert len(FALLBACK_CLIP_IDS) >= 5

    def test_fallback_clip_ids_are_positive_integers(self) -> None:
        assert all(isinstance(fid, int) and fid > 0 for fid in FALLBACK_CLIP_IDS)


class TestFillFromFallback:
    @patch("src.media.pixabay_fetcher.httpx.get")
    @patch("src.media.pixabay_fetcher.PixabayFetcher._download_verified", return_value=True)
    def test_fills_slot_from_fallback_api(self, mock_dl, mock_get, tmp_path) -> None:
        fid = FALLBACK_CLIP_IDS[0]
        hit = _make_hit_dims(fid, 1080, 1920)
        mock_get.return_value = _mock_api_response([hit])

        fetcher = PixabayFetcher(api_key="fake", output_dir=tmp_path)
        result = fetcher._fill_from_fallback("t1", [], set(), 2)
        assert len(result) >= 1

    @patch("src.media.pixabay_fetcher.httpx.get")
    def test_handles_api_error_per_clip_gracefully(self, mock_get, tmp_path) -> None:
        import httpx as _httpx
        mock_get.side_effect = _httpx.RequestError("network error")

        fetcher = PixabayFetcher(api_key="fake", output_dir=tmp_path)
        result = fetcher._fill_from_fallback("t1", [], set(), 2)
        assert result == []  # fails gracefully, returns what we had

    @patch("src.media.pixabay_fetcher.httpx.get")
    @patch("src.media.pixabay_fetcher.PixabayFetcher._download_verified", return_value=True)
    def test_skips_already_seen_ids(self, mock_dl, mock_get, tmp_path) -> None:
        """Fallback IDs already in seen_ids must be skipped."""
        all_seen = set(FALLBACK_CLIP_IDS)  # mark every fallback as already used
        fetcher = PixabayFetcher(api_key="fake", output_dir=tmp_path)
        result = fetcher._fill_from_fallback("t1", [], all_seen, 2)
        mock_get.assert_not_called()
        assert result == []

    @patch("src.media.pixabay_fetcher.httpx.get")
    @patch("src.media.pixabay_fetcher.PixabayFetcher._download_verified", return_value=True)
    def test_returns_existing_paths_plus_new(self, mock_dl, mock_get, tmp_path) -> None:
        """Pre-existing path must be preserved in output."""
        fid = FALLBACK_CLIP_IDS[0]
        hit = _make_hit_dims(fid, 1080, 1920)
        mock_get.return_value = _mock_api_response([hit])

        fetcher = PixabayFetcher(api_key="fake", output_dir=tmp_path)
        existing = [str(tmp_path / "t1_stock_0.mp4")]
        result = fetcher._fill_from_fallback("t1", existing, set(), 2)
        assert existing[0] in result


# ---------------------------------------------------------------------------
# fetch_multiple with FIX 2 + FIX 3 integration
# ---------------------------------------------------------------------------


class TestFetchMultiple:
    @patch("src.media.pixabay_fetcher.httpx.get")
    @patch("src.media.pixabay_fetcher.PixabayFetcher._download_verified", return_value=True)
    def test_basic_fetch_multiple_returns_clips(self, mock_dl, mock_get, tmp_path) -> None:
        hits = [_make_hit(i) for i in range(1, 4)]
        mock_get.return_value = _mock_api_response(hits)
        fetcher = PixabayFetcher(api_key="fake", output_dir=tmp_path)
        paths = fetcher.fetch_multiple("t1", ["phrase1", "phrase2"], count=2)
        assert len(paths) == 2
        assert all(p.endswith(".mp4") for p in paths)

    @patch("src.media.pixabay_fetcher.httpx.get")
    def test_raises_without_api_key(self, mock_get, tmp_path) -> None:
        fetcher = PixabayFetcher(api_key="", output_dir=tmp_path)
        with pytest.raises(ValueError, match="PIXABAY_API_KEY not set"):
            fetcher.fetch_multiple("t1", ["phrase"])

    @patch("src.media.pixabay_fetcher.httpx.get")
    def test_calls_fallback_when_no_candidates(self, mock_get, tmp_path) -> None:
        """No API hits → fewer than 2 clips → _fill_from_fallback is called."""
        mock_get.return_value = _mock_api_response([])
        fetcher = PixabayFetcher(api_key="fake", output_dir=tmp_path)
        with patch.object(
            fetcher, "_fill_from_fallback", return_value=["a.mp4", "b.mp4"]
        ) as mock_fill:
            paths = fetcher.fetch_multiple("t1", ["empty"], count=2)
        mock_fill.assert_called_once()
        assert paths == ["a.mp4", "b.mp4"]

    @patch("src.media.pixabay_fetcher.httpx.get")
    @patch("src.media.pixabay_fetcher.PixabayFetcher._download_verified", return_value=True)
    def test_skips_fallback_when_enough_clips(self, mock_dl, mock_get, tmp_path) -> None:
        """2+ clips downloaded → _fill_from_fallback is NOT called."""
        hits = [_make_hit(i) for i in range(1, 4)]
        mock_get.return_value = _mock_api_response(hits)
        fetcher = PixabayFetcher(api_key="fake", output_dir=tmp_path)
        with patch.object(fetcher, "_fill_from_fallback") as mock_fill:
            paths = fetcher.fetch_multiple("t1", ["phrase"], count=2)
        mock_fill.assert_not_called()
        assert len(paths) == 2

    @patch("src.media.pixabay_fetcher.httpx.get")
    @patch("src.media.pixabay_fetcher.PixabayFetcher._download_verified", return_value=True)
    def test_deduplicates_candidates_across_phrases(self, mock_dl, mock_get, tmp_path) -> None:
        """Same clips returned for two phrases should not be downloaded twice."""
        hits = [_make_hit(1), _make_hit(2)]
        mock_get.return_value = _mock_api_response(hits)
        fetcher = PixabayFetcher(api_key="fake", output_dir=tmp_path)
        paths = fetcher.fetch_multiple("t1", ["phrase1", "phrase2"], count=4)
        # Only 2 unique clips exist
        assert len(paths) == 2

    @patch("src.media.pixabay_fetcher.httpx.get")
    @patch("src.media.pixabay_fetcher.PixabayFetcher._download_verified", return_value=True)
    def test_relevance_scoring_called_when_topic_set(self, mock_dl, mock_get, tmp_path) -> None:
        """When topic is provided and anthropic_api_key is set, scoring is invoked."""
        hits = [_make_hit(i) for i in range(1, 4)]
        mock_get.return_value = _mock_api_response(hits)
        fetcher = PixabayFetcher(api_key="fake", output_dir=tmp_path, anthropic_api_key="k")
        with patch.object(fetcher, "score_clip_relevance", return_value=fetcher._query_api.__func__(fetcher, "x") if False else [PixabayVideo(1, 15, 1080, 1920, "u", "p", "t")]) as mock_score:
            # Just verify it's called
            mock_score.return_value = []
            with patch.object(fetcher, "_fill_from_fallback", return_value=[]):
                fetcher.fetch_multiple("t1", ["phrase"], count=2, topic="money")
        mock_score.assert_called_once()

    @patch("src.media.pixabay_fetcher.httpx.get")
    @patch("src.media.pixabay_fetcher.PixabayFetcher._download_verified", return_value=True)
    def test_relevance_scoring_skipped_without_topic(self, mock_dl, mock_get, tmp_path) -> None:
        """When topic is empty, relevance scoring is not called."""
        hits = [_make_hit(i) for i in range(1, 4)]
        mock_get.return_value = _mock_api_response(hits)
        fetcher = PixabayFetcher(api_key="fake", output_dir=tmp_path, anthropic_api_key="k")
        with patch.object(fetcher, "score_clip_relevance") as mock_score:
            fetcher.fetch_multiple("t1", ["phrase"], count=2)  # no topic
        mock_score.assert_not_called()

# ---------------------------------------------------------------------------
# fetch_photos — portrait photo API
# ---------------------------------------------------------------------------


def _make_photo_hit(pid: int, width: int, height: int) -> dict:
    """Build a minimal Pixabay image API hit dict."""
    return {
        "id": pid,
        "imageWidth": width,
        "imageHeight": height,
        "tags": f"photo_tag_{pid}",
        "largeImageURL": f"https://cdn.pixabay.com/photo/{pid}.jpg",
        "webformatURL":  f"https://cdn.pixabay.com/wf/{pid}.jpg",
    }


def _mock_photo_api_response(hits: list) -> MagicMock:
    m = MagicMock()
    m.raise_for_status = MagicMock()
    m.json.return_value = {"hits": hits, "total": len(hits)}
    return m


class TestFetchPhotos:
    @patch("src.media.pixabay_fetcher.httpx.get")
    @patch("src.media.pixabay_fetcher.httpx.stream")
    def test_fetch_photos_returns_portrait_images_only(self, mock_stream, mock_get, tmp_path) -> None:
        """width/height = 0.5625 (portrait) -> accepted."""
        hit = _make_photo_hit(pid=10, width=1080, height=1920)
        mock_get.return_value = _mock_photo_api_response([hit])
        # Mock stream download
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ctx.raise_for_status = MagicMock()
        mock_ctx.iter_bytes = MagicMock(return_value=[b"x" * 2048])
        mock_stream.return_value = mock_ctx

        fetcher = PixabayFetcher(api_key="fake", output_dir=tmp_path)
        results = fetcher.fetch_photos(topic_id="t1", phrase="portrait test")
        assert len(results) == 1
        assert results[0]["id"] == 10
        assert results[0]["width"] == 1080
        assert results[0]["height"] == 1920

    @patch("src.media.pixabay_fetcher.httpx.get")
    def test_fetch_photos_rejects_landscape(self, mock_get, tmp_path) -> None:
        """width/height = 1.5 (landscape) -> rejected."""
        hit = _make_photo_hit(pid=20, width=1920, height=1080)  # ratio 1.778 >= 0.65
        mock_get.return_value = _mock_photo_api_response([hit])
        fetcher = PixabayFetcher(api_key="fake", output_dir=tmp_path)
        results = fetcher.fetch_photos(topic_id="t1", phrase="landscape test")
        assert results == []

    @patch("src.media.pixabay_fetcher.httpx.get")
    def test_fetch_photos_rejects_near_square(self, mock_get, tmp_path) -> None:
        """width/height = 0.90 (near square, >= 0.65) -> rejected."""
        hit = _make_photo_hit(pid=30, width=900, height=1000)  # ratio 0.90 >= 0.65
        mock_get.return_value = _mock_photo_api_response([hit])
        fetcher = PixabayFetcher(api_key="fake", output_dir=tmp_path)
        results = fetcher.fetch_photos(topic_id="t1", phrase="square test")
        assert results == []

    def test_fetch_photos_raises_without_api_key(self, tmp_path) -> None:
        fetcher = PixabayFetcher(api_key="", output_dir=tmp_path)
        with pytest.raises(ValueError, match="PIXABAY_API_KEY not set"):
            fetcher.fetch_photos(topic_id="t1", phrase="test")

    @patch("src.media.pixabay_fetcher.httpx.get")
    def test_fetch_photos_returns_empty_on_api_error(self, mock_get, tmp_path) -> None:
        import httpx as _httpx
        mock_get.side_effect = _httpx.RequestError("network error")
        fetcher = PixabayFetcher(api_key="fake", output_dir=tmp_path)
        results = fetcher.fetch_photos(topic_id="t1", phrase="error test")
        assert results == []

    @patch("src.media.pixabay_fetcher.httpx.get")
    @patch("src.media.pixabay_fetcher.httpx.stream")
    def test_fetch_photos_respects_count(self, mock_stream, mock_get, tmp_path) -> None:
        """Should return at most count photos."""
        hits = [_make_photo_hit(pid=i, width=1080, height=1920) for i in range(1, 6)]
        mock_get.return_value = _mock_photo_api_response(hits)
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ctx.raise_for_status = MagicMock()
        mock_ctx.iter_bytes = MagicMock(return_value=[b"x" * 2048])
        mock_stream.return_value = mock_ctx

        fetcher = PixabayFetcher(api_key="fake", output_dir=tmp_path)
        results = fetcher.fetch_photos(topic_id="t1", phrase="test", count=2)
        assert len(results) == 2

    @patch("src.media.pixabay_fetcher.httpx.get")
    @patch("src.media.pixabay_fetcher.httpx.stream")
    def test_fetch_photos_result_dict_has_required_keys(self, mock_stream, mock_get, tmp_path) -> None:
        hit = _make_photo_hit(pid=99, width=1080, height=1920)
        mock_get.return_value = _mock_photo_api_response([hit])
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ctx.raise_for_status = MagicMock()
        mock_ctx.iter_bytes = MagicMock(return_value=[b"x" * 2048])
        mock_stream.return_value = mock_ctx

        fetcher = PixabayFetcher(api_key="fake", output_dir=tmp_path)
        results = fetcher.fetch_photos(topic_id="t1", phrase="test")
        assert len(results) == 1
        for key in ("id", "local_path", "width", "height", "tags"):
            assert key in results[0]

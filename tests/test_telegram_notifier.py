"""
Tests for src/notifications/telegram_notifier.py

All HTTP calls are mocked — no real Telegram API activity.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.notifications.telegram_notifier import TelegramNotifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _notifier(token: str = "tok", chat_id: str = "123") -> TelegramNotifier:
    return TelegramNotifier(token=token, chat_id=chat_id)


def _mock_httpx_ok():
    """Return a context-manager-compatible mock for a successful httpx.post."""
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# TelegramNotifier.send
# ---------------------------------------------------------------------------

class TestSend:
    def test_returns_true_on_success(self) -> None:
        n = _notifier()
        with patch("src.notifications.telegram_notifier.httpx") as mock_httpx:
            mock_httpx.post.return_value = _mock_httpx_ok()
            result = n.send("hello")
        assert result is True

    def test_returns_false_silently_on_http_failure(self) -> None:
        n = _notifier()
        with patch("src.notifications.telegram_notifier.httpx") as mock_httpx:
            mock_httpx.post.side_effect = Exception("connection refused")
            result = n.send("hello")
        assert result is False

    def test_returns_false_when_token_missing(self) -> None:
        n = TelegramNotifier(token="", chat_id="123")
        result = n.send("hello")
        assert result is False

    def test_returns_false_when_chat_id_missing(self) -> None:
        n = TelegramNotifier(token="tok", chat_id="")
        result = n.send("hello")
        assert result is False

    def test_returns_false_when_both_missing(self) -> None:
        n = TelegramNotifier(token="", chat_id="")
        result = n.send("hello")
        assert result is False

    def test_post_called_with_correct_url(self) -> None:
        n = TelegramNotifier(token="mytoken", chat_id="42")
        with patch("src.notifications.telegram_notifier.httpx") as mock_httpx:
            mock_httpx.post.return_value = _mock_httpx_ok()
            n.send("test")
        call_args = mock_httpx.post.call_args
        assert "mytoken" in call_args[0][0]   # URL contains token

    def test_post_body_has_parse_mode_html(self) -> None:
        n = _notifier()
        with patch("src.notifications.telegram_notifier.httpx") as mock_httpx:
            mock_httpx.post.return_value = _mock_httpx_ok()
            n.send("test")
        payload = mock_httpx.post.call_args[1]["json"]
        assert payload["parse_mode"] == "HTML"

    def test_reads_token_from_env(self) -> None:
        with patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "envtok", "TELEGRAM_CHAT_ID": "99"}):
            n = TelegramNotifier()
            assert n.token == "envtok"
            assert n.chat_id == "99"

    def test_raise_for_status_called(self) -> None:
        n = _notifier()
        resp = _mock_httpx_ok()
        with patch("src.notifications.telegram_notifier.httpx") as mock_httpx:
            mock_httpx.post.return_value = resp
            n.send("test")
        resp.raise_for_status.assert_called_once()


# ---------------------------------------------------------------------------
# Message formatters
# ---------------------------------------------------------------------------

class TestFmtVideoUploaded:
    def test_contains_title(self) -> None:
        msg = TelegramNotifier.fmt_video_uploaded(
            title="Rich People Don't Work Harder",
            youtube_url="https://youtube.com/watch?v=x",
            duration=13.5,
            topic="passive income",
        )
        assert "Rich People Don't Work Harder" in msg

    def test_contains_url(self) -> None:
        msg = TelegramNotifier.fmt_video_uploaded(
            title="T", youtube_url="https://youtube.com/watch?v=abc", duration=13.0, topic="t"
        )
        assert "https://youtube.com/watch?v=abc" in msg

    def test_contains_duration(self) -> None:
        msg = TelegramNotifier.fmt_video_uploaded(
            title="T", youtube_url="u", duration=13.5, topic="t"
        )
        assert "13.5" in msg

    def test_contains_topic(self) -> None:
        msg = TelegramNotifier.fmt_video_uploaded(
            title="T", youtube_url="u", duration=13.0, topic="stoic quotes"
        )
        assert "stoic quotes" in msg

    def test_custom_timestamp_used(self) -> None:
        msg = TelegramNotifier.fmt_video_uploaded(
            title="T", youtube_url="u", duration=0, topic="t", timestamp="2025-01-01 08:00"
        )
        assert "2025-01-01 08:00" in msg


class TestFmtElevenLabsWarning:
    def test_contains_chars_used(self) -> None:
        msg = TelegramNotifier.fmt_elevenlabs_warning(
            chars_used=20_000, monthly_limit=30_000, pct=66.7,
            videos_left=5, reset_date="April 1, 2025"
        )
        assert "20,000" in msg
        assert "30,000" in msg

    def test_contains_pct(self) -> None:
        msg = TelegramNotifier.fmt_elevenlabs_warning(
            chars_used=20_000, monthly_limit=30_000, pct=66.7,
            videos_left=5, reset_date="April 1"
        )
        assert "66.7%" in msg

    def test_contains_videos_left(self) -> None:
        msg = TelegramNotifier.fmt_elevenlabs_warning(
            chars_used=1, monthly_limit=30_000, pct=1.0, videos_left=12, reset_date="April 1"
        )
        assert "12" in msg

    def test_contains_reset_date(self) -> None:
        msg = TelegramNotifier.fmt_elevenlabs_warning(
            chars_used=1, monthly_limit=30_000, pct=1.0, videos_left=1, reset_date="May 5, 2025"
        )
        assert "May 5, 2025" in msg


class TestFmtElevenLabsCritical:
    def test_contains_chars_remaining(self) -> None:
        msg = TelegramNotifier.fmt_elevenlabs_critical(chars_remaining=1_500, reset_date="April 1")
        assert "1,500" in msg

    def test_contains_reset_date(self) -> None:
        msg = TelegramNotifier.fmt_elevenlabs_critical(chars_remaining=100, reset_date="June 1, 2025")
        assert "June 1, 2025" in msg

    def test_contains_elevenlabs_link(self) -> None:
        msg = TelegramNotifier.fmt_elevenlabs_critical(chars_remaining=100, reset_date="soon")
        assert "elevenlabs.io" in msg


class TestFmtYouTubeQuotaWarning:
    def test_contains_units_used(self) -> None:
        msg = TelegramNotifier.fmt_youtube_quota_warning(units_used=8_000, uploads_left=1)
        assert "8,000" in msg

    def test_contains_uploads_left(self) -> None:
        msg = TelegramNotifier.fmt_youtube_quota_warning(units_used=8_000, uploads_left=1)
        assert "1" in msg

    def test_mentions_reset_time(self) -> None:
        msg = TelegramNotifier.fmt_youtube_quota_warning(units_used=8_000, uploads_left=1)
        assert "08:00 WAT" in msg


class TestFmtYouTubeQuotaExceeded:
    def test_contains_queued_count(self) -> None:
        msg = TelegramNotifier.fmt_youtube_quota_exceeded(queued_videos=3)
        assert "3" in msg

    def test_mentions_reset(self) -> None:
        msg = TelegramNotifier.fmt_youtube_quota_exceeded(queued_videos=1)
        assert "08:00 WAT" in msg


class TestFmtProductionError:
    def test_contains_topic(self) -> None:
        msg = TelegramNotifier.fmt_production_error(
            topic="stoic_001", step_name="voiceover", error_message="API timeout"
        )
        assert "stoic_001" in msg

    def test_contains_step_name(self) -> None:
        msg = TelegramNotifier.fmt_production_error(
            topic="t", step_name="video_build", error_message="ffmpeg error"
        )
        assert "video_build" in msg

    def test_contains_error_message(self) -> None:
        msg = TelegramNotifier.fmt_production_error(
            topic="t", step_name="s", error_message="connection refused"
        )
        assert "connection refused" in msg

    def test_custom_timestamp_used(self) -> None:
        msg = TelegramNotifier.fmt_production_error(
            topic="t", step_name="s", error_message="err", timestamp="2025-01-01 09:00"
        )
        assert "2025-01-01 09:00" in msg


class TestFmtDailySummary:
    def _make(self, **kw) -> str:
        defaults = dict(
            date="2025-01-01",
            videos_uploaded=3,
            views_today=1_200,
            new_subs=15,
            total_subs=5_000,
            chars_used=18_000,
            chars_limit=30_000,
            pct_elevenlabs=60.0,
            units_used=4_800,
            next_run_time="01:00 WAT",
        )
        defaults.update(kw)
        return TelegramNotifier.fmt_daily_summary(**defaults)

    def test_contains_date(self) -> None:
        assert "2025-01-01" in self._make()

    def test_contains_videos_uploaded(self) -> None:
        assert "3" in self._make(videos_uploaded=3)

    def test_contains_views(self) -> None:
        assert "1,200" in self._make(views_today=1_200)

    def test_contains_subscriber_count(self) -> None:
        assert "5,000" in self._make(total_subs=5_000)

    def test_contains_elevenlabs_pct(self) -> None:
        assert "60.0%" in self._make(pct_elevenlabs=60.0)

    def test_contains_youtube_quota(self) -> None:
        assert "4,800" in self._make(units_used=4_800)

    def test_contains_next_run_time(self) -> None:
        assert "01:00 WAT" in self._make(next_run_time="01:00 WAT")

    def test_contains_running_smoothly(self) -> None:
        assert "running smoothly" in self._make()


class TestFmtSchedulerStarted:
    def test_contains_scrape_time(self) -> None:
        msg = TelegramNotifier.fmt_scheduler_started(
            next_scrape_time="00:00 WAT", next_production_time="01:00 WAT"
        )
        assert "00:00 WAT" in msg

    def test_contains_production_time(self) -> None:
        msg = TelegramNotifier.fmt_scheduler_started(
            next_scrape_time="00:00", next_production_time="01:00 WAT"
        )
        assert "01:00 WAT" in msg

    def test_custom_timestamp(self) -> None:
        msg = TelegramNotifier.fmt_scheduler_started(
            next_scrape_time="x", next_production_time="y", timestamp="2025-03-01 12:00"
        )
        assert "2025-03-01 12:00" in msg


class TestFmtHotLead:
    def test_contains_commenter_name(self) -> None:
        msg = TelegramNotifier.fmt_hot_lead(
            commenter_name="Jordan", video_title="VT", comment_text="CT", video_url="VU"
        )
        assert "Jordan" in msg

    def test_contains_video_title(self) -> None:
        msg = TelegramNotifier.fmt_hot_lead(
            commenter_name="J", video_title="Passive Income Guide", comment_text="C", video_url="U"
        )
        assert "Passive Income Guide" in msg

    def test_contains_comment_text(self) -> None:
        msg = TelegramNotifier.fmt_hot_lead(
            commenter_name="J", video_title="V", comment_text="build it for me", video_url="U"
        )
        assert "build it for me" in msg

    def test_contains_video_url(self) -> None:
        msg = TelegramNotifier.fmt_hot_lead(
            commenter_name="J", video_title="V", comment_text="C",
            video_url="https://youtube.com/watch?v=xyz"
        )
        assert "https://youtube.com/watch?v=xyz" in msg

    def test_contains_cta(self) -> None:
        msg = TelegramNotifier.fmt_hot_lead(
            commenter_name="J", video_title="V", comment_text="C", video_url="U"
        )
        assert "Reply personally" in msg


# ---------------------------------------------------------------------------
# notify_* convenience methods call send() with correct formatted message
# ---------------------------------------------------------------------------

class TestNotifyMethods:
    """Each notify_* method should call send() with the corresponding fmt_* output."""

    def _send_spy(self, n: TelegramNotifier):
        """Patch send() to capture what was sent and return True."""
        sent = []
        original_send = n.send
        def fake_send(msg):
            sent.append(msg)
            return True
        n.send = fake_send
        return sent

    def test_notify_video_uploaded(self) -> None:
        n = _notifier()
        captured = self._send_spy(n)
        n.notify_video_uploaded(title="T", youtube_url="U", duration=13.5, topic="tp")
        assert len(captured) == 1
        assert "T" in captured[0]
        assert "13.5" in captured[0]

    def test_notify_elevenlabs_warning(self) -> None:
        n = _notifier()
        captured = self._send_spy(n)
        n.notify_elevenlabs_warning(
            chars_used=20_000, monthly_limit=30_000, pct=66.7,
            videos_left=5, reset_date="April 1"
        )
        assert "20,000" in captured[0]

    def test_notify_elevenlabs_critical(self) -> None:
        n = _notifier()
        captured = self._send_spy(n)
        n.notify_elevenlabs_critical(chars_remaining=1_500, reset_date="April 1")
        assert "1,500" in captured[0]

    def test_notify_youtube_quota_warning(self) -> None:
        n = _notifier()
        captured = self._send_spy(n)
        n.notify_youtube_quota_warning(units_used=8_000, uploads_left=1)
        assert "8,000" in captured[0]

    def test_notify_youtube_quota_exceeded(self) -> None:
        n = _notifier()
        captured = self._send_spy(n)
        n.notify_youtube_quota_exceeded(queued_videos=2)
        assert "2" in captured[0]

    def test_notify_production_error(self) -> None:
        n = _notifier()
        captured = self._send_spy(n)
        n.notify_production_error(topic="t", step_name="hook", error_message="timeout")
        assert "hook" in captured[0]
        assert "timeout" in captured[0]

    def test_notify_daily_summary(self) -> None:
        n = _notifier()
        captured = self._send_spy(n)
        n.notify_daily_summary(
            date="2025-01-01", videos_uploaded=3, views_today=500, new_subs=10,
            total_subs=1_000, chars_used=5_000, chars_limit=30_000, pct_elevenlabs=16.7,
            units_used=3_200, next_run_time="01:00 WAT",
        )
        assert "2025-01-01" in captured[0]
        assert "running smoothly" in captured[0]

    def test_notify_scheduler_started(self) -> None:
        n = _notifier()
        captured = self._send_spy(n)
        n.notify_scheduler_started(
            next_scrape_time="00:00 WAT", next_production_time="01:00 WAT"
        )
        assert "ChannelForge Started" in captured[0]

    def test_notify_hot_lead(self) -> None:
        n = _notifier()
        captured = self._send_spy(n)
        n.notify_hot_lead(
            commenter_name="Jordan",
            video_title="Rich People Don't Work Harder",
            comment_text="Can you build this system for me?",
            video_url="https://youtube.com/watch?v=abc",
        )
        assert "Jordan" in captured[0]
        assert "Hot Lead" in captured[0]

"""
telegram_notifier.py — TelegramNotifier

Sends HTML-formatted messages to a Telegram chat via the Bot API.
All sends fail silently: if Telegram is unavailable or not configured
the pipeline continues uninterrupted.

Configuration (add to .env):
    TELEGRAM_BOT_TOKEN=<your_bot_token>
    TELEGRAM_CHAT_ID=<your_chat_id>

Usage:
    from src.notifications.telegram_notifier import TelegramNotifier

    n = TelegramNotifier()
    n.notify_video_uploaded(
        title="Rich People Don't Work Harder",
        youtube_url="https://www.youtube.com/watch?v=abc123",
        duration=13.5,
        topic="passive income myths",
    )
"""

import logging
import os
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_API_BASE = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramNotifier:
    """
    Sends Telegram messages via the Bot API using httpx.

    Args:
        token:   Bot token. Defaults to TELEGRAM_BOT_TOKEN env var.
        chat_id: Target chat / channel ID. Defaults to TELEGRAM_CHAT_ID env var.
    """

    def __init__(
        self,
        token: str | None = None,
        chat_id: str | None = None,
    ) -> None:
        self.token   = token   if token   is not None else os.getenv("TELEGRAM_BOT_TOKEN",  "")
        # Support multiple recipients via TELEGRAM_CHAT_IDS (comma-separated)
        # Falls back to single TELEGRAM_CHAT_ID for backwards compatibility
        if chat_id is not None:
            self.chat_ids = [chat_id]
        else:
            multi = os.getenv("TELEGRAM_CHAT_IDS", "")
            single = os.getenv("TELEGRAM_CHAT_ID", "")
            ids = multi if multi else single
            self.chat_ids = [c.strip() for c in ids.split(",") if c.strip()]
        self.chat_id = self.chat_ids[0] if self.chat_ids else ""

    # ------------------------------------------------------------------
    # Core send
    # ------------------------------------------------------------------

    def send(self, message: str) -> bool:
        """
        Send a raw HTML message to the configured Telegram chat.

        Returns True on success, False on any failure (never raises).
        Returns False silently if token or chat_id are not configured.
        """
        if not self.token or not self.chat_ids:
            logger.debug("Telegram not configured — skipping notification")
            return False
        success = False
        for cid in self.chat_ids:
            try:
                url = _API_BASE.format(token=self.token)
                payload = {
                    "chat_id":    cid,
                    "text":       message,
                    "parse_mode": "HTML",
                }
                response = httpx.post(url, json=payload, timeout=10.0)
                response.raise_for_status()
                logger.debug("Telegram sent to %s (%d chars)", cid, len(message))
                success = True
            except Exception as exc:
                logger.warning("Telegram failed for %s: %s", cid, exc)
        return success

        try:
            url = _API_BASE.format(token=self.token)
            payload = {
                "chat_id":    self.chat_id,
                "text":       message,
                "parse_mode": "HTML",
            }
            response = httpx.post(url, json=payload, timeout=10.0)
            response.raise_for_status()
            logger.debug("Telegram notification sent (%d chars)", len(message))
            return True

        except Exception as exc:
            logger.warning("Telegram notification failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Notification 1 — Video uploaded successfully
    # ------------------------------------------------------------------

    @staticmethod
    def fmt_video_uploaded(
        title: str,
        youtube_url: str,
        duration: float,
        topic: str,
        timestamp: str = "",
    ) -> str:
        ts = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        return (
            f"✅ <b>New Video Uploaded</b>\n"
            f"📺 {title}\n"
            f"🔗 {youtube_url}\n"
            f"⏱ {duration}s\n"
            f"📊 Topic: {topic}\n"
            f"🕐 {ts}"
        )

    def notify_video_uploaded(
        self,
        title: str,
        youtube_url: str,
        duration: float,
        topic: str,
    ) -> bool:
        return self.send(self.fmt_video_uploaded(title, youtube_url, duration, topic))

    # ------------------------------------------------------------------
    # Notification 2 — ElevenLabs usage warning (67 %)
    # ------------------------------------------------------------------

    @staticmethod
    def fmt_elevenlabs_warning(
        chars_used: int,
        monthly_limit: int,
        pct: float,
        videos_left: int,
        reset_date: str,
    ) -> str:
        return (
            f"⚠️ <b>ElevenLabs Usage Warning</b>\n"
            f"Used: {chars_used:,}/{monthly_limit:,} chars\n"
            f"Percentage: {pct:.1f}%\n"
            f"Videos remaining: ~{videos_left}\n"
            f"Reset date: {reset_date}"
        )

    def notify_elevenlabs_warning(
        self,
        chars_used: int,
        monthly_limit: int,
        pct: float,
        videos_left: int,
        reset_date: str,
    ) -> bool:
        return self.send(
            self.fmt_elevenlabs_warning(chars_used, monthly_limit, pct, videos_left, reset_date)
        )

    # ------------------------------------------------------------------
    # Notification 3 — ElevenLabs critical (95 %)
    # ------------------------------------------------------------------

    @staticmethod
    def fmt_elevenlabs_critical(
        chars_remaining: int,
        reset_date: str,
    ) -> str:
        return (
            f"🚨 <b>ElevenLabs CRITICAL</b>\n"
            f"Only {chars_remaining:,} chars remaining\n"
            f"Production will stop soon.\n"
            f"Reset date: {reset_date}\n"
            f"👉 Buy more credits at elevenlabs.io"
        )

    def notify_elevenlabs_critical(self, chars_remaining: int, reset_date: str) -> bool:
        return self.send(self.fmt_elevenlabs_critical(chars_remaining, reset_date))

    # ------------------------------------------------------------------
    # Notification 4 — YouTube quota warning (80 %)
    # ------------------------------------------------------------------

    @staticmethod
    def fmt_youtube_quota_warning(units_used: int, uploads_left: int) -> str:
        return (
            f"⚠️ <b>YouTube Quota Warning</b>\n"
            f"Used: {units_used:,}/10,000 units today\n"
            f"Uploads remaining today: ~{uploads_left}\n"
            f"Quota resets at 08:00 WAT"
        )

    def notify_youtube_quota_warning(self, units_used: int, uploads_left: int) -> bool:
        return self.send(self.fmt_youtube_quota_warning(units_used, uploads_left))

    # ------------------------------------------------------------------
    # Notification 5 — YouTube quota exceeded (100 %)
    # ------------------------------------------------------------------

    @staticmethod
    def fmt_youtube_quota_exceeded(queued_videos: int) -> str:
        return (
            f"🚫 <b>YouTube Quota Exceeded</b>\n"
            f"No more uploads today.\n"
            f"{queued_videos} video(s) queued for tomorrow.\n"
            f"Quota resets at 08:00 WAT"
        )

    def notify_youtube_quota_exceeded(self, queued_videos: int) -> bool:
        return self.send(self.fmt_youtube_quota_exceeded(queued_videos))

    # ------------------------------------------------------------------
    # Notification 6 — Production error
    # ------------------------------------------------------------------

    @staticmethod
    def fmt_production_error(
        topic: str,
        step_name: str,
        error_message: str,
        timestamp: str = "",
    ) -> str:
        ts = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        return (
            f"❌ <b>Production Error</b>\n"
            f"Topic: {topic}\n"
            f"Failed step: {step_name}\n"
            f"Error: {error_message}\n"
            f"🕐 {ts}"
        )

    def notify_production_error(
        self,
        topic: str,
        step_name: str,
        error_message: str,
    ) -> bool:
        return self.send(self.fmt_production_error(topic, step_name, error_message))

    # ------------------------------------------------------------------
    # Notification 7 — Daily summary
    # ------------------------------------------------------------------

    @staticmethod
    def fmt_daily_summary(
        date: str,
        videos_uploaded: int,
        views_today: int,
        new_subs: int,
        total_subs: int,
        chars_used: int,
        chars_limit: int,
        pct_elevenlabs: float,
        units_used: int,
        next_run_time: str,
    ) -> str:
        return (
            f"📊 <b>Daily Summary — {date}</b>\n\n"
            f"Videos uploaded: {videos_uploaded}\n"
            f"Total views today: {views_today:,}\n"
            f"New subscribers: {new_subs:,}\n"
            f"Total subscribers: {total_subs:,}\n\n"
            f"ElevenLabs: {chars_used:,}/{chars_limit:,} ({pct_elevenlabs:.1f}%)\n"
            f"YouTube quota: {units_used:,}/10,000\n\n"
            f"Next production: {next_run_time}\n"
            f"🤖 ChannelForge running smoothly"
        )

    def notify_daily_summary(
        self,
        date: str,
        videos_uploaded: int,
        views_today: int,
        new_subs: int,
        total_subs: int,
        chars_used: int,
        chars_limit: int,
        pct_elevenlabs: float,
        units_used: int,
        next_run_time: str,
    ) -> bool:
        return self.send(
            self.fmt_daily_summary(
                date, videos_uploaded, views_today, new_subs, total_subs,
                chars_used, chars_limit, pct_elevenlabs, units_used, next_run_time,
            )
        )

    # ------------------------------------------------------------------
    # Notification 8 — Scheduler started
    # ------------------------------------------------------------------

    @staticmethod
    def fmt_scheduler_started(
        next_scrape_time: str,
        next_production_time: str,
        timestamp: str = "",
    ) -> str:
        ts = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        return (
            f"🚀 <b>ChannelForge Started</b>\n"
            f"Scheduler is running.\n"
            f"Next scrape: {next_scrape_time}\n"
            f"Next production: {next_production_time}\n"
            f"🕐 {ts}"
        )

    def notify_scheduler_started(
        self,
        next_scrape_time: str,
        next_production_time: str,
    ) -> bool:
        return self.send(self.fmt_scheduler_started(next_scrape_time, next_production_time))

    # ------------------------------------------------------------------
    # Notification 9 — Hot lead alert
    # ------------------------------------------------------------------

    @staticmethod
    def fmt_hot_lead(
        commenter_name: str,
        video_title: str,
        comment_text: str,
        video_url: str,
    ) -> str:
        return (
            f"🔥 <b>Hot Lead Alert</b>\n"
            f"Someone commented SYSTEM on your video!\n"
            f"Channel: @{commenter_name}\n"
            f"Video: {video_title}\n"
            f"Comment: {comment_text}\n"
            f"🔗 {video_url}\n"
            f"👉 Reply personally to convert this lead!"
        )

    def notify_hot_lead(
        self,
        commenter_name: str,
        video_title: str,
        comment_text: str,
        video_url: str,
    ) -> bool:
        return self.send(self.fmt_hot_lead(commenter_name, video_title, comment_text, video_url))

    # ------------------------------------------------------------------
    # Notification 10 — New comment alert (approval flow)
    # ------------------------------------------------------------------

    @staticmethod
    def fmt_new_comment_alert(
        commenter: str,
        comment_text: str,
        video_title: str,
        comment_id: str,
        suggested_reply: str,
    ) -> str:
        return (
            f"\U0001f4ac <b>New Comment</b>\n"
            f"\U0001f464 {commenter}\n"
            f"\U0001f4fa {video_title}\n\n"
            f"Comment:\n<i>{comment_text}</i>\n\n"
            f"Suggested reply:\n<i>{suggested_reply}</i>\n\n"
            f"Commands:\n"
            f"/approve_{comment_id} \u2014 post as-is\n"
            f"/edit_{comment_id} \u2014 edit before posting\n"
            f"/skip_{comment_id} \u2014 skip this comment"
        )

    def send_new_comment_alert(
        self,
        commenter: str,
        comment_text: str,
        video_title: str,
        video_id: str,
        comment_id: str,
        suggested_reply: str,
    ) -> bool:
        """Send a comment alert and return True on success."""
        return self.send(
            self.fmt_new_comment_alert(
                commenter, comment_text, video_title, comment_id, suggested_reply,
            )
        )

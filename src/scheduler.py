"""
scheduler.py — APScheduler configuration for ChannelForge.

Registers four recurring jobs:

    run_all_channel_scrapers()    — every 6 h at 00:00, 06:00, 12:00, 18:00
    run_all_channel_production()  — every 6 h offset +1 h: 01:00, 07:00, 13:00, 19:00
    run_daily_analytics()         — daily at 00:30
    run_weekly_optimization()     — every Sunday at 02:00

Timezone is read from the UPLOAD_TIMEZONE environment variable
(default "Africa/Lagos").

Usage:
    from src.scheduler import build_scheduler, run_startup_tasks
    run_startup_tasks()     # seed topics + immediate scrape
    scheduler = build_scheduler()
    scheduler.start()       # blocks until Ctrl-C
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv

from config.constants import (
    ANALYTICS_HOUR,
    ANALYTICS_MINUTE,
    OPTIMIZATION_DAY_OF_WEEK,
    OPTIMIZATION_HOUR,
    OPTIMIZATION_MINUTE,
    PRODUCTION_HOURS,
    SCRAPER_HOURS,
)

load_dotenv()

logger = logging.getLogger(__name__)

_DEFAULT_DB = Path(os.getenv("DB_PATH", "data/processed/channel_forge.db"))


# ---------------------------------------------------------------------------
# Startup helpers
# ---------------------------------------------------------------------------


def seed_scored_topics_if_empty(db_path: Path | None = None) -> int:
    """
    Seed scored_topics with FALLBACK_TOPICS if the table is empty.

    Creates the table if it doesn't exist, then inserts all FALLBACK_TOPICS
    entries so that the production pipeline always has topics to work with
    even before the first scheduled scrape fires.

    Args:
        db_path: Path to the SQLite database. Defaults to DB_PATH env var or
                 ``data/processed/channel_forge.db``.

    Returns:
        Number of rows inserted (0 when the table was already populated).
    """
    import sqlite3

    from config.constants import FALLBACK_TOPICS

    target = db_path or _DEFAULT_DB
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        conn = sqlite3.connect(target)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scored_topics (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    keyword    TEXT    NOT NULL,
                    category   TEXT    NOT NULL DEFAULT 'success',
                    score      REAL    NOT NULL DEFAULT 0,
                    source     TEXT    NOT NULL DEFAULT 'manual',
                    used       INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT    NOT NULL DEFAULT (datetime('now'))
                )
            """)
            count = conn.execute("SELECT COUNT(*) FROM scored_topics").fetchone()[0]
            if count > 0:
                logger.info(
                    "[scheduler] scored_topics already has %d row(s) — skipping seed",
                    count,
                )
                return 0

            rows = [
                (keyword, category, 50.0, "FALLBACK")
                for category, keywords in FALLBACK_TOPICS.items()
                for keyword in keywords
            ]
            conn.executemany(
                "INSERT INTO scored_topics (keyword, category, score, source) VALUES (?, ?, ?, ?)",
                rows,
            )
            conn.commit()
            logger.info(
                "[scheduler] Seeded scored_topics with %d fallback topic(s)", len(rows)
            )
            return len(rows)
        finally:
            conn.close()
    except Exception as exc:
        logger.error("[scheduler] seed_scored_topics_if_empty failed: %s", exc)
        return 0


def run_startup_tasks(db_path: Path | None = None) -> None:
    """
    Run once immediately when the scheduler process starts.

    Steps:
      1. Seed scored_topics with FALLBACK_TOPICS if the table is empty, so
         production never fires with zero topics in the queue.
      2. Run all channel scrapers immediately — don't wait for the first
         scheduled cron hour.

    Args:
        db_path: Passed through to ``seed_scored_topics_if_empty``.
    """
    logger.info("[scheduler] Running startup tasks…")
    seed_scored_topics_if_empty(db_path)
    try:
        run_all_channel_scrapers()
    except Exception as exc:
        logger.error("[scheduler] Startup scrape failed (non-fatal): %s", exc)

    # Force-seed fallback topics if scored_topics is still empty after scraping
    # (handles datacenter IP blocks, e.g. Reddit 403 on Railway)
    import sqlite3 as _sqlite3

    from config.constants import FALLBACK_TOPICS

    target = db_path or _DEFAULT_DB
    try:
        conn = _sqlite3.connect(target)
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM scored_topics WHERE used=0"
            ).fetchone()[0]
            if count == 0:
                logger.warning(
                    "[scheduler] Scored topics still empty after scrape "
                    "— force seeding fallback topics"
                )
                rows = [
                    (keyword, category, 50.0, "fallback")
                    for category, keywords in FALLBACK_TOPICS.items()
                    for keyword in keywords
                ]
                conn.executemany(
                    "INSERT OR IGNORE INTO scored_topics "
                    "(keyword, category, score, source) VALUES (?,?,?,?)",
                    rows,
                )
                conn.commit()
                logger.info(
                    "[scheduler] Force seeded %d fallback topics", len(rows)
                )
        finally:
            conn.close()
    except Exception as exc:
        logger.error("[scheduler] Force-seed fallback check failed: %s", exc)

    # Check manual topic queue status
    try:
        conn = _sqlite3.connect(target)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS manual_topics (
                    seq INTEGER PRIMARY KEY, title TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'money',
                    hook_angle TEXT NOT NULL DEFAULT '', priority TEXT NOT NULL DEFAULT 'MEDIUM',
                    notes TEXT NOT NULL DEFAULT '', status TEXT NOT NULL DEFAULT 'QUEUED',
                    loaded_at TEXT NOT NULL DEFAULT (datetime('now')),
                    used_at TEXT, video_id TEXT NOT NULL DEFAULT ''
                )
            """)
            manual_count = conn.execute(
                "SELECT COUNT(*) FROM manual_topics WHERE status = 'QUEUED'"
            ).fetchone()[0]
            if manual_count > 0:
                logger.info("[startup] manual_topics: %d QUEUED topics ready", manual_count)
            else:
                logger.warning(
                    "[startup] manual_topics empty — production will use AI fallback queue"
                )
        finally:
            conn.close()
    except Exception as exc:
        logger.debug("[startup] manual_topics check failed: %s", exc)

    logger.info("[scheduler] Startup tasks complete")
    # Notification 8 — scheduler started
    try:
        from src.notifications.telegram_notifier import TelegramNotifier
        TelegramNotifier().notify_scheduler_started(
            next_scrape_time="next scheduled hour",
            next_production_time="next scheduled hour + 1",
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Job functions — must be importable top-level callables (not lambdas)
# ---------------------------------------------------------------------------


def run_all_channel_scrapers() -> None:
    """Fetch fresh trend signals and autocomplete topics for every configured channel."""
    start = datetime.now(timezone.utc)
    logger.info("[scheduler] run_all_channel_scrapers START %s", start.isoformat())
    try:
        from src.crawler.trend_scraper import TrendScrapingEngine  # lazy
        from src.crawler.competitor_scraper import CompetitorScraper  # lazy

        engine = TrendScrapingEngine()
        scraper = CompetitorScraper()
        from config.channels import CHANNELS  # lazy

        for channel in CHANNELS:
            category = getattr(channel, "category", "money")
            try:
                logger.info(
                    "[scheduler] Scraping trends for channel '%s'", channel.channel_key
                )
                engine.fetch_all(keywords=[category])
            except Exception as exc:
                logger.error(
                    "[scheduler] Scraper failed for channel '%s': %s",
                    channel.channel_key, exc,
                )

            # Autocomplete scraping runs alongside trends (every 6 h)
            try:
                ac_topics = scraper.scrape_search_autocomplete(category)
                logger.info(
                    "[scheduler] Autocomplete topics for '%s': %d",
                    channel.channel_key, len(ac_topics),
                )
            except Exception as exc:
                logger.error(
                    "[scheduler] Autocomplete scrape failed for '%s': %s",
                    channel.channel_key, exc,
                )
    except Exception as exc:
        logger.error("[scheduler] run_all_channel_scrapers ERROR: %s", exc)
    finally:
        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        logger.info(
            "[scheduler] run_all_channel_scrapers END (%.1fs)", elapsed
        )


def run_scraped_topic_processor() -> None:
    """Promote approved scraped topics into the production queue."""
    from src.crawler.scraped_topic_processor import ScrapedTopicProcessor
    start = datetime.now(timezone.utc)
    logger.info("[scheduler] run_scraped_topic_processor START %s", start.isoformat())
    try:
        processor = ScrapedTopicProcessor()
        stats = processor.run()
        logger.info("[scheduler] Scraped topic processor stats: %s", stats)
    except Exception as exc:
        logger.error("[scheduler] run_scraped_topic_processor ERROR: %s", exc)
    finally:
        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        logger.info("[scheduler] run_scraped_topic_processor END (%.1fs)", elapsed)

def run_all_channel_production() -> None:
    """Run the full production pipeline for every configured channel."""
    import sqlite3 as _sqlite3

    start = datetime.now(timezone.utc)
    logger.info(
        "[scheduler] run_all_channel_production START %s", start.isoformat()
    )

    # Check for empty topic queue before starting production
    try:
        conn = _sqlite3.connect(_DEFAULT_DB)
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM scored_topics WHERE used=0"
            ).fetchone()[0]
        finally:
            conn.close()
        if count == 0:
            next_time = "next scheduled scrape"
            logger.warning(
                "[scheduler] Empty topic queue — no topics available for production. "
                "Scrapers running now to refill queue."
            )
            try:
                from src.notifications.telegram_notifier import TelegramNotifier
                TelegramNotifier().send(
                    "⚠️ <b>Empty Topic Queue</b>\n\n"
                    "No topics available for production.\n"
                    "Scrapers running now to refill queue.\n"
                    f"Next attempt: {next_time}"
                )
            except Exception:
                pass
            run_all_channel_scrapers()
    except Exception as exc:
        logger.warning("[scheduler] Queue check failed (non-fatal): %s", exc)

    try:
        from src.pipeline.multi_channel_orchestrator import MultiChannelOrchestrator  # lazy

        orchestrator = MultiChannelOrchestrator()
        results = orchestrator.run_all()
        for r in results:
            logger.info(
                "[scheduler] Channel '%s': succeeded=%d failed=%d valid=%s",
                r.channel_key, r.topics_succeeded, r.topics_failed, r.is_valid,
            )
    except Exception as exc:
        logger.error("[scheduler] run_all_channel_production ERROR: %s", exc)
    finally:
        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        logger.info(
            "[scheduler] run_all_channel_production END (%.1fs)", elapsed
        )


def run_daily_analytics() -> None:
    """Track analytics for all uploaded videos across every channel."""
    start = datetime.now(timezone.utc)
    logger.info("[scheduler] run_daily_analytics START %s", start.isoformat())
    try:
        from src.analytics.analytics_tracker import AnalyticsTracker  # lazy
        from config.channels import CHANNELS  # lazy

        for channel in CHANNELS:
            try:
                tracker = AnalyticsTracker()
                results = tracker.track_all(channel_key=channel.channel_key)
                logger.info(
                    "[scheduler] Analytics for '%s': %d videos tracked",
                    channel.channel_key, len(results),
                )
                try:
                    from scripts.harvest_analytics import harvest  # lazy
                    harvest(channel=channel.channel_key)
                except Exception as harvest_exc:
                    logger.warning(
                        "[scheduler] Harvest failed for '%s': %s",
                        channel.channel_key, harvest_exc,
                    )
            except Exception as exc:
                logger.error(
                    "[scheduler] Analytics failed for channel '%s': %s",
                    channel.channel_key, exc,
                )
    except Exception as exc:
        logger.error("[scheduler] run_daily_analytics ERROR: %s", exc)
    finally:
        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        logger.info("[scheduler] run_daily_analytics END (%.1fs)", elapsed)


def run_elevenlabs_usage_check() -> None:
    """Check ElevenLabs monthly usage and log a prominent warning if above 67%."""
    start = datetime.now(timezone.utc)
    logger.info("[scheduler] run_elevenlabs_usage_check START %s", start.isoformat())
    try:
        from scripts.check_elevenlabs_usage import get_usage_report  # lazy
        report = get_usage_report()
        pct = report["pct_used"]
        logger.info(
            "[scheduler] ElevenLabs usage: %d/%d chars (%.1f%%) — %s",
            report["monthly_total"], report["monthly_limit"], pct, report["status"],
        )
        if pct >= 67:
            logger.warning(
                "===================================\n"
                "ELEVENLABS WARNING: %.0f%% used\n"
                "%d chars remaining\n"
                "~%d videos left\n"
                "===================================",
                pct,
                report["chars_remaining"],
                report["videos_remaining"],
            )
    except Exception as exc:
        logger.error("[scheduler] run_elevenlabs_usage_check ERROR: %s", exc)
    finally:
        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        logger.info("[scheduler] run_elevenlabs_usage_check END (%.1fs)", elapsed)


def run_competitor_research() -> None:
    """Scrape competitor channels and trending finance topics (daily at 07:30 WAT)."""
    start = datetime.now(timezone.utc)
    logger.info("[scheduler] run_competitor_research START %s", start.isoformat())
    try:
        from src.crawler.competitor_scraper import CompetitorScraper  # lazy
        from config.channels import CHANNELS  # lazy

        scraper = CompetitorScraper()
        for channel in CHANNELS:
            category = getattr(channel, "category", "money")
            try:
                topics = scraper.scrape_competitor_topics(category)
                logger.info(
                    "[scheduler] Competitor topics for '%s': %d",
                    channel.channel_key, len(topics),
                )
            except Exception as exc:
                logger.error(
                    "[scheduler] Competitor scrape failed for '%s': %s",
                    channel.channel_key, exc,
                )

        # Also scrape trending finance topics (shared across all channels)
        try:
            trending = scraper.scrape_trending_finance_topics()
            logger.info("[scheduler] Trending finance topics: %d", len(trending))
        except Exception as exc:
            logger.error("[scheduler] Trending finance scrape failed: %s", exc)

        # Trending search (recent high-view Shorts) — runs daily at 07:30 WAT
        try:
            trending_search = scraper.scrape_trending_search_topics()
            logger.info(
                "[scheduler] Trending search topics: %d", len(trending_search)
            )
        except Exception as exc:
            logger.error("[scheduler] Trending search scrape failed: %s", exc)

        # Sync scraped topics to Google Sheet
        try:
            import sqlite3 as _sq
            conn = _sq.connect(_DEFAULT_DB)
            try:
                topic_rows = conn.execute(
                    "SELECT source, channel_name, original_title, extracted_topic, "
                    "view_count, category, scraped_at "
                    "FROM competitor_topics "
                    "ORDER BY scraped_at DESC LIMIT 500"
                ).fetchall()
                cols = ["source", "channel_name", "original_title", "extracted_topic",
                        "view_count", "category", "scraped_at"]
                row_dicts = [
                    {**dict(zip(cols, r)), "score": ""}
                    for r in topic_rows
                ]
            finally:
                conn.close()
            from src.crawler.gsheet_topic_sync import GSheetTopicSync
            GSheetTopicSync().sync_scraped_topics(row_dicts)
        except Exception as sync_exc:
            logger.warning("[scheduler] Scraped topics GSheet sync failed: %s", sync_exc)

    except Exception as exc:
        logger.error("[scheduler] run_competitor_research ERROR: %s", exc)
    finally:
        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        logger.info("[scheduler] run_competitor_research END (%.1fs)", elapsed)


def run_reddit_scraper() -> None:
    """Scrape Reddit finance/career/success subreddits for fresh topics (every 6 h)."""
    start = datetime.now(timezone.utc)
    logger.info("[scheduler] run_reddit_scraper START %s", start.isoformat())
    try:
        from src.crawler.reddit_scraper import RedditScraper  # lazy

        scraper = RedditScraper()
        for category in ("money", "career", "success"):
            try:
                topics = scraper.scrape_finance_subreddits(category=category)
                logger.info(
                    "[scheduler] Reddit topics saved for '%s': %d",
                    category, len(topics),
                )
            except Exception as exc:
                logger.error(
                    "[scheduler] Reddit scrape failed for '%s': %s", category, exc
                )
    except Exception as exc:
        logger.error("[scheduler] run_reddit_scraper ERROR: %s", exc)
    finally:
        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        logger.info("[scheduler] run_reddit_scraper END (%.1fs)", elapsed)


def run_weekly_disk_cleanup() -> None:
    """Delete output files older than 7 days to prevent Railway disk from filling up."""
    import glob
    from datetime import timedelta

    start = datetime.now(timezone.utc)
    logger.info("[scheduler] run_weekly_disk_cleanup START %s", start.isoformat())
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).timestamp()
        patterns = [
            "data/output/*_final.mp4",
            "data/output/*_thumb.jpg",
        ]
        deleted = 0
        for pattern in patterns:
            for path in glob.glob(pattern):
                try:
                    if os.path.getmtime(path) < cutoff:
                        os.remove(path)
                        logger.info("[scheduler] Cleaned up: %s", path)
                        deleted += 1
                except Exception as exc:
                    logger.warning("[scheduler] Could not delete %s: %s", path, exc)
        logger.info("[scheduler] Disk cleanup complete: %d file(s) removed", deleted)
    except Exception as exc:
        logger.error("[scheduler] run_weekly_disk_cleanup ERROR: %s", exc)
    finally:
        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        logger.info("[scheduler] run_weekly_disk_cleanup END (%.1fs)", elapsed)


def run_weekly_optimization() -> None:
    """Analyze performance data and inject optimized topics (runs weekly)."""
    start = datetime.now(timezone.utc)
    logger.info(
        "[scheduler] run_weekly_optimization START %s", start.isoformat()
    )
    try:
        from src.optimizer.optimization_loop import OptimizationLoop  # lazy

        loop = OptimizationLoop()
        result = loop.run()
        logger.info(
            "[scheduler] Optimization: winners=%d losers=%d injected=%d valid=%s",
            result.winners_count, result.losers_count,
            result.topics_injected, result.is_valid,
        )
        if not result.is_valid:
            logger.error(
                "[scheduler] Optimization returned invalid: %s", result.error
            )
    except Exception as exc:
        logger.error("[scheduler] run_weekly_optimization ERROR: %s", exc)
    finally:
        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        logger.info(
            "[scheduler] run_weekly_optimization END (%.1fs)", elapsed
        )


def run_comment_check() -> None:
    """Check for new YouTube comments and send Telegram alerts (every 60 min)."""
    start = datetime.now(timezone.utc)
    logger.info("[scheduler] run_comment_check START %s", start.isoformat())
    try:
        # Quota gate — skip if daily quota is exhausted
        try:
            from src.publisher.youtube_uploader import QuotaTracker, QUOTA_UNITS  # lazy

            tracker = QuotaTracker(db_path=_DEFAULT_DB)
            if tracker.is_quota_exceeded():
                logger.info("[comments] Quota exhausted — skipping check")
                return
        except Exception as q_exc:
            logger.debug("[comments] Quota check skipped: %s", q_exc)
            tracker = None

        from src.publisher.comment_responder import CommentResponder  # lazy

        # Fetch recent comments from YouTube API
        comments: list[dict[str, str]] = []
        try:
            from googleapiclient.discovery import build as yt_build  # lazy
            from google.oauth2.credentials import Credentials  # lazy
            from config.channels import CHANNELS  # lazy
            import json

            base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
            creds_dir = base_dir / ".credentials"

            for channel in CHANNELS:
                token_path = creds_dir / f"{channel.channel_key}_token.json"
                if not token_path.exists():
                    continue
                try:
                    token_data = json.loads(token_path.read_text())
                    creds = Credentials.from_authorized_user_info(token_data)
                    service = yt_build("youtube", "v3", credentials=creds)

                    # Get channel's uploaded videos (1 quota unit)
                    ch_resp = service.channels().list(
                        part="contentDetails", mine=True
                    ).execute()
                    if tracker:
                        tracker.record("channels_list", QUOTA_UNITS["channels_list"])

                    uploads_id = (
                        ch_resp.get("items", [{}])[0]
                        .get("contentDetails", {})
                        .get("relatedPlaylists", {})
                        .get("uploads", "")
                    )
                    if not uploads_id:
                        continue

                    # Get recent videos (1 quota unit)
                    pl_resp = service.playlistItems().list(
                        part="snippet", playlistId=uploads_id, maxResults=10
                    ).execute()
                    if tracker:
                        tracker.record(
                            "playlist_items_list", QUOTA_UNITS["playlist_items_list"]
                        )

                    for item in pl_resp.get("items", []):
                        vid = item["snippet"]["resourceId"]["videoId"]
                        vtitle = item["snippet"]["title"]

                        # Check privacy status before fetching comments (1 quota unit)
                        try:
                            vid_resp = service.videos().list(
                                part="status", id=vid
                            ).execute()
                            if tracker:
                                tracker.record("videos_list", QUOTA_UNITS.get("videos_list", 1))
                            vid_items = vid_resp.get("items", [])
                            if vid_items:
                                privacy = vid_items[0].get("status", {}).get("privacyStatus", "public")
                                if privacy in ("private", "unlisted"):
                                    logger.debug(
                                        "[comment] Skipping %s — %s", vid, privacy,
                                    )
                                    continue
                        except Exception as priv_exc:
                            logger.debug(
                                "[comment] Privacy check failed for %s, proceeding: %s",
                                vid, priv_exc,
                            )

                        # Get comment threads for this video (1 quota unit per video)
                        try:
                            ct_resp = service.commentThreads().list(
                                part="snippet", videoId=vid,
                                maxResults=20, order="time",
                            ).execute()
                            if tracker:
                                tracker.record(
                                    "comment_threads_list",
                                    QUOTA_UNITS["comment_threads_list"],
                                )
                            for thread in ct_resp.get("items", []):
                                snippet = thread["snippet"]["topLevelComment"]["snippet"]
                                c_id = thread["snippet"]["topLevelComment"]["id"]
                                c_author = snippet.get("authorDisplayName", "")
                                logger.debug(
                                    "[comments] Found: video=%s comment=%s author=%r",
                                    vid, c_id, c_author,
                                )
                                comments.append({
                                    "comment_id": c_id,
                                    "video_id": vid,
                                    "commenter": c_author,
                                    "comment_text": snippet.get("textOriginal", ""),
                                    "video_title": vtitle,
                                    "category": getattr(channel, "category", "money"),
                                    "trigger_type": "GENERAL",
                                })
                        except Exception as ct_exc:
                            logger.warning(
                                "[scheduler] Comment fetch failed for video %s: %s",
                                vid, ct_exc,
                            )
                except Exception as ch_exc:
                    logger.error(
                        "[scheduler] Comment check failed for channel '%s': %s",
                        channel.channel_key, ch_exc,
                    )
        except Exception as exc:
            logger.warning("[scheduler] YouTube comment fetch unavailable: %s", exc)

        if comments:
            responder = CommentResponder()
            processed = responder.detect_and_alert(comments)
            logger.info(
                "[scheduler] Comment check: %d new comments found, %d processed",
                len(comments), len(processed),
            )
        else:
            logger.info("[scheduler] Comment check: no new comments")

    except Exception as exc:
        logger.error("[scheduler] run_comment_check ERROR: %s", exc)
    finally:
        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        logger.info("[scheduler] run_comment_check END (%.1fs)", elapsed)


def run_quota_recovery() -> None:
    """Retry pending uploads after YouTube quota resets at 08:00 WAT (runs at 08:05)."""
    import sqlite3 as _sqlite3

    start = datetime.now(timezone.utc)
    logger.info("[scheduler] run_quota_recovery START %s", start.isoformat())
    try:
        from src.publisher.youtube_uploader import YouTubeUploader, QuotaTracker  # lazy
        import json as _json

        db = _DEFAULT_DB
        if not db.exists():
            logger.info("[scheduler] quota_recovery: DB not found — nothing to retry")
            return

        conn = _sqlite3.connect(db)
        try:
            rows = conn.execute(
                "SELECT id, topic_id, channel_key, video_path, thumbnail_path, "
                "metadata_json, publish_at FROM pending_uploads WHERE status = 'pending'"
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            logger.info("[scheduler] quota_recovery: no pending uploads")
            return

        logger.info("[scheduler] quota_recovery: %d pending upload(s) to retry", len(rows))

        for row in rows:
            row_id, topic_id, channel_key, video_path, thumb_path, meta_json, publish_at = row
            try:
                metadata = _json.loads(meta_json)
                tracker = QuotaTracker(db_path=db)

                if tracker.is_quota_exceeded():
                    logger.warning(
                        "[scheduler] quota_recovery: quota already exhausted — "
                        "stopping retries"
                    )
                    break

                uploader = YouTubeUploader(
                    channel_key=channel_key,
                    quota_tracker=tracker,
                )
                result = uploader.upload(
                    topic_id=topic_id,
                    video_path=video_path,
                    metadata=metadata,
                    publish_at=publish_at or None,
                    thumbnail_path=thumb_path,
                )

                if result.is_valid:
                    # Mark as completed in DB
                    conn = _sqlite3.connect(db)
                    try:
                        conn.execute(
                            "UPDATE pending_uploads SET status = 'completed', "
                            "completed_at = datetime('now') WHERE id = ?",
                            (row_id,),
                        )
                        conn.commit()
                    finally:
                        conn.close()
                    logger.info(
                        "[scheduler] quota_recovery: uploaded %s -> %s",
                        topic_id, result.youtube_url,
                    )
                else:
                    # Increment retry count
                    conn = _sqlite3.connect(db)
                    try:
                        conn.execute(
                            "UPDATE pending_uploads SET retry_count = retry_count + 1 "
                            "WHERE id = ?",
                            (row_id,),
                        )
                        conn.commit()
                    finally:
                        conn.close()
                    logger.warning(
                        "[scheduler] quota_recovery: retry failed for %s: %s",
                        topic_id, result.validation_errors,
                    )
            except Exception as exc:
                logger.error(
                    "[scheduler] quota_recovery: error retrying %s: %s", topic_id, exc
                )
    except Exception as exc:
        logger.error("[scheduler] run_quota_recovery ERROR: %s", exc)
    finally:
        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        logger.info("[scheduler] run_quota_recovery END (%.1fs)", elapsed)


def run_daily_summary() -> None:
    """Send a daily summary Telegram notification at 23:00 WAT."""
    import sqlite3
    from datetime import date

    start = datetime.now(timezone.utc)
    logger.info("[scheduler] run_daily_summary START %s", start.isoformat())
    try:
        today_str = date.today().isoformat()
        db = _DEFAULT_DB

        # Videos uploaded today
        videos_uploaded = 0
        chars_used = 0
        units_used = 0
        try:
            conn = sqlite3.connect(db)
            try:
                row = conn.execute(
                    "SELECT COUNT(*) FROM production_results WHERE is_valid=1 AND completed_at LIKE ?",
                    (f"{today_str}%",),
                ).fetchone()
                videos_uploaded = int(row[0] or 0) if row else 0

                row = conn.execute(
                    "SELECT COALESCE(SUM(chars_used),0) FROM elevenlabs_usage WHERE date=?",
                    (today_str,),
                ).fetchone()
                chars_used = int(row[0] or 0) if row else 0

                row = conn.execute(
                    "SELECT COALESCE(SUM(units_used),0) FROM youtube_quota_usage WHERE date=?",
                    (today_str,),
                ).fetchone()
                units_used = int(row[0] or 0) if row else 0
            finally:
                conn.close()
        except Exception as db_exc:
            logger.warning("[scheduler] daily_summary DB query failed: %s", db_exc)

        monthly_limit = int(os.getenv("ELEVENLABS_MONTHLY_LIMIT", "30000"))
        pct_el = chars_used / monthly_limit * 100 if monthly_limit > 0 else 0.0

        from src.notifications.telegram_notifier import TelegramNotifier
        TelegramNotifier().notify_daily_summary(
            date=today_str,
            videos_uploaded=videos_uploaded,
            views_today=0,         # requires YouTube Analytics API; reported separately
            new_subs=0,
            total_subs=0,
            chars_used=chars_used,
            chars_limit=monthly_limit,
            pct_elevenlabs=pct_el,
            units_used=units_used,
            next_run_time="01:00 WAT",
        )
    except Exception as exc:
        logger.error("[scheduler] run_daily_summary ERROR: %s", exc)
    finally:
        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        logger.info("[scheduler] run_daily_summary END (%.1fs)", elapsed)


def run_topic_queue_sync() -> None:
    """Sync READY topics from Google Sheet into local manual_topics table.

    Runs weekly (Monday 05:00 WAT).  Silently skips if Google Sheet
    credentials are not configured.
    """
    import sqlite3 as _sqlite3

    start = datetime.now(timezone.utc)
    logger.info("[scheduler] run_topic_queue_sync START %s", start.isoformat())
    try:
        sheet_id = os.getenv("GOOGLE_SHEET_ID", "")
        creds_b64 = os.getenv("GOOGLE_CREDENTIALS_B64", "")

        if not sheet_id or not creds_b64:
            logger.warning("[topic_sync] Google Sheet not configured — skipping sync")
            return

        from src.crawler.gsheet_topic_sync import GSheetTopicSync  # lazy
        sync = GSheetTopicSync(sheet_id=sheet_id, credentials_b64=creds_b64)

        # Get last synced SEQ from settings
        db = _DEFAULT_DB
        conn = _sqlite3.connect(db)
        try:
            # Ensure tables exist
            conn.execute("""
                CREATE TABLE IF NOT EXISTS manual_topics (
                    seq INTEGER PRIMARY KEY, title TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'money',
                    hook_angle TEXT NOT NULL DEFAULT '',
                    priority TEXT NOT NULL DEFAULT 'MEDIUM',
                    notes TEXT NOT NULL DEFAULT '', status TEXT NOT NULL DEFAULT 'QUEUED',
                    loaded_at TEXT NOT NULL DEFAULT (datetime('now')),
                    used_at TEXT, video_id TEXT NOT NULL DEFAULT ''
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY, value TEXT,
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)
            conn.execute(
                "INSERT OR IGNORE INTO settings (key, value) VALUES ('last_manual_seq', '0')"
            )
            conn.commit()

            row = conn.execute(
                "SELECT value FROM settings WHERE key = 'last_manual_seq'"
            ).fetchone()
            last_seq = int(row[0]) if row else 0
        finally:
            conn.close()

        # Fetch next 28 READY topics from Sheet
        topics = sync.get_next_batch(last_seq=last_seq, count=28)

        if not topics:
            logger.warning(
                "[topic_sync] No READY topics after SEQ %d — AI queue will be used",
                last_seq,
            )
            try:
                from src.notifications.telegram_notifier import TelegramNotifier
                TelegramNotifier().send(
                    f"⚠️ Topic queue empty — no READY topics after SEQ {last_seq}. "
                    "Please add topics to Google Sheet."
                )
            except Exception:
                pass
            return

        # Insert into manual_topics
        conn = _sqlite3.connect(db)
        try:
            for t in topics:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO manual_topics
                    (seq, title, category, hook_angle, priority, notes, status)
                    VALUES (?, ?, ?, ?, ?, ?, 'QUEUED')
                    """,
                    (t["seq"], t["title"], t["category"],
                     t["hook_angle"], t["priority"], t["notes"]),
                )

            max_seq = max(t["seq"] for t in topics)
            conn.execute(
                "INSERT OR REPLACE INTO settings (key, value, updated_at) "
                "VALUES ('last_manual_seq', ?, datetime('now'))",
                (str(max_seq),),
            )
            conn.commit()
        finally:
            conn.close()

        # Update sync log on Sheet
        min_seq = min(t["seq"] for t in topics)
        max_seq = max(t["seq"] for t in topics)
        try:
            sync.update_sync_log(
                len(topics), min_seq, max_seq,
                "Auto-loaded by ChannelForge weekly sync",
            )
        except Exception as log_exc:
            logger.warning("[topic_sync] Sync log update failed: %s", log_exc)

        msg = (
            f"✅ Topic queue synced — {len(topics)} topics "
            f"loaded (SEQ {min_seq}→{max_seq})\n"
            f"Next: {topics[0]['title']}"
        )
        logger.info("[topic_sync] %s", msg)
        try:
            from src.notifications.telegram_notifier import TelegramNotifier
            TelegramNotifier().send(msg)
        except Exception:
            pass

    except Exception as exc:
        logger.error("[topic_sync] Sync failed: %s", exc)
        try:
            from src.notifications.telegram_notifier import TelegramNotifier
            TelegramNotifier().send(f"❌ Topic queue sync failed: {exc}")
        except Exception:
            pass
    finally:
        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        logger.info("[scheduler] run_topic_queue_sync END (%.1fs)", elapsed)


# ---------------------------------------------------------------------------
# Scheduler factory
# ---------------------------------------------------------------------------


def build_scheduler(timezone_name: str | None = None) -> BlockingScheduler:
    """
    Create and configure a BlockingScheduler with all four ChannelForge jobs.

    Args:
        timezone_name: Override UPLOAD_TIMEZONE env var (used in tests).

    Returns:
        A configured but not-yet-started BlockingScheduler.
    """
    # One-time startup check: warn if SEQ 6 is missing from manual_topics
    try:
        import sqlite3 as _sq_check
        if _DEFAULT_DB.exists():
            _conn = _sq_check.connect(_DEFAULT_DB)
            try:
                row = _conn.execute(
                    "SELECT seq FROM manual_topics WHERE seq = 6"
                ).fetchone()
                if row is None:
                    logger.warning(
                        "[WARNING] SEQ 6 missing from manual_topics — add via Telegram: "
                        "/addtopic money The 5 money habits of people who retire early"
                    )
            except _sq_check.OperationalError:
                pass
            finally:
                _conn.close()
    except Exception:
        pass

    tz_name = timezone_name or os.getenv("UPLOAD_TIMEZONE", "Africa/Lagos")
    tz = ZoneInfo(tz_name)

    scheduler = BlockingScheduler(timezone=tz)

    # --- Scraping: 00:00, 06:00, 12:00, 18:00 ---
    scheduler.add_job(
        run_all_channel_scrapers,
        trigger=CronTrigger(hour=SCRAPER_HOURS, minute=0, timezone=tz),
        id="scraper",
        name="Channel Scrapers",
        replace_existing=True,
        misfire_grace_time=300,
    )

    # --- Scraped topic processor: 30 min after scrapers (00:30, 06:30, 12:30, 18:30) ---
    scheduler.add_job(
        run_scraped_topic_processor,
        trigger=CronTrigger(hour=SCRAPER_HOURS, minute=30, timezone=tz),
        id="scraped_topic_processor",
        name="Scraped Topic Processor",
        replace_existing=True,
        misfire_grace_time=300,
    )
    # --- Production: 01:00, 07:00, 13:00, 19:00 ---
    scheduler.add_job(
        run_all_channel_production,
        trigger=CronTrigger(hour=PRODUCTION_HOURS, minute=0, timezone=tz),
        id="production",
        name="Channel Production",
        replace_existing=True,
        misfire_grace_time=300,
    )

    # --- Analytics: daily at 00:30 ---
    scheduler.add_job(
        run_daily_analytics,
        trigger=CronTrigger(
            hour=ANALYTICS_HOUR, minute=ANALYTICS_MINUTE, timezone=tz
        ),
        id="analytics",
        name="Daily Analytics",
        replace_existing=True,
        misfire_grace_time=300,
    )

    # --- ElevenLabs usage check: daily at 09:00 ---
    scheduler.add_job(
        run_elevenlabs_usage_check,
        trigger=CronTrigger(hour=9, minute=0, timezone=tz),
        id="elevenlabs_usage",
        name="ElevenLabs Usage Check",
        replace_existing=True,
        misfire_grace_time=300,
    )

    # --- Competitor research: daily at 06:30 UTC = 07:30 WAT (before 08:00 quota reset) ---
    scheduler.add_job(
        run_competitor_research,
        trigger=CronTrigger(hour=6, minute=30, timezone=tz),
        id="competitor_research",
        name="Competitor Research",
        replace_existing=True,
        misfire_grace_time=600,
    )

    # --- Reddit scraper: every 6 h at 02:00, 08:00, 14:00, 20:00 ---
    scheduler.add_job(
        run_reddit_scraper,
        trigger=CronTrigger(hour="2,8,14,20", minute=0, timezone=tz),
        id="reddit_scraper",
        name="Reddit Scraper",
        replace_existing=True,
        misfire_grace_time=300,
    )

    # --- Optimization: every Sunday at 02:00 ---
    scheduler.add_job(
        run_weekly_optimization,
        trigger=CronTrigger(
            day_of_week=OPTIMIZATION_DAY_OF_WEEK,
            hour=OPTIMIZATION_HOUR,
            minute=OPTIMIZATION_MINUTE,
            timezone=tz,
        ),
        id="optimization",
        name="Weekly Optimization",
        replace_existing=True,
        misfire_grace_time=600,
    )

    # --- Disk cleanup: every Sunday at 03:00 ---
    scheduler.add_job(
        run_weekly_disk_cleanup,
        trigger=CronTrigger(day_of_week="sun", hour=3, minute=0, timezone=tz),
        id="disk_cleanup",
        name="Weekly Disk Cleanup",
        replace_existing=True,
        misfire_grace_time=600,
    )

    # --- Daily summary: 23:00 WAT ---
    scheduler.add_job(
        run_daily_summary,
        trigger=CronTrigger(hour=23, minute=0, timezone=tz),
        id="daily_summary",
        name="Daily Summary",
        replace_existing=True,
        misfire_grace_time=300,
    )

    # --- Quota recovery: daily at 08:05 (5 min after YouTube quota reset) ---
    scheduler.add_job(
        run_quota_recovery,
        trigger=CronTrigger(hour=8, minute=5, timezone=tz),
        id="quota_recovery",
        name="Quota Recovery",
        replace_existing=True,
        misfire_grace_time=300,
    )

    # --- Comment check: every 60 minutes (hourly at :00) ---
    scheduler.add_job(
        run_comment_check,
        trigger=CronTrigger(hour="*", minute=0, timezone=tz),
        id="comment_check",
        name="Comment Check",
        replace_existing=True,
        misfire_grace_time=120,
    )

    # --- Topic Queue Sync: every Monday at 05:00 ---
    scheduler.add_job(
        run_topic_queue_sync,
        trigger=CronTrigger(day_of_week="mon", hour=5, minute=0, timezone=tz),
        id="topic_queue_sync",
        name="Topic Queue Sync",
        replace_existing=True,
        misfire_grace_time=600,
    )

    n_jobs = len(scheduler.get_jobs())
    logger.info("[scheduler] Built scheduler with timezone '%s', %d jobs", tz_name, n_jobs)
    return scheduler

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
    logger.info("[scheduler] Startup tasks complete")


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


def run_all_channel_production() -> None:
    """Run the full production pipeline for every configured channel."""
    start = datetime.now(timezone.utc)
    logger.info(
        "[scheduler] run_all_channel_production START %s", start.isoformat()
    )
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


def run_competitor_research() -> None:
    """Scrape competitor channels and trending finance topics (runs every 12 h)."""
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

        # Trending search (recent high-view Shorts) — runs every 12 h
        try:
            trending_search = scraper.scrape_trending_search_topics()
            logger.info(
                "[scheduler] Trending search topics: %d", len(trending_search)
            )
        except Exception as exc:
            logger.error("[scheduler] Trending search scrape failed: %s", exc)

    except Exception as exc:
        logger.error("[scheduler] run_competitor_research ERROR: %s", exc)
    finally:
        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        logger.info("[scheduler] run_competitor_research END (%.1fs)", elapsed)


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

    # --- Competitor research: every 12 h at 00:00 and 12:00 ---
    scheduler.add_job(
        run_competitor_research,
        trigger=CronTrigger(hour="0,12", minute=0, timezone=tz),
        id="competitor_research",
        name="Competitor Research",
        replace_existing=True,
        misfire_grace_time=600,
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

    n_jobs = len(scheduler.get_jobs())
    logger.info("[scheduler] Built scheduler with timezone '%s', %d jobs", tz_name, n_jobs)
    return scheduler

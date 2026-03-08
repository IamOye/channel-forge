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
    from src.scheduler import build_scheduler
    scheduler = build_scheduler()
    scheduler.start()   # blocks until Ctrl-C
"""

import logging
import os
from datetime import datetime, timezone
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


# ---------------------------------------------------------------------------
# Job functions — must be importable top-level callables (not lambdas)
# ---------------------------------------------------------------------------


def run_all_channel_scrapers() -> None:
    """Fetch fresh trend signals for every configured channel."""
    start = datetime.now(timezone.utc)
    logger.info("[scheduler] run_all_channel_scrapers START %s", start.isoformat())
    try:
        from src.crawler.trend_scraper import TrendScrapingEngine  # lazy

        engine = TrendScrapingEngine()
        from config.channels import CHANNELS  # lazy

        for channel in CHANNELS:
            try:
                logger.info(
                    "[scheduler] Scraping trends for channel '%s'", channel.channel_key
                )
                engine.fetch_all(keywords=[channel.category])
            except Exception as exc:
                logger.error(
                    "[scheduler] Scraper failed for channel '%s': %s",
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

    logger.info(
        "[scheduler] Built scheduler with timezone '%s', %d jobs",
        tz_name, len(scheduler.get_jobs()),
    )
    return scheduler

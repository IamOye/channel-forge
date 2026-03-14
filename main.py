"""
main.py — ChannelForge CLI entry point.

Commands:
    python main.py run               Start the full APScheduler (blocks)
    python main.py crawl <url>       Run a single trend crawl
    python main.py produce           Produce one video end-to-end (--topic required)
    python main.py test-pipeline     Dry run of the full pipeline with mocked outputs
    python main.py analytics         Run analytics manually for all uploaded videos
    python main.py optimize          Run optimization loop manually
    python main.py status            Show queue status, video counts, last run times
"""

import argparse
import logging
import os
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "main.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

DB_PATH = Path(os.getenv("DB_PATH", "data/processed/channel_forge.db"))


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------


def _check_ffmpeg() -> None:
    """Verify ffmpeg is accessible via imageio_ffmpeg; log path or error."""
    try:
        import imageio_ffmpeg
        path = imageio_ffmpeg.get_ffmpeg_exe()
        logger.info("ffmpeg ready at %s", path)
    except Exception as exc:
        logger.error(
            "ffmpeg not available via imageio_ffmpeg: %s — "
            "video normalization will be skipped. "
            "Install imageio-ffmpeg: pip install imageio-ffmpeg",
            exc,
        )


def cmd_run() -> int:
    """Start the blocking APScheduler. Press Ctrl-C to stop."""
    _check_ffmpeg()
    logger.info("Starting ChannelForge scheduler…")
    from src.scheduler import build_scheduler, run_startup_tasks  # lazy

    run_startup_tasks()  # seed fallback topics + immediate scrape
    scheduler = build_scheduler()
    try:
        logger.info("Scheduler running. Press Ctrl-C to exit.")
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped by user.")
    return 0


def cmd_crawl(url: str) -> int:
    """Run trend scraping for the given URL/keyword."""
    logger.info("Running trend crawl for: %s", url)
    try:
        from src.crawler.trend_scraper import TrendScrapingEngine  # lazy

        engine = TrendScrapingEngine()
        results = engine.fetch_all(keywords=[url])
        logger.info("Crawl complete. Signals fetched: %d", len(results))
        print(f"Crawl complete — {len(results)} signals fetched.")
        return 0
    except Exception as exc:
        logger.error("Crawl failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def cmd_produce(topic: str, channel: str = "money_debate") -> int:
    """Produce one video end-to-end for the given topic text."""
    from config.channels import CHANNELS  # resolve category from config

    channel_cfg = next((c for c in CHANNELS if c.channel_key == channel), None)
    category = channel_cfg.category if channel_cfg else "money"

    logger.info("Producing video for topic=%r channel=%s category=%s", topic, channel, category)
    print(f"Channel: {channel}  |  Category: {category}")
    try:
        from src.pipeline.production_pipeline import ProductionPipeline  # lazy

        pipeline = ProductionPipeline(youtube_channel_key=channel)
        result = pipeline.run({
            "topic_id": "cli_produce_001",
            "keyword":  topic,
            "category": category,
            "score":    80.0,
        })
        if result.is_valid:
            logger.info(
                "Production complete. Video ID: %s", result.youtube_video_id
            )
            print(f"Production complete — YouTube ID: {result.youtube_video_id}")
        else:
            logger.error("Production failed: %s", result.validation_errors)
            print(f"Production failed: {result.validation_errors}", file=sys.stderr)
            return 1
        return 0
    except Exception as exc:
        logger.error("Produce command failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def cmd_test_pipeline() -> int:
    """Dry run — validate the pipeline wiring without live API calls."""
    logger.info("Running test-pipeline dry run…")
    try:
        from unittest.mock import MagicMock, patch

        from src.pipeline.production_pipeline import ProductionPipeline

        mock_result = MagicMock()
        mock_result.is_valid = True
        mock_result.youtube_video_id = "DRY_RUN_NO_UPLOAD"
        mock_result.validation_errors = []

        pipeline = ProductionPipeline(
            anthropic_api_key="test",
            elevenlabs_api_key="test",
            pixabay_api_key="test",
        )

        with patch.object(pipeline, "run", return_value=mock_result):
            result = pipeline.run({
                "topic_id": "dryrun_001",
                "keyword":  "stoic wisdom",
                "category": "success",
                "score":    85.0,
            })

        print(f"Dry run OK — is_valid={result.is_valid}")
        logger.info("test-pipeline complete: is_valid=%s", result.is_valid)
        return 0
    except Exception as exc:
        logger.error("test-pipeline failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def cmd_analytics() -> int:
    """Run analytics manually for all uploaded videos across every channel."""
    logger.info("Running manual analytics…")
    try:
        from src.analytics.analytics_tracker import AnalyticsTracker  # lazy
        from config.channels import CHANNELS  # lazy

        total = 0
        for channel in CHANNELS:
            tracker = AnalyticsTracker()
            results = tracker.track_all(channel_key=channel.channel_key)
            total += len(results)
            logger.info(
                "Analytics for '%s': %d videos tracked",
                channel.channel_key, len(results),
            )

        print(f"Analytics complete — {total} videos tracked across {len(CHANNELS)} channel(s).")
        return 0
    except Exception as exc:
        logger.error("Analytics command failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def cmd_optimize() -> int:
    """Run the optimization loop manually."""
    logger.info("Running manual optimization loop…")
    try:
        from src.optimizer.optimization_loop import OptimizationLoop  # lazy

        loop = OptimizationLoop()
        result = loop.run()
        logger.info(
            "Optimization complete: winners=%d losers=%d injected=%d",
            result.winners_count, result.losers_count, result.topics_injected,
        )
        print(
            f"Optimization complete — "
            f"winners={result.winners_count} "
            f"losers={result.losers_count} "
            f"injected={result.topics_injected} "
            f"valid={result.is_valid}"
        )
        if not result.is_valid:
            print(f"Error: {result.error}", file=sys.stderr)
            return 1
        return 0
    except Exception as exc:
        logger.error("Optimize command failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def cmd_status() -> int:
    """Show queue status, uploaded video count, and last run times."""
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        print("Run `python scripts/init_db.py` first.")
        return 1

    conn = sqlite3.connect(DB_PATH)
    try:
        # Production queue
        try:
            queue_rows = conn.execute(
                "SELECT status, COUNT(*) FROM production_queue GROUP BY status"
            ).fetchall()
        except sqlite3.OperationalError:
            queue_rows = []

        # Uploaded videos
        try:
            video_count = conn.execute(
                "SELECT COUNT(*) FROM uploaded_videos"
            ).fetchone()[0]
        except sqlite3.OperationalError:
            video_count = 0

        # Last optimization run
        try:
            last_opt = conn.execute(
                "SELECT MAX(run_at) FROM optimization_log"
            ).fetchone()[0]
        except sqlite3.OperationalError:
            last_opt = None

        # Last analytics fetch
        try:
            last_analytics = conn.execute(
                "SELECT MAX(fetched_at) FROM video_metrics"
            ).fetchone()[0]
        except sqlite3.OperationalError:
            last_analytics = None

        # Scored topics count
        try:
            topic_count = conn.execute(
                "SELECT COUNT(*) FROM scored_topics"
            ).fetchone()[0]
        except sqlite3.OperationalError:
            topic_count = 0

    finally:
        conn.close()

    print("\n=== ChannelForge Status ===")
    print(f"Database       : {DB_PATH}")
    print(f"Uploaded videos: {video_count}")
    print(f"Scored topics  : {topic_count}")
    print(f"Last analytics : {last_analytics or 'never'}")
    print(f"Last optimization: {last_opt or 'never'}")
    print("\nProduction queue:")
    if queue_rows:
        for status, count in queue_rows:
            print(f"  {status:<12} {count}")
    else:
        print("  (empty)")
    print()
    return 0


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="channelforge",
        description="ChannelForge — automated faceless YouTube channel system",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # run
    sub.add_parser("run", help="Start the full APScheduler (blocks until Ctrl-C)")

    # crawl
    crawl_p = sub.add_parser("crawl", help="Run a single trend crawl")
    crawl_p.add_argument("url", help="URL or keyword to crawl")

    # produce
    produce_p = sub.add_parser("produce", help="Produce one video end-to-end")
    produce_p.add_argument(
        "--topic", required=True, metavar="TEXT",
        help='Topic text, e.g. "stoic morning routine"',
    )
    produce_p.add_argument(
        "--channel", default="money_debate", metavar="KEY",
        help="Channel key to use for upload credentials (default: money_debate)",
    )

    # test-pipeline
    sub.add_parser(
        "test-pipeline", help="Dry run of the full pipeline with mocked outputs"
    )

    # analytics
    sub.add_parser("analytics", help="Run analytics manually for all uploaded videos")

    # optimize
    sub.add_parser("optimize", help="Run the optimization loop manually")

    # status
    sub.add_parser("status", help="Show queue status, video counts, last run times")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    dispatch = {
        "run":           lambda: cmd_run(),
        "crawl":         lambda: cmd_crawl(args.url),
        "produce":       lambda: cmd_produce(args.topic, args.channel),
        "test-pipeline": lambda: cmd_test_pipeline(),
        "analytics":     lambda: cmd_analytics(),
        "optimize":      lambda: cmd_optimize(),
        "status":        lambda: cmd_status(),
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler()


if __name__ == "__main__":
    sys.exit(main())

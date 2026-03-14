"""
multi_channel_orchestrator.py — MultiChannelOrchestrator

Runs the full production pipeline independently for every channel defined
in config/channels.py. One channel failure does NOT stop other channels.

Each channel gets:
  - Isolated output directory:  data/output/{channel_key}/
  - Per-channel SQLite database: data/processed/{channel_key}.db
  - Per-channel daily_quota respected (via scored_topics limit)

Usage:
    orchestrator = MultiChannelOrchestrator()
    results = orchestrator.run_all()
    for r in results:
        print(r.channel_key, r.topics_succeeded, r.topics_failed)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ChannelRunResult:
    """Outcome of running the pipeline for one channel."""

    channel_key: str
    channel_name: str
    topics_processed: int = 0
    topics_succeeded: int = 0
    topics_failed: int = 0
    is_valid: bool = True
    error: str = ""
    completed_at: str = ""

    def __post_init__(self) -> None:
        if not self.completed_at:
            self.completed_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel_key":      self.channel_key,
            "channel_name":     self.channel_name,
            "topics_processed": self.topics_processed,
            "topics_succeeded": self.topics_succeeded,
            "topics_failed":    self.topics_failed,
            "is_valid":         self.is_valid,
            "error":            self.error,
            "completed_at":     self.completed_at,
        }


# ---------------------------------------------------------------------------
# MultiChannelOrchestrator
# ---------------------------------------------------------------------------

class MultiChannelOrchestrator:
    """
    Orchestrates the full production pipeline across multiple YouTube channels.

    Loads channel configurations from config.channels.CHANNELS and runs each
    channel's pipeline independently. A failure in one channel does not affect
    the others. Each channel gets isolated output and database paths.

    Args:
        anthropic_api_key:  Anthropic API key (or reads from env).
        elevenlabs_api_key: ElevenLabs API key (or reads from env).
        pixabay_api_key:    Pixabay API key (or reads from env).
        topics_per_channel: Max topics to process per channel per run.
        base_output_dir:    Root output directory (default "data/output").
        base_db_dir:        Root database directory (default "data/processed").
    """

    def __init__(
        self,
        anthropic_api_key: str | None = None,
        elevenlabs_api_key: str | None = None,
        pixabay_api_key: str | None = None,
        topics_per_channel: int = 3,
        base_output_dir: str | Path = "data/output",
        base_db_dir: str | Path = "data/processed",
    ) -> None:
        self.anthropic_api_key  = anthropic_api_key
        self.elevenlabs_api_key = elevenlabs_api_key
        self.pixabay_api_key    = pixabay_api_key
        self.topics_per_channel = topics_per_channel
        self.base_output_dir    = Path(base_output_dir)
        self.base_db_dir        = Path(base_db_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all(self) -> list[ChannelRunResult]:
        """
        Run the pipeline for every channel in config.channels.CHANNELS.

        Returns:
            List of ChannelRunResult, one per channel. Never raises —
            individual channel failures are captured in ChannelRunResult.error.
        """
        from config.channels import CHANNELS  # lazy — allows test mocking
        channels = CHANNELS

        if not channels:
            logger.warning("No channels configured in config/channels.py")
            return []

        logger.info("Running pipeline for %d channel(s)", len(channels))
        results: list[ChannelRunResult] = []
        for channel_cfg in channels:
            result = self.run_channel(channel_cfg)
            results.append(result)
            logger.info(
                "Channel '%s' done: succeeded=%d failed=%d",
                channel_cfg.channel_key,
                result.topics_succeeded,
                result.topics_failed,
            )
        return results

    def run_channel(self, channel_cfg) -> ChannelRunResult:
        """
        Run the production pipeline for a single channel configuration.

        Args:
            channel_cfg: A ChannelConfig from config.channels.

        Returns:
            ChannelRunResult. Never raises — exceptions are caught and stored
            in ChannelRunResult.error.
        """
        channel_key  = channel_cfg.channel_key
        channel_name = channel_cfg.name
        daily_quota  = getattr(channel_cfg, "daily_quota", self.topics_per_channel)
        db_path      = self.base_db_dir / f"{channel_key}.db"
        output_dir   = self.base_output_dir / channel_key

        logger.info(
            "Starting channel '%s' (%s) — quota=%d output=%s",
            channel_key, channel_name, daily_quota, output_dir,
        )

        result = ChannelRunResult(channel_key=channel_key, channel_name=channel_name)

        try:
            self._setup_channel_output(output_dir)
            topics = self._load_topics(channel_cfg, daily_quota)
            result.topics_processed = len(topics)

            pipeline = self._build_pipeline(channel_cfg, db_path)

            for topic_item in topics:
                try:
                    run_result = pipeline.run(topic_item)
                    if run_result.is_valid:
                        result.topics_succeeded += 1
                    else:
                        result.topics_failed += 1
                        logger.warning(
                            "Topic '%s' failed in channel '%s': %s",
                            topic_item.get("topic_id", "?"),
                            channel_key,
                            run_result.validation_errors,
                        )
                except Exception as exc:
                    result.topics_failed += 1
                    logger.error(
                        "Unhandled error for topic in channel '%s': %s",
                        channel_key, exc,
                    )

        except Exception as exc:
            logger.error("Channel '%s' setup failed: %s", channel_key, exc)
            result.is_valid = False
            result.error = str(exc)

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_topics(self, channel_cfg, max_topics: int) -> list[dict[str, Any]]:
        """
        Load up to max_topics unused topics for a channel, ordered by priority.

        Uses TopicQueue which:
          1. Queries competitor_topics + scored_topics across all sources
          2. Orders by SOURCE_PRIORITIES (VIEWER_REQUESTED first, FALLBACK last)
          3. Deduplicates against upload history (difflib similarity > 70%)
          4. Falls back to FALLBACK_TOPICS constants when DB is empty
          5. Calls Claude as a last resort to generate a fresh topic

        Also ensures the scored_topics table exists in the target DB.
        """
        import sqlite3

        from src.pipeline.topic_queue import TopicQueue

        channel_key = channel_cfg.channel_key
        category    = getattr(channel_cfg, "category", "money")

        db_path   = self.base_db_dir / f"{channel_key}.db"
        main_db   = self.base_db_dir / "channel_forge.db"
        db_to_use = db_path if db_path.exists() else main_db

        # Ensure scored_topics table exists in whichever DB we'll use
        try:
            db_to_use.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(db_to_use)
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS scored_topics (
                        id         INTEGER PRIMARY KEY AUTOINCREMENT,
                        keyword    TEXT    NOT NULL,
                        category   TEXT    NOT NULL DEFAULT 'success',
                        score      REAL    NOT NULL DEFAULT 0,
                        source     TEXT    NOT NULL DEFAULT 'manual',
                        created_at TEXT    NOT NULL DEFAULT (datetime('now'))
                    )
                """)
                conn.commit()
            finally:
                conn.close()
        except Exception as exc:
            logger.warning("Could not ensure scored_topics table for '%s': %s", channel_key, exc)

        # Build topic queue backed by the channel's DB
        queue = TopicQueue(db_path=db_to_use)
        uploaded = queue.get_uploaded_topics()

        raw_topics = queue.get_next_topics(
            category=category,
            max_count=max_topics,
            uploaded_topics=uploaded,
        )

        # Stamp with stable topic_ids for this channel
        topics: list[dict[str, Any]] = []
        for i, t in enumerate(raw_topics):
            topics.append({
                "topic_id": t.get("topic_id") or f"{channel_key}_topic_{i:03d}",
                "keyword":  t["keyword"],
                "category": t.get("category", category),
                "score":    t.get("score", 50.0),
            })

        if not topics:
            logger.warning("No topics available for channel '%s'", channel_key)

        return topics

    def _build_pipeline(self, channel_cfg, db_path: Path):
        """Instantiate ProductionPipeline for the given channel."""
        from src.pipeline.production_pipeline import ProductionPipeline  # lazy
        return ProductionPipeline(
            anthropic_api_key=self.anthropic_api_key,
            elevenlabs_api_key=self.elevenlabs_api_key,
            pixabay_api_key=self.pixabay_api_key,
            youtube_channel_key=channel_cfg.channel_key,
            db_path=db_path,
        )

    def _setup_channel_output(self, output_dir: Path) -> None:
        """Create the channel's isolated output directory."""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.debug("Channel output dir ready: %s", output_dir)
        except Exception as exc:
            logger.warning("Could not create output dir %s: %s", output_dir, exc)

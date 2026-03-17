"""
init_db.py — Initialize all SQLite databases for ChannelForge.

Creates:
  data/processed/channel_forge.db   — main unified database
  data/processed/trends.db          — raw trend signals
  data/processed/safety.db          — safety filter decisions
  data/processed/scores.db          — engagement scores

Run:
  python scripts/init_db.py
"""

import logging
import sqlite3
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("init_db")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
LOGS_DIR = PROJECT_ROOT / "logs"


# ---------------------------------------------------------------------------
# Schema definitions — Phases 1–10
# ---------------------------------------------------------------------------

TRENDS_DDL = """
CREATE TABLE IF NOT EXISTS trend_signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    keyword         TEXT    NOT NULL,
    source          TEXT    NOT NULL,           -- 'google' | 'youtube'
    region          TEXT    NOT NULL DEFAULT 'US',
    interest_score  REAL    NOT NULL DEFAULT 0,
    related_query   TEXT,                       -- JSON array of related queries
    fetched_at      TEXT    NOT NULL,           -- ISO-8601 UTC
    raw_json        TEXT                        -- full API response
);

CREATE INDEX IF NOT EXISTS idx_trend_keyword ON trend_signals (keyword);
CREATE INDEX IF NOT EXISTS idx_trend_source  ON trend_signals (source);
CREATE INDEX IF NOT EXISTS idx_trend_fetched ON trend_signals (fetched_at);
"""

SAFETY_DDL = """
CREATE TABLE IF NOT EXISTS safety_decisions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    keyword         TEXT    NOT NULL,
    is_safe         INTEGER NOT NULL,           -- 1 = safe, 0 = blocked
    block_reason    TEXT,                       -- NULL if safe
    method          TEXT    NOT NULL,           -- 'blocklist' | 'claude_api'
    confidence      REAL    NOT NULL DEFAULT 1.0,
    checked_at      TEXT    NOT NULL            -- ISO-8601 UTC
);

CREATE INDEX IF NOT EXISTS idx_safety_keyword ON safety_decisions (keyword);
CREATE INDEX IF NOT EXISTS idx_safety_safe    ON safety_decisions (is_safe);
"""

SCORES_DDL = """
CREATE TABLE IF NOT EXISTS engagement_scores (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    keyword         TEXT    NOT NULL,
    title_score     REAL    NOT NULL DEFAULT 0, -- 0–100
    trend_score     REAL    NOT NULL DEFAULT 0,
    competition_score REAL  NOT NULL DEFAULT 0,
    monetization_score REAL NOT NULL DEFAULT 0,
    composite_score REAL    NOT NULL DEFAULT 0, -- weighted average
    rationale       TEXT,                       -- Claude explanation
    scored_at       TEXT    NOT NULL            -- ISO-8601 UTC
);

CREATE INDEX IF NOT EXISTS idx_score_keyword   ON engagement_scores (keyword);
CREATE INDEX IF NOT EXISTS idx_score_composite ON engagement_scores (composite_score DESC);
"""

TITLES_DDL = """
CREATE TABLE IF NOT EXISTS titles (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    url         TEXT    NOT NULL,
    title       TEXT    NOT NULL,
    heading     TEXT,
    frequency   INTEGER NOT NULL DEFAULT 1,
    relevance   REAL    NOT NULL DEFAULT 0,
    crawled_at  TEXT    NOT NULL                -- ISO-8601 UTC
);

CREATE INDEX IF NOT EXISTS idx_titles_url   ON titles (url);
CREATE INDEX IF NOT EXISTS idx_titles_freq  ON titles (frequency DESC);
"""

VIDEO_IDEAS_DDL = """
CREATE TABLE IF NOT EXISTS video_ideas (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    keyword         TEXT    NOT NULL,
    video_title     TEXT    NOT NULL,
    description     TEXT,
    tags            TEXT,                       -- JSON array
    script_outline  TEXT,
    thumbnail_text  TEXT,
    composite_score REAL    NOT NULL DEFAULT 0,
    generated_at    TEXT    NOT NULL            -- ISO-8601 UTC
);

CREATE INDEX IF NOT EXISTS idx_video_keyword ON video_ideas (keyword);
CREATE INDEX IF NOT EXISTS idx_video_score   ON video_ideas (composite_score DESC);
"""

# ---------------------------------------------------------------------------
# Schema definitions — Phases 11–13
# ---------------------------------------------------------------------------

SCORED_TOPICS_DDL = """
CREATE TABLE IF NOT EXISTS scored_topics (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    keyword     TEXT    NOT NULL,
    category    TEXT    NOT NULL DEFAULT 'success',
    score       REAL    NOT NULL DEFAULT 0,
    source      TEXT    NOT NULL DEFAULT 'manual',
    used        INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_scored_keyword ON scored_topics (keyword);
CREATE INDEX IF NOT EXISTS idx_scored_score   ON scored_topics (score DESC);
CREATE INDEX IF NOT EXISTS idx_scored_used    ON scored_topics (used);
"""

UPLOADED_VIDEOS_DDL = """
CREATE TABLE IF NOT EXISTS uploaded_videos (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id    TEXT    NOT NULL UNIQUE,
    channel_key TEXT    NOT NULL DEFAULT 'default',
    topic_id    TEXT    NOT NULL DEFAULT '',
    title       TEXT    NOT NULL DEFAULT '',
    uploaded_at TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_uploaded_video   ON uploaded_videos (video_id);
CREATE INDEX IF NOT EXISTS idx_uploaded_channel ON uploaded_videos (channel_key);
"""

VIDEO_METRICS_DDL = """
CREATE TABLE IF NOT EXISTS video_metrics (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id            TEXT    NOT NULL,
    channel_key         TEXT    NOT NULL DEFAULT 'default',
    views               INTEGER NOT NULL DEFAULT 0,
    watch_time_minutes  REAL    NOT NULL DEFAULT 0,
    likes               INTEGER NOT NULL DEFAULT 0,
    comments            INTEGER NOT NULL DEFAULT 0,
    shares              INTEGER NOT NULL DEFAULT 0,
    impressions         INTEGER NOT NULL DEFAULT 0,
    ctr                 REAL    NOT NULL DEFAULT 0,
    subscribers_gained  INTEGER NOT NULL DEFAULT 0,
    subscribers_lost    INTEGER NOT NULL DEFAULT 0,
    engagement_rate     REAL    NOT NULL DEFAULT 0,
    virality_score      REAL    NOT NULL DEFAULT 0,
    tier                TEXT    NOT NULL DEFAULT 'F',
    fetched_at          TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_metrics_video   ON video_metrics (video_id);
CREATE INDEX IF NOT EXISTS idx_metrics_tier    ON video_metrics (tier);
CREATE INDEX IF NOT EXISTS idx_metrics_fetched ON video_metrics (fetched_at);
"""

OPTIMIZATION_LOG_DDL = """
CREATE TABLE IF NOT EXISTS optimization_log (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    winners_count     INTEGER NOT NULL DEFAULT 0,
    losers_count      INTEGER NOT NULL DEFAULT 0,
    topics_generated  INTEGER NOT NULL DEFAULT 0,
    topics_injected   INTEGER NOT NULL DEFAULT 0,
    pattern_analysis  TEXT    NOT NULL DEFAULT '',
    is_valid          INTEGER NOT NULL DEFAULT 1,
    error             TEXT    NOT NULL DEFAULT '',
    run_at            TEXT    NOT NULL
);
"""

PRODUCTION_RESULTS_DDL = """
CREATE TABLE IF NOT EXISTS production_results (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id         TEXT    NOT NULL,
    keyword          TEXT    NOT NULL,
    category         TEXT    NOT NULL DEFAULT 'success',
    hook             TEXT    NOT NULL DEFAULT '',
    script           TEXT    NOT NULL DEFAULT '',
    voiceover_path   TEXT    NOT NULL DEFAULT '',
    video_path       TEXT    NOT NULL DEFAULT '',
    youtube_video_id TEXT    NOT NULL DEFAULT '',
    is_valid         INTEGER NOT NULL DEFAULT 0,
    validation_errors TEXT   NOT NULL DEFAULT '[]',
    created_at       TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_results_topic ON production_results (topic_id);
CREATE INDEX IF NOT EXISTS idx_results_valid ON production_results (is_valid);
"""

COMPETITOR_TOPICS_DDL = """
CREATE TABLE IF NOT EXISTS competitor_topics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_name    TEXT    NOT NULL,
    original_title  TEXT    NOT NULL,
    extracted_topic TEXT    NOT NULL,
    view_count      INTEGER NOT NULL DEFAULT 0,
    category        TEXT    NOT NULL DEFAULT 'money',
    source          TEXT    NOT NULL DEFAULT 'competitor',
    used            INTEGER NOT NULL DEFAULT 0,          -- 1 = already produced
    scraped_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_comp_category  ON competitor_topics (category);
CREATE INDEX IF NOT EXISTS idx_comp_views     ON competitor_topics (view_count DESC);
CREATE INDEX IF NOT EXISTS idx_comp_used      ON competitor_topics (used);
"""

YOUTUBE_QUOTA_DDL = """
CREATE TABLE IF NOT EXISTS youtube_quota_usage (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    date             TEXT    NOT NULL,           -- YYYY-MM-DD UTC
    operation        TEXT    NOT NULL,           -- 'video_upload' | 'thumbnail_upload' | 'metadata_update'
    units_used       INTEGER NOT NULL,
    cumulative_daily INTEGER NOT NULL,
    created_at       TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_quota_date ON youtube_quota_usage (date);
"""

ELEVENLABS_USAGE_DDL = """
CREATE TABLE IF NOT EXISTS elevenlabs_usage (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    date          TEXT    NOT NULL,
    topic_id      TEXT    NOT NULL,
    chars_used    INTEGER NOT NULL,
    voice_name    TEXT    NOT NULL,
    monthly_total INTEGER DEFAULT 0,
    pct_used      REAL    DEFAULT 0,
    created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_usage_date    ON elevenlabs_usage (date);
CREATE INDEX IF NOT EXISTS idx_usage_topic   ON elevenlabs_usage (topic_id);
"""

PRODUCTION_LOCK_DDL = """
CREATE TABLE IF NOT EXISTS production_lock (
    id        INTEGER PRIMARY KEY CHECK (id = 1),
    locked    INTEGER NOT NULL DEFAULT 0,
    locked_at TEXT,
    locked_by TEXT
);
INSERT OR IGNORE INTO production_lock (id, locked) VALUES (1, 0);
"""

# Unified main DB gets all tables
MAIN_DDL_PARTS = [
    TRENDS_DDL, SAFETY_DDL, SCORES_DDL, TITLES_DDL, VIDEO_IDEAS_DDL,
    SCORED_TOPICS_DDL, UPLOADED_VIDEOS_DDL, VIDEO_METRICS_DDL,
    OPTIMIZATION_LOG_DDL, PRODUCTION_RESULTS_DDL, COMPETITOR_TOPICS_DDL,
    ELEVENLABS_USAGE_DDL, YOUTUBE_QUOTA_DDL, PRODUCTION_LOCK_DDL,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_db(db_path: Path, ddl_statements: list[str]) -> None:
    """Create a SQLite database and apply DDL statements."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.executescript("\n".join(ddl_statements))
        conn.commit()
        logger.info("Initialized: %s", db_path)
    finally:
        conn.close()


def verify_db(db_path: Path) -> list[str]:
    """Return list of table names in the database."""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


def migrate_db(db_path: Path) -> None:
    """
    Apply incremental schema migrations to an existing database.

    Safe to run on any ChannelForge DB — each migration is idempotent.
    """
    if not db_path.exists():
        return
    conn = sqlite3.connect(db_path)
    try:
        # Phase 12 migration: add `used` column to scored_topics
        try:
            conn.execute(
                "ALTER TABLE scored_topics ADD COLUMN used INTEGER DEFAULT 0"
            )
            conn.commit()
            logger.info("Migration applied: scored_topics.used column added (%s)", db_path)
        except sqlite3.OperationalError:
            pass  # column already exists

        # ElevenLabs usage migration: add monthly_total and pct_used columns
        for col, coltype in [("monthly_total", "INTEGER DEFAULT 0"), ("pct_used", "REAL DEFAULT 0")]:
            try:
                conn.execute(f"ALTER TABLE elevenlabs_usage ADD COLUMN {col} {coltype}")
                conn.commit()
                logger.info("Migration applied: elevenlabs_usage.%s column added (%s)", col, db_path)
            except sqlite3.OperationalError:
                pass  # column already exists or table doesn't exist yet

        # YouTube quota migration: create youtube_quota_usage table if absent
        try:
            conn.executescript(YOUTUBE_QUOTA_DDL)
            conn.commit()
            logger.info("Migration applied: youtube_quota_usage table ensured (%s)", db_path)
        except sqlite3.OperationalError:
            pass

        # Production lock migration: create production_lock table if absent
        try:
            conn.executescript(PRODUCTION_LOCK_DDL)
            conn.commit()
            logger.info("Migration applied: production_lock table ensured (%s)", db_path)
        except sqlite3.OperationalError:
            pass
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Create all ChannelForge databases."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    databases: dict[str, tuple[Path, list[str]]] = {
        "channel_forge.db": (DATA_DIR / "channel_forge.db", MAIN_DDL_PARTS),
        "trends.db":        (DATA_DIR / "trends.db",        [TRENDS_DDL]),
        "safety.db":        (DATA_DIR / "safety.db",        [SAFETY_DDL]),
        "scores.db":        (DATA_DIR / "scores.db",        [SCORES_DDL]),
    }

    for name, (path, ddl_parts) in databases.items():
        create_db(path, ddl_parts)
        migrate_db(path)
        tables = verify_db(path)
        logger.info("  %s tables: %s", name, tables)

    logger.info("All databases initialized successfully.")


if __name__ == "__main__":
    main()

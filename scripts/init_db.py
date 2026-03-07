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
# Schema definitions
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

# Unified main DB gets all tables
MAIN_DDL_PARTS = [TRENDS_DDL, SAFETY_DDL, SCORES_DDL, TITLES_DDL, VIDEO_IDEAS_DDL]


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Create all ChannelForge databases."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    databases: dict[str, tuple[Path, list[str]]] = {
        "channel_forge.db": (DATA_DIR / "channel_forge.db", MAIN_DDL_PARTS),
        "trends.db":        (DATA_DIR / "trends.db",        [TRENDS_DDL]),
        "safety.db":        (DATA_DIR / "safety.db",         [SAFETY_DDL]),
        "scores.db":        (DATA_DIR / "scores.db",         [SCORES_DDL]),
    }

    for name, (path, ddl_parts) in databases.items():
        create_db(path, ddl_parts)
        tables = verify_db(path)
        logger.info("  %s tables: %s", name, tables)

    logger.info("All databases initialized successfully.")


if __name__ == "__main__":
    main()

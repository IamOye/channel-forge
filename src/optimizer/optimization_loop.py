"""
optimization_loop.py — OptimizationLoop

Analyzes video performance data, generates new optimized topics via Claude API,
and injects them into the scored_topics table. Designed to run weekly.

Usage:
    loop = OptimizationLoop()
    result = loop.run()
    print(result.topics_injected, result.pattern_analysis[:100])
"""

import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-5"
DB_PATH = Path("data/processed/channel_forge.db")

_PATTERN_SYSTEM_PROMPT = """You are an expert YouTube analytics strategist.

Given lists of top-performing (Tier S and A) and worst-performing (Tier F) YouTube Shorts
videos, analyze the winning and losing patterns.

Return ONLY a JSON object in this exact format (no markdown):
{
  "winning_patterns": ["pattern 1", "pattern 2", "pattern 3"],
  "losing_patterns":  ["pattern 1", "pattern 2"],
  "content_insights": "brief strategic summary",
  "recommended_category": "success"
}"""

_TOPIC_SYSTEM_PROMPT = """You are an expert YouTube Shorts content strategist.

Given performance analysis data, generate exactly 20 new optimized video topics.
Each topic should apply the winning patterns and avoid the losing patterns.

Return ONLY a JSON array of exactly 20 objects (no markdown):
[
  {
    "keyword": "topic keyword phrase",
    "category": "success",
    "predicted_tier": "A",
    "rationale": "why this will perform well"
  },
  ...
]

predicted_tier must be one of: S, A, B, C, F"""

_OPTIMIZATION_LOG_DDL = """
CREATE TABLE IF NOT EXISTS optimization_log (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    winners_count    INTEGER NOT NULL DEFAULT 0,
    losers_count     INTEGER NOT NULL DEFAULT 0,
    pattern_analysis TEXT,
    topics_generated INTEGER NOT NULL DEFAULT 0,
    topics_injected  INTEGER NOT NULL DEFAULT 0,
    run_at           TEXT    NOT NULL
);
"""

_SCORED_TOPICS_DDL = """
CREATE TABLE IF NOT EXISTS scored_topics (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    keyword     TEXT    NOT NULL,
    category    TEXT    NOT NULL DEFAULT 'success',
    score       REAL    NOT NULL DEFAULT 0,
    source      TEXT    NOT NULL DEFAULT 'manual',
    created_at  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_scored_keyword ON scored_topics (keyword);
CREATE INDEX IF NOT EXISTS idx_scored_score   ON scored_topics (score DESC);
"""

_HIGH_TIERS = {"S", "A"}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    """Result of one OptimizationLoop.run() execution."""

    winners_count: int = 0
    losers_count: int = 0
    pattern_analysis: str = ""
    topics_generated: int = 0
    topics_injected: int = 0
    is_valid: bool = True
    error: str = ""
    run_at: str = ""

    def __post_init__(self) -> None:
        if not self.run_at:
            self.run_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "winners_count":    self.winners_count,
            "losers_count":     self.losers_count,
            "pattern_analysis": self.pattern_analysis,
            "topics_generated": self.topics_generated,
            "topics_injected":  self.topics_injected,
            "is_valid":         self.is_valid,
            "error":            self.error,
            "run_at":           self.run_at,
        }


# ---------------------------------------------------------------------------
# OptimizationLoop
# ---------------------------------------------------------------------------

class OptimizationLoop:
    """
    Weekly optimization loop: analyzes winner/loser patterns via Claude,
    generates 20 new topics, and injects high-predicted-tier topics into the DB.

    Args:
        api_key:      Anthropic API key. If None, reads ANTHROPIC_API_KEY.
        db_path:      Path to SQLite database.
        inject_score: Score assigned to injected topics (default 85).
        num_topics:   Number of topics to request from Claude (default 20).
    """

    INJECT_SCORE = 85.0

    def __init__(
        self,
        api_key: str | None = None,
        db_path: str | Path = DB_PATH,
        inject_score: float = 85.0,
        num_topics: int = 20,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("ANTHROPIC_API_KEY", "")
        self.db_path = Path(db_path)
        self.inject_score = inject_score
        self.num_topics = num_topics
        self._client: anthropic.Anthropic | None = None
        self._ensure_tables()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> OptimizationResult:
        """
        Execute the full optimization loop:
          1. Pull S/A winners and F losers from video_metrics
          2. Call Claude to analyze patterns
          3. Call Claude to generate num_topics new topics
          4. Inject topics with predicted_tier S or A into scored_topics
          5. Save run log to optimization_log

        Returns:
            OptimizationResult summarizing the run.
        """
        logger.info("OptimizationLoop started")
        result = OptimizationResult()

        try:
            winners = self.pull_performers(tiers=["S", "A"])
            losers  = self.pull_performers(tiers=["F"])
            result.winners_count = len(winners)
            result.losers_count  = len(losers)

            pattern_data = self.analyze_patterns(winners, losers)
            result.pattern_analysis = json.dumps(pattern_data)

            topics = self.generate_topics(pattern_data)
            result.topics_generated = len(topics)

            injected = self.inject_topics(topics)
            result.topics_injected = injected

        except Exception as exc:
            logger.error("OptimizationLoop failed: %s", exc)
            result.is_valid = False
            result.error = str(exc)

        self._save_log(result)
        logger.info(
            "OptimizationLoop done: winners=%d losers=%d injected=%d",
            result.winners_count, result.losers_count, result.topics_injected,
        )
        return result

    def pull_performers(self, tiers: list[str]) -> list[dict[str, Any]]:
        """
        Return video metric rows for the given performance tiers.

        Fetches most recent data ordered by fetched_at descending.
        """
        if not tiers:
            return []
        placeholders = ", ".join("?" * len(tiers))
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(
                f"""
                SELECT video_id, tier, views, engagement_rate, virality_score, fetched_at
                FROM video_metrics
                WHERE tier IN ({placeholders})
                ORDER BY fetched_at DESC
                """,
                tiers,
            ).fetchall()
        except sqlite3.OperationalError:
            # video_metrics table doesn't exist yet (AnalyticsTracker not run)
            rows = []
        finally:
            conn.close()

        return [
            {
                "video_id":        r[0],
                "tier":            r[1],
                "views":           r[2],
                "engagement_rate": r[3],
                "virality_score":  r[4],
            }
            for r in rows
        ]

    def analyze_patterns(
        self, winners: list[dict], losers: list[dict]
    ) -> dict[str, Any]:
        """
        Call Claude to analyze winning and losing patterns.

        Returns a dict with keys: winning_patterns, losing_patterns,
        content_insights, recommended_category.
        """
        prompt = (
            f"Top performers (Tier S/A) — {len(winners)} videos:\n"
            + json.dumps(winners[:10], indent=2)
            + f"\n\nWorst performers (Tier F) — {len(losers)} videos:\n"
            + json.dumps(losers[:10], indent=2)
        )

        client = self._get_client()
        message = client.messages.create(
            model=_MODEL,
            max_tokens=600,
            system=_PATTERN_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
        return self._parse_json(
            raw,
            default={
                "winning_patterns":     [],
                "losing_patterns":      [],
                "content_insights":     "",
                "recommended_category": "success",
            },
        )

    def generate_topics(self, pattern_data: dict) -> list[dict[str, Any]]:
        """
        Call Claude to generate num_topics optimized topics based on pattern data.

        Returns a list filtered to predicted_tier S or A only.
        """
        prompt = (
            f"Performance analysis:\n{json.dumps(pattern_data, indent=2)}\n\n"
            f"Generate exactly {self.num_topics} optimized YouTube Shorts topics "
            "that apply the winning patterns."
        )

        client = self._get_client()
        message = client.messages.create(
            model=_MODEL,
            max_tokens=1500,
            system=_TOPIC_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
        topics = self._parse_json(raw, default=[])

        if not isinstance(topics, list):
            return []

        return [
            t for t in topics
            if isinstance(t, dict) and t.get("predicted_tier") in _HIGH_TIERS
        ]

    def inject_topics(self, topics: list[dict]) -> int:
        """
        Insert topics into scored_topics with inject_score.

        Returns the count of successfully injected topics.
        """
        if not topics:
            return 0

        now = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(self.db_path)
        injected = 0
        try:
            for t in topics:
                keyword  = str(t.get("keyword", "")).strip()
                category = str(t.get("category", "success")).strip()
                if not keyword:
                    continue
                conn.execute(
                    """
                    INSERT INTO scored_topics (keyword, category, score, source, created_at)
                    VALUES (?, ?, ?, 'optimization_loop', ?)
                    """,
                    (keyword, category, self.inject_score, now),
                )
                injected += 1
            conn.commit()
            logger.info("Injected %d topics into scored_topics", injected)
        finally:
            conn.close()
        return injected

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            if not self.api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    @staticmethod
    def _parse_json(raw: str, default: Any) -> Any:
        """Parse JSON from Claude response, stripping markdown fences if present."""
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                line for line in lines if not line.strip().startswith("```")
            ).strip()
        try:
            return json.loads(text)
        except Exception as exc:
            logger.error("Failed to parse JSON: %s | raw=%s", exc, raw[:200])
            return default

    def _save_log(self, result: OptimizationResult) -> None:
        """Persist the run summary to optimization_log. Failures are swallowed."""
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute(
                    """
                    INSERT INTO optimization_log
                        (winners_count, losers_count, pattern_analysis,
                         topics_generated, topics_injected, run_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        result.winners_count,
                        result.losers_count,
                        result.pattern_analysis,
                        result.topics_generated,
                        result.topics_injected,
                        result.run_at,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as exc:
            logger.warning("Failed to save optimization log: %s", exc)

    def _ensure_tables(self) -> None:
        """Create optimization_log and scored_topics tables if they don't exist."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path)
            try:
                conn.executescript(_OPTIMIZATION_LOG_DDL + _SCORED_TOPICS_DDL)
                conn.commit()
            finally:
                conn.close()
        except Exception as exc:
            logger.warning("Could not initialise optimization tables: %s", exc)

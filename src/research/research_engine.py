"""
research_engine.py — ResearchEngine

Reusable research pipeline for ChannelForge topic discovery.
Works both locally (CLI via tools/research.py) and on Railway (Telegram).

Phases:
  1. Scrape  — Reddit, autocomplete, Google Trends, competitors
  2. Dedup   — vs uploaded_videos, manual_topics, research_reviewed
  3. Score   — 5-dimension weighted scoring via Claude Haiku
  4. Rewrite — all topics rewritten for scroll-stopping hook power

Usage:
    engine = ResearchEngine(exclude_reddit=True)  # Railway
    topics = engine.run(source="all", category="money")
"""

import json
import logging
import os
import re
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DB = Path(os.getenv("DB_PATH", "data/processed/channel_forge.db"))

# ---------------------------------------------------------------------------
# Safe conversion helpers
# ---------------------------------------------------------------------------

def _safe_str(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value) if value is not None else default
    except (ValueError, TypeError):
        return default


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AUTOCOMPLETE_SEEDS = [
    "why your boss", "how to save money",
    "salary negotiation", "passive income",
    "why most people", "how to invest",
    "financial freedom", "side hustle",
    "why the rich", "how to build wealth",
]

TREND_KEYWORDS = ["money", "salary", "investing", "career", "financial freedom"]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class RawTopic:
    """A single raw topic from any source."""
    title: str
    source: str
    source_detail: str = ""
    score_hint: float = 0.0
    extra: dict = field(default_factory=dict)


@dataclass
class ScoredTopic:
    """A topic after Claude Haiku scoring and optional rewrite."""
    title: str
    score: float
    category: str
    hook_angle: str
    reason: str
    source: str
    source_detail: str
    score_hint: float = 0.0
    # Sub-scores
    hook_strength: float = 0.0
    contrarian: float = 0.0
    specificity: float = 0.0
    brand_fit: float = 0.0
    search_demand: float = 0.0
    # Rewrite fields
    original_title: str = ""
    rewritten_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title, "score": self.score,
            "category": self.category, "hook_angle": self.hook_angle,
            "reason": self.reason, "source": self.source,
            "source_detail": self.source_detail, "score_hint": self.score_hint,
            "hook_strength": self.hook_strength, "contrarian": self.contrarian,
            "specificity": self.specificity, "brand_fit": self.brand_fit,
            "search_demand": self.search_demand,
            "original_title": self.original_title,
            "rewritten_score": self.rewritten_score,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ScoredTopic":
        return cls(
            title=d.get("title", ""), score=_safe_float(d.get("score")),
            category=d.get("category", "money"),
            hook_angle=d.get("hook_angle", ""), reason=d.get("reason", ""),
            source=d.get("source", ""), source_detail=d.get("source_detail", ""),
            score_hint=_safe_float(d.get("score_hint")),
            hook_strength=_safe_float(d.get("hook_strength")),
            contrarian=_safe_float(d.get("contrarian")),
            specificity=_safe_float(d.get("specificity")),
            brand_fit=_safe_float(d.get("brand_fit")),
            search_demand=_safe_float(d.get("search_demand")),
            original_title=d.get("original_title", ""),
            rewritten_score=_safe_float(d.get("rewritten_score")),
        )


def _normalise_title(title: str | None) -> str:
    if not title:
        return ""
    t = str(title).lower().strip()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


# ---------------------------------------------------------------------------
# ResearchEngine
# ---------------------------------------------------------------------------

class ResearchEngine:
    """Multi-source research pipeline: scrape, dedup, score, rewrite.

    Args:
        db_path:        SQLite DB path for dedup + reviewed topics.
        api_key:        Anthropic API key (reads env if None).
        enable_rewrite: Whether to auto-rewrite titles after scoring.
        exclude_reddit: Skip Reddit scraping (required on Railway).
    """

    def __init__(
        self,
        db_path: Path | None = None,
        api_key: str | None = None,
        enable_rewrite: bool = True,
        exclude_reddit: bool = False,
    ) -> None:
        self.db_path = db_path or _DEFAULT_DB
        self.api_key = api_key if api_key is not None else os.getenv("ANTHROPIC_API_KEY", "")
        self.enable_rewrite = enable_rewrite
        self.exclude_reddit = exclude_reddit

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        source: str | None = None,
        category: str | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> list[ScoredTopic]:
        """Run the full 4-phase pipeline.

        Args:
            source:   "reddit", "autocomplete", "trends", "competitor", or None for all.
            category: Filter to this category after scoring.
            progress_callback: Called with status messages after each phase.

        Returns:
            Sorted list of scored (and optionally rewritten) topics.
        """
        cb = progress_callback or (lambda msg: None)

        # Phase 1: Scrape
        sources = [source] if source else None
        raw = self.scrape(sources)
        cb(f"📡 Scraped {len(raw)} topics")

        if not raw:
            cb("⚠️ No topics found")
            return []

        # Phase 2: Dedup
        clean = self.deduplicate(raw)
        removed = len(raw) - len(clean)
        cb(f"🧹 {len(clean)} topics after dedup ({removed} already seen)")

        if not clean:
            cb("⚠️ No topics remaining after dedup")
            return []

        # Phase 3: Score
        cb(f"🧠 Scoring {len(clean)} topics...")
        scored = self.score(clean)
        cb(f"✅ Scored {len(scored)} topics")

        # Phase 4: Rewrite
        if self.enable_rewrite and self.api_key:
            cb(f"✏️ Rewriting {len(scored)} topics...")
            scored = self.rewrite(scored)
            cb(f"✅ Rewritten {len(scored)} topics")

        # Category filter
        if category:
            scored = [s for s in scored if s.category == category]

        scored.sort(key=lambda s: s.score, reverse=True)
        cb(f"✅ Ready! {len(scored)} topics ranked.")
        return scored

    # ------------------------------------------------------------------
    # Phase 1: Scrape
    # ------------------------------------------------------------------

    def scrape(self, sources: list[str] | None = None) -> list[RawTopic]:
        """Run scrapers in parallel, return raw topics."""
        scrapers: dict[str, Callable] = {
            "reddit": self._scrape_reddit,
            "autocomplete": self._scrape_autocomplete,
            "trends": self._scrape_trends,
            "competitor": self._scrape_competitors,
        }

        if self.exclude_reddit:
            scrapers.pop("reddit", None)

        if sources:
            scrapers = {k: v for k, v in scrapers.items() if k in sources}

        all_topics: list[RawTopic] = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(fn): name for name, fn in scrapers.items()}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    all_topics.extend(future.result())
                except Exception as exc:
                    logger.error("[scrape] %s raised: %s", name, exc)

        logger.info("[scrape] Total raw topics: %d", len(all_topics))
        return all_topics

    def _scrape_reddit(self) -> list[RawTopic]:
        topics: list[RawTopic] = []
        try:
            from src.crawler.reddit_scraper import RedditScraper
            scraper = RedditScraper()
            for r in scraper.scrape_finance_subreddits():
                title = _safe_str(r.keyword)
                if title:
                    topics.append(RawTopic(
                        title=title, source="reddit",
                        source_detail=f"r/{_safe_str(r.subreddit)}",
                        score_hint=_safe_float(r.upvotes),
                        extra={"category": _safe_str(r.category), "upvotes": _safe_int(r.upvotes)},
                    ))
            logger.info("[scrape] Reddit: %d topics", len(topics))
        except Exception as exc:
            logger.warning("[scrape] Reddit failed: %s", exc)
        return topics

    def _scrape_autocomplete(self) -> list[RawTopic]:
        topics: list[RawTopic] = []
        try:
            import httpx
            for seed in AUTOCOMPLETE_SEEDS:
                try:
                    resp = httpx.get(
                        "https://suggestqueries-clients6.youtube.com/complete/search",
                        params={"client": "youtube", "q": seed, "ds": "yt"},
                        timeout=10.0,
                    )
                    text = resp.text
                    start = text.index("[")
                    data = json.loads(text[start:])
                    suggestions = [s[0] for s in data[1]] if len(data) > 1 else []
                    for sug in suggestions:
                        s = _safe_str(sug)
                        if s and s.lower() != seed.lower():
                            topics.append(RawTopic(title=s, source="autocomplete", source_detail=seed))
                except Exception:
                    pass
            logger.info("[scrape] Autocomplete: %d topics", len(topics))
        except Exception as exc:
            logger.warning("[scrape] Autocomplete failed: %s", exc)
        return topics

    def _scrape_trends(self) -> list[RawTopic]:
        topics: list[RawTopic] = []
        try:
            from src.crawler.trend_scraper import TrendScrapingEngine
            engine = TrendScrapingEngine()
            for s in engine.fetch_all(TREND_KEYWORDS):
                kw = _safe_str(s.keyword)
                if kw:
                    topics.append(RawTopic(
                        title=kw, source="trends",
                        source_detail=_safe_str(s.source),
                        score_hint=_safe_float(s.interest_score),
                    ))
                for rq in (s.related_queries or []):
                    rq_str = _safe_str(rq)
                    if rq_str:
                        topics.append(RawTopic(title=rq_str, source="trends", source_detail=f"related/{kw}"))
            logger.info("[scrape] Trends: %d topics", len(topics))
        except Exception as exc:
            logger.warning("[scrape] Trends failed: %s", exc)
        return topics

    def _scrape_competitors(self) -> list[RawTopic]:
        topics: list[RawTopic] = []
        try:
            from src.crawler.competitor_scraper import CompetitorScraper
            scraper = CompetitorScraper()
            for cat in ("money", "career", "success"):
                try:
                    for title in scraper.scrape_competitor_topics(cat):
                        t = _safe_str(title)
                        if t:
                            topics.append(RawTopic(title=t, source="competitor", source_detail=cat))
                except Exception:
                    pass
                try:
                    for sug in scraper.scrape_search_autocomplete(cat):
                        s = _safe_str(sug)
                        if s:
                            topics.append(RawTopic(title=s, source="autocomplete", source_detail=f"yt/{cat}"))
                except Exception:
                    pass
            logger.info("[scrape] Competitors: %d topics", len(topics))
        except Exception as exc:
            logger.warning("[scrape] Competitors failed: %s", exc)
        return topics

    # ------------------------------------------------------------------
    # Phase 2: Deduplicate
    # ------------------------------------------------------------------

    def deduplicate(self, topics: list[RawTopic]) -> list[RawTopic]:
        """Remove duplicates, short titles, non-English, and already-reviewed."""
        existing = self._load_exclusion_set()
        seen: set[str] = set()
        result: list[RawTopic] = []

        for t in topics:
            title = _safe_str(t.title)
            if not title or len(title.split()) < 5:
                continue
            ascii_ratio = sum(1 for c in title if ord(c) < 128) / max(len(title), 1)
            if ascii_ratio < 0.7:
                continue
            norm = _normalise_title(title)
            if not norm or norm in seen or norm in existing:
                continue
            seen.add(norm)
            result.append(t)

        logger.info("[dedup] %d topics after dedup (from %d)", len(result), len(topics))
        return result

    def _load_exclusion_set(self) -> set[str]:
        """Load titles to exclude: uploaded_videos, production_results, manual_topics, research_reviewed."""
        existing: set[str] = set()
        db = self.db_path
        if not Path(db).exists():
            return existing
        try:
            conn = sqlite3.connect(db)
            try:
                queries = [
                    "SELECT title FROM uploaded_videos",
                    "SELECT keyword FROM production_results WHERE keyword != ''",
                    "SELECT title FROM manual_topics WHERE status IN ('USED', 'QUEUED', 'HOLD')",
                    "SELECT normalised_title FROM research_reviewed",
                ]
                for q in queries:
                    try:
                        for (val,) in conn.execute(q).fetchall():
                            existing.add(_normalise_title(val))
                    except sqlite3.OperationalError:
                        pass
            finally:
                conn.close()
        except Exception as exc:
            logger.warning("[dedup] DB load failed: %s", exc)
        existing.discard("")
        return existing

    # ------------------------------------------------------------------
    # Phase 3: Score (5-dimension weighted)
    # ------------------------------------------------------------------

    def score(self, topics: list[RawTopic]) -> list[ScoredTopic]:
        """Score topics with 5-dimension weighted system via Claude Haiku."""
        if not self.api_key:
            logger.warning("[score] No API key — returning unscored")
            return [
                ScoredTopic(
                    title=_safe_str(t.title) or "untitled", score=0.0,
                    category="money", hook_angle="", reason="unscored",
                    source=_safe_str(t.source), source_detail=_safe_str(t.source_detail),
                    score_hint=_safe_float(t.score_hint),
                )
                for t in topics
            ]

        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        scored: list[ScoredTopic] = []
        batch_size = 20

        for i in range(0, len(topics), batch_size):
            batch = topics[i:i + batch_size]
            titles = [_safe_str(t.title) for t in batch]

            prompt = (
                "Score each YouTube Shorts topic idea for @moneyheresy "
                "(money/career/success, contrarian angle, Western audience US/UK/CA/AU).\n\n"
                "For each topic, score 5 dimensions (1-10):\n"
                "1. hook_strength: Creates instant curiosity or emotion in 3 seconds? Would someone stop scrolling?\n"
                "2. contrarian: Challenges something people believe or were told?\n"
                "3. specificity: Concrete and specific vs vague? Specific titles outperform generic.\n"
                "4. brand_fit: Suits Money Heresy — system-focused, anti-conventional, Western audience?\n"
                "5. search_demand: Evidence people search this (upvotes, autocomplete frequency)?\n\n"
                "Final score = hook_strength×0.35 + contrarian×0.25 + specificity×0.20 + brand_fit×0.10 + search_demand×0.10\n\n"
                "Return JSON per topic:\n"
                '{"title": str, "hook_strength": 1-10, "contrarian": 1-10, "specificity": 1-10,\n'
                ' "brand_fit": 1-10, "search_demand": 1-10, "score": weighted_avg,\n'
                ' "category": "money"|"career"|"success", "hook_angle": str, "reason": str (15 words max)}\n\n'
                "Return ONLY a JSON array.\n\n"
                f"Topics:\n{json.dumps(titles, indent=2)}"
            )

            try:
                message = client.messages.create(
                    model="claude-haiku-4-5-20251001", max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = message.content[0].text.strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                items = json.loads(raw.strip())

                title_to_raw = {_safe_str(t.title): t for t in batch}
                for item in items:
                    title = _safe_str(item.get("title"))
                    raw_t = title_to_raw.get(title) or (batch[0] if batch else None)
                    hs = _safe_float(item.get("hook_strength"))
                    ct = _safe_float(item.get("contrarian"))
                    sp = _safe_float(item.get("specificity"))
                    bf = _safe_float(item.get("brand_fit"))
                    sd = _safe_float(item.get("search_demand"))
                    weighted = hs * 0.35 + ct * 0.25 + sp * 0.20 + bf * 0.10 + sd * 0.10

                    scored.append(ScoredTopic(
                        title=title or "untitled",
                        score=round(weighted, 1),
                        category=_safe_str(item.get("category")) or "money",
                        hook_angle=_safe_str(item.get("hook_angle")),
                        reason=_safe_str(item.get("reason")),
                        source=_safe_str(raw_t.source) if raw_t else "unknown",
                        source_detail=_safe_str(raw_t.source_detail) if raw_t else "",
                        score_hint=_safe_float(raw_t.score_hint) if raw_t else 0.0,
                        hook_strength=hs, contrarian=ct, specificity=sp,
                        brand_fit=bf, search_demand=sd,
                    ))
            except Exception as exc:
                logger.warning("[score] Batch %d failed: %s", i, exc)
                for t in batch:
                    scored.append(ScoredTopic(
                        title=_safe_str(t.title) or "untitled", score=0.0,
                        category="money", hook_angle="", reason="scoring failed",
                        source=_safe_str(t.source), source_detail=_safe_str(t.source_detail),
                        score_hint=_safe_float(t.score_hint),
                    ))

        scored.sort(key=lambda s: s.score, reverse=True)
        logger.info("[score] Scored %d topics", len(scored))
        return scored

    # ------------------------------------------------------------------
    # Phase 4: Rewrite
    # ------------------------------------------------------------------

    def rewrite(self, scored: list[ScoredTopic]) -> list[ScoredTopic]:
        """Rewrite all topics for scroll-stopping hook power via Claude Haiku."""
        if not self.api_key:
            return scored

        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        rewritten: list[ScoredTopic] = []
        batch_size = 10

        for i in range(0, len(scored), batch_size):
            batch = scored[i:i + batch_size]
            titles = [_safe_str(t.title) for t in batch]

            prompt = (
                "You are a YouTube Shorts title specialist for @moneyheresy — "
                "a finance channel with a contrarian, system-critique angle "
                "targeting US/UK/CA/AU audiences.\n\n"
                "Rewrite each title to maximise scroll-stopping power:\n"
                "- Under 60 characters ideally\n"
                "- Lead with 'you' or 'your' — makes it personal and immediate "
                "(e.g. 'Your savings is losing value')\n"
                "- Replace passive phrasing with active present-tense tension "
                "(e.g. 'is costing you' not 'can cost you')\n"
                "- Add specificity where possible — numbers, ages, amounts signal "
                "real data and increase credibility "
                "(e.g. 'half your raise', 'by age 30', 'a dollar a day')\n"
                "- Plant a curiosity gap — viewer must watch to resolve the tension "
                "the title creates\n"
                "- Soften extreme language where needed to stay credible without losing "
                "punch ('can destroy' not 'destroys', 'is stealing from' not 'steals from')\n"
                "- Feels like a secret or truth being revealed\n"
                "- Avoids clickbait — must be fully deliverable in a 60-second video\n"
                "- Keep the core insight but sharpen the hook\n\n"
                "Return JSON array:\n"
                '{"original": str, "rewritten": str, "rewrite_score": 1-10, '
                '"improvement_reason": str (15 words max)}\n\n'
                "Return ONLY a JSON array.\n\n"
                f"Titles:\n{json.dumps(titles, indent=2)}"
            )

            try:
                message = client.messages.create(
                    model="claude-haiku-4-5-20251001", max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = message.content[0].text.strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                items = json.loads(raw.strip())

                title_to_topic = {_safe_str(t.title): t for t in batch}
                used: set[str] = set()
                for item in items:
                    orig = _safe_str(item.get("original"))
                    topic = title_to_topic.get(orig)
                    if not topic:
                        continue
                    used.add(orig)
                    new_title = _safe_str(item.get("rewritten")) or topic.title
                    rw_score = _safe_float(item.get("rewrite_score"))
                    reason = _safe_str(item.get("improvement_reason"))

                    rewritten.append(ScoredTopic(
                        title=new_title,
                        score=topic.score,
                        category=topic.category,
                        hook_angle=topic.hook_angle,
                        reason=reason or topic.reason,
                        source=topic.source,
                        source_detail=topic.source_detail,
                        score_hint=topic.score_hint,
                        hook_strength=topic.hook_strength,
                        contrarian=topic.contrarian,
                        specificity=topic.specificity,
                        brand_fit=topic.brand_fit,
                        search_demand=topic.search_demand,
                        original_title=topic.title,
                        rewritten_score=rw_score,
                    ))

                # Add any topics not matched in the rewrite response
                for t in batch:
                    if _safe_str(t.title) not in used:
                        rewritten.append(t)

            except Exception as exc:
                logger.warning("[rewrite] Batch %d failed: %s", i, exc)
                rewritten.extend(batch)

        rewritten.sort(key=lambda s: s.score, reverse=True)
        logger.info("[rewrite] Rewrote %d topics", len(rewritten))
        return rewritten

    # ------------------------------------------------------------------
    # Reviewed topics memory
    # ------------------------------------------------------------------

    def _ensure_reviewed_table(self, conn: sqlite3.Connection) -> None:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS research_reviewed (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL, normalised_title TEXT NOT NULL,
                original_title TEXT NOT NULL DEFAULT '',
                score REAL NOT NULL DEFAULT 0, category TEXT NOT NULL DEFAULT 'money',
                source TEXT NOT NULL DEFAULT '', action TEXT NOT NULL,
                session_id TEXT NOT NULL DEFAULT '',
                reviewed_at TEXT NOT NULL DEFAULT (datetime('now')),
                synced_to_sheet INTEGER NOT NULL DEFAULT 0
            )
        """)

    def mark_reviewed(
        self,
        title: str,
        action: str,
        session_id: str = "",
        score: float = 0.0,
        category: str = "money",
        source: str = "",
        original_title: str = "",
    ) -> None:
        """Record a topic as reviewed (added or skipped)."""
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                self._ensure_reviewed_table(conn)
                conn.execute(
                    "INSERT INTO research_reviewed "
                    "(title, normalised_title, original_title, score, category, source, action, session_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (title, _normalise_title(title), original_title, score, category, source, action, session_id),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as exc:
            logger.warning("[reviewed] Failed to mark reviewed: %s", exc)

    def mark_batch_reviewed(self, topics: list[ScoredTopic], action: str, session_id: str = "") -> None:
        """Mark multiple topics as reviewed."""
        for t in topics:
            self.mark_reviewed(
                title=t.title, action=action, session_id=session_id,
                score=t.score, category=t.category, source=t.source,
                original_title=t.original_title,
            )

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _ensure_sessions_table(self, conn: sqlite3.Connection) -> None:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS research_sessions (
                id TEXT PRIMARY KEY, chat_id TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT 'all', category TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'active',
                current_index INTEGER NOT NULL DEFAULT 0,
                total_topics INTEGER NOT NULL DEFAULT 0,
                topics_added INTEGER NOT NULL DEFAULT 0,
                topics_skipped INTEGER NOT NULL DEFAULT 0,
                topics_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)

    def create_session(self, chat_id: str, topics: list[ScoredTopic], source: str = "all", category: str = "") -> str:
        """Create a research session and return its ID."""
        session_id = str(uuid.uuid4())[:8]
        topics_json = json.dumps([t.to_dict() for t in topics])
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                self._ensure_sessions_table(conn)
                conn.execute(
                    "INSERT INTO research_sessions "
                    "(id, chat_id, source, category, status, total_topics, topics_json) "
                    "VALUES (?, ?, ?, ?, 'active', ?, ?)",
                    (session_id, chat_id, source, category, len(topics), topics_json),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as exc:
            logger.error("[session] Create failed: %s", exc)
        return session_id

    def get_session(self, chat_id: str) -> dict | None:
        """Get the active session for a chat_id."""
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                self._ensure_sessions_table(conn)
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT * FROM research_sessions WHERE chat_id = ? AND status = 'active' "
                    "ORDER BY created_at DESC LIMIT 1",
                    (chat_id,),
                ).fetchone()
                return dict(row) if row else None
            finally:
                conn.close()
        except Exception:
            return None

    def update_session(self, session_id: str, **kwargs: Any) -> None:
        """Update session fields."""
        if not kwargs:
            return
        sets = [f"{k} = ?" for k in kwargs]
        sets.append("updated_at = datetime('now')")
        vals = list(kwargs.values()) + [session_id]
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute(
                    f"UPDATE research_sessions SET {', '.join(sets)} WHERE id = ?",
                    vals,
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as exc:
            logger.warning("[session] Update failed: %s", exc)

    def get_session_topics(self, session: dict) -> list[ScoredTopic]:
        """Deserialize topics from session dict."""
        try:
            items = json.loads(session.get("topics_json", "[]"))
            return [ScoredTopic.from_dict(d) for d in items]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Google Sheet sync for reviewed topics
    # ------------------------------------------------------------------

    def sync_reviewed_to_sheet(self) -> int:
        """Push un-synced reviewed topics to Google Sheet 'Reviewed Log' tab.

        Returns count of synced rows.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                self._ensure_reviewed_table(conn)
                rows = conn.execute(
                    "SELECT id, title, original_title, score, category, source, action, reviewed_at "
                    "FROM research_reviewed WHERE synced_to_sheet = 0"
                ).fetchall()
                if not rows:
                    return 0

                from src.crawler.gsheet_topic_sync import GSheetTopicSync
                sync = GSheetTopicSync()
                sync._connect()

                # Get or create Reviewed Log tab
                import gspread
                try:
                    log_tab = sync._sheet.worksheet("Reviewed Log")
                except gspread.exceptions.WorksheetNotFound:
                    log_tab = sync._sheet.add_worksheet("Reviewed Log", rows=500, cols=8)
                    log_tab.update("A1:G1", [["Date", "Title", "Original", "Score", "Category", "Source", "Action"]])
                    log_tab.format("A1:G1", {"textFormat": {"bold": True}})

                sheet_rows = []
                ids_to_mark: list[int] = []
                for row_id, title, orig, score, cat, src, action, reviewed_at in rows:
                    sheet_rows.append([reviewed_at[:10], title, orig, score, cat, src, action])
                    ids_to_mark.append(row_id)

                if sheet_rows:
                    log_tab.append_rows(sheet_rows, value_input_option="USER_ENTERED")

                # Mark synced
                for rid in ids_to_mark:
                    conn.execute("UPDATE research_reviewed SET synced_to_sheet = 1 WHERE id = ?", (rid,))
                conn.commit()

                logger.info("[reviewed] Synced %d reviewed topics to sheet", len(ids_to_mark))
                return len(ids_to_mark)
            finally:
                conn.close()
        except Exception as exc:
            logger.warning("[reviewed] Sheet sync failed: %s", exc)
            return 0

    def import_reviewed_from_sheet(self) -> int:
        """Import reviewed topics from Google Sheet 'Reviewed Log' tab into local DB.

        Returns count of new rows imported.
        """
        try:
            from src.crawler.gsheet_topic_sync import GSheetTopicSync
            import gspread
            sync = GSheetTopicSync()
            sync._connect()

            try:
                log_tab = sync._sheet.worksheet("Reviewed Log")
            except gspread.exceptions.WorksheetNotFound:
                return 0

            rows = log_tab.get_all_records()
            if not rows:
                return 0

            conn = sqlite3.connect(self.db_path)
            try:
                self._ensure_reviewed_table(conn)
                existing = {
                    r[0] for r in conn.execute("SELECT normalised_title FROM research_reviewed").fetchall()
                }

                count = 0
                for row in rows:
                    title = _safe_str(row.get("Title"))
                    if not title:
                        continue
                    norm = _normalise_title(title)
                    if norm in existing:
                        continue
                    conn.execute(
                        "INSERT INTO research_reviewed "
                        "(title, normalised_title, original_title, score, category, source, action, synced_to_sheet) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, 1)",
                        (title, norm, _safe_str(row.get("Original", "")),
                         _safe_float(row.get("Score")), _safe_str(row.get("Category", "money")),
                         _safe_str(row.get("Source", "")), _safe_str(row.get("Action", "skipped"))),
                    )
                    existing.add(norm)
                    count += 1

                conn.commit()
                logger.info("[reviewed] Imported %d reviewed topics from sheet", count)
                return count
            finally:
                conn.close()
        except Exception as exc:
            logger.warning("[reviewed] Sheet import failed: %s", exc)
            return 0

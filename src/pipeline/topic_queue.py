"""
topic_queue.py — TopicQueue

Pulls topics from all available sources, orders them by SOURCE_PRIORITIES,
deduplicates against upload history, and returns the top N for production.

Priority order (highest first):
  VIEWER_REQUESTED       100  — viewers asked via comment
  COMPETITOR_HIGH_SIGNAL  90  — competitor video > 100k views / 30 days
  YOUTUBE_TRENDING        80  — trending / high-view search result
  GOOGLE_TRENDS           70  — pytrends interest signal
  YOUTUBE_KEYWORD         60  — general YouTube keyword signal
  FALLBACK                50  — pre-written fallback list

Usage:
    queue = TopicQueue(db_path="data/processed/channel_forge.db")
    topics = queue.get_next_topics(
        category="money",
        max_count=3,
        uploaded_topics=["why saving money keeps you poor"],
    )
"""

import logging
import os
import random
import sqlite3
from pathlib import Path
from typing import Any

from config.constants import FALLBACK_TOPICS, SOURCE_PRIORITIES
from src.utils.topic_dedup import filter_new_topics

logger = logging.getLogger(__name__)

_DEFAULT_DB = Path("data/processed/channel_forge.db")


class TopicQueue:
    """
    Multi-source priority topic selector with upload-history deduplication.

    Args:
        db_path: SQLite database containing scored_topics and competitor_topics.
        anthropic_api_key: Used only when generating a fresh Claude topic as
                           last resort. If None, reads ANTHROPIC_API_KEY from env.
    """

    def __init__(
        self,
        db_path: str | Path = _DEFAULT_DB,
        anthropic_api_key: str | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.anthropic_api_key = (
            anthropic_api_key if anthropic_api_key is not None
            else os.getenv("ANTHROPIC_API_KEY", "")
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_next_topics(
        self,
        category: str,
        max_count: int = 3,
        uploaded_topics: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return up to max_count topics, ordered by priority, excluding duplicates.

        Sources queried in priority order:
          1. competitor_topics (COMPETITOR_HIGH_SIGNAL / YOUTUBE_TRENDING / VIEWER_REQUESTED)
          2. scored_topics     (GOOGLE_TRENDS / YOUTUBE_KEYWORD / OPTIMIZATION etc.)
          3. FALLBACK_TOPICS   constant list
          4. Claude API        (last resort, only if all above are exhausted)

        Args:
            category: Channel category slug ("money", "career", "success").
            max_count: Maximum topics to return.
            uploaded_topics: Previously produced topic strings (for dedup).

        Returns:
            List of topic dicts: {"topic_id", "keyword", "category", "score", "source"}.
        """
        uploaded = uploaded_topics or []

        # Priority 0: Manual topics from Google Sheet queue (highest priority)
        manual = self._get_manual_topics(category, max_count)
        if manual:
            logger.info(
                "[topic_queue] Returning %d manual topic(s) before AI queue",
                len(manual),
            )
            if len(manual) >= max_count:
                return manual[:max_count]
            # Fill remaining slots from AI queue below
            max_count -= len(manual)

        all_candidates = self._gather_all_candidates(category)

        # Sort by priority score descending
        all_candidates.sort(key=lambda t: t["priority_score"], reverse=True)

        # Filter duplicates
        keywords = [t["keyword"] for t in all_candidates]
        fresh_keywords = filter_new_topics(keywords, uploaded)

        # Rebuild ordered list using only fresh keywords
        seen: set[str] = set(fresh_keywords)
        ordered = [t for t in all_candidates if t["keyword"] in seen]

        # Remove inter-candidate duplicates (filter_new_topics already handles this,
        # but the dict list may have dupes from multiple sources)
        deduped: list[dict[str, Any]] = []
        used_keys: set[str] = set()
        for t in ordered:
            kw = t["keyword"].lower().strip()
            if kw not in used_keys:
                used_keys.add(kw)
                deduped.append(t)

        selected = deduped[:max_count]

        # If still short, try fallbacks
        if len(selected) < max_count:
            fallback_candidates = FALLBACK_TOPICS.get(category, []) + FALLBACK_TOPICS.get("money", [])
            fresh_fallbacks = filter_new_topics(
                fallback_candidates,
                uploaded + [t["keyword"] for t in selected],
            )
            for i, keyword in enumerate(fresh_fallbacks):
                if len(selected) >= max_count:
                    break
                selected.append({
                    "topic_id":       f"fallback_{i:03d}",
                    "keyword":        keyword,
                    "category":       category,
                    "score":          float(SOURCE_PRIORITIES["FALLBACK"]),
                    "priority_score": SOURCE_PRIORITIES["FALLBACK"],
                    "source":         "FALLBACK",
                })

        # Last resort: generate a fresh topic via Claude
        if not selected:
            fresh = self._generate_fresh_topic(category)
            if fresh:
                selected.append({
                    "topic_id":       "claude_fresh_000",
                    "keyword":        fresh,
                    "category":       category,
                    "score":          float(SOURCE_PRIORITIES["FALLBACK"]),
                    "priority_score": SOURCE_PRIORITIES["FALLBACK"],
                    "source":         "CLAUDE_GENERATED",
                })

        # Prepend any manual topics found earlier
        if manual:
            selected = manual + selected

        logger.info(
            "TopicQueue returned %d topics for category='%s'",
            len(selected), category,
        )
        return selected

    def get_uploaded_topics(self) -> list[str]:
        """
        Query the DB for all previously produced topic keywords and video titles.

        Returns:
            List of topic strings (raw, not normalised — dedup normalises internally).
        """
        topics: list[str] = []
        if not self.db_path.exists():
            return topics

        try:
            conn = sqlite3.connect(self.db_path)
            try:
                # From production_results (keyword field)
                try:
                    rows = conn.execute(
                        "SELECT keyword FROM production_results WHERE keyword != ''"
                    ).fetchall()
                    topics.extend(r[0] for r in rows)
                except sqlite3.OperationalError:
                    pass

                # From uploaded_videos (title field)
                try:
                    rows = conn.execute(
                        "SELECT title FROM uploaded_videos WHERE title != ''"
                    ).fetchall()
                    topics.extend(r[0] for r in rows)
                except sqlite3.OperationalError:
                    pass
            finally:
                conn.close()
        except Exception as exc:
            logger.warning("Could not query upload history: %s", exc)

        return topics

    def mark_topic_used(self, keyword: str, category: str) -> None:
        """
        Mark a scored_topics row as used so it is excluded from future runs.

        Safe to call even if the table or DB don't exist yet — errors are
        logged and swallowed.

        Args:
            keyword: The topic keyword string to mark.
            category: The category the topic belongs to.
        """
        if not self.db_path.exists():
            return
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute(
                    "UPDATE scored_topics SET used = 1 WHERE keyword = ? AND category = ?",
                    (keyword, category),
                )
                conn.commit()
                logger.debug("Marked scored_topic used: %r (%s)", keyword, category)
            finally:
                conn.close()
        except Exception as exc:
            logger.warning("Could not mark topic '%s' as used: %s", keyword, exc)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_manual_topics(
        self,
        category: str,
        max_count: int,
    ) -> list[dict[str, Any]]:
        """Fetch QUEUED manual topics from the manual_topics table.

        First tries the per-channel DB (self.db_path).  If that returns no
        QUEUED rows, falls back to channel_forge.db in the same directory,
        because the GSheetTopicSync job always writes to channel_forge.db.

        All QUEUED topics are returned regardless of category — manual topics
        are user-curated and should always be produced in SEQ order.

        Returns up to ``max_count`` topics ordered by SEQ ascending.
        Each consumed topic is immediately marked USED in DB and Google Sheet.
        """
        results = self._query_and_mark_manual(self.db_path, category, max_count)

        # Fallback: per-channel DB had no QUEUED rows — try channel_forge.db
        if not results and self.db_path.name != "channel_forge.db":
            fallback_path = self.db_path.parent / "channel_forge.db"
            if fallback_path.exists() and fallback_path != self.db_path:
                results = self._query_and_mark_manual(
                    fallback_path, category, max_count, is_fallback=True
                )

        return results

    def _query_and_mark_manual(
        self,
        db_path: Path,
        category: str,
        max_count: int,
        is_fallback: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Open ``db_path``, fetch up to ``max_count`` QUEUED manual topics, mark
        them USED immediately, and return them as topic dicts.

        Manual topics are user-curated and produced regardless of category.
        The category label is preserved for downstream use (b-roll, metadata)
        but does not filter which topics get produced.

        Args:
            db_path: SQLite file to query.
            category: Channel category (used as fallback if topic has none).
            max_count: Maximum rows to return.
            is_fallback: When True, log messages indicate this is the fallback DB.
        """
        if not db_path.exists():
            logger.info("[topic_queue] manual_topics: DB does not exist at %s", db_path)
            return []

        results: list[dict[str, Any]] = []
        label = f"fallback {db_path.name}" if is_fallback else db_path.name

        try:
            conn = sqlite3.connect(db_path)
            try:
                # Fetch next QUEUED topic(s) regardless of category
                rows = conn.execute(
                    """
                    SELECT seq, title, category, hook_angle
                    FROM manual_topics
                    WHERE status = 'QUEUED'
                    ORDER BY seq ASC
                    LIMIT ?
                    """,
                    (max_count,),
                ).fetchall()

                logger.info(
                    "[topic_queue] manual_topics query (%s): %d QUEUED rows found",
                    label, len(rows),
                )

                # Log total queued count for visibility
                total_queued = conn.execute(
                    "SELECT COUNT(*) FROM manual_topics WHERE status = 'QUEUED'"
                ).fetchone()[0]
                logger.info(
                    "[topic_queue] manual_topics total QUEUED (%s): %d", label, total_queued,
                )

                for seq, title, cat, hook in rows:
                    conn.execute(
                        "UPDATE manual_topics SET status = 'USED', "
                        "used_at = datetime('now') WHERE seq = ?",
                        (seq,),
                    )
                    results.append({
                        "topic_id":       f"manual_{seq:03d}",
                        "keyword":        title,
                        "category":       cat or category,
                        "score":          100.0,
                        "priority_score": 100,
                        "source":         "MANUAL",
                        "manual_seq":     seq,
                        "hook_angle":     hook or "",
                    })
                    logger.info("[topic_queue] MANUAL SEQ %d: %s (cat=%s)", seq, title, cat)

                if results:
                    conn.commit()
                    # Write back USED status to Google Sheet
                    self._gsheet_mark_used(results)
                    if is_fallback:
                        logger.info(
                            "[topic_queue] manual_topics: using fallback channel_forge.db "
                            "(%d rows found)",
                            len(results),
                        )
            except sqlite3.OperationalError as op_exc:
                logger.info(
                    "[topic_queue] manual_topics table not found in %s "
                    "(will use AI queue): %s",
                    label, op_exc,
                )
            finally:
                conn.close()
        except Exception as exc:
            logger.warning("[topic_queue] Manual topic query failed (%s): %s", label, exc)

        return results

    def _gsheet_mark_used(self, topics: list[dict[str, Any]]) -> None:
        """Write USED status back to Google Sheet for consumed manual topics."""
        try:
            from src.crawler.gsheet_topic_sync import GSheetTopicSync
            sync = GSheetTopicSync()
            for t in topics:
                seq = t.get("manual_seq")
                if seq is not None:
                    sync.mark_used(seq=seq)
                    logger.info("[topic_queue] GSheet writeback: SEQ %d → USED", seq)
        except Exception as exc:
            logger.warning("[topic_queue] GSheet writeback failed (non-blocking): %s", exc)

    def _gather_all_candidates(self, category: str) -> list[dict[str, Any]]:
        """
        Query all DB topic sources and return a unified candidate list.

        Each item has keys: keyword, category, score, priority_score, source.
        """
        candidates: list[dict[str, Any]] = []
        if not self.db_path.exists():
            return candidates

        try:
            conn = sqlite3.connect(self.db_path)
            try:
                # Source 1: competitor_topics
                try:
                    rows = conn.execute(
                        """
                        SELECT extracted_topic, category, view_count, source
                        FROM competitor_topics
                        WHERE category = ? AND used = 0
                        ORDER BY view_count DESC
                        """,
                        (category,),
                    ).fetchall()
                    for row in rows:
                        src = row[3]
                        priority = SOURCE_PRIORITIES.get(src, SOURCE_PRIORITIES["YOUTUBE_KEYWORD"])
                        candidates.append({
                            "keyword":        row[0],
                            "category":       row[1],
                            "score":          float(priority),
                            "priority_score": priority,
                            "source":         src,
                        })
                except sqlite3.OperationalError:
                    pass

                # Source 2: scored_topics (from optimizer / sentiment analyzer)
                try:
                    rows = conn.execute(
                        """
                        SELECT keyword, category, score, source
                        FROM scored_topics
                        WHERE category = ? AND used = 0
                        ORDER BY score DESC
                        """,
                        (category,),
                    ).fetchall()
                    for row in rows:
                        src = row[3] if row[3] else "GOOGLE_TRENDS"
                        priority = SOURCE_PRIORITIES.get(src, SOURCE_PRIORITIES["GOOGLE_TRENDS"])
                        candidates.append({
                            "keyword":        row[0],
                            "category":       row[1] or category,
                            "score":          float(row[2]),
                            "priority_score": priority,
                            "source":         src,
                        })
                except sqlite3.OperationalError:
                    pass
            finally:
                conn.close()
        except Exception as exc:
            logger.warning("TopicQueue DB query failed: %s", exc)

        return candidates

    def _generate_fresh_topic(self, category: str) -> str:
        """
        Ask Claude to generate one fresh topic for the given category.

        Used only when all other sources (DB + fallbacks) are exhausted.
        Returns empty string if Claude API is unavailable.
        """
        if not self.anthropic_api_key:
            logger.warning("All topics exhausted and no Claude API key — returning empty")
            return ""

        prompt = (
            f"Generate one YouTube Shorts topic about {category} finance for a US audience. "
            f"Write it in plain lowercase English, 5 to 10 words, no hashtags. "
            f"Return only the topic, no explanation."
        )
        try:
            import anthropic as _ant
            client = _ant.Anthropic(api_key=self.anthropic_api_key)
            message = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=40,
                messages=[{"role": "user", "content": prompt}],
            )
            topic = message.content[0].text.strip().strip('"').strip("'")
            logger.info("Claude generated fresh topic: %r", topic)
            return topic
        except Exception as exc:
            logger.error("Claude fresh topic generation failed: %s", exc)
            return ""

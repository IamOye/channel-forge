"""
scraped_topic_processor.py — Scraped Topic Promotion Pipeline

Reads raw topics from the 'Scraped Topics' GSheet tab, scores them via Claude,
and promotes approved topics into:
  1. scored_topics SQLite table (immediate pipeline use)
  2. Topic Queue GSheet tab as READY rows (manual visibility + override)

Marks processed rows in col I of Scraped Topics tab to prevent reprocessing.

Schedule: runs every 6 hours alongside channel scrapers.

Manual override: topics added directly to Topic Queue GSheet are always
respected — this processor only fills the automated queue layer beneath them.
"""

import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_DB = Path("data/processed/channel_forge.db")

# Minimum view count to consider a scraped topic worth scoring
MIN_VIEWS_THRESHOLD = 5_000

# Maximum topics to promote per run per category (prevents flooding queue)
MAX_PROMOTE_PER_CATEGORY = 10

# Column index (1-based) for the "Processed" flag we add to Scraped Topics tab
PROCESSED_COL = 9  # col I

# Batch size for Claude scoring calls
CLAUDE_BATCH_SIZE = 10

# Categories the pipeline handles
VALID_CATEGORIES = {"money", "career", "success"}


# ---------------------------------------------------------------------------
# Weak topic filters (no Claude needed — fast pre-filter)
# ---------------------------------------------------------------------------

def _is_weak_topic(topic: str, original_title: str, views: int) -> tuple[bool, str]:
    """
    Return (True, reason) if topic should be discarded without Claude scoring.
    Fast heuristic checks only — Claude handles nuanced quality scoring.
    """
    if views < MIN_VIEWS_THRESHOLD:
        return True, f"low views ({views})"

    t = topic.strip()

    if len(t) < 10:
        return True, "too short"

    if len(t) > 120:
        return True, "too long"

    # Reject if topic appears to be in a non-English language
    # Simple heuristic: high ratio of non-ASCII characters
    non_ascii = sum(1 for c in t if ord(c) > 127)
    if non_ascii / max(len(t), 1) > 0.3:
        return True, "non-English"

    # Reject hashtag-heavy titles extracted as topics
    if t.count("#") > 2:
        return True, "hashtag-heavy"

    # Reject pure product/brand promotional content
    promo_signals = ["buy now", "discount", "sale ", "promo code", "coupon"]
    t_lower = t.lower()
    if any(p in t_lower for p in promo_signals):
        return True, "promotional"

    return False, ""


# ---------------------------------------------------------------------------
# Claude batch scorer
# ---------------------------------------------------------------------------

def _score_topics_with_claude(
    topics: list[dict[str, Any]],
    api_key: str,
) -> list[dict[str, Any]]:
    """
    Score a batch of topics via Claude. Returns topics with score >= 60.

    Each topic dict must have: topic, category, views, source.
    Returns list of approved topics with 'claude_score' added.
    """
    if not topics:
        return []

    try:
        import anthropic
    except ImportError:
        logger.error("[processor] anthropic package not installed")
        return []

    client = anthropic.Anthropic(api_key=api_key)

    # Build batch prompt
    topic_lines = "\n".join(
        f"{i+1}. [{t['category'].upper()}] {t['topic']} (views: {t['views']:,})"
        for i, t in enumerate(topics)
    )

    prompt = f"""You are scoring YouTube Shorts topics for a financial education channel called Money Heresy.

The channel's style: contrarian, provocative, real — it challenges conventional financial wisdom.
Target audience: English-speaking Western viewers aged 22-40 who feel financially stuck.
Winning topics: expose uncomfortable financial truths, challenge mainstream advice, create urgency.
Weak topics: generic advice, vague claims, non-English content, product promotions, celebrity gossip.

Score each topic from 0-100 based on:
- Relevance to personal finance struggles (0-30 pts)
- Provocative / curiosity-gap angle (0-30 pts)  
- Specificity and clarity (0-20 pts)
- Viral potential for Shorts format (0-20 pts)

Topics to score:
{topic_lines}

Respond with ONLY a JSON array. One object per topic in the same order:
[{{"score": 85, "reason": "exposes painful truth about salary"}}, ...]

No other text. Just the JSON array."""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        scores = json.loads(raw)

        approved = []
        for i, score_obj in enumerate(scores):
            if i >= len(topics):
                break
            score = int(score_obj.get("score", 0))
            if score >= 60:
                topic = topics[i].copy()
                topic["claude_score"] = score
                topic["claude_reason"] = score_obj.get("reason", "")
                approved.append(topic)
                logger.info(
                    "[processor] ✓ score=%d '%s' — %s",
                    score, topic["topic"][:60], topic["claude_reason"]
                )
            else:
                logger.debug(
                    "[processor] ✗ score=%d '%s'",
                    score, topics[i]["topic"][:60]
                )

        return approved

    except Exception as exc:
        logger.error("[processor] Claude scoring failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# DB writer
# ---------------------------------------------------------------------------

def _insert_into_scored_topics(
    topics: list[dict[str, Any]],
    db_path: Path,
) -> int:
    """
    Insert approved topics into scored_topics table.
    Skips duplicates (keyword + category).
    Returns count of rows inserted.
    """
    if not topics or not db_path.exists():
        return 0

    inserted = 0
    try:
        conn = sqlite3.connect(db_path)
        try:
            for t in topics:
                try:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO scored_topics
                            (keyword, category, score, source, used, created_at)
                        VALUES (?, ?, ?, ?, 0, datetime('now'))
                        """,
                        (
                            t["topic"],
                            t["category"],
                            float(t["claude_score"]),
                            t.get("source", "SCRAPED_PROMOTED"),
                        ),
                    )
                    inserted += 1
                except sqlite3.Error as row_exc:
                    logger.warning("[processor] DB insert skipped: %s", row_exc)
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:
        logger.error("[processor] DB write failed: %s", exc)

    return inserted


# ---------------------------------------------------------------------------
# GSheet operations
# ---------------------------------------------------------------------------

def _get_scraped_tab(spreadsheet):
    """Return the Scraped Topics worksheet."""
    import gspread
    try:
        return spreadsheet.worksheet("Scraped Topics")
    except gspread.WorksheetNotFound:
        logger.error("[processor] 'Scraped Topics' tab not found")
        return None


def _ensure_processed_header(ws) -> None:
    """Add 'Processed' header to col I row 1 if missing."""
    try:
        existing = ws.cell(1, PROCESSED_COL).value
        if not existing or not str(existing).strip():
            ws.update_cell(1, PROCESSED_COL, "Processed")
            logger.info("[processor] Added 'Processed' header to col I")
    except Exception as exc:
        logger.warning("[processor] Could not set col I header: %s", exc)


def _read_unprocessed_rows(ws) -> list[dict[str, Any]]:
    """
    Read all rows from Scraped Topics tab that haven't been processed yet.
    Row 1 = headers. Data starts row 2.
    Col I (index 8) = Processed flag.
    """
    try:
        all_values = ws.get_all_values()
    except Exception as exc:
        logger.error("[processor] Failed to read Scraped Topics tab: %s", exc)
        return []

    if len(all_values) < 2:
        return []

    # Headers in row 1
    headers = all_values[0]
    data_rows = all_values[1:]

    results = []
    for idx, row in enumerate(data_rows):
        # Pad to at least 9 cols
        padded = row + [""] * (PROCESSED_COL - len(row))

        processed_flag = padded[PROCESSED_COL - 1].strip().upper()
        if processed_flag in ("YES", "QUEUED", "SKIP"):
            continue  # already handled

        # Parse columns: A=Source, B=Channel, C=Original Title, D=Topic,
        #                E=Views, F=Category, G=Score, H=Date Scraped
        try:
            views_raw = padded[4].replace(",", "").strip()
            views = int(float(views_raw)) if views_raw else 0
        except (ValueError, TypeError):
            views = 0

        category = padded[5].strip().lower()
        if category not in VALID_CATEGORIES:
            category = "money"  # default fallback

        results.append({
            "sheet_row": idx + 2,  # 1-based, row 1 = headers
            "source": padded[0].strip(),
            "channel": padded[1].strip(),
            "original_title": padded[2].strip(),
            "topic": padded[3].strip(),
            "views": views,
            "category": category,
            "raw_score": padded[6].strip(),
            "scraped_at": padded[7].strip(),
        })

    return results


def _mark_rows_processed(ws, sheet_rows: list[int], flag: str = "YES") -> None:
    """Batch-update col I for the given sheet row numbers."""
    if not sheet_rows:
        return
    try:
        cell_updates = []
        for row_num in sheet_rows:
            cell_updates.append({
                "range": f"I{row_num}",
                "values": [[flag]],
            })
        ws.batch_update(cell_updates)
        logger.info("[processor] Marked %d rows as '%s' in col I", len(sheet_rows), flag)
    except Exception as exc:
        logger.error("[processor] Failed to mark processed rows: %s", exc)


# ---------------------------------------------------------------------------
# Main processor
# ---------------------------------------------------------------------------


def _read_reddit_tab(spreadsheet) -> list[dict]:
    """Read unprocessed rows from Reddit Topics GSheet tab."""
    try:
        import gspread as _gspread
        try:
            ws = spreadsheet.worksheet("Reddit Topics")
        except _gspread.WorksheetNotFound:
            return []
        return _read_unprocessed_rows(ws)
    except Exception as exc:
        logger.warning("[processor] Failed to read Reddit Topics tab: %s", exc)
        return []

class ScrapedTopicProcessor:
    """
    Promotes raw scraped topics into the production pipeline.

    Workflow:
      1. Read unprocessed rows from Scraped Topics GSheet tab
      2. Pre-filter weak topics (fast, no API)
      3. Score remaining topics via Claude in batches
      4. Insert approved topics into scored_topics DB
      5. Append approved topics to Topic Queue GSheet as READY
      6. Mark all processed rows in Scraped Topics tab col I

    Manual override: rows you add directly to Topic Queue GSheet are
    never touched by this processor. They are consumed first by the pipeline.
    """

    def __init__(
        self,
        db_path: str | Path = _DEFAULT_DB,
        sheet_id: str | None = None,
        credentials_b64: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.sheet_id = sheet_id or os.getenv("GOOGLE_SHEET_ID", "")
        self.credentials_b64 = credentials_b64 or os.getenv("GOOGLE_CREDENTIALS_B64", "")
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")

    def run(self) -> dict[str, int]:
        """
        Execute one full processing cycle.

        Returns:
            Dict with keys: total_read, pre_filtered, scored, approved, inserted, queued
        """
        stats = {
            "total_read": 0,
            "pre_filtered": 0,
            "scored": 0,
            "approved": 0,
            "inserted": 0,
            "queued": 0,
        }

        # Connect to GSheet
        try:
            from src.crawler.gsheet_topic_sync import get_gsheet_client, GSheetTopicSync
            _, spreadsheet = get_gsheet_client(self.sheet_id, self.credentials_b64)
        except Exception as exc:
            logger.error("[processor] GSheet connection failed: %s", exc)
            return stats

        ws = _get_scraped_tab(spreadsheet)
        if ws is None:
            return stats

        _ensure_processed_header(ws)

        # Read unprocessed rows from Scraped Topics tab
        raw_rows = _read_unprocessed_rows(ws)

        # Also read from Reddit Topics tab
        reddit_rows = _read_reddit_tab(spreadsheet)
        raw_rows = raw_rows + reddit_rows

        stats["total_read"] = len(raw_rows)
        logger.info(
            "[processor] Read %d unprocessed rows (scraped=%d, reddit=%d)",
            len(raw_rows), len(raw_rows) - len(reddit_rows), len(reddit_rows),
        )

        if not raw_rows:
            logger.info("[processor] No new rows to process")
            return stats

        # Pre-filter weak topics
        candidates = []
        skipped_rows = []

        for row in raw_rows:
            weak, reason = _is_weak_topic(row["topic"], row["original_title"], row["views"])
            if weak:
                logger.debug("[processor] PRE-FILTER '%s': %s", row["topic"][:50], reason)
                stats["pre_filtered"] += 1
                skipped_rows.append(row["sheet_row"])
            else:
                candidates.append(row)

        # Mark skipped rows in GSheet
        if skipped_rows:
            _mark_rows_processed(ws, skipped_rows, flag="SKIP")

        if not candidates:
            logger.info("[processor] All rows pre-filtered — nothing to score")
            return stats

        logger.info("[processor] %d candidates pass pre-filter, sending to Claude", len(candidates))
        stats["scored"] = len(candidates)

        # Category-aware promotion limit: track how many approved per category
        approved_per_category: dict[str, int] = {c: 0 for c in VALID_CATEGORIES}

        # Score in batches
        all_approved = []
        processed_rows = []  # all rows we've attempted (for col I marking)

        for i in range(0, len(candidates), CLAUDE_BATCH_SIZE):
            batch = candidates[i : i + CLAUDE_BATCH_SIZE]
            approved_batch = _score_topics_with_claude(batch, self.api_key)
            all_approved.extend(approved_batch)
            processed_rows.extend(r["sheet_row"] for r in batch)

        stats["approved"] = len(all_approved)
        logger.info("[processor] Claude approved %d / %d topics", len(all_approved), len(candidates))

        # Apply per-category promotion cap and dedup against existing queue
        existing_topics = self._get_existing_queue_topics(spreadsheet)
        sync = GSheetTopicSync(self.sheet_id, self.credentials_b64)

        promote = []
        for t in sorted(all_approved, key=lambda x: x["claude_score"], reverse=True):
            cat = t["category"]
            if approved_per_category.get(cat, 0) >= MAX_PROMOTE_PER_CATEGORY:
                continue

            # Dedup: skip if very similar topic already in queue
            if self._is_duplicate(t["topic"], existing_topics):
                logger.debug("[processor] DEDUP skip: '%s'", t["topic"][:60])
                continue

            promote.append(t)
            approved_per_category[cat] = approved_per_category.get(cat, 0) + 1
            existing_topics.add(t["topic"].lower().strip())

        # Insert into scored_topics DB
        db_inserted = _insert_into_scored_topics(promote, self.db_path)
        stats["inserted"] = db_inserted

        # Append to Topic Queue GSheet tab as READY
        queued_count = 0
        for t in promote:
            try:
                notes = (
                    f"Auto-promoted | source={t.get('source','')} | "
                    f"views={t['views']:,} | claude_score={t['claude_score']}"
                )
                sync.append_topic(
                    title=t["topic"],
                    category=t["category"],
                    hook_angle="",
                    notes=notes,
                )
                queued_count += 1
            except Exception as exc:
                logger.error("[processor] GSheet append failed for '%s': %s", t["topic"][:50], exc)

        stats["queued"] = queued_count
        logger.info(
            "[processor] Promoted %d topics to queue (%d to DB, %d to GSheet)",
            len(promote), db_inserted, queued_count,
        )

        # Mark all processed rows in Scraped Topics tab
        if processed_rows:
            _mark_rows_processed(ws, processed_rows, flag="YES")

        # Log summary per category
        for cat, count in approved_per_category.items():
            if count > 0:
                logger.info("[processor] Category '%s': %d topics promoted", cat, count)

        return stats

    def _get_existing_queue_topics(self, spreadsheet) -> set[str]:
        """Return lowercase set of topics already in Topic Queue tab."""
        existing = set()
        try:
            ws = spreadsheet.worksheet("Topic Queue")
            all_vals = ws.get_all_values()
            # Headers in row 3 (index 2), data from row 4
            if len(all_vals) > 3:
                for row in all_vals[3:]:
                    if len(row) > 2:
                        topic = str(row[2]).strip().lower()
                        if topic:
                            existing.add(topic)
        except Exception as exc:
            logger.warning("[processor] Could not read existing queue topics: %s", exc)
        return existing

    def _is_duplicate(self, topic: str, existing: set[str]) -> bool:
        """
        Simple duplicate check — exact match or very high word overlap.
        """
        t_lower = topic.lower().strip()
        if t_lower in existing:
            return True

        # Word overlap check: if 80%+ of words match an existing topic
        t_words = set(t_lower.split())
        if len(t_words) < 3:
            return t_lower in existing

        for existing_topic in existing:
            e_words = set(existing_topic.split())
            if not e_words:
                continue
            overlap = len(t_words & e_words) / max(len(t_words), len(e_words))
            if overlap >= 0.8:
                return True

        return False


# ---------------------------------------------------------------------------
# Standalone runner (for testing / manual trigger)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    from dotenv import load_dotenv
    load_dotenv()

    processor = ScrapedTopicProcessor()
    result = processor.run()

    print("\n=== Scraped Topic Processor Results ===")
    print(f"  Total rows read:   {result['total_read']}")
    print(f"  Pre-filtered out:  {result['pre_filtered']}")
    print(f"  Sent to Claude:    {result['scored']}")
    print(f"  Claude approved:   {result['approved']}")
    print(f"  Inserted to DB:    {result['inserted']}")
    print(f"  Added to queue:    {result['queued']}")

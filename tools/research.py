#!/usr/bin/env python3
"""
research.py — ChannelForge Local Research Tool

Standalone script for topic research. Scrapes Reddit, YouTube autocomplete,
Google Trends, and competitor channels, then scores with Claude Haiku and
presents an interactive ranked list for manual curation.

Usage:
    python tools/research.py                     # full scrape, top 50
    python tools/research.py --source reddit     # Reddit only
    python tools/research.py --source competitor # competitor channels only
    python tools/research.py --category money    # filter to money
    python tools/research.py --count 100         # show top 100
    python tools/research.py --no-score          # skip Claude scoring

Requires: pip install rich gspread google-auth anthropic httpx
"""

import argparse
import json
import logging
import os
import re
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path so we can import src/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

load_dotenv(_PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("research")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_PATH = _PROJECT_ROOT / "data" / "processed" / "channel_forge.db"


def _safe_str(value: object) -> str:
    """Convert any value to a stripped string, treating None as empty."""
    if value is None:
        return ""
    return str(value).strip()

AUTOCOMPLETE_SEEDS = [
    "why your boss", "how to save money",
    "salary negotiation", "passive income",
    "why most people", "how to invest",
    "financial freedom", "side hustle",
    "why the rich", "how to build wealth",
]

TREND_KEYWORDS = ["money", "salary", "investing", "career", "financial freedom"]

COMPETITOR_CHANNELS = [
    "ImpactTheory", "LewisHowes", "GrahamStephan",
    "AliAbdaal", "MinorityMindset", "AndreiJikh",
]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RawTopic:
    """A single raw topic from any source."""
    title: str
    source: str             # "reddit", "autocomplete", "trends", "competitor"
    source_detail: str = "" # subreddit name, channel name, keyword, etc.
    score_hint: float = 0.0 # upvotes, views, rising_score — for display
    extra: dict = field(default_factory=dict)


@dataclass
class ScoredTopic:
    """A topic after Claude Haiku scoring."""
    title: str
    score: float
    category: str
    hook_angle: str
    reason: str
    source: str
    source_detail: str
    score_hint: float = 0.0


# ---------------------------------------------------------------------------
# Phase 1: Scrape
# ---------------------------------------------------------------------------

def _scrape_reddit() -> list[RawTopic]:
    """Scrape finance subreddits using existing RedditScraper."""
    topics: list[RawTopic] = []
    try:
        from src.crawler.reddit_scraper import RedditScraper
        scraper = RedditScraper()
        results = scraper.scrape_finance_subreddits()
        for r in results:
            title = _safe_str(r.keyword)
            if not title:
                continue
            topics.append(RawTopic(
                title=title,
                source="reddit",
                source_detail=f"r/{_safe_str(r.subreddit)}",
                score_hint=float(r.upvotes or 0),
                extra={"category": _safe_str(r.category), "upvotes": r.upvotes or 0},
            ))
        logger.info("[scrape] Reddit: %d topics", len(topics))
    except Exception as exc:
        logger.warning("[scrape] Reddit failed: %s", exc)
    return topics


def _scrape_autocomplete() -> list[RawTopic]:
    """Scrape YouTube autocomplete suggestions."""
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
                # Response is JSONP — extract JSON array
                text = resp.text
                # Find first '[' and parse from there
                start = text.index("[")
                data = json.loads(text[start:])
                suggestions = [s[0] for s in data[1]] if len(data) > 1 else []
                for sug in suggestions:
                    s = _safe_str(sug)
                    if s and s.lower() != seed.lower():
                        topics.append(RawTopic(
                            title=s,
                            source="autocomplete",
                            source_detail=seed,
                        ))
            except Exception as exc:
                logger.debug("[scrape] Autocomplete failed for '%s': %s", seed, exc)
        logger.info("[scrape] Autocomplete: %d topics", len(topics))
    except Exception as exc:
        logger.warning("[scrape] Autocomplete failed: %s", exc)
    return topics


def _scrape_trends() -> list[RawTopic]:
    """Scrape Google Trends rising queries."""
    topics: list[RawTopic] = []
    try:
        from src.crawler.trend_scraper import TrendScrapingEngine
        engine = TrendScrapingEngine()
        signals = engine.fetch_all(TREND_KEYWORDS)
        for s in signals:
            kw = _safe_str(s.keyword)
            if not kw:
                continue
            topics.append(RawTopic(
                title=kw,
                source="trends",
                source_detail=_safe_str(s.source),
                score_hint=float(s.interest_score or 0),
            ))
            # Also include related queries as topics
            for rq in (s.related_queries or []):
                rq_str = _safe_str(rq)
                if rq_str:
                    topics.append(RawTopic(
                        title=rq_str,
                        source="trends",
                        source_detail=f"related/{kw}",
                    ))
        logger.info("[scrape] Trends: %d topics", len(topics))
    except Exception as exc:
        logger.warning("[scrape] Trends failed: %s", exc)
    return topics


def _scrape_competitors() -> list[RawTopic]:
    """Scrape recent uploads from competitor channels."""
    topics: list[RawTopic] = []
    try:
        from src.crawler.competitor_scraper import CompetitorScraper
        scraper = CompetitorScraper()
        for category in ("money", "career", "success"):
            try:
                extracted = scraper.scrape_competitor_topics(category)
                for title in extracted:
                    t = _safe_str(title)
                    if not t:
                        continue
                    topics.append(RawTopic(
                        title=t,
                        source="competitor",
                        source_detail=category,
                    ))
            except Exception as exc:
                logger.debug("[scrape] Competitor %s failed: %s", category, exc)

        # Also scrape autocomplete via competitor scraper
        for category in ("money", "career", "success"):
            try:
                suggestions = scraper.scrape_search_autocomplete(category)
                for sug in suggestions:
                    s = _safe_str(sug)
                    if not s:
                        continue
                    topics.append(RawTopic(
                        title=s,
                        source="autocomplete",
                        source_detail=f"yt/{category}",
                    ))
            except Exception:
                pass

        logger.info("[scrape] Competitors: %d topics", len(topics))
    except Exception as exc:
        logger.warning("[scrape] Competitors failed: %s", exc)
    return topics


def run_scrape(sources: list[str] | None = None) -> list[RawTopic]:
    """Run all scrapers in parallel and collect raw topics."""
    scrapers = {
        "reddit": _scrape_reddit,
        "autocomplete": _scrape_autocomplete,
        "trends": _scrape_trends,
        "competitor": _scrape_competitors,
    }

    if sources:
        scrapers = {k: v for k, v in scrapers.items() if k in sources}

    all_topics: list[RawTopic] = []

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fn): name for name, fn in scrapers.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                all_topics.extend(result)
            except Exception as exc:
                logger.error("[scrape] %s raised: %s", name, exc)

    logger.info("[scrape] Total raw topics: %d", len(all_topics))
    return all_topics


# ---------------------------------------------------------------------------
# Phase 2: Deduplicate & Clean
# ---------------------------------------------------------------------------

def _normalise_title(title: str | None) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    if not title:
        return ""
    t = str(title).lower().strip()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _load_existing_topics() -> set[str]:
    """Load normalised titles from uploaded_videos + manual_topics in DB.

    Checks against:
      - uploaded_videos: already produced as a video
      - manual_topics (USED/QUEUED/HOLD): already in the Sheet queue

    Does NOT check scored_topics — that table is the research pool, not a filter.
    """
    existing: set[str] = set()
    if not DB_PATH.exists():
        return existing
    try:
        conn = sqlite3.connect(DB_PATH)
        try:
            # Already-published videos
            try:
                rows = conn.execute("SELECT title FROM uploaded_videos").fetchall()
                for (val,) in rows:
                    existing.add(_normalise_title(val))
            except sqlite3.OperationalError:
                pass

            # Already-published via production_results
            try:
                rows = conn.execute(
                    "SELECT keyword FROM production_results WHERE keyword != ''"
                ).fetchall()
                for (val,) in rows:
                    existing.add(_normalise_title(val))
            except sqlite3.OperationalError:
                pass

            # Already in manual queue (USED, QUEUED, or HOLD)
            try:
                rows = conn.execute(
                    "SELECT title FROM manual_topics "
                    "WHERE status IN ('USED', 'QUEUED', 'HOLD')"
                ).fetchall()
                for (val,) in rows:
                    existing.add(_normalise_title(val))
            except sqlite3.OperationalError:
                pass
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("[dedup] DB load failed: %s", exc)

    # Remove empty string if present
    existing.discard("")
    return existing


def deduplicate(topics: list[RawTopic]) -> list[RawTopic]:
    """Remove duplicates, too-short titles, and already-used topics."""
    existing = _load_existing_topics()
    seen: set[str] = set()
    result: list[RawTopic] = []

    for t in topics:
        title = _safe_str(t.title)
        if not title:
            continue

        # Skip titles under 5 words
        if len(title.split()) < 5:
            continue

        # Skip non-ASCII-heavy titles (rough non-English filter)
        ascii_ratio = sum(1 for c in title if ord(c) < 128) / max(len(title), 1)
        if ascii_ratio < 0.7:
            continue

        norm = _normalise_title(title)
        if not norm:
            continue

        # Exact duplicate
        if norm in seen:
            continue

        # Already in DB
        if norm in existing:
            continue

        seen.add(norm)
        result.append(t)

    logger.info("[dedup] %d topics after deduplication (from %d)", len(result), len(topics))
    return result


# ---------------------------------------------------------------------------
# Phase 3: Score with Claude Haiku
# ---------------------------------------------------------------------------

def score_topics(topics: list[RawTopic]) -> list[ScoredTopic]:
    """Score topics in batches of 20 using Claude Haiku."""
    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.warning("[score] ANTHROPIC_API_KEY not set — returning unscored")
        return [
            ScoredTopic(
                title=_safe_str(t.title) or "untitled", score=0.0,
                category="money", hook_angle="", reason="unscored",
                source=_safe_str(t.source), source_detail=_safe_str(t.source_detail),
                score_hint=float(t.score_hint or 0),
            )
            for t in topics
        ]

    client = anthropic.Anthropic(api_key=api_key)
    scored: list[ScoredTopic] = []
    batch_size = 20

    from rich.progress import Progress
    with Progress() as progress:
        task = progress.add_task("[cyan]Scoring with Claude Haiku...", total=len(topics))

        for i in range(0, len(topics), batch_size):
            batch = topics[i:i + batch_size]
            titles = [t.title for t in batch]

            prompt = (
                "Score each of these YouTube Shorts topic ideas "
                "for the @moneyheresy channel (money/career/success, "
                "contrarian angle, Western audience US/UK/CA/AU).\n\n"
                "For each topic return JSON:\n"
                "{\n"
                '  "title": string,\n'
                '  "score": 1-10,\n'
                '  "category": "money"|"career"|"success",\n'
                '  "hook_angle": one sentence hook,\n'
                '  "reason": why this works (15 words max)\n'
                "}\n\n"
                "Scoring criteria:\n"
                "- Emotional trigger: curiosity/anger/fear/aspiration\n"
                "- Contrarian potential: challenges conventional wisdom\n"
                "- Search volume signal: people are searching this\n"
                "- Brand fit: suits Money Heresy style\n\n"
                "Return ONLY a JSON array. No other text.\n\n"
                f"Topics:\n{json.dumps(titles, indent=2)}"
            )

            try:
                message = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = message.content[0].text.strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                items = json.loads(raw.strip())

                # Map scored items back to source info
                title_to_raw = {t.title: t for t in batch}
                for item in items:
                    title = _safe_str(item.get("title"))
                    raw_topic = title_to_raw.get(title) or (batch[0] if batch else None)
                    scored.append(ScoredTopic(
                        title=title or "untitled",
                        score=float(item.get("score") or 0),
                        category=_safe_str(item.get("category")) or "money",
                        hook_angle=_safe_str(item.get("hook_angle")),
                        reason=_safe_str(item.get("reason")),
                        source=_safe_str(raw_topic.source) if raw_topic else "unknown",
                        source_detail=_safe_str(raw_topic.source_detail) if raw_topic else "",
                        score_hint=float(raw_topic.score_hint or 0) if raw_topic else 0,
                    ))
            except Exception as exc:
                logger.warning("[score] Batch %d failed: %s — adding unscored", i, exc)
                for t in batch:
                    scored.append(ScoredTopic(
                        title=_safe_str(t.title) or "untitled", score=0.0,
                        category="money", hook_angle="", reason="scoring failed",
                        source=_safe_str(t.source), source_detail=_safe_str(t.source_detail),
                        score_hint=float(t.score_hint or 0),
                    ))

            progress.update(task, advance=len(batch))

    scored.sort(key=lambda s: s.score, reverse=True)
    logger.info("[score] Scored %d topics", len(scored))
    return scored


# ---------------------------------------------------------------------------
# Phase 4: Display with Rich
# ---------------------------------------------------------------------------

def display_topics(scored: list[ScoredTopic], count: int = 50, offset: int = 0) -> None:
    """Print a ranked table of scored topics using rich."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    today = datetime.now().strftime("%Y-%m-%d")
    total = len(scored)
    showing = scored[offset:offset + count]

    console.print()
    console.print("─" * 65, style="bold blue")
    console.print(
        f"CHANNELFORGE TOPIC RESEARCH — {today}",
        style="bold white",
    )
    console.print(
        f"Scored: {total} | Showing {offset + 1}–{offset + len(showing)} of {total}",
        style="dim",
    )
    console.print("─" * 65, style="bold blue")
    console.print()

    for i, t in enumerate(showing, start=offset + 1):
        # Score colour
        if t.score >= 8:
            score_style = "bold green"
        elif t.score >= 6:
            score_style = "yellow"
        else:
            score_style = "dim"

        cat = _safe_str(t.category) or "money"
        cat_colors = {"money": "green", "career": "cyan", "success": "magenta"}
        cat_style = cat_colors.get(cat, "white")
        title_display = _safe_str(t.title) or "(untitled)"
        hook = _safe_str(t.hook_angle)

        console.print(
            f" [{score_style}]{i:>3}[/]  "
            f"[{score_style}]{t.score:.1f}[/]  "
            f"[{cat_style}]{cat:<8}[/] "
            f"[bold white]{title_display}[/]"
        )
        if hook:
            console.print(f"              [dim italic]Hook: {hook}[/]")

        # Source line
        source_str = _safe_str(t.source) or "unknown"
        detail = _safe_str(t.source_detail)
        if detail:
            source_str += f"/{detail}"
        if t.score_hint > 0:
            if t.source == "reddit":
                source_str += f" ({int(t.score_hint)} upvotes)"
            elif t.source == "competitor":
                source_str += f" ({int(t.score_hint):,} views)"
        console.print(f"              [dim]From: {source_str}[/]")
        console.print()

    console.print("─" * 65, style="bold blue")


# ---------------------------------------------------------------------------
# Phase 5: Google Sheets integration
# ---------------------------------------------------------------------------

def _get_gsheet():
    """Get the Google Sheet worksheet (Topic Queue tab)."""
    import gspread
    from google.oauth2.service_account import Credentials
    import base64

    sheet_id = os.getenv("GOOGLE_SHEET_ID", "")
    if not sheet_id:
        raise ValueError("GOOGLE_SHEET_ID not set in .env")

    # Try base64-encoded credentials first, then file path
    creds_b64 = os.getenv("GOOGLE_CREDENTIALS_B64", "")
    creds_file = os.getenv("GOOGLE_CREDENTIALS_FILE", "")

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]

    if creds_b64:
        creds_json = json.loads(base64.b64decode(creds_b64))
        creds = Credentials.from_service_account_info(creds_json, scopes=scopes)
    elif creds_file:
        creds = Credentials.from_service_account_file(creds_file, scopes=scopes)
    else:
        raise ValueError(
            "Set GOOGLE_CREDENTIALS_B64 or GOOGLE_CREDENTIALS_FILE in .env"
        )

    gc = gspread.authorize(creds)
    spreadsheet = gc.open_by_key(sheet_id)

    # Get or create "Topic Queue" worksheet
    try:
        ws = spreadsheet.worksheet("Topic Queue")
    except gspread.exceptions.WorksheetNotFound:
        ws = spreadsheet.add_worksheet("Topic Queue", rows=500, cols=10)
        ws.update("A1:G1", [["SEQ", "Title", "Category", "Status",
                              "Date Added", "Hook Angle", "Notes"]])
        ws.format("A1:G1", {"textFormat": {"bold": True}})

    return ws


def _next_seq(ws) -> int:
    """Get the next SEQ number from the sheet."""
    col_a = ws.col_values(1)
    # Filter to numeric values (skip header)
    nums = []
    for v in col_a[1:]:
        try:
            nums.append(int(v))
        except (ValueError, TypeError):
            pass
    return max(nums, default=0) + 1


def add_to_sheet(topics: list[ScoredTopic]) -> tuple[int, int]:
    """Append scored topics to Google Sheet. Returns (first_seq, last_seq)."""
    ws = _get_gsheet()
    seq = _next_seq(ws)
    first_seq = seq
    today = datetime.now().strftime("%Y-%m-%d")

    rows = []
    for t in topics:
        notes = f"Source: {t.source}"
        if t.source_detail:
            notes += f"/{t.source_detail}"
        notes += f" | Score: {t.score:.1f}"
        if t.reason:
            notes += f" | {t.reason}"

        rows.append([
            seq,
            t.title,
            t.category,
            "READY",
            today,
            t.hook_angle,
            notes,
        ])
        seq += 1

    if rows:
        # Append below last row
        ws.append_rows(rows, value_input_option="USER_ENTERED")

    return first_seq, seq - 1


# ---------------------------------------------------------------------------
# Phase 5: Interactive Review
# ---------------------------------------------------------------------------

def interactive_review(scored: list[ScoredTopic], count: int = 50) -> None:
    """Interactive loop: display topics, accept commands."""
    from rich.console import Console
    console = Console()
    offset = 0

    while True:
        showing = scored[offset:offset + count]
        if not showing:
            console.print("[yellow]No more topics to show.[/]")
            break

        display_topics(scored, count=count, offset=offset)

        console.print(
            "[bold]Commands:[/] Enter topic #s (e.g. 1,3,5-8) then "
            "[green][A]dd to sheet[/]  [yellow][S]kip[/]  "
            "[cyan][E]dit # title[/]  [blue][R]efresh next batch[/]  "
            "[red][Q]uit[/]"
        )
        console.print()

        selected_indices: list[int] = []

        while True:
            try:
                cmd = input("→ ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Goodbye.[/]")
                return

            if not cmd:
                continue

            upper = cmd.upper()

            if upper == "Q":
                console.print("[dim]Goodbye.[/]")
                return

            if upper == "R":
                offset += count
                break  # redraw

            if upper.startswith("E"):
                # Edit: "E 3" or "E3"
                parts = cmd.split(None, 1)
                num_str = parts[1] if len(parts) > 1 else parts[0][1:]
                try:
                    idx = int(num_str) - 1
                    if 0 <= idx < len(scored):
                        new_title = input(f"  New title [{scored[idx].title}]: ").strip()
                        if new_title:
                            scored[idx] = ScoredTopic(
                                title=new_title,
                                score=scored[idx].score,
                                category=scored[idx].category,
                                hook_angle=scored[idx].hook_angle,
                                reason=scored[idx].reason,
                                source=scored[idx].source,
                                source_detail=scored[idx].source_detail,
                                score_hint=scored[idx].score_hint,
                            )
                            console.print(f"  [green]Updated #{idx + 1}[/]")
                    else:
                        console.print(f"  [red]#{idx + 1} out of range[/]")
                except ValueError:
                    console.print("  [red]Usage: E <number>[/]")
                continue

            if upper == "A":
                if not selected_indices:
                    console.print("[yellow]No topics selected. Enter numbers first.[/]")
                    continue
                to_add = [scored[i] for i in selected_indices if 0 <= i < len(scored)]
                if not to_add:
                    console.print("[yellow]No valid topics selected.[/]")
                    continue
                try:
                    first, last = add_to_sheet(to_add)
                    console.print(
                        f"[bold green]Added {len(to_add)} topics to Google Sheet "
                        f"(SEQ {first}–{last})[/]"
                    )
                except Exception as exc:
                    console.print(f"[bold red]Google Sheet error: {exc}[/]")
                selected_indices = []
                continue

            if upper == "S":
                selected_indices = []
                console.print("[dim]Selection cleared.[/]")
                continue

            # Try to parse as topic numbers: "1,3,5-8"
            try:
                for part in cmd.split(","):
                    part = part.strip()
                    if "-" in part:
                        lo, hi = part.split("-", 1)
                        for n in range(int(lo), int(hi) + 1):
                            selected_indices.append(n - 1)
                    else:
                        selected_indices.append(int(part) - 1)
                # Deduplicate
                selected_indices = sorted(set(selected_indices))
                titles = [scored[i].title for i in selected_indices
                          if 0 <= i < len(scored)]
                console.print(
                    f"[cyan]Selected {len(titles)} topic(s).[/] "
                    f"Press [green]A[/] to add to sheet, [yellow]S[/] to clear."
                )
            except ValueError:
                console.print("[red]Invalid input. Enter numbers, A, S, E #, R, or Q.[/]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ChannelForge Topic Research Tool",
    )
    parser.add_argument(
        "--source", type=str, default=None,
        choices=["reddit", "autocomplete", "trends", "competitor"],
        help="Scrape only this source (default: all)",
    )
    parser.add_argument(
        "--category", type=str, default=None,
        choices=["money", "career", "success"],
        help="Filter to this category after scoring",
    )
    parser.add_argument(
        "--count", type=int, default=50,
        help="Number of topics to display (default: 50)",
    )
    parser.add_argument(
        "--no-score", action="store_true",
        help="Skip Claude scoring, show raw scraped topics",
    )
    args = parser.parse_args()

    from rich.console import Console
    console = Console()

    # Phase 1: Scrape
    console.print("\n[bold cyan]Phase 1: Scraping sources...[/]\n")
    sources = [args.source] if args.source else None
    raw = run_scrape(sources)

    if not raw:
        console.print("[bold red]No topics found. Check your API keys and network.[/]")
        return

    console.print(f"[green]Scraped {len(raw)} raw topics[/]\n")

    # Phase 2: Deduplicate
    console.print("[bold cyan]Phase 2: Deduplicating...[/]\n")
    clean = deduplicate(raw)
    console.print(f"[green]{len(clean)} topics after dedup[/]\n")

    if not clean:
        console.print("[bold red]No topics remaining after dedup.[/]")
        return

    # Phase 3: Score
    if args.no_score:
        scored = [
            ScoredTopic(
                title=_safe_str(t.title) or "untitled", score=0.0,
                category="unknown", hook_angle="", reason="",
                source=_safe_str(t.source), source_detail=_safe_str(t.source_detail),
                score_hint=float(t.score_hint or 0),
            )
            for t in clean
        ]
        # Sort by score_hint (upvotes/views) as fallback
        scored.sort(key=lambda s: s.score_hint, reverse=True)
    else:
        console.print("[bold cyan]Phase 3: Scoring with Claude Haiku...[/]\n")
        scored = score_topics(clean)

    # Category filter
    if args.category:
        scored = [s for s in scored if s.category == args.category]
        console.print(f"[dim]Filtered to {args.category}: {len(scored)} topics[/]\n")

    if not scored:
        console.print("[bold red]No scored topics to display.[/]")
        return

    # Phase 4+5: Display & Interactive Review
    interactive_review(scored, count=args.count)


if __name__ == "__main__":
    main()

"""
check_elevenlabs_usage.py — Print a monthly ElevenLabs usage report.

Usage:
    .venv\\Scripts\\python.exe scripts\\check_elevenlabs_usage.py

Reads from the elevenlabs_usage table in channel_forge.db.
Respects ELEVENLABS_MONTHLY_LIMIT and ELEVENLABS_RESET_DAY from .env.
"""

import os
import sqlite3
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_PATH = Path(os.getenv("DB_PATH", "data/processed/channel_forge.db"))
MONTHLY_LIMIT = int(os.getenv("ELEVENLABS_MONTHLY_LIMIT", "30000"))
RESET_DAY = int(os.getenv("ELEVENLABS_RESET_DAY", "1"))


# ---------------------------------------------------------------------------
# Report logic (importable for tests and scheduler)
# ---------------------------------------------------------------------------

def get_usage_report(db_path: Path = DB_PATH, monthly_limit: int = MONTHLY_LIMIT, reset_day: int = RESET_DAY) -> dict:
    """
    Query elevenlabs_usage and return a structured usage dict.

    Returns:
        {
            "month_label":        "March 2026",
            "monthly_limit":      30000,
            "monthly_total":      12450,
            "pct_used":           41.5,
            "chars_remaining":    17550,
            "videos_produced":    15,
            "avg_chars_per_video": 830,
            "videos_remaining":   21,
            "reset_date":         "April 1, 2026",
            "status":             "OK — plenty of headroom",
            "daily_breakdown":    [{"date": "2026-03-08", "chars": 2400, "videos": 3}, ...],
        }
    """
    today = date.today()

    # Determine the start of the current billing cycle
    month_start = today.replace(day=reset_day)
    if today.day < reset_day:
        if today.month == 1:
            month_start = month_start.replace(year=today.year - 1, month=12)
        else:
            month_start = month_start.replace(month=today.month - 1)

    # Determine the next reset date
    if today.day < reset_day:
        reset_date = today.replace(day=reset_day)
    elif today.month == 12:
        reset_date = today.replace(year=today.year + 1, month=1, day=reset_day)
    else:
        reset_date = today.replace(month=today.month + 1, day=reset_day)

    # Query DB
    monthly_total = 0
    videos_produced = 0
    daily_breakdown: list[dict] = []

    if db_path.exists():
        try:
            conn = sqlite3.connect(db_path)
            try:
                # Ensure table exists (graceful on fresh DB)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS elevenlabs_usage (
                        id         INTEGER PRIMARY KEY AUTOINCREMENT,
                        date       TEXT    NOT NULL,
                        topic_id   TEXT    NOT NULL,
                        chars_used INTEGER NOT NULL,
                        voice_name TEXT    NOT NULL,
                        created_at TEXT    NOT NULL DEFAULT (datetime('now'))
                    )
                """)
                row = conn.execute(
                    "SELECT SUM(chars_used), COUNT(*) FROM elevenlabs_usage WHERE date >= ?",
                    (month_start.isoformat(),),
                ).fetchone()
                monthly_total = int(row[0] or 0)
                videos_produced = int(row[1] or 0)

                rows = conn.execute(
                    """
                    SELECT date, SUM(chars_used), COUNT(*)
                    FROM elevenlabs_usage
                    WHERE date >= ?
                    GROUP BY date
                    ORDER BY date
                    """,
                    (month_start.isoformat(),),
                ).fetchall()
                daily_breakdown = [
                    {"date": r[0], "chars": int(r[1]), "videos": int(r[2])}
                    for r in rows
                ]
            finally:
                conn.close()
        except Exception as exc:
            daily_breakdown = []
            print(f"[warning] DB query failed: {exc}")

    chars_remaining = max(0, monthly_limit - monthly_total)
    pct_used = monthly_total / monthly_limit * 100 if monthly_limit > 0 else 0.0
    avg_chars = int(monthly_total / videos_produced) if videos_produced > 0 else 0
    videos_remaining = int(chars_remaining / avg_chars) if avg_chars > 0 else 0

    if pct_used >= 95:
        status = "CRITICAL — production at risk"
    elif pct_used >= 85:
        status = "CAUTION — consider upgrading"
    elif pct_used >= 67:
        status = "WARNING — monitor usage closely"
    else:
        status = "OK — plenty of headroom"

    return {
        "month_label":         month_start.strftime("%B %Y"),
        "monthly_limit":       monthly_limit,
        "monthly_total":       monthly_total,
        "pct_used":            round(pct_used, 1),
        "chars_remaining":     chars_remaining,
        "videos_produced":     videos_produced,
        "avg_chars_per_video": avg_chars,
        "videos_remaining":    videos_remaining,
        "reset_date":          reset_date.strftime("%B %-d, %Y") if os.name != "nt" else reset_date.strftime("%B %d, %Y").replace(" 0", " "),
        "status":              status,
        "daily_breakdown":     daily_breakdown,
    }


def print_report(report: dict) -> None:
    """Print a formatted usage report to stdout."""
    print()
    print("=" * 48)
    print(f"ELEVENLABS USAGE REPORT — {report['month_label']}")
    print("=" * 48)
    print(f"Plan limit:        {report['monthly_limit']:,} chars/month")
    print(f"Used this month:   {report['monthly_total']:,} chars ({report['pct_used']:.1f}%)")
    print(f"Remaining:         {report['chars_remaining']:,} chars")
    print(f"Videos produced:   {report['videos_produced']} videos this month")
    print(f"Avg per video:     {report['avg_chars_per_video']:,} chars")
    if report['videos_remaining'] > 0:
        print(f"Videos remaining:  ~{report['videos_remaining']} videos at current rate")
    else:
        print("Videos remaining:  N/A (no videos produced yet)")
    print(f"Reset date:        {report['reset_date']}")
    print("=" * 48)

    if report["daily_breakdown"]:
        print("DAILY BREAKDOWN:")
        for entry in report["daily_breakdown"]:
            day_label = entry["date"][5:]  # MM-DD portion
            print(f"  {day_label}:  {entry['chars']:,} chars ({entry['videos']} video{'s' if entry['videos'] != 1 else ''})")
    else:
        print("DAILY BREAKDOWN: no data yet")

    print("=" * 48)
    print(f"STATUS: {report['status']}")
    print("=" * 48)
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    report = get_usage_report()
    print_report(report)

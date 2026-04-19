"""
constants.py — System-wide constants for ChannelForge.

All tunable parameters are centralised here so they can be adjusted
without touching business logic.  Environment-specific values (API keys,
file paths) live in .env — these constants are code-level defaults.
"""

# ---------------------------------------------------------------------------
# Content scoring
# ---------------------------------------------------------------------------

# Minimum composite engagement score to be promoted to the production queue
PRODUCTION_THRESHOLD: int = 70

# Default score assigned to topics injected by the OptimizationLoop
OPTIMIZATION_INJECT_SCORE: float = 85.0

# Score assigned to viral followup topics injected by SentimentAnalyzer
VIRAL_FOLLOWUP_SCORE: float = 92.0

# ---------------------------------------------------------------------------
# Upload scheduling
# ---------------------------------------------------------------------------

# Maximum videos uploaded per channel per day
MAX_DAILY_VIDEOS: int = 3

# Upload time slots (24-h "HH:MM" in UPLOAD_TIMEZONE)
UPLOAD_SLOTS: list[str] = ["08:00", "12:00", "18:00"]

# Default timezone for all scheduling and upload operations
DEFAULT_TIMEZONE: str = "Africa/Lagos"

# ---------------------------------------------------------------------------
# Performance tier thresholds
# ---------------------------------------------------------------------------

# Tier S — both conditions must be met
TIER_S_MIN_VIEWS: int = 50_000
TIER_S_MIN_ENGAGEMENT: float = 8.0       # %

# Tier A — either condition
TIER_A_MIN_VIEWS: int = 20_000
TIER_A_MIN_ENGAGEMENT: float = 6.0       # %

# Tier B — either condition
TIER_B_MIN_VIEWS: int = 5_000
TIER_B_MIN_ENGAGEMENT: float = 3.0       # %

# Tier C — views only
TIER_C_MIN_VIEWS: int = 1_000

# ---------------------------------------------------------------------------
# APScheduler job schedule
# ---------------------------------------------------------------------------

# Scraping runs every 6 hours starting at midnight
SCRAPER_HOURS: str = "0,6,12,18"

# Production runs 1 hour after scraping
PRODUCTION_HOURS: str = "1,7,13,19"

# Analytics run daily at 00:30
ANALYTICS_HOUR: int = 0
ANALYTICS_MINUTE: int = 30

# Optimization runs every Sunday at 02:00
OPTIMIZATION_DAY_OF_WEEK: str = "sun"
OPTIMIZATION_HOUR: int = 2
OPTIMIZATION_MINUTE: int = 0

# ---------------------------------------------------------------------------
# Caption timing
# ---------------------------------------------------------------------------

CAPTION_TIMING: dict[str, float] = {
    "hook_start":       0.0,
    "hook_end":         10.0,
    "point1_start":     10.0,
    "point1_end":       22.0,
    "point2_start":     22.0,
    "point2_end":       35.0,
    "cta_start":        35.0,
    "cta_end":          50.0,
}

# ---------------------------------------------------------------------------
# Voice profiles (ElevenLabs)
# ---------------------------------------------------------------------------

VOICE_PROFILES: dict[str, dict[str, str]] = {
    "money": {
        "voice_id":   "pNInz6obpgDQGcFmaJgB",
        "voice_name": "Adam",
    },
    "career": {
        "voice_id":   "TxGEqnHWrfWFTfGW9XjX",
        "voice_name": "Josh",
    },
    "success": {
        "voice_id":   "21m00Tcm4TlvDq8ikWAM",
        "voice_name": "Rachel",
    },
}

# ---------------------------------------------------------------------------
# Pixabay keyword map (category → search term)
# ---------------------------------------------------------------------------

PIXABAY_KEYWORD_MAP: dict[str, str] = {
    "money":    "money finance wealth",
    "career":   "professional office business",
    "success":  "motivation inspiration achievement",
    "health":   "fitness wellness lifestyle",
    "mindset":  "mindfulness meditation focus",
}

# ---------------------------------------------------------------------------
# CTA products (category → product config for script/metadata/overlay)
# Set GUMROAD_URL_MONEY / GUMROAD_URL_CAREER / GUMROAD_URL_SUCCESS in .env
# to replace the placeholder URLs before going live.
# ---------------------------------------------------------------------------

import os as _os

# CTA trigger keywords per category — used in comment detection + CTA validation
CTA_TRIGGER_KEYWORDS: dict[str, str] = {
    "money":   "SYSTEM",
    "career":  "AUTOMATE",
    "success": "BLUEPRINT",
}

# ---------------------------------------------------------------------------
# Rotating CTA system — single ask per video
# ---------------------------------------------------------------------------

# Subscribe CTA (used 80% of videos — every video except every 5th)
CTA_SUBSCRIBE = "Subscribe — we say the money things your bank hopes you never hear."
# Lead-magnet CTAs per category (used every 5th video)
CTA_LEAD_MAGNET: dict[str, str] = {
    "money":   "Comment SYSTEM below. I will send you the 5-day money reset free.",
    "career":  "Comment AUTOMATE below. I will send you the salary playbook free.",
    "success": "Comment BLUEPRINT below. I will send you the AI advantage guide free.",
}


def get_cta_mode(total_videos_produced: int) -> str:
    """Return 'lead_magnet' every 5th video, 'subscribe' otherwise."""
    if total_videos_produced > 0 and total_videos_produced % 5 == 0:
        return "lead_magnet"
    return "subscribe"


def get_cta_script(category: str, total_videos_produced: int) -> str:
    """Return the CTA script for this video based on rotation."""
    mode = get_cta_mode(total_videos_produced)
    if mode == "lead_magnet":
        return CTA_LEAD_MAGNET.get(category, CTA_LEAD_MAGNET["money"])
    return CTA_SUBSCRIBE


PRODUCTS: dict[str, dict[str, str]] = {
    "money": {
        "name":       "The 5 Money Systems Millionaires Use While They Sleep",
        "short_name": "Wealth Systems Blueprint",
        "gumroad_url": _os.getenv(
            "GUMROAD_URL_MONEY", "https://gumroad.com/l/placeholder1"
        ),
        "cta_script":  CTA_SUBSCRIBE,
        "cta_overlay": "FREE WEALTH SYSTEMS BLUEPRINT",
    },
    "career": {
        "name":       "Salary Escape Blueprint — 3 Income Streams to Start This Weekend",
        "short_name": "Salary Escape Blueprint",
        "gumroad_url": _os.getenv(
            "GUMROAD_URL_CAREER", "https://gumroad.com/l/placeholder2"
        ),
        "cta_script":  CTA_SUBSCRIBE,
        "cta_overlay": "FREE SALARY ESCAPE GUIDE",
    },
    "success": {
        "name":       "Success Myths Exposed — The Beliefs Keeping You Broke",
        "short_name": "Success Myths Guide",
        "gumroad_url": _os.getenv(
            "GUMROAD_URL_SUCCESS", "https://gumroad.com/l/placeholder3"
        ),
        "cta_script":  CTA_SUBSCRIBE,
        "cta_overlay": "FREE SUCCESS MYTHS BREAKDOWN",
    },
}

# ---------------------------------------------------------------------------
# Fallback topics per category
# Used by the production pipeline when scored_topics table is empty or
# the scraper returned nothing — ensures production always runs.
# ---------------------------------------------------------------------------

FALLBACK_TOPICS: dict[str, list[str]] = {
    "money": [
        "why your salary will never make you rich",
        "the debt trap nobody talks about",
        "why saving money keeps you poor",
        "how the wealthy avoid paying taxes legally",
        "why your boss will always earn more than you",
    ],
    "career": [
        "why working hard guarantees you stay broke",
        "the salary negotiation secret nobody teaches",
        "why loyal employees get paid the least",
        "how to escape the paycheck to paycheck cycle",
        "why your degree is making you poor",
    ],
    "success": [
        "the biggest lie about becoming successful",
        "why talented people stay broke",
        "the morning routine myth that wastes your time",
        "why most successful people were told no first",
        "the real reason you are not making progress",
    ],
}

# ---------------------------------------------------------------------------
# Competitor channels to monitor for topic research
# ---------------------------------------------------------------------------

COMPETITOR_CHANNELS: dict[str, list[dict[str, str]]] = {
    "money": [
        {"name": "GrahamStephan",        "id": "UCV6KDgJskWaEckne5aPA0aQ"},
        {"name": "AndreiJikh",           "id": "UCGy7SkBjcIAgTiwkXEtPnYg"},
        {"name": "MinorityMindset",      "id": "UCT3EznhW_CNFcfOlyDNTLLQ"},
        {"name": "WallStreetMillennial", "id": "UCeugSTiqaIoqCVHkkGQYDjg"},
        {"name": "JarradMorrow",         "id": "UCamg5A4wQRIVEFDXSBVmtkg"},
    ],
    "career": [
        {"name": "AliAbdaal",    "id": "UCoOae5nYA7VqaXzerajD0lg"},
        {"name": "JeffSu",       "id": "UC8wqCr7GfXWRpFbMIQWbVAQ"},
        {"name": "LindaRaynier", "id": "UCKrqnTQILduMtTQTGGmFRcQ"},
    ],
    "success": [
        {"name": "ImpactTheory", "id": "UCnYMOamNKLGVlJgRUbamveA"},
        {"name": "LewisHowes",  "id": "UCKnzDQGO2bNy0sXFPLGT1OA"},
        {"name": "BrianTracy",  "id": "UCkKNXIzv86vLhoBiMRzV_9A"},
    ],
}

# Finance keywords cycled through for YouTube trending/Shorts searches
FINANCE_SEARCH_KEYWORDS: list[str] = [
    "money", "salary", "wealth", "investing",
    "passive income", "financial freedom",
    "debt", "stocks", "real estate", "side hustle",
]

# ---------------------------------------------------------------------------
# Topic source priority scores (higher = pick first)
# ---------------------------------------------------------------------------

SOURCE_PRIORITIES: dict[str, int] = {
    "VIEWER_REQUESTED":        100,   # viewers asked via comment
    "COMPETITOR_HIGH_SIGNAL":   90,   # competitor video > 100k views / 30 days
    "AUTOCOMPLETE":             85,   # YouTube search autocomplete suggestions
    "TRENDING_SEARCH":          80,   # recent high-view Shorts via YouTube search
    "YOUTUBE_TRENDING":         80,   # trending / high-view search result
    "RISING_GOOGLE_TRENDS":     75,   # rising related queries (gaining momentum)
    "GOOGLE_TRENDS":            70,   # pytrends interest signal (top queries)
    "YOUTUBE_KEYWORD":          60,   # general YouTube keyword signal
    "FALLBACK":                 50,   # pre-written fallback list
}

# ---------------------------------------------------------------------------
# Autocomplete seed keywords (per category) for YouTube suggest scraping
# ---------------------------------------------------------------------------

AUTOCOMPLETE_SEED_KEYWORDS: dict[str, list[str]] = {
    "money": [
        "why am i", "how to make money", "why rich people", "salary truth",
        "financial freedom", "passive income", "how to invest", "why saving",
    ],
    "career": [
        "why your job", "salary negotiation", "how to get promoted",
        "quit your job", "side hustle", "work from home income",
    ],
    "success": [
        "why successful people", "morning routine", "how to be successful",
        "millionaire habits", "why most people fail", "growth mindset",
    ],
}

# Keywords cycled for trending-search Shorts discovery (Source 2)
TRENDING_SEARCH_KEYWORDS: list[str] = [
    "salary", "investing", "passive income", "financial freedom",
    "debt", "rich vs poor", "money mindset", "side hustle",
    "stock market", "real estate",
]

# Minimum view count for a competitor video to be treated as a HIGH_SIGNAL topic
COMPETITOR_HIGH_SIGNAL_MIN_VIEWS: int = 100_000

# ---------------------------------------------------------------------------
# Script limits
# ---------------------------------------------------------------------------

MAX_SCRIPT_WORDS: int = 150     # hard cap — script must be < 150 words
VIDEO_DURATION_SECONDS: float = 50.0

"""
hyperframes_renderer.py — HyperFramesRenderer

Generates HyperFrames HTML compositions from script data and renders to MP4
via the HyperFrames CLI (npx hyperframes render). Replaces PIL-based
MorphRenderer as Slot 1 in the production pipeline.

Interface matches MorphRenderer.build() exactly.
"""
from __future__ import annotations

import html as _html_mod
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUTPUT_DIR        = Path("data/output")
BEAT_ORDER        = ("hook", "statement", "twist", "landing", "cta")
ICON_IN           = 0.4    # scale + fade in
ICON_HOLD         = 4.8   # icon AND text held together
ICON_OUT          = 0.15   # simultaneous fade out
BEAT_START_OFFSET = 0.1    # small delay at each beat start
DEFAULT_DURATION  = 30.0   # fallback if audio probe fails

# ---------------------------------------------------------------------------
# Morph dictionary — keyword groups map to icon names
# ---------------------------------------------------------------------------
_MORPH_TABLE: list[tuple[tuple[str, ...], str]] = [
    (("coffee", "serve", "barista"),                         "coffee_mug"),
    (("retire", "retired", "freedom"),                       "open_door"),
    (("money", "dollars", "earn", "income", "salary"),       "dollar_bill"),
    (("spend", "spent", "cost", "pay"),                      "dollar_bill_down"),
    (("apartment", "home", "house", "live", "lived"),        "house"),
    (("car", "drive", "drove", "vehicle"),                   "car"),
    (("think", "thought", "mind", "brain", "smart"),         "brain"),
    (("time", "waiting", "clock", "hours"),                  "clock"),
    (("growth", "gain", "grew", "rise", "rising"),           "bar_chart"),
    (("loss", "drop", "losing", "fell", "down"),             "arrow_down"),
    (("subscribe", "follow", "join"),                        "smartphone"),
    (("scale", "balance", "decision", "compare"),            "balance_scale"),
    (("coin", "invest", "asset", "assets"),                  "coin"),
    (("chain", "habit", "trap", "stuck"),                    "chain_links"),
    (("key", "unlock", "secret", "system"),                  "key"),
    (("lock", "risk", "hidden"),                             "padlock"),
    (("eye", "see", "notice", "look"),                       "eye"),
    (("heart", "want", "desire", "love"),                    "heart"),
    (("seed", "start", "begin", "born"),                     "sprout"),
    (("door", "opportunity", "open"),                        "door_handle"),
    # --- expanded library ---
    (("save", "saving", "savings", "deposit"),                             "piggy_bank"),
    (("grow", "increase", "profit", "gains"),                              "graph_rising"),
    (("trapped", "escape", "free", "break", "broke"),                     "broken_chain"),
    (("win", "winner", "success", "succeed", "achieve", "achievement"),   "trophy"),
    (("protect", "safe", "safety", "security", "secure", "risky"),        "shield"),
    (("delay", "late", "deadline", "age", "years", "months"),             "hourglass"),
    (("wallet", "spending", "budget", "afford", "price"),                 "wallet"),
    (("tax", "taxes", "debt", "bill", "bills", "owe", "payment"),        "receipt"),
    (("family", "relationship", "marriage", "partner", "children", "kids"), "family"),
    (("career", "promotion", "climb", "level", "rank", "position", "job"), "ladder"),
    (("stable", "stability", "steady", "foundation", "grounded"),         "anchor"),
    (("discover", "reveal", "revealed", "unlocks"),                       "lock_open"),
    (("wealth", "rich", "wealthy", "millionaire", "status", "power", "elite"), "crown"),
]

_FINANCIAL = frozenset({
    "coffee", "money", "dollars", "earn", "income", "salary",
    "spend", "spent", "cost", "pay", "retire", "retired",
    "invest", "asset", "assets", "coin", "bank", "debt",
})
_ACTION_VERBS = frozenset({
    "serve", "drive", "drove", "think", "thought",
    "subscribe", "follow", "join", "unlock", "see",
    "start", "begin", "open", "scale", "balance",
})

# ---------------------------------------------------------------------------
# Icon family taxonomy — used for semantic fallback when enforcing diversity
# ---------------------------------------------------------------------------
_ICON_FAMILIES: dict[str, frozenset[str]] = {
    "financial":  frozenset({
        "dollar_bill", "dollar_bill_down", "coin", "piggy_bank",
        "wallet", "receipt", "bar_chart", "graph_rising", "arrow_down",
    }),
    "structural": frozenset({
        "house", "car", "padlock", "lock_open", "key",
        "shield", "anchor", "ladder", "crown",
    }),
    "temporal":   frozenset({"clock", "hourglass"}),
    "relational": frozenset({"brain", "heart", "eye", "family", "smartphone"}),
    "action":     frozenset({
        "coffee_mug", "open_door", "door_handle", "chain_links",
        "broken_chain", "balance_scale", "sprout", "trophy",
    }),
}
_FAMILY_ORDER = ("financial", "structural", "temporal", "relational", "action")

# ---------------------------------------------------------------------------
# SVG icon library — viewBox="0 0 300 300", stroke #F0F2FF, no fill
# Glow is applied via CSS (.beat-icon svg { filter: drop-shadow(...) })
# ---------------------------------------------------------------------------
ICON_SVG_MAP: dict[str, str] = {

    "coffee_mug": (
        '<svg viewBox="0 0 300 300" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<path d="M100 115 Q110 78 100 38" stroke="#F0F2FF" stroke-width="2.5" stroke-linecap="round"/>'
        '<path d="M150 115 Q160 78 150 38" stroke="#F0F2FF" stroke-width="2.5" stroke-linecap="round"/>'
        '<path d="M200 115 Q210 78 200 38" stroke="#F0F2FF" stroke-width="2.5" stroke-linecap="round"/>'
        '<rect x="50" y="120" width="200" height="155" rx="12" stroke="#F0F2FF" stroke-width="2.5"/>'
        '<path d="M250 148 C296 148 296 248 250 248" stroke="#F0F2FF" stroke-width="2.5" stroke-linecap="round"/>'
        '</svg>'
    ),

    "coin": (
        '<svg viewBox="0 0 300 300" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<circle cx="150" cy="150" r="138" stroke="#F0F2FF" stroke-width="2.5"/>'
        '<circle cx="150" cy="150" r="108" stroke="#F0F2FF" stroke-width="1.5" opacity="0.4"/>'
        '<text x="150" y="150" font-family="Georgia,serif" font-size="128" fill="#F0F2FF"'
        ' text-anchor="middle" dominant-baseline="central" font-weight="bold">$</text>'
        '</svg>'
    ),

    "dollar_bill": (
        '<svg viewBox="0 0 300 300" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<rect x="20" y="90" width="260" height="120" rx="4" stroke="#F0F2FF" stroke-width="2.5"/>'
        '<line x1="20" y1="102" x2="280" y2="102" stroke="#F0F2FF" stroke-width="1"/>'
        '<line x1="20" y1="198" x2="280" y2="198" stroke="#F0F2FF" stroke-width="1"/>'
        '<text x="150" y="150" font-family="Georgia,serif" font-size="80" fill="#F0F2FF"'
        ' text-anchor="middle" dominant-baseline="central" font-weight="bold">$</text>'
        '</svg>'
    ),

    "dollar_bill_down": (
        '<svg viewBox="0 0 300 300" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<rect x="20" y="28" width="260" height="160" rx="4" stroke="#F0F2FF" stroke-width="2"/>'
        '<line x1="20" y1="40" x2="280" y2="40" stroke="#F0F2FF" stroke-width="1"/>'
        '<line x1="20" y1="176" x2="280" y2="176" stroke="#F0F2FF" stroke-width="1"/>'
        '<text x="150" y="108" font-family="Georgia,serif" font-size="80" fill="#F0F2FF"'
        ' text-anchor="middle" dominant-baseline="central" font-weight="bold">$</text>'
        '<line x1="150" y1="194" x2="150" y2="244" stroke="#F0F2FF" stroke-width="2.5" stroke-linecap="round"/>'
        '<polygon points="138,244 162,244 150,264" fill="#F0F2FF"/>'
        '</svg>'
    ),

    "brain": (
        '<svg viewBox="26 42 246 216" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<path d="M150,252 C84,250 38,212 36,170 C34,140 50,114 72,100'
        ' C64,80 70,56 92,48 C112,40 130,52 144,68 C146,58 150,50 150,50"'
        ' stroke="#F0F2FF" stroke-width="2.5" stroke-linecap="round"/>'
        '<path d="M150,50 C150,50 154,58 156,68'
        ' C170,52 188,40 208,48 C230,56 236,80 228,100'
        ' C250,114 266,140 264,170 C262,212 216,250 150,252"'
        ' stroke="#F0F2FF" stroke-width="2.5" stroke-linecap="round"/>'
        '<line x1="150" y1="50" x2="150" y2="252" stroke="#F0F2FF"'
        ' stroke-width="1.5" stroke-dasharray="6,5" opacity="0.5"/>'
        '<path d="M86,128 Q106,114 114,132 Q122,150 105,160"'
        ' stroke="#F0F2FF" stroke-width="2" stroke-linecap="round"/>'
        '<path d="M70,180 Q91,164 100,182 Q108,198 90,207"'
        ' stroke="#F0F2FF" stroke-width="2" stroke-linecap="round"/>'
        '<path d="M214,128 Q194,114 186,132 Q178,150 195,160"'
        ' stroke="#F0F2FF" stroke-width="2" stroke-linecap="round"/>'
        '<path d="M230,180 Q209,164 200,182 Q192,198 210,207"'
        ' stroke="#F0F2FF" stroke-width="2" stroke-linecap="round"/>'
        '</svg>'
    ),

    "smartphone": (
        '<svg viewBox="0 0 300 300" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<rect x="60" y="17" width="180" height="266" rx="22" stroke="#F0F2FF" stroke-width="2.5"/>'
        '<rect x="80" y="62" width="140" height="180" rx="6" stroke="#F0F2FF" stroke-width="1.5"/>'
        '<rect x="110" y="33" width="80" height="12" rx="6" stroke="#F0F2FF" stroke-width="1.5"/>'
        '<rect x="115" y="258" width="70" height="9" rx="4.5" fill="#F0F2FF"/>'
        '<polygon points="118,132 118,172 168,152"'
        ' stroke="#F0F2FF" stroke-width="2" stroke-linejoin="round"/>'
        '</svg>'
    ),

    "house": (
        '<svg viewBox="0 0 300 300" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<polyline points="20,160 150,40 280,160"'
        ' stroke="#F0F2FF" stroke-width="2.5" stroke-linejoin="round" fill="none"/>'
        '<rect x="55" y="160" width="190" height="110" stroke="#F0F2FF" stroke-width="2.5"/>'
        '<rect x="120" y="210" width="60" height="60" stroke="#F0F2FF" stroke-width="2"/>'
        '</svg>'
    ),

    "car": (
        '<svg viewBox="0 0 300 300" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<rect x="20" y="155" width="260" height="75" rx="10" stroke="#F0F2FF" stroke-width="2.5"/>'
        '<path d="M55,155 L85,95 L215,95 L245,155"'
        ' stroke="#F0F2FF" stroke-width="2.5" stroke-linejoin="round"/>'
        '<circle cx="82" cy="238" r="26" stroke="#F0F2FF" stroke-width="2.5"/>'
        '<circle cx="218" cy="238" r="26" stroke="#F0F2FF" stroke-width="2.5"/>'
        '</svg>'
    ),

    "clock": (
        '<svg viewBox="0 0 300 300" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<circle cx="150" cy="150" r="128" stroke="#F0F2FF" stroke-width="2.5"/>'
        '<line x1="150" y1="150" x2="150" y2="62"'
        ' stroke="#F0F2FF" stroke-width="3" stroke-linecap="round"/>'
        '<line x1="150" y1="150" x2="218" y2="185"'
        ' stroke="#F0F2FF" stroke-width="3" stroke-linecap="round"/>'
        '<circle cx="150" cy="150" r="7" fill="#F0F2FF"/>'
        '</svg>'
    ),

    "bar_chart": (
        '<svg viewBox="0 0 300 300" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<rect x="30"  y="170" width="58" height="100" rx="3" stroke="#F0F2FF" stroke-width="2.5"/>'
        '<rect x="121" y="120" width="58" height="150" rx="3" stroke="#F0F2FF" stroke-width="2.5"/>'
        '<rect x="212" y="50"  width="58" height="220" rx="3" stroke="#F0F2FF" stroke-width="2.5"/>'
        '<line x1="18" y1="272" x2="282" y2="272" stroke="#F0F2FF" stroke-width="2"/>'
        '</svg>'
    ),

    "arrow_down": (
        '<svg viewBox="0 0 300 300" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<line x1="150" y1="40" x2="150" y2="210"'
        ' stroke="#F0F2FF" stroke-width="3" stroke-linecap="round"/>'
        '<polygon points="88,200 212,200 150,272" fill="#F0F2FF"/>'
        '</svg>'
    ),

    "open_door": (
        '<svg viewBox="0 0 300 300" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<rect x="55" y="30" width="145" height="240" rx="4" stroke="#F0F2FF" stroke-width="2.5"/>'
        '<path d="M200,30 L262,58 L262,270 L200,270"'
        ' stroke="#F0F2FF" stroke-width="2.5" stroke-linejoin="round"/>'
        '<circle cx="185" cy="155" r="9" fill="#F0F2FF"/>'
        '</svg>'
    ),

    "door_handle": (
        '<svg viewBox="0 0 300 300" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<rect x="50" y="20" width="200" height="260" rx="6" stroke="#F0F2FF" stroke-width="2.5"/>'
        '<line x1="150" y1="20" x2="150" y2="280"'
        ' stroke="#F0F2FF" stroke-width="1" opacity="0.25"/>'
        '<circle cx="212" cy="150" r="13" stroke="#F0F2FF" stroke-width="2.5"/>'
        '</svg>'
    ),

    "balance_scale": (
        '<svg viewBox="0 0 300 300" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<line x1="150" y1="55" x2="150" y2="245" stroke="#F0F2FF" stroke-width="2.5"/>'
        '<line x1="55"  y1="105" x2="245" y2="105" stroke="#F0F2FF" stroke-width="2.5"/>'
        '<path d="M55,105 Q28,138 55,158 Q82,138 55,105 Z" stroke="#F0F2FF" stroke-width="2"/>'
        '<path d="M245,105 Q218,138 245,158 Q272,138 245,105 Z" stroke="#F0F2FF" stroke-width="2"/>'
        '<line x1="108" y1="245" x2="192" y2="245" stroke="#F0F2FF" stroke-width="3"/>'
        '</svg>'
    ),

    "chain_links": (
        '<svg viewBox="0 0 300 300" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<ellipse cx="88"  cy="150" rx="62" ry="36" stroke="#F0F2FF" stroke-width="2.5"/>'
        '<ellipse cx="212" cy="150" rx="62" ry="36" stroke="#F0F2FF" stroke-width="2.5"/>'
        '</svg>'
    ),

    "key": (
        '<svg viewBox="0 0 300 300" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<circle cx="98" cy="108" r="72" stroke="#F0F2FF" stroke-width="2.5"/>'
        '<line x1="155" y1="142" x2="272" y2="255"'
        ' stroke="#F0F2FF" stroke-width="3" stroke-linecap="round"/>'
        '<line x1="218" y1="218" x2="244" y2="193"'
        ' stroke="#F0F2FF" stroke-width="3" stroke-linecap="round"/>'
        '<line x1="240" y1="240" x2="266" y2="215"'
        ' stroke="#F0F2FF" stroke-width="3" stroke-linecap="round"/>'
        '</svg>'
    ),

    "padlock": (
        '<svg viewBox="0 0 300 300" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<path d="M88,145 L88,100 Q88,48 150,48 Q212,48 212,100 L212,145"'
        ' stroke="#F0F2FF" stroke-width="2.5" fill="none"/>'
        '<rect x="48" y="145" width="204" height="130" rx="10" stroke="#F0F2FF" stroke-width="2.5"/>'
        '<circle cx="150" cy="198" r="19" stroke="#F0F2FF" stroke-width="2.5"/>'
        '<line x1="150" y1="217" x2="150" y2="244" stroke="#F0F2FF" stroke-width="2.5"/>'
        '</svg>'
    ),

    "eye": (
        '<svg viewBox="0 0 300 300" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<path d="M18,150 Q150,48 282,150 Q150,252 18,150 Z"'
        ' stroke="#F0F2FF" stroke-width="2.5"/>'
        '<circle cx="150" cy="150" r="46" stroke="#F0F2FF" stroke-width="2.5"/>'
        '<circle cx="150" cy="150" r="21" fill="#F0F2FF"/>'
        '</svg>'
    ),

    "heart": (
        '<svg viewBox="0 0 300 300" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<path d="M150,238 C48,178 18,115 50,76'
        ' C70,48 108,48 150,90'
        ' C192,48 230,48 250,76'
        ' C282,115 252,178 150,238 Z"'
        ' stroke="#F0F2FF" stroke-width="2.5"/>'
        '</svg>'
    ),

    "sprout": (
        '<svg viewBox="0 0 300 300" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<line x1="150" y1="268" x2="150" y2="115"'
        ' stroke="#F0F2FF" stroke-width="2.5" stroke-linecap="round"/>'
        '<path d="M150,178 Q78,158 58,98 Q118,78 150,128"'
        ' stroke="#F0F2FF" stroke-width="2.5" stroke-linecap="round" fill="none"/>'
        '<path d="M150,148 Q222,128 242,68 Q182,48 150,98"'
        ' stroke="#F0F2FF" stroke-width="2.5" stroke-linecap="round" fill="none"/>'
        '<circle cx="150" cy="268" r="12" fill="#F0F2FF"/>'
        '</svg>'
    ),

    # ---- expanded icon library ----

    "piggy_bank": (
        '<svg viewBox="0 0 300 300" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<ellipse cx="148" cy="155" rx="88" ry="68" stroke="#F0F2FF" stroke-width="2.5"/>'
        '<circle cx="210" cy="100" r="18" stroke="#F0F2FF" stroke-width="2"/>'
        '<rect x="226" y="145" width="30" height="22" rx="8" stroke="#F0F2FF" stroke-width="2"/>'
        '<line x1="130" y1="87" x2="162" y2="87" stroke="#F0F2FF" stroke-width="3" stroke-linecap="round"/>'
        '<rect x="85" y="218" width="18" height="38" rx="4" stroke="#F0F2FF" stroke-width="2"/>'
        '<rect x="115" y="218" width="18" height="38" rx="4" stroke="#F0F2FF" stroke-width="2"/>'
        '<rect x="162" y="218" width="18" height="38" rx="4" stroke="#F0F2FF" stroke-width="2"/>'
        '<rect x="192" y="218" width="18" height="38" rx="4" stroke="#F0F2FF" stroke-width="2"/>'
        '<circle cx="196" cy="142" r="5" fill="#F0F2FF"/>'
        '</svg>'
    ),

    "graph_rising": (
        '<svg viewBox="0 0 300 300" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<line x1="48" y1="260" x2="270" y2="260" stroke="#F0F2FF" stroke-width="2.5" stroke-linecap="round"/>'
        '<line x1="48" y1="260" x2="48" y2="40" stroke="#F0F2FF" stroke-width="2.5" stroke-linecap="round"/>'
        '<polyline points="68,232 108,196 148,215 192,160 238,108 268,60"'
        ' stroke="#F0F2FF" stroke-width="2.5" fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
        '<polygon points="256,74 278,52 274,76" fill="#F0F2FF"/>'
        '</svg>'
    ),

    "broken_chain": (
        '<svg viewBox="0 0 300 300" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<ellipse cx="72" cy="130" rx="50" ry="28" stroke="#F0F2FF" stroke-width="2.5"/>'
        '<ellipse cx="106" cy="168" rx="50" ry="28" stroke="#F0F2FF" stroke-width="2.5"/>'
        '<ellipse cx="194" cy="132" rx="50" ry="28" stroke="#F0F2FF" stroke-width="2.5"/>'
        '<ellipse cx="228" cy="170" rx="50" ry="28" stroke="#F0F2FF" stroke-width="2.5"/>'
        '<polyline points="132,142 142,150 130,158" stroke="#F0F2FF" stroke-width="2.5" fill="none" stroke-linecap="round"/>'
        '<polyline points="168,142 158,150 170,158" stroke="#F0F2FF" stroke-width="2.5" fill="none" stroke-linecap="round"/>'
        '</svg>'
    ),

    "trophy": (
        '<svg viewBox="0 0 300 300" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<path d="M82,75 Q80,195 150,215 Q220,195 218,75 Z" stroke="#F0F2FF" stroke-width="2.5" fill="none"/>'
        '<path d="M82,100 C52,100 48,168 82,168" stroke="#F0F2FF" stroke-width="2.5" fill="none"/>'
        '<path d="M218,100 C248,100 252,168 218,168" stroke="#F0F2FF" stroke-width="2.5" fill="none"/>'
        '<line x1="150" y1="215" x2="150" y2="248" stroke="#F0F2FF" stroke-width="2.5"/>'
        '<rect x="100" y="248" width="100" height="22" rx="4" stroke="#F0F2FF" stroke-width="2.5"/>'
        '</svg>'
    ),

    "shield": (
        '<svg viewBox="0 0 300 300" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<polygon points="150,48 232,105 204,252 96,252 68,105"'
        ' stroke="#F0F2FF" stroke-width="2.5" fill="none"/>'
        '<polyline points="108,155 140,187 198,118"'
        ' stroke="#F0F2FF" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" fill="none"/>'
        '</svg>'
    ),

    "hourglass": (
        '<svg viewBox="0 0 300 300" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<rect x="62" y="42" width="176" height="216" rx="6" stroke="#F0F2FF" stroke-width="1.5" opacity="0.5"/>'
        '<polygon points="72,52 228,52 150,150"'
        ' stroke="#F0F2FF" stroke-width="2.5" fill="none" stroke-linejoin="round"/>'
        '<polygon points="72,248 228,248 150,150"'
        ' stroke="#F0F2FF" stroke-width="2.5" fill="none" stroke-linejoin="round"/>'
        '<circle cx="143" cy="150" r="4" fill="#F0F2FF"/>'
        '<circle cx="157" cy="150" r="4" fill="#F0F2FF"/>'
        '</svg>'
    ),

    "wallet": (
        '<svg viewBox="0 0 300 300" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<rect x="48" y="82" width="204" height="136" rx="12" stroke="#F0F2FF" stroke-width="2.5"/>'
        '<line x1="48" y1="150" x2="252" y2="150" stroke="#F0F2FF" stroke-width="1.5" opacity="0.6"/>'
        '<rect x="158" y="96" width="80" height="50" rx="8" stroke="#F0F2FF" stroke-width="2"/>'
        '<circle cx="195" cy="121" r="10" stroke="#F0F2FF" stroke-width="2"/>'
        '</svg>'
    ),

    "receipt": (
        '<svg viewBox="0 0 300 300" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<polyline points="80,32 80,255 95,270 110,255 125,270 140,255 155,270 170,255 185,270 200,255 215,270 220,255 220,32 80,32"'
        ' stroke="#F0F2FF" stroke-width="2.5" fill="none" stroke-linejoin="round"/>'
        '<text x="98" y="78" font-family="Georgia,serif" font-size="38" fill="#F0F2FF" font-weight="bold">$</text>'
        '<line x1="98" y1="110" x2="200" y2="110" stroke="#F0F2FF" stroke-width="2" opacity="0.7"/>'
        '<line x1="98" y1="140" x2="200" y2="140" stroke="#F0F2FF" stroke-width="2" opacity="0.7"/>'
        '<line x1="98" y1="170" x2="200" y2="170" stroke="#F0F2FF" stroke-width="2" opacity="0.7"/>'
        '</svg>'
    ),

    "family": (
        '<svg viewBox="0 0 300 300" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<circle cx="80" cy="105" r="38" stroke="#F0F2FF" stroke-width="2.5"/>'
        '<circle cx="172" cy="105" r="38" stroke="#F0F2FF" stroke-width="2.5"/>'
        '<circle cx="248" cy="115" r="26" stroke="#F0F2FF" stroke-width="2.5"/>'
        '<line x1="48" y1="148" x2="270" y2="148" stroke="#F0F2FF" stroke-width="2.5"/>'
        '</svg>'
    ),

    "ladder": (
        '<svg viewBox="0 0 300 300" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<line x1="90" y1="50" x2="90" y2="254" stroke="#F0F2FF" stroke-width="2.5" stroke-linecap="round"/>'
        '<line x1="210" y1="50" x2="210" y2="254" stroke="#F0F2FF" stroke-width="2.5" stroke-linecap="round"/>'
        '<line x1="90" y1="90" x2="210" y2="90" stroke="#F0F2FF" stroke-width="2.5" stroke-linecap="round"/>'
        '<line x1="90" y1="130" x2="210" y2="130" stroke="#F0F2FF" stroke-width="2.5" stroke-linecap="round"/>'
        '<line x1="90" y1="170" x2="210" y2="170" stroke="#F0F2FF" stroke-width="2.5" stroke-linecap="round"/>'
        '<line x1="90" y1="210" x2="210" y2="210" stroke="#F0F2FF" stroke-width="2.5" stroke-linecap="round"/>'
        '<line x1="90" y1="250" x2="210" y2="250" stroke="#F0F2FF" stroke-width="2.5" stroke-linecap="round"/>'
        '</svg>'
    ),

    "anchor": (
        '<svg viewBox="0 0 300 300" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<circle cx="150" cy="70" r="30" stroke="#F0F2FF" stroke-width="2.5"/>'
        '<line x1="150" y1="100" x2="150" y2="222" stroke="#F0F2FF" stroke-width="2.5" stroke-linecap="round"/>'
        '<line x1="98" y1="122" x2="202" y2="122" stroke="#F0F2FF" stroke-width="2.5" stroke-linecap="round"/>'
        '<path d="M150,222 C122,222 80,240 78,262" stroke="#F0F2FF" stroke-width="2.5" fill="none" stroke-linecap="round"/>'
        '<path d="M150,222 C178,222 220,240 222,262" stroke="#F0F2FF" stroke-width="2.5" fill="none" stroke-linecap="round"/>'
        '<circle cx="78" cy="262" r="8" stroke="#F0F2FF" stroke-width="2"/>'
        '<circle cx="222" cy="262" r="8" stroke="#F0F2FF" stroke-width="2"/>'
        '</svg>'
    ),

    "lock_open": (
        '<svg viewBox="0 0 300 300" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<rect x="48" y="145" width="204" height="130" rx="10" stroke="#F0F2FF" stroke-width="2.5"/>'
        '<circle cx="150" cy="198" r="19" stroke="#F0F2FF" stroke-width="2.5"/>'
        '<line x1="150" y1="217" x2="150" y2="244" stroke="#F0F2FF" stroke-width="2.5"/>'
        '<path d="M88,145 L88,90 Q88,38 150,38 Q212,38 212,90 L212,110"'
        ' stroke="#F0F2FF" stroke-width="2.5" fill="none" stroke-linecap="round"/>'
        '</svg>'
    ),

    "crown": (
        '<svg viewBox="0 0 300 300" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<polyline points="58,222 58,158 112,222 150,105 188,222 242,158 242,222"'
        ' stroke="#F0F2FF" stroke-width="2.5" stroke-linejoin="round" fill="none"/>'
        '<rect x="58" y="222" width="184" height="26" rx="4" stroke="#F0F2FF" stroke-width="2.5"/>'
        '<circle cx="58" cy="158" r="9" fill="#F0F2FF"/>'
        '<circle cx="242" cy="158" r="9" fill="#F0F2FF"/>'
        '<circle cx="150" cy="105" r="9" fill="#F0F2FF"/>'
        '</svg>'
    ),
}

# Fallback icon when lookup fails
_FALLBACK_ICON = "coin"


# ---------------------------------------------------------------------------
# BuildResult — same interface as MorphRenderer and SlidesRenderer
# ---------------------------------------------------------------------------
@dataclass
class BuildResult:
    topic_id:          str
    output_path:       str
    duration_seconds:  float
    is_valid:          bool
    validation_errors: list[str] = field(default_factory=list)
    built_at:          str = ""

    def __post_init__(self) -> None:
        if not self.built_at:
            self.built_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic_id":          self.topic_id,
            "output_path":       self.output_path,
            "duration_seconds":  self.duration_seconds,
            "is_valid":          self.is_valid,
            "validation_errors": self.validation_errors,
            "built_at":          self.built_at,
        }


# ---------------------------------------------------------------------------
# HyperFramesRenderer
# ---------------------------------------------------------------------------
class HyperFramesRenderer:
    """
    Semantic morph video renderer using HyperFrames HTML + GSAP + SVG.
    Produces 1080x1920 YouTube Shorts with icon-per-beat storyboard.
    Drop-in replacement for MorphRenderer (same build() interface).
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        topic_id:          str,
        script_dict:       dict[str, str],
        audio_path:        str | Path,
        word_timestamps:   list[dict] | None = None,
        cta_overlay:       str = "",
        anthropic_api_key: str = "",
        stock_video_path:  Any = None,
        **kwargs: Any,
    ) -> BuildResult:
        """
        Generate a HyperFrames morph video for one topic.

        Args:
            topic_id:    Unique identifier (used for output filename).
            script_dict: Parts dict with keys hook/statement/twist/landing/cta
                         (also accepts 'question' for the final beat).
            audio_path:  Path to voiceover MP3.
            word_timestamps, cta_overlay, anthropic_api_key, stock_video_path:
                         Accepted for interface compatibility; not used.

        Returns:
            BuildResult with output_path and is_valid flag.
        """
        start_ts   = time.time()
        audio_path = Path(audio_path)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"{topic_id}_morph.mp4"

        duration = self._audio_duration(audio_path)
        if duration <= 0:
            duration = DEFAULT_DURATION
            logger.warning("[hf] Could not read audio duration — using %.1fs", duration)

        beats = self._plan_beats(script_dict)

        logger.info("[hf] === STORYBOARD ===")
        for i, b in enumerate(beats):
            logger.info(
                "[hf] beat %d %-10s | anchor=%-14s | icon=%s",
                i, b["name"], b["anchor"], b["icon"],
            )

        try:
            html = self._generate_html(beats, duration)
            self._render_html(html, output_path, audio_path)
            elapsed = time.time() - start_ts
            logger.info("[hf] Built %s in %.1fs", output_path.name, elapsed)
            return BuildResult(
                topic_id=topic_id,
                output_path=str(output_path),
                duration_seconds=duration,
                is_valid=True,
            )
        except Exception as exc:
            logger.error("[hf] Render failed: %s", exc, exc_info=True)
            return BuildResult(
                topic_id=topic_id,
                output_path="",
                duration_seconds=0.0,
                is_valid=False,
                validation_errors=[str(exc)],
            )

    # ------------------------------------------------------------------
    # Beat planning
    # ------------------------------------------------------------------

    def _plan_beats(self, script_dict: dict[str, str]) -> list[dict]:
        """Map each of the 5 beat slots to text, anchor keyword, and icon name.

        Scores all keyword candidates per beat by semantic specificity
        (smaller keyword groups = more specific = higher score). Enforces
        strict icon diversity: each icon appears at most once across all 5 beats.
        If all keyword-matched icons are exhausted, cascades through semantic
        families: financial → structural → temporal → relational → action.
        """
        beats: list[dict] = []
        used_icons: set[str] = set()

        for name in BEAT_ORDER:
            if name == "cta":
                text = script_dict.get("cta", "") or script_dict.get("question", "")
            else:
                text = script_dict.get(name, "")

            candidates = self._score_beat_keywords(text)

            cand_str = ", ".join(
                f"{kw}:{icon}" for _, kw, icon in candidates[:6]
            ) or "(none)"

            # Select best non-duplicate icon from scored candidates
            selected_kw: str | None = None
            selected_icon: str | None = None
            for _score, kw, icon in candidates:
                if icon not in used_icons and icon in ICON_SVG_MAP:
                    selected_kw, selected_icon = kw, icon
                    break

            # No unique candidate found — use family-aware fallback
            if selected_icon is None:
                original_icon = candidates[0][2] if candidates else _FALLBACK_ICON
                selected_kw   = candidates[0][1] if candidates else "money"
                selected_icon = self._pick_fallback_icon(original_icon, used_icons)

            # Absolute safety net
            if selected_icon not in ICON_SVG_MAP:
                selected_icon = _FALLBACK_ICON
            if not selected_kw:
                words = text.split()
                selected_kw = words[0].lower().strip(".,!?;:") if words else "money"

            logger.info(
                "[hf] beat %d (%s): '%.40s' -> candidates: [%s] -> selected: %s (%s)",
                len(beats), name, text, cand_str, selected_kw, selected_icon,
            )

            used_icons.add(selected_icon)
            beats.append({
                "name":   name,
                "text":   text,
                "anchor": selected_kw,
                "icon":   selected_icon,
            })

        return beats

    def _score_beat_keywords(self, text: str) -> list[tuple[float, str, str]]:
        """Return scored keyword candidates for the given beat text.

        Score = 3.0 / len(keyword_group) so smaller (more specific) groups
        score higher. Returns list sorted descending by score.
        """
        words = {w.lower().strip(".,!?;:'\"") for w in text.split()}
        results: list[tuple[float, str, str]] = []
        for kws, icon in _MORPH_TABLE:
            for kw in kws:
                if kw in words:
                    results.append((3.0 / len(kws), kw, icon))
        results.sort(key=lambda x: x[0], reverse=True)
        return results

    def _pick_fallback_icon(self, excluded: str, used_icons: set[str]) -> str:
        """Return an unused icon, preferring a different semantic family.

        Cascades through: financial → structural → temporal → relational → action,
        skipping the family of the excluded icon on the first pass.
        """
        excluded_family: str | None = None
        for family_name, members in _ICON_FAMILIES.items():
            if excluded in members:
                excluded_family = family_name
                break

        # First pass: skip excluded family to prefer semantic variety
        for family_name in _FAMILY_ORDER:
            if family_name == excluded_family:
                continue
            for candidate in _ICON_FAMILIES.get(family_name, frozenset()):
                if candidate not in used_icons and candidate in ICON_SVG_MAP:
                    return candidate

        # Second pass: allow any unused icon including same family
        for candidate in ICON_SVG_MAP:
            if candidate not in used_icons:
                return candidate

        return _FALLBACK_ICON

    def _extract_anchor(self, text: str) -> str:
        """Extract the primary keyword from beat text using priority ordering.

        Kept for backward compatibility. New code uses _score_beat_keywords().
        """
        words = [w.lower().strip(".,!?;:") for w in text.split()]
        for priority in (_FINANCIAL, _ACTION_VERBS, None):
            for w in words:
                if priority is not None and w not in priority:
                    continue
                for kws, _ in _MORPH_TABLE:
                    if w in kws:
                        return w
        return words[0] if words else "money"

    def _lookup_icon(self, anchor: str) -> str:
        """Return icon name for anchor keyword; falls back to coin."""
        for kws, icon in _MORPH_TABLE:
            if anchor in kws:
                return icon if icon in ICON_SVG_MAP else _FALLBACK_ICON
        return _FALLBACK_ICON

    # ------------------------------------------------------------------
    # HTML generation
    # ------------------------------------------------------------------

    def _generate_html(self, beats: list[dict], total_duration: float) -> str:
        """
        Build a complete HyperFrames HTML composition string with kinetic
        word-reveal typography. Each beat shows a glowing icon + static gold
        anchor word + word-by-word staggered secondary text.
        """
        n_beats  = len(beats)
        beat_dur = total_duration / n_beats

        # Beat HTML blocks — secondary text split into individual word spans
        beat_blocks: list[str] = []
        for i, beat in enumerate(beats):
            svg    = ICON_SVG_MAP.get(beat["icon"], ICON_SVG_MAP[_FALLBACK_ICON])
            anchor = _html_mod.escape(beat["anchor"].upper())
            words  = beat["text"].split()
            word_spans = " ".join(
                f'<span class="word">{_html_mod.escape(w)}</span>'
                for w in words
            )
            beat_blocks.append(
                f'      <div id="icon-{i}" class="beat-icon">{svg}</div>\n'
                f'      <div id="anchor-{i}" class="beat-anchor">{anchor}</div>\n'
                f'      <div id="text-{i}" class="beat-text">{word_spans}</div>'
            )

        beats_html = "\n\n".join(beat_blocks)

        # GSAP timeline — icon scale-in, anchor fade-in, kinetic word stagger
        gsap_script = (
            f"      window.__timelines = window.__timelines || {{}};\n"
            f"      const tl = gsap.timeline({{ paused: true }});\n\n"
            f"      const BEAT_DUR  = {beat_dur:.4f};\n"
            f"      const ICON_IN   = {ICON_IN};\n"
            f"      const ICON_HOLD = {ICON_HOLD};\n"
            f"      const ICON_OUT  = {ICON_OUT};\n\n"
            f"      for (let b = 0; b < {n_beats}; b++) {{\n"
            f"        const bs     = b * BEAT_DUR;\n"
            f"        const icon   = '#icon-'   + b;\n"
            f"        const anchor = '#anchor-' + b;\n"
            f"        const words  = '#text-'   + b + ' .word';\n"
            f"        const textEl = '#text-'   + b;\n"
            f"        const inT    = bs + {BEAT_START_OFFSET};\n"
            f"        const outT   = bs + {BEAT_START_OFFSET} + ICON_IN + ICON_HOLD;\n\n"
            f"        tl.fromTo(icon,\n"
            f"          {{ scale: 0.6, opacity: 0 }},\n"
            f"          {{ scale: 1.0, opacity: 1, duration: ICON_IN, ease: 'power2.out' }},\n"
            f"          inT\n"
            f"        );\n"
            f"        tl.fromTo(anchor,\n"
            f"          {{ opacity: 0, y: -24 }},\n"
            f"          {{ opacity: 1, y: 0, duration: ICON_IN, ease: 'power3.out' }},\n"
            f"          inT\n"
            f"        );\n"
            f"        tl.fromTo(words,\n"
            f"          {{ opacity: 0, y: 15 }},\n"
            f"          {{ opacity: 1, y: 0, duration: 0.3, stagger: 0.08, ease: 'power2.out' }},\n"
            f"          inT + 0.5\n"
            f"        );\n"
            f"        tl.to(icon,\n"
            f"          {{ opacity: 0, scale: 0.8, duration: ICON_OUT, ease: 'power2.in' }},\n"
            f"          outT\n"
            f"        );\n"
            f"        tl.to([anchor, textEl],\n"
            f"          {{ opacity: 0, duration: ICON_OUT, ease: 'power2.in' }},\n"
            f"          outT\n"
            f"        );\n"
            f"      }}\n\n"
            f"      window.__timelines['main'] = tl;"
        )

        duration_int = int(total_duration)

        return (
            "<!doctype html>\n"
            '<html lang="en">\n'
            "  <head>\n"
            '    <meta charset="UTF-8" />\n'
            '    <meta name="viewport" content="width=1080, height=1920" />\n'
            '    <link rel="preconnect" href="https://fonts.googleapis.com" />\n'
            '    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />\n'
            '    <link href="https://fonts.googleapis.com/css2?family=Bebas+Neue'
            '&family=Montserrat:wght@400;600&display=block" rel="stylesheet" />\n'
            '    <script src="https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js"></script>\n'
            "    <style>\n"
            "      * { margin: 0; padding: 0; box-sizing: border-box; }\n"
            "      html, body {\n"
            "        margin: 0; width: 1080px; height: 1920px;\n"
            "        overflow: hidden; background: #000000;\n"
            "      }\n"
            "      .beat-icon {\n"
            "        position: absolute; left: 540px; top: 960px;\n"
            "        width: 300px; height: 300px;\n"
            "        margin-left: -150px; margin-top: -150px;\n"
            "        opacity: 0; display: flex;\n"
            "        align-items: center; justify-content: center;\n"
            "      }\n"
            "      .beat-icon svg {\n"
            "        width: 300px; height: 300px; overflow: visible;\n"
            "        filter: drop-shadow(0 0 12px rgba(240,242,255,0.9))"
            " drop-shadow(0 0 28px rgba(240,242,255,0.45));\n"
            "      }\n"
            "      .beat-anchor {\n"
            "        position: absolute; left: 0; width: 1080px;\n"
            "        text-align: center; top: 704px;\n"
            "        font-family: 'Bebas Neue', Impact, sans-serif;\n"
            "        font-size: 72px; color: #f5c518;\n"
            "        letter-spacing: 0.1em; line-height: 1; opacity: 0;\n"
            "        text-shadow: 0 0 20px rgba(245,197,24,0.6), 0 0 40px rgba(245,197,24,0.3);\n"
            "      }\n"
            "      .beat-text {\n"
            "        position: absolute; left: 80px; width: 920px;\n"
            "        text-align: center; top: 1120px;\n"
            "        font-family: 'Montserrat', Arial, sans-serif;\n"
            "        font-size: 36px; color: #a0a2af;\n"
            "        line-height: 1.5; opacity: 1;\n"
            "      }\n"
            "      .word {\n"
            "        display: inline-block; opacity: 0;\n"
            "      }\n"
            "    </style>\n"
            "  </head>\n"
            "  <body>\n"
            "    <div\n"
            '      id="root"\n'
            '      data-composition-id="main"\n'
            '      data-start="0"\n'
            f'      data-duration="{duration_int}"\n'
            '      data-width="1080"\n'
            '      data-height="1920"\n'
            "    >\n\n"
            f"{beats_html}\n\n"
            "    </div>\n\n"
            "    <script>\n"
            f"{gsap_script}\n"
            "    </script>\n"
            "  </body>\n"
            "</html>\n"
        )

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _render_html(
        self,
        html: str,
        output_path: Path,
        audio_path: Path | None = None,
    ) -> None:
        """
        Write HTML to a temp HyperFrames project dir and invoke the CLI renderer.
        If audio_path is supplied, mixes the voiceover into the visual MP4 via ffmpeg.
        Cleans up the temp dir regardless of success or failure.
        """
        tmpdir = Path(tempfile.mkdtemp(prefix="hf_morph_"))
        try:
            self._write_project_files(tmpdir, html, output_path.stem)

            env = os.environ.copy()
            ffmpeg_dir = self._ensure_ffmpeg()
            if ffmpeg_dir:
                sep = ";" if os.name == "nt" else ":"
                env["PATH"] = str(ffmpeg_dir) + sep + env.get("PATH", "")

            abs_out = str(output_path.resolve())
            # shell=True required on Windows so npx.cmd resolves correctly
            cmd = f'npx --yes hyperframes@0.6.0 render --output "{abs_out}"'
            logger.info("[hf] %s (cwd=%s)", cmd, tmpdir)

            result = subprocess.run(
                cmd,
                cwd=str(tmpdir),
                env=env,
                shell=True,
                capture_output=True,
                text=True,
                timeout=600,
            )
            stdout = result.stdout[-800:] if result.stdout else ""
            stderr = result.stderr[-400:] if result.stderr else ""
            if stdout.strip():
                logger.info("[hf] stdout: %s", stdout)
            if result.returncode != 0:
                raise RuntimeError(
                    f"hyperframes render exit {result.returncode}.\n"
                    f"stdout: {stdout}\nstderr: {stderr}"
                )
            if not output_path.exists() or output_path.stat().st_size == 0:
                raise RuntimeError(
                    f"hyperframes render returned 0 but {output_path} not found or empty."
                )

            # Mix voiceover audio into the visual-only MP4
            if audio_path is not None and audio_path.exists() and audio_path.stat().st_size > 0:
                ffmpeg_exe: Path | str | None = None
                if ffmpeg_dir:
                    candidate = ffmpeg_dir / "ffmpeg.exe"
                    if candidate.exists():
                        ffmpeg_exe = candidate
                if ffmpeg_exe is None:
                    found = shutil.which("ffmpeg")
                    if found:
                        ffmpeg_exe = found
                if ffmpeg_exe:
                    mix_start  = time.time()
                    tmp_mixed  = output_path.with_suffix(".mixed.mp4")
                    mix_result = subprocess.run(
                        [
                            str(ffmpeg_exe),
                            "-i",  str(output_path),
                            "-i",  str(audio_path),
                            "-c:v", "copy",
                            "-c:a", "aac",
                            "-shortest",
                            "-map", "0:v:0",
                            "-map", "1:a:0",
                            "-y",
                            str(tmp_mixed),
                        ],
                        capture_output=True, text=True, timeout=120,
                    )
                    if mix_result.returncode == 0 and tmp_mixed.exists() and tmp_mixed.stat().st_size > 0:
                        tmp_mixed.replace(output_path)
                        logger.info("[hf] Mixed voiceover into output (%.1fs)", time.time() - mix_start)
                    else:
                        logger.warning(
                            "[hf] ffmpeg voiceover mix failed (rc=%d) — visual-only output kept",
                            mix_result.returncode,
                        )
                        if tmp_mixed.exists():
                            tmp_mixed.unlink(missing_ok=True)
                else:
                    logger.warning("[hf] ffmpeg not available — voiceover not mixed into output")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _write_project_files(self, tmpdir: Path, html: str, stem: str) -> None:
        """Write the minimal HyperFrames project structure to tmpdir."""
        (tmpdir / "index.html").write_text(html, encoding="utf-8")
        (tmpdir / "hyperframes.json").write_text(
            json.dumps({
                "$schema": "https://hyperframes.heygen.com/schema/hyperframes.json",
                "registry": "https://raw.githubusercontent.com/heygen-com/hyperframes/main/registry",
                "paths": {
                    "blocks":     "compositions",
                    "components": "compositions/components",
                    "assets":     "assets",
                },
            }, indent=2),
            encoding="utf-8",
        )
        (tmpdir / "meta.json").write_text(
            json.dumps({
                "id":        f"hf-{stem}",
                "name":      f"hf-{stem}",
                "createdAt": datetime.now(timezone.utc).isoformat(),
            }, indent=2),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _audio_duration(self, audio_path: Path) -> float:
        """Return audio duration in seconds.

        Tries ffprobe first, then ffmpeg -i stderr parsing, then mutagen.
        Returns 0.0 only if all strategies fail.
        """
        # Strategy 1: ffprobe
        ffprobe = self._find_ffprobe()
        if ffprobe:
            try:
                r = subprocess.run(
                    [ffprobe, "-v", "quiet", "-show_entries", "format=duration",
                     "-of", "csv=p=0", str(audio_path)],
                    capture_output=True, text=True, timeout=10,
                )
                val = float(r.stdout.strip())
                if val > 0:
                    return val
            except Exception:
                pass

        # Strategy 2: ffmpeg -i parses duration from stderr
        ffmpeg_dir = self._ensure_ffmpeg()
        if ffmpeg_dir:
            ffmpeg_exe = ffmpeg_dir / "ffmpeg.exe"
            if ffmpeg_exe.exists():
                try:
                    import re as _re
                    r = subprocess.run(
                        [str(ffmpeg_exe), "-i", str(audio_path)],
                        capture_output=True, text=True, timeout=10,
                    )
                    m = _re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", r.stderr)
                    if m:
                        h, mi, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
                        return h * 3600 + mi * 60 + s
                except Exception:
                    pass

        # Strategy 3: mutagen (pure Python, no subprocess)
        try:
            from mutagen.mp3 import MP3
            audio = MP3(str(audio_path))
            if audio.info.length > 0:
                return audio.info.length
        except Exception:
            pass

        return 0.0

    def _ensure_ffmpeg(self) -> Path | None:
        """
        Copy imageio_ffmpeg binary to a temp dir as ffmpeg.exe so the
        HyperFrames CLI can find it. Returns the dir path, or None.
        """
        try:
            import imageio_ffmpeg
            src  = Path(imageio_ffmpeg.get_ffmpeg_exe())
            dest_dir = Path(tempfile.gettempdir()) / "hf_ffmpeg_bin"
            dest_dir.mkdir(exist_ok=True)
            dest = dest_dir / "ffmpeg.exe"
            if not dest.exists() or dest.stat().st_size != src.stat().st_size:
                shutil.copy2(src, dest)
            return dest_dir
        except Exception as exc:
            logger.debug("[hf] Could not stage imageio_ffmpeg: %s", exc)
            return None

    def _find_ffprobe(self) -> str | None:
        """Locate ffprobe: try PATH first, then same dir as imageio_ffmpeg."""
        if shutil.which("ffprobe"):
            return "ffprobe"
        try:
            import imageio_ffmpeg
            exe_dir = Path(imageio_ffmpeg.get_ffmpeg_exe()).parent
            for name in ("ffprobe.exe", "ffprobe"):
                candidate = exe_dir / name
                if candidate.exists():
                    return str(candidate)
        except Exception:
            pass
        # Last resort: use ffmpeg itself (same dir as the staged binary)
        ffmpeg_dir = self._ensure_ffmpeg()
        if ffmpeg_dir:
            ffmpeg = ffmpeg_dir / "ffmpeg.exe"
            if ffmpeg.exists():
                # ffmpeg can serve as ffprobe with -i flag
                return None  # caller falls back to 0.0, which triggers DEFAULT_DURATION
        return None

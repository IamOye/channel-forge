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
# Morph dictionary — identical to morph_renderer.py
# ---------------------------------------------------------------------------
_MORPH_TABLE: list[tuple[tuple[str, ...], str]] = [
    (("coffee", "serve", "barista"),                    "coffee_mug"),
    (("retire", "retired", "freedom"),                  "open_door"),
    (("money", "dollars", "earn", "income", "salary"),  "dollar_bill"),
    (("spend", "spent", "cost", "pay"),                 "dollar_bill_down"),
    (("apartment", "home", "house", "live", "lived"),   "house"),
    (("car", "drive", "drove", "vehicle"),              "car"),
    (("think", "thought", "mind", "brain", "smart"),    "brain"),
    (("time", "waiting", "clock", "hours"),             "clock"),
    (("growth", "gain", "grew", "rise", "rising"),      "bar_chart"),
    (("loss", "drop", "losing", "fell", "down"),        "arrow_down"),
    (("subscribe", "follow", "join"),                   "smartphone"),
    (("scale", "balance", "decision", "compare"),       "balance_scale"),
    (("coin", "invest", "asset", "assets"),             "coin"),
    (("chain", "habit", "trap", "stuck"),               "chain_links"),
    (("key", "unlock", "secret", "system"),             "key"),
    (("lock", "risk", "hidden"),                        "padlock"),
    (("eye", "see", "notice", "look"),                  "eye"),
    (("heart", "want", "desire", "love"),               "heart"),
    (("seed", "start", "begin", "born"),                "sprout"),
    (("door", "opportunity", "open"),                   "door_handle"),
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
            self._render_html(html, output_path)
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
        """Map each of the 5 beat slots to text, anchor keyword, and icon name."""
        beats: list[dict] = []
        for name in BEAT_ORDER:
            if name == "cta":
                text = script_dict.get("cta", "") or script_dict.get("question", "")
            else:
                text = script_dict.get(name, "")
            anchor = self._extract_anchor(text)
            icon   = self._lookup_icon(anchor)
            beats.append({"name": name, "text": text, "anchor": anchor, "icon": icon})
        return beats

    def _extract_anchor(self, text: str) -> str:
        """Extract the primary keyword from beat text using priority ordering."""
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
        Build a complete HyperFrames HTML composition string.
        BEAT_DUR is derived from total_duration / number of beats.
        """
        n_beats  = len(beats)
        beat_dur = total_duration / n_beats

        # Beat HTML blocks
        beat_blocks: list[str] = []
        for i, beat in enumerate(beats):
            svg    = ICON_SVG_MAP.get(beat["icon"], ICON_SVG_MAP[_FALLBACK_ICON])
            anchor = _html_mod.escape(beat["anchor"].upper())
            body   = _html_mod.escape(beat["text"])
            beat_blocks.append(
                f'      <div id="icon-{i}" class="beat-icon">{svg}</div>\n'
                f'      <div id="anchor-{i}" class="beat-anchor">{anchor}</div>\n'
                f'      <div id="text-{i}" class="beat-text">{body}</div>'
            )

        beats_html = "\n\n".join(beat_blocks)

        # GSAP timeline script (curly braces escaped for f-string)
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
            f"        const text   = '#text-'   + b;\n"
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
            f"        tl.fromTo(text,\n"
            f"          {{ opacity: 0, y: 24 }},\n"
            f"          {{ opacity: 1, y: 0, duration: ICON_IN, ease: 'power3.out' }},\n"
            f"          inT\n"
            f"        );\n"
            f"        tl.to(icon,\n"
            f"          {{ opacity: 0, scale: 0.8, duration: ICON_OUT, ease: 'power2.in' }},\n"
            f"          outT\n"
            f"        );\n"
            f"        tl.to([anchor, text],\n"
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
            "      }\n"
            "      .beat-text {\n"
            "        position: absolute; left: 80px; width: 920px;\n"
            "        text-align: center; top: 1120px;\n"
            "        font-family: 'Montserrat', Arial, sans-serif;\n"
            "        font-size: 36px; color: #a0a2af;\n"
            "        line-height: 1.5; opacity: 0;\n"
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

    def _render_html(self, html: str, output_path: Path) -> None:
        """
        Write HTML to a temp HyperFrames project dir and invoke the CLI renderer.
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
            if not output_path.exists():
                raise RuntimeError(
                    f"hyperframes render returned 0 but {output_path} not found."
                )
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

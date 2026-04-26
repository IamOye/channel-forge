"""
style_classifier.py — picks a StylePreset for a given script.

Calls Anthropic API once per video at script-analysis time to classify
the script tone, then maps to a StylePreset. Falls back to DEFAULT_STYLE
on any error (network, parse, classification miss).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

from anthropic import Anthropic

from src.media.kinetic_renderer import StylePreset, DEFAULT_STYLE

logger = logging.getLogger(__name__)

# Map tone keys -> StylePreset. Currently all map to default; G.1 adds variants.
PRESET_REGISTRY: dict[str, StylePreset] = {
    "high_energy":    DEFAULT_STYLE,   # G.1 will replace
    "minimal_clean":  DEFAULT_STYLE,   # G.1 will replace
    "dark_cinematic": DEFAULT_STYLE,   # G.1 will replace
    "default":        DEFAULT_STYLE,
}


CLASSIFY_PROMPT = """You are a style classifier for a YouTube Shorts kinetic
typography pipeline. Given a script, pick exactly ONE style category that best
matches the script's tone:

- high_energy: punchy, fast, exciting, lots of urgency or excitement
- minimal_clean: calm, educational, factual, restrained
- dark_cinematic: serious, contrarian, exposing, conspiratorial

Respond with ONLY a JSON object: {"style": "<category>", "reason": "<one sentence>"}

Script:
%s
"""


def classify_script_style(script_text: str) -> StylePreset:
    """
    Returns the StylePreset matching the script's tone. Falls back to
    DEFAULT_STYLE on any error. Never raises.
    """
    if not script_text or not script_text.strip():
        logger.info("[style] empty script, using default preset")
        return DEFAULT_STYLE

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("[style] ANTHROPIC_API_KEY missing, using default preset")
        return DEFAULT_STYLE

    raw = ""
    try:
        client = Anthropic(api_key=api_key)
        # Truncate very long scripts to first 2000 chars for cost control
        truncated = script_text[:2000]
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[
                {"role": "user", "content": CLASSIFY_PROMPT % truncated},
            ],
        )
        raw = response.content[0].text.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        parsed = json.loads(raw)
        style_key = parsed.get("style", "default")
        reason = parsed.get("reason", "")
        preset = PRESET_REGISTRY.get(style_key, DEFAULT_STYLE)
        logger.info(f"[style] classified='{style_key}' preset='{preset.name}' reason={reason!r}")
        return preset
    except json.JSONDecodeError as e:
        logger.warning(f"[style] JSON parse failed: {e}, raw={raw!r}, using default")
        return DEFAULT_STYLE
    except Exception as e:
        logger.warning(f"[style] classification failed: {type(e).__name__}: {e}, using default")
        return DEFAULT_STYLE

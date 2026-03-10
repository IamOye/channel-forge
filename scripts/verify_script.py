"""
Verify script output for a topic — runs HookGenerator + ScriptGenerator only.
No video, no upload. Used to confirm Western audience targeting before production.

Usage:
    PYTHONPATH=. .venv/Scripts/python.exe scripts/verify_script.py "topic here"
"""
import sys
import logging
import os

logging.basicConfig(level=logging.WARNING)  # suppress info noise

sys.path.insert(0, ".")

topic = " ".join(sys.argv[1:]) or "why your salary is designed to keep you poor"

from src.content.hook_generator import HookGenerator
from src.content.script_generator import ScriptGenerator

api_key = os.getenv("ANTHROPIC_API_KEY", "")
if not api_key:
    print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
    sys.exit(1)

print(f"\nTopic: {topic}\n")
print("Generating hooks...")
hook_gen = HookGenerator(api_key=api_key)
hook_result = hook_gen.generate(topic=topic, score=80.0, emotion="shock")

print(f"\n--- Hook variants ---")
for i, v in enumerate(hook_result.variants, 1):
    marker = " <-- SELECTED" if v is hook_result.best else ""
    print(f"  {i}. [{v.combined_score:.1f}] {v.text}{marker}")

print(f"\nBest hook: {hook_result.best.text}")
print("\nGenerating script...")
script_gen = ScriptGenerator(api_key=api_key)
script_result = script_gen.generate(topic=topic, hook=hook_result.best.text)

print(f"\n{'='*60}")
print(f"GENERATED SCRIPT  ({script_result.word_count} words, valid={script_result.is_valid})")
print(f"{'='*60}")
print(f"\n[HOOK]\n{script_result.hook}")
print(f"\n[STATEMENT]\n{script_result.statement}")
print(f"\n[TWIST]\n{script_result.twist}")
print(f"\n[QUESTION]\n{script_result.question}")
print(f"\n{'='*60}")
print(f"Full script:\n{script_result.full_script}")
print(f"{'='*60}\n")

if script_result.validation_errors:
    print(f"Validation errors: {script_result.validation_errors}")

# Check for banned content
banned = ["india", "rupee", "inr", "rs.", "₹", "chai", "cricket", "bollywood",
          "asian market", "african market"]
full_lower = script_result.full_script.lower()
found = [b for b in banned if b in full_lower]
if found:
    print(f"WARNING: Banned terms found: {found}")
else:
    print("Audience check: PASS -- no banned regional terms detected")

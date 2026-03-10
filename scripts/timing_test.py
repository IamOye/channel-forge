"""
Timing test for VideoBuilder._write_two_pass fast render path.
Uses real assets from data/raw/ to measure end-to-end export time.
"""
import logging
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)

from src.media.video_builder import VideoBuilder

TOPIC_ID = "money_salary_001"
AUDIO    = Path("data/raw/money_salary_001_voice.mp3")
CLIPS    = [
    Path("data/raw/money_salary_001_stock_0.mp4"),
    Path("data/raw/money_salary_001_stock_1.mp4"),
    Path("data/raw/money_salary_001_stock_2.mp4"),
    Path("data/raw/money_salary_001_stock_3.mp4"),
]

# Minimal fake word_timestamps to trigger fast path
WORD_TS = [
    {"text": "Your",    "start_time": 0.0,  "end_time": 0.4},
    {"text": "salary",  "start_time": 0.5,  "end_time": 1.0},
    {"text": "is",      "start_time": 1.1,  "end_time": 1.3},
    {"text": "lying",   "start_time": 1.4,  "end_time": 1.8},
    {"text": "to",      "start_time": 2.0,  "end_time": 2.2},
    {"text": "you",     "start_time": 2.3,  "end_time": 2.6},
    {"text": "every",   "start_time": 3.0,  "end_time": 3.3},
    {"text": "single",  "start_time": 3.4,  "end_time": 3.7},
    {"text": "day",     "start_time": 3.8,  "end_time": 4.1},
]

SCRIPT = {
    "hook":      "Your salary is lying to you",
    "statement": "Every pay cheque hides the real cost of your time",
    "twist":     "Your employer sells your hour for 10× what they pay",
    "question":  "So who's actually getting rich here?",
}

builder = VideoBuilder(output_dir="data/output")

print("\n=== VideoBuilder timing test ===")
print(f"Clips: {[str(c) for c in CLIPS]}")
print(f"Audio: {AUDIO}")
print()

t0 = time.perf_counter()
result = builder.build(
    topic_id=TOPIC_ID,
    script_dict=SCRIPT,
    audio_path=AUDIO,
    stock_video_path=CLIPS,
    cta_overlay="Watch full video ↗",
    word_timestamps=WORD_TS,
)
elapsed = time.perf_counter() - t0

print(f"\n{'='*40}")
print(f"  is_valid       : {result.is_valid}")
print(f"  output_path    : {result.output_path}")
print(f"  duration_secs  : {result.duration_seconds:.2f}s")
print(f"  TOTAL EXPORT   : {elapsed:.1f}s  ({elapsed/60:.1f} min)")
if result.validation_errors:
    print(f"  errors         : {result.validation_errors}")
print(f"{'='*40}\n")

TARGET = 240  # 4 minutes
if elapsed < TARGET:
    print(f"PASS -- under {TARGET}s target ({elapsed:.0f}s)")
else:
    print(f"FAIL -- over {TARGET}s target ({elapsed:.0f}s)")

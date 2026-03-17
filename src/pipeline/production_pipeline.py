"""
production_pipeline.py — ProductionPipeline

Orchestrates the full Phase 6–10 production flow for a single topic:
  HookGenerator → ScriptGenerator → VoiceoverGenerator →
  PixabayFetcher → VideoBuilder → MetadataGenerator → YouTubeUploader

All component imports are lazy (inside step methods) so the module loads
cleanly even when optional dependencies (moviepy, ElevenLabs, etc.) are absent.

Usage:
    pipeline = ProductionPipeline(
        anthropic_api_key="...",
        elevenlabs_api_key="...",
        pixabay_api_key="...",
    )
    result = pipeline.run({
        "topic_id":  "stoic_001",
        "keyword":   "stoic quotes",
        "category":  "success",
        "score":     82.5,
    })
    print(result.youtube_video_id)
"""

import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_PATH = Path("data/processed/channel_forge.db")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class _MultiFetchResult:
    """Internal result wrapping multiple Pixabay stock video paths."""

    video_paths: list[str]
    is_valid: bool
    validation_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_paths":        self.video_paths,
            "is_valid":           self.is_valid,
            "validation_errors":  self.validation_errors,
        }


@dataclass
class StepResult:
    """Outcome of a single pipeline step."""

    step: str
    success: bool
    output: Any = None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        output_data = None
        if self.output is not None and hasattr(self.output, "to_dict"):
            output_data = self.output.to_dict()
        elif self.output is not None:
            output_data = str(self.output)
        return {
            "step":    self.step,
            "success": self.success,
            "output":  output_data,
            "error":   self.error,
        }


@dataclass
class PipelineResult:
    """Full result of a ProductionPipeline.run() call."""

    topic_id: str
    keyword: str
    youtube_video_id: str
    youtube_url: str
    is_valid: bool
    steps: list[StepResult] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)
    completed_at: str = ""

    def __post_init__(self) -> None:
        if not self.completed_at:
            self.completed_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic_id":          self.topic_id,
            "keyword":           self.keyword,
            "youtube_video_id":  self.youtube_video_id,
            "youtube_url":       self.youtube_url,
            "is_valid":          self.is_valid,
            "steps":             [s.to_dict() for s in self.steps],
            "validation_errors": self.validation_errors,
            "completed_at":      self.completed_at,
        }


# ---------------------------------------------------------------------------
# ProductionPipeline
# ---------------------------------------------------------------------------

class ProductionPipeline:
    """
    Runs the full 7-step production pipeline for a single topic item.

    Args:
        anthropic_api_key:   Anthropic key for Hook/Script/Metadata generators.
                             If None, reads ANTHROPIC_API_KEY from env.
        elevenlabs_api_key:  ElevenLabs key for VoiceoverGenerator.
                             If None, reads ELEVENLABS_API_KEY from env.
        pixabay_api_key:     Pixabay key for PixabayFetcher.
                             If None, reads PIXABAY_API_KEY from env.
        youtube_channel_key: Key name for YouTubeUploader credentials file.
        db_path:             SQLite path for saving pipeline results.
    """

    def __init__(
        self,
        anthropic_api_key: str | None = None,
        elevenlabs_api_key: str | None = None,
        pixabay_api_key: str | None = None,
        youtube_channel_key: str = "default",
        db_path: str | Path = DB_PATH,
    ) -> None:
        self.anthropic_api_key   = anthropic_api_key
        self.elevenlabs_api_key  = elevenlabs_api_key
        self.pixabay_api_key     = pixabay_api_key
        self.youtube_channel_key = youtube_channel_key
        self.db_path = Path(db_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, topic_item: dict[str, Any]) -> PipelineResult:
        """
        Execute all 7 production steps for one topic.

        The pipeline aborts at the first step that fails, persists the partial
        result to the database, and returns is_valid=False.

        Args:
            topic_item: Dict with keys:
                - topic_id  (str)  — unique identifier, e.g. "stoic_001"
                - keyword   (str)  — topic text, e.g. "stoic quotes"
                - category  (str)  — category slug, e.g. "success"
                - score     (float)— engagement score 0–100

        Returns:
            PipelineResult with youtube_video_id on full success.
        """
        topic_id     = topic_item.get("topic_id", "unknown")
        keyword      = topic_item.get("keyword", "")
        category     = topic_item.get("category", "default")
        score        = float(topic_item.get("score", 0.0))
        video_number = int(topic_item.get("video_number", 0))

        from config.constants import PRODUCTS
        _product    = PRODUCTS.get(category, {})
        cta_script  = _product.get("cta_script", "")
        cta_overlay = _product.get("cta_overlay", "")

        logger.info("Pipeline start: topic_id=%s keyword='%s'", topic_id, keyword)

        steps: list[StepResult] = []
        errors: list[str] = []

        # --- Step 1: Hook ---
        hook_result = self._run_step(
            "hook", steps, errors,
            lambda: self._run_hook_generator(keyword, category, score),
        )
        if hook_result is None:
            return self._fail(topic_id, keyword, steps, errors)

        hook_text = hook_result.best.text

        # --- Step 2: Script ---
        script_result = self._run_step(
            "script", steps, errors,
            lambda: self._run_script_generator(keyword, hook_text, cta_script),
        )
        if script_result is None:
            return self._fail(topic_id, keyword, steps, errors)

        script_dict = {
            "hook":      script_result.hook,
            "statement": script_result.statement,
            "twist":     script_result.twist,
            "question":  script_result.question,
        }

        # --- Step 3: Voiceover ---
        voice_result = self._run_step(
            "voiceover", steps, errors,
            lambda: self._run_voiceover(topic_id, script_dict, category),
        )
        if voice_result is None or not voice_result.is_valid:
            errors.append(f"voiceover failed: {getattr(voice_result, 'validation_errors', [])}")
            return self._fail(topic_id, keyword, steps, errors)

        audio_path = voice_result.audio_path

        # Load word timestamps for caption sync and cut timing
        word_timestamps: list[dict] = []
        words_json_path = getattr(voice_result, "words_path", "")
        if words_json_path:
            import json as _json
            try:
                word_timestamps = _json.loads(Path(words_json_path).read_text(encoding="utf-8"))
                logger.info("Loaded %d word timestamps from %s", len(word_timestamps), words_json_path)
            except Exception as exc:
                logger.warning("Could not load word timestamps: %s", exc)

        # --- Step 4: Stock video(s) ---
        fetch_result = self._run_step(
            "pixabay", steps, errors,
            lambda: self._run_pixabay(topic_id, script_dict, category),
        )
        if fetch_result is None or not fetch_result.is_valid:
            errors.append(f"pixabay failed: {getattr(fetch_result, 'validation_errors', [])}")
            return self._fail(topic_id, keyword, steps, errors)

        stock_video_paths = fetch_result.video_paths

        # --- Step 5: Video build ---
        build_result = self._run_step(
            "video_build", steps, errors,
            lambda: self._run_video_builder(topic_id, script_dict, audio_path, stock_video_paths, cta_overlay, word_timestamps),
        )
        if build_result is None or not build_result.is_valid:
            errors.append(f"video_build failed: {getattr(build_result, 'validation_errors', [])}")
            return self._fail(topic_id, keyword, steps, errors)

        video_path = build_result.output_path

        # --- Step 5.5: Thumbnail ---
        thumbnail_path = ""
        thumb_result = self._run_step(
            "thumbnail", steps, errors,
            lambda: self._run_thumbnail(topic_id, hook_text, keyword, category),
        )
        if thumb_result:
            thumbnail_path = thumb_result if isinstance(thumb_result, str) else ""

        # --- Step 6: Metadata ---
        meta_result = self._run_step(
            "metadata", steps, errors,
            lambda: self._run_metadata(keyword, script_result.full_script, category=category, video_number=video_number),
        )
        if meta_result is None:
            return self._fail(topic_id, keyword, steps, errors)

        metadata = {
            "title":       meta_result.title,
            "description": meta_result.description,
            "tags":        meta_result.hashtags,
            "category_id": "22",    # People & Blogs
        }

        # --- Step 7: YouTube upload ---
        upload_result = self._run_step(
            "youtube_upload", steps, errors,
            lambda: self._run_uploader(topic_id, video_path, metadata, thumbnail_path),
        )
        if upload_result is None or not upload_result.is_valid:
            errors.append(f"upload failed: {getattr(upload_result, 'validation_errors', [])}")
            return self._fail(topic_id, keyword, steps, errors)

        result = PipelineResult(
            topic_id=topic_id,
            keyword=keyword,
            youtube_video_id=upload_result.youtube_video_id,
            youtube_url=upload_result.youtube_url,
            is_valid=True,
            steps=steps,
        )
        self._save_to_db(result)
        logger.info(
            "[production] SUCCESS: video_id=%s url=%s",
            result.youtube_video_id, result.youtube_url,
        )
        self._cleanup_raw_files(topic_id)
        return result

    # ------------------------------------------------------------------
    # Step runners (lazy imports keep module import fast)
    # ------------------------------------------------------------------

    def _run_hook_generator(self, keyword: str, category: str, score: float):
        from src.content.hook_generator import HookGenerator
        gen = HookGenerator(api_key=self.anthropic_api_key)
        return gen.generate(topic=keyword, score=score, emotion=category)

    def _run_script_generator(self, keyword: str, hook_text: str, cta_script: str = ""):
        from src.content.script_generator import ScriptGenerator
        gen = ScriptGenerator(api_key=self.anthropic_api_key)
        return gen.generate(topic=keyword, hook=hook_text, cta_script=cta_script)

    def _run_voiceover(self, topic_id: str, script_dict: dict, category: str):
        from src.media.voiceover import VoiceoverGenerator
        gen = VoiceoverGenerator(api_key=self.elevenlabs_api_key)
        return gen.generate(script_dict=script_dict, topic_id=topic_id, category=category)

    # Words that indicate a high-energy (confrontational / urgent) topic
    HIGH_ENERGY_WORDS: list[str] = [
        "broke", "trap", "salary", "boss",
        "debt", "paycheck", "fired", "poor",
        "steal", "lie", "scam", "warning",
    ]

    def _detect_energy(self, hook: str) -> str:
        """Return 'high' if hook contains a HIGH_ENERGY_WORDS keyword, else 'reflective'."""
        hook_lower = hook.lower()
        for word in self.HIGH_ENERGY_WORDS:
            if word in hook_lower:
                return "high"
        return "reflective"

    def _run_pixabay(self, topic_id: str, script_dict: dict, category: str) -> _MultiFetchResult:
        """Fetch video + Ken Burns photo clips and interleave them.

        Clip mix is dynamic based on hook energy level:
          HIGH energy  → 3 video clips + 1 photo clip
          Reflective   → 2 video clips + 2 photo clips

        Clip order: video → photo → video → (photo or video) → ...
        """
        from src.media.pixabay_fetcher import PixabayFetcher
        from src.media.video_builder import VideoBuilder, VIDEO_DURATION

        fetcher = PixabayFetcher(api_key=self.pixabay_api_key)

        # ── Determine clip mix based on hook energy ────────────────────────────
        hook = script_dict.get("hook", "")
        energy = self._detect_energy(hook)
        if energy == "high":
            n_video, n_photo = 3, 1
        else:
            n_video, n_photo = 2, 2

        logger.info(
            "[pipeline] Clip mix: %d video + %d photo (energy: %s)",
            n_video, n_photo, energy,
        )

        # ── Video clips (action / situational) ────────────────────────────────
        video_phrases = self._extract_broll_keywords(script_dict, count=n_video)
        video_paths = fetcher.fetch_multiple(
            topic_id=topic_id,
            keywords_list=video_phrases,
            count=n_video,
        )

        # ── Photo clips with Ken Burns (conceptual / emotional) ────────────────
        photo_phrases = self._extract_photo_phrases(script_dict)
        builder = VideoBuilder()
        ken_burns_paths: list[str] = []

        for i, phrase in enumerate(photo_phrases[:n_photo]):
            photos = fetcher.fetch_photos(
                topic_id=f"{topic_id}_kb{i}",
                phrase=phrase,
                count=1,
            )
            if photos:
                kb_path = Path("data/raw") / f"{topic_id}_ken_burns_{i}.mp4"
                ok = builder.write_ken_burns_mp4(
                    image_path=photos[0]["local_path"],
                    output_path=kb_path,
                    duration=VIDEO_DURATION,
                )
                if ok:
                    ken_burns_paths.append(str(kb_path))

        # ── Interleave: video0, photo0, video1, photo1 ─────────────────────────
        all_paths: list[str] = []
        v_list = list(video_paths)
        k_list = list(ken_burns_paths)
        for v, k in zip(v_list, k_list):
            all_paths.extend([v, k])
        # Append any remaining clips if counts differ
        all_paths.extend(v_list[len(k_list):])
        all_paths.extend(k_list[len(v_list):])

        if not all_paths:
            return _MultiFetchResult(
                video_paths=[],
                is_valid=False,
                validation_errors=["no stock media found for any b-roll phrase"],
            )
        return _MultiFetchResult(video_paths=all_paths, is_valid=True)

    def _run_video_builder(
        self,
        topic_id: str,
        script_dict: dict,
        audio_path: str,
        stock_paths: list[str],
        cta_overlay: str = "",
        word_timestamps: list[dict] | None = None,
    ):
        from src.media.video_builder import VideoBuilder
        builder = VideoBuilder()
        return builder.build(
            topic_id=topic_id,
            script_dict=script_dict,
            audio_path=audio_path,
            stock_video_path=stock_paths,
            cta_overlay=cta_overlay,
            word_timestamps=word_timestamps,
            anthropic_api_key=self.anthropic_api_key or "",
        )

    def _run_thumbnail(self, topic_id: str, hook: str, keyword: str, category: str) -> str:
        from src.media.thumbnail_generator import ThumbnailGenerator
        gen = ThumbnailGenerator()
        return gen.generate(hook=hook, topic=topic_id, category=category)

    def _extract_broll_keywords(self, script_dict: dict, count: int = 4) -> list[str]:
        """Use Claude to derive visually concrete Pixabay search phrases from the script.

        Splits the script into 4 parts (hook/statement/twist/question) and asks
        Claude to suggest one 2–3 word search query per part that a stock video
        camera could actually film.  Falls back to generic phrases if the API
        call fails.
        """
        import json as _json
        import anthropic

        parts = [
            ("hook",      script_dict.get("hook", "")),
            ("statement", script_dict.get("statement", "")),
            ("twist",     script_dict.get("twist", "")),
            ("question",  script_dict.get("question", "")),
        ]
        parts_text = "\n".join(
            f"Part {i+1} ({label}): {text}" for i, (label, text) in enumerate(parts) if text
        )

        prompt = (
            f"You are a video director selecting b-roll footage for a finance YouTube Short.\n\n"
            f"Script (4 segments):\n{parts_text}\n\n"
            "Generate EXACTLY 4 Pixabay search phrases in this specific mix:\n"
            "  [0] Situational — EMOTION or SITUATION from Part 1 (hook), features a human\n"
            "  [1] Situational — EMOTION or SITUATION from Part 2 (statement), features a human\n"
            "  [2] Famous figure — pick ONE: 'Elon Musk speaking', 'Warren Buffett interview',\n"
            "      'Jeff Bezos presentation', 'Mark Zuckerberg', 'Steve Jobs keynote',\n"
            "      'Bill Gates interview'\n"
            "  [3] Financial context — a finance/wealth/work scene matching Parts 3-4\n\n"
            "RULES for situational clips [0] and [1]:\n"
            "- Must feature a human being in the shot\n"
            "- Show the EMOTION or SITUATION, not literal script words\n"
            "- NEVER: animals, nature, mountains, ocean, abstract, empty rooms, sky, clouds\n"
            "- 2-3 words only, human faces visible, subject centred\n\n"
            "EMOTIONAL MAPPING:\n"
            "  earning money / salary -> 'person receiving paycheck', 'worker getting paid'\n"
            "  boss / wealthy person -> 'confident businessman office', 'successful entrepreneur'\n"
            "  working hard / long hours -> 'tired office worker', 'person working late night'\n"
            "  passive income -> 'person relaxing laptop income'\n"
            "  financial freedom -> 'happy person financial success'\n"
            "  debt / broke -> 'person worried bills', 'stressed person looking at phone'\n"
            "  investment -> 'person reviewing investment portfolio'\n"
            "  inequality -> 'frustrated employee meeting', 'worker vs executive'\n\n"
            "Return a JSON array of exactly 4 strings only, no explanation, no markdown:\n"
            '[\"situational\",\"situational\",\"famous_figure\",\"financial_context\"]'
        )

        try:
            client = anthropic.Anthropic(api_key=self.anthropic_api_key)
            message = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=128,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text.strip()
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            phrases = _json.loads(raw.strip())
            if isinstance(phrases, list) and phrases:
                logger.info("Claude b-roll phrases: %s", phrases)
                return [str(p) for p in phrases[:count]]
        except Exception as exc:
            logger.warning("Claude b-roll extraction failed (%s) — using fallback", exc)

        # Fallback: generic category-based phrases
        from src.media.pixabay_fetcher import KEYWORD_MAP
        base = KEYWORD_MAP.get("default")
        return (base * count)[:count]

    def _extract_photo_phrases(self, script_dict: dict) -> list[str]:
        """Use Claude to generate 2 conceptual/emotional stock photo search phrases.

        Distinct from video b-roll phrases — photos are symbolic/lifestyle-oriented.
        Falls back to generic phrases if API call fails.
        """
        import json as _json
        import anthropic

        full_script = " ".join(filter(None, [
            script_dict.get("hook", ""),
            script_dict.get("statement", ""),
            script_dict.get("twist", ""),
            script_dict.get("question", ""),
        ]))

        prompt = (
            "Generate 2 search phrases for stock PHOTOS (not videos) that visually "
            "represent the emotion and concept of this script.\n"
            "Think lifestyle, mood, symbolic.\n"
            "Examples: 'wealthy lifestyle luxury', 'stressed person bills', "
            "'financial freedom beach laptop', 'empty wallet poverty'\n"
            f"Script: {full_script}\n"
            "Return JSON array of 2 strings only."
        )

        try:
            client = anthropic.Anthropic(api_key=self.anthropic_api_key)
            message = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=80,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text.strip()
            if raw.startswith("```"):
                raw = "\n".join(
                    line for line in raw.splitlines()
                    if not line.strip().startswith("```")
                ).strip()
            phrases = _json.loads(raw)
            if isinstance(phrases, list):
                return [str(p).strip() for p in phrases[:2] if str(p).strip()]
        except Exception as exc:
            logger.warning("[pipeline] Photo phrase extraction failed: %s", exc)

        return ["successful person lifestyle", "financial freedom concept"]

    def _run_metadata(self, keyword: str, script: str, category: str = "", video_number: int = 0):
        from src.content.metadata_generator import MetadataGenerator
        gen = MetadataGenerator(api_key=self.anthropic_api_key)
        return gen.generate(topic=keyword, script=script, category=category, video_number=video_number)

    def _run_uploader(self, topic_id: str, video_path: str, metadata: dict, thumbnail_path: str = ""):
        from src.publisher.youtube_uploader import YouTubeUploader
        uploader = YouTubeUploader(channel_key=self.youtube_channel_key)
        return uploader.upload(topic_id=topic_id, video_path=video_path, metadata=metadata, thumbnail_path=thumbnail_path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run_step(
        self,
        step_name: str,
        steps: list[StepResult],
        errors: list[str],
        fn,
    ):
        """
        Execute fn(), record a StepResult, and return the output.
        Returns None and appends to errors if fn raises.
        """
        try:
            output = fn()
            steps.append(StepResult(step=step_name, success=True, output=output))
            logger.info("Step '%s' succeeded", step_name)
            return output
        except Exception as exc:
            msg = f"{step_name} raised: {exc}"
            logger.error(msg)
            steps.append(StepResult(step=step_name, success=False, error=str(exc)))
            errors.append(msg)
            # Notification 6 — production error
            try:
                from src.notifications.telegram_notifier import TelegramNotifier
                TelegramNotifier().notify_production_error(
                    topic=step_name,
                    step_name=step_name,
                    error_message=str(exc),
                )
            except Exception:
                pass
            return None

    def _fail(
        self,
        topic_id: str,
        keyword: str,
        steps: list[StepResult],
        errors: list[str],
    ) -> PipelineResult:
        result = PipelineResult(
            topic_id=topic_id,
            keyword=keyword,
            youtube_video_id="",
            youtube_url="",
            is_valid=False,
            steps=steps,
            validation_errors=errors,
        )
        self._save_to_db(result)
        reason = "; ".join(errors) if errors else "unknown"
        logger.error("[production] FAILED: %s", reason)
        return result

    def _cleanup_raw_files(self, topic_id: str) -> None:
        """Delete temporary raw media files for topic_id after successful upload."""
        import glob as _glob

        patterns = [
            f"data/raw/{topic_id}_voice.mp3",
            f"data/raw/{topic_id}_words.json",
            f"data/raw/{topic_id}_stock_*.mp4",
            f"data/raw/{topic_id}_photo_*.jpg",
            f"data/raw/{topic_id}_ken_burns_*.mp4",
            f"data/raw/{topic_id}_kb*_photo_*.jpg",
        ]
        deleted = 0
        for pattern in patterns:
            for path in _glob.glob(pattern):
                try:
                    Path(path).unlink(missing_ok=True)
                    deleted += 1
                except Exception as exc:
                    logger.warning("[production] Could not delete %s: %s", path, exc)
        if deleted:
            logger.info("[production] Cleaned up %d raw file(s) for topic_id=%s", deleted, topic_id)

    def _save_to_db(self, result: PipelineResult) -> None:
        """Persist the pipeline result to the production_results table."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS production_results (
                        id                     INTEGER PRIMARY KEY AUTOINCREMENT,
                        topic_id               TEXT NOT NULL,
                        keyword                TEXT NOT NULL,
                        youtube_video_id       TEXT,
                        youtube_url            TEXT,
                        is_valid               INTEGER NOT NULL DEFAULT 0,
                        steps_json             TEXT,
                        validation_errors_json TEXT,
                        completed_at           TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO production_results
                        (topic_id, keyword, youtube_video_id, youtube_url, is_valid,
                         steps_json, validation_errors_json, completed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        result.topic_id,
                        result.keyword,
                        result.youtube_video_id,
                        result.youtube_url,
                        int(result.is_valid),
                        json.dumps([s.to_dict() for s in result.steps]),
                        json.dumps(result.validation_errors),
                        result.completed_at,
                    ),
                )
                conn.commit()
                logger.info("Pipeline result saved: topic_id=%s", result.topic_id)
            finally:
                conn.close()
        except Exception as exc:
            logger.warning("Failed to save pipeline result to DB: %s", exc)

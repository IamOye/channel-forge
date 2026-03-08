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
        topic_id = topic_item.get("topic_id", "unknown")
        keyword  = topic_item.get("keyword", "")
        category = topic_item.get("category", "default")
        score    = float(topic_item.get("score", 0.0))

        from config.constants import PRODUCTS
        cta_product = PRODUCTS.get(category, "")

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
            lambda: self._run_script_generator(keyword, hook_text, cta_product),
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
            lambda: self._run_video_builder(topic_id, script_dict, audio_path, stock_video_paths),
        )
        if build_result is None or not build_result.is_valid:
            errors.append(f"video_build failed: {getattr(build_result, 'validation_errors', [])}")
            return self._fail(topic_id, keyword, steps, errors)

        video_path = build_result.output_path

        # --- Step 6: Metadata ---
        meta_result = self._run_step(
            "metadata", steps, errors,
            lambda: self._run_metadata(keyword, script_result.full_script, cta_product),
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
            lambda: self._run_uploader(topic_id, video_path, metadata),
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
        logger.info("Pipeline complete: topic_id=%s → %s", topic_id, result.youtube_url)
        return result

    # ------------------------------------------------------------------
    # Step runners (lazy imports keep module import fast)
    # ------------------------------------------------------------------

    def _run_hook_generator(self, keyword: str, category: str, score: float):
        from src.content.hook_generator import HookGenerator
        gen = HookGenerator(api_key=self.anthropic_api_key)
        return gen.generate(topic=keyword, score=score, emotion=category)

    def _run_script_generator(self, keyword: str, hook_text: str, cta_product: str = ""):
        from src.content.script_generator import ScriptGenerator
        gen = ScriptGenerator(api_key=self.anthropic_api_key)
        return gen.generate(topic=keyword, hook=hook_text, cta_product=cta_product)

    def _run_voiceover(self, topic_id: str, script_dict: dict, category: str):
        from src.media.voiceover import VoiceoverGenerator
        gen = VoiceoverGenerator(api_key=self.elevenlabs_api_key)
        return gen.generate(script_dict=script_dict, topic_id=topic_id, category=category)

    def _run_pixabay(self, topic_id: str, script_dict: dict, category: str) -> _MultiFetchResult:
        from src.media.pixabay_fetcher import PixabayFetcher
        fetcher = PixabayFetcher(api_key=self.pixabay_api_key)
        phrases = self._extract_broll_keywords(script_dict, count=4)
        paths = fetcher.fetch_multiple(topic_id=topic_id, keywords_list=phrases, count=4)
        if not paths:
            return _MultiFetchResult(
                video_paths=[],
                is_valid=False,
                validation_errors=["no stock videos found for any b-roll phrase"],
            )
        return _MultiFetchResult(video_paths=paths, is_valid=True)

    def _run_video_builder(
        self, topic_id: str, script_dict: dict, audio_path: str, stock_paths: list[str]
    ):
        from src.media.video_builder import VideoBuilder
        builder = VideoBuilder()
        return builder.build(
            topic_id=topic_id,
            script_dict=script_dict,
            audio_path=audio_path,
            stock_video_path=stock_paths,
        )

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
            f"You are a video director choosing b-roll footage for a 4-part script.\n\n"
            f"Script (4 equal time segments):\n{parts_text}\n\n"
            "For each segment, choose ONE Pixabay search phrase (2–3 words) that a "
            "camera crew could actually film on location. The phrase must represent the "
            "single most visually concrete action or object from that segment.\n\n"
            "STRICT RULES:\n"
            "- Prefer human actions over static objects: "
            "\"man counting cash\" beats \"money\"\n"
            "- No abstract nouns: ban 'success', 'wealth', 'mindset', 'inequality', "
            "'freedom', 'passive income'\n"
            "- No animals unless the script explicitly mentions animals\n"
            "- Must be filmable in real life (no metaphors, no concepts)\n"
            "- 2–3 words only\n\n"
            "EXAMPLES:\n"
            "BAD: \"passive income sleeping\", \"dog napping\", \"money abstract\"\n"
            "GOOD: \"investor checking portfolio\", \"businessman reading financial report\", "
            "\"entrepreneur on laptop\", \"luxury apartment interior\"\n\n"
            "Return a JSON array only, no explanation, no markdown:\n"
            '[\"phrase1\",\"phrase2\",\"phrase3\",\"phrase4\"]'
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

    def _run_metadata(self, keyword: str, script: str, cta_product: str = ""):
        from src.content.metadata_generator import MetadataGenerator
        gen = MetadataGenerator(api_key=self.anthropic_api_key)
        return gen.generate(topic=keyword, script=script, cta_product=cta_product)

    def _run_uploader(self, topic_id: str, video_path: str, metadata: dict):
        from src.publisher.youtube_uploader import YouTubeUploader
        uploader = YouTubeUploader(channel_key=self.youtube_channel_key)
        return uploader.upload(topic_id=topic_id, video_path=video_path, metadata=metadata)

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
        return result

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

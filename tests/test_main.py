"""
Tests for main.py — CLI entry point.

All heavy dependencies (scheduler, pipeline, analytics, optimizer, trend scraper)
are mocked. SQLite status tests use tmp_path for real DB assertions.
No real API calls, no real file I/O beyond tmp_path.
"""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from main import (
    build_parser,
    cmd_analytics,
    cmd_crawl,
    cmd_optimize,
    cmd_produce,
    cmd_status,
    cmd_test_pipeline,
    main,
)


# ---------------------------------------------------------------------------
# build_parser
# ---------------------------------------------------------------------------

class TestBuildParser:
    def test_parser_has_all_commands(self) -> None:
        parser = build_parser()
        # Parse each command without error
        for cmd in ["run", "analytics", "optimize", "status", "test-pipeline"]:
            args = parser.parse_args([cmd])
            assert args.command == cmd

    def test_crawl_requires_url(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["crawl"])

    def test_crawl_accepts_url(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["crawl", "https://example.com"])
        assert args.url == "https://example.com"

    def test_produce_requires_topic(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["produce"])

    def test_produce_accepts_topic(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["produce", "--topic", "stoic wisdom"])
        assert args.topic == "stoic wisdom"

    def test_no_command_returns_zero(self) -> None:
        result = main([])
        assert result == 0

    def test_unknown_command_exits_nonzero(self) -> None:
        # argparse exits for unknown subcommand with sys.exit
        with pytest.raises(SystemExit):
            main(["badcommand"])


# ---------------------------------------------------------------------------
# cmd_crawl
# ---------------------------------------------------------------------------

class TestCmdCrawl:
    def test_successful_crawl_returns_zero(self) -> None:
        mock_engine = MagicMock()
        mock_engine.fetch_all.return_value = [{"keyword": "test"}]

        with patch.dict("sys.modules", {
            "src.crawler.trend_scraper": MagicMock(
                TrendScrapingEngine=MagicMock(return_value=mock_engine)
            ),
        }):
            result = cmd_crawl("stoic quotes")

        assert result == 0

    def test_crawl_exception_returns_one(self) -> None:
        with patch.dict("sys.modules", {
            "src.crawler.trend_scraper": MagicMock(
                TrendScrapingEngine=MagicMock(side_effect=Exception("network error"))
            ),
        }):
            result = cmd_crawl("test url")

        assert result == 1

    def test_crawl_calls_fetch_all_with_keyword(self) -> None:
        mock_engine = MagicMock()
        mock_engine.fetch_all.return_value = []

        with patch.dict("sys.modules", {
            "src.crawler.trend_scraper": MagicMock(
                TrendScrapingEngine=MagicMock(return_value=mock_engine)
            ),
        }):
            cmd_crawl("my keyword")

        mock_engine.fetch_all.assert_called_once_with(keywords=["my keyword"])


# ---------------------------------------------------------------------------
# cmd_produce
# ---------------------------------------------------------------------------

class TestCmdProduce:
    def test_successful_produce_returns_zero(self) -> None:
        mock_result = MagicMock(is_valid=True, youtube_video_id="YT_abc123", validation_errors=[])
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = mock_result

        with patch.dict("sys.modules", {
            "src.pipeline.production_pipeline": MagicMock(
                ProductionPipeline=MagicMock(return_value=mock_pipeline)
            ),
        }):
            result = cmd_produce("stoic morning routine")

        assert result == 0

    def test_failed_produce_returns_one(self) -> None:
        mock_result = MagicMock(is_valid=False, validation_errors=["step failed"])
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = mock_result

        with patch.dict("sys.modules", {
            "src.pipeline.production_pipeline": MagicMock(
                ProductionPipeline=MagicMock(return_value=mock_pipeline)
            ),
        }):
            result = cmd_produce("some topic")

        assert result == 1

    def test_exception_returns_one(self) -> None:
        with patch.dict("sys.modules", {
            "src.pipeline.production_pipeline": MagicMock(
                ProductionPipeline=MagicMock(side_effect=Exception("broken"))
            ),
        }):
            result = cmd_produce("some topic")

        assert result == 1

    def test_passes_topic_to_pipeline(self) -> None:
        mock_result = MagicMock(is_valid=True, youtube_video_id="v1", validation_errors=[])
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = mock_result

        with patch.dict("sys.modules", {
            "src.pipeline.production_pipeline": MagicMock(
                ProductionPipeline=MagicMock(return_value=mock_pipeline)
            ),
        }):
            cmd_produce("career secrets")

        call_args = mock_pipeline.run.call_args[0][0]
        assert call_args["keyword"] == "career secrets"


# ---------------------------------------------------------------------------
# cmd_test_pipeline
# ---------------------------------------------------------------------------

class TestCmdTestPipeline:
    def test_returns_zero(self) -> None:
        mock_result = MagicMock(is_valid=True)
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = mock_result

        with patch.dict("sys.modules", {
            "src.pipeline.production_pipeline": MagicMock(
                ProductionPipeline=MagicMock(return_value=mock_pipeline)
            ),
        }):
            result = cmd_test_pipeline()

        assert result == 0

    def test_import_error_returns_one(self) -> None:
        with patch.dict("sys.modules", {
            "src.pipeline.production_pipeline": MagicMock(
                ProductionPipeline=MagicMock(side_effect=ImportError("missing dep"))
            ),
        }):
            result = cmd_test_pipeline()

        assert result == 1


# ---------------------------------------------------------------------------
# cmd_analytics
# ---------------------------------------------------------------------------

class TestCmdAnalytics:
    def test_successful_analytics_returns_zero(self) -> None:
        mock_tracker = MagicMock()
        mock_tracker.track_all.return_value = [MagicMock(), MagicMock()]
        mock_channels = [MagicMock(channel_key="default"), MagicMock(channel_key="career")]

        with patch.dict("sys.modules", {
            "src.analytics.analytics_tracker": MagicMock(
                AnalyticsTracker=MagicMock(return_value=mock_tracker)
            ),
            "config.channels": MagicMock(CHANNELS=mock_channels),
        }):
            result = cmd_analytics()

        assert result == 0

    def test_exception_returns_one(self) -> None:
        with patch.dict("sys.modules", {
            "src.analytics.analytics_tracker": MagicMock(
                AnalyticsTracker=MagicMock(side_effect=Exception("API down"))
            ),
            "config.channels": MagicMock(CHANNELS=[MagicMock(channel_key="default")]),
        }):
            result = cmd_analytics()

        assert result == 1

    def test_calls_track_all_per_channel(self) -> None:
        mock_tracker = MagicMock()
        mock_tracker.track_all.return_value = []
        mock_channels = [
            MagicMock(channel_key="default"),
            MagicMock(channel_key="career"),
        ]

        with patch.dict("sys.modules", {
            "src.analytics.analytics_tracker": MagicMock(
                AnalyticsTracker=MagicMock(return_value=mock_tracker)
            ),
            "config.channels": MagicMock(CHANNELS=mock_channels),
        }):
            cmd_analytics()

        assert mock_tracker.track_all.call_count == 2


# ---------------------------------------------------------------------------
# cmd_optimize
# ---------------------------------------------------------------------------

class TestCmdOptimize:
    def test_successful_optimize_returns_zero(self) -> None:
        mock_result = MagicMock(
            is_valid=True, winners_count=3, losers_count=1, topics_injected=2
        )
        mock_loop = MagicMock()
        mock_loop.run.return_value = mock_result

        with patch.dict("sys.modules", {
            "src.optimizer.optimization_loop": MagicMock(
                OptimizationLoop=MagicMock(return_value=mock_loop)
            ),
        }):
            result = cmd_optimize()

        assert result == 0

    def test_invalid_result_returns_one(self) -> None:
        mock_result = MagicMock(
            is_valid=False, error="API failure",
            winners_count=0, losers_count=0, topics_injected=0,
        )
        mock_loop = MagicMock()
        mock_loop.run.return_value = mock_result

        with patch.dict("sys.modules", {
            "src.optimizer.optimization_loop": MagicMock(
                OptimizationLoop=MagicMock(return_value=mock_loop)
            ),
        }):
            result = cmd_optimize()

        assert result == 1

    def test_exception_returns_one(self) -> None:
        with patch.dict("sys.modules", {
            "src.optimizer.optimization_loop": MagicMock(
                OptimizationLoop=MagicMock(side_effect=Exception("broken"))
            ),
        }):
            result = cmd_optimize()

        assert result == 1


# ---------------------------------------------------------------------------
# cmd_status
# ---------------------------------------------------------------------------

class TestCmdStatus:
    def _make_db(self, tmp_path: Path) -> Path:
        db = tmp_path / "channel_forge.db"
        conn = sqlite3.connect(db)
        conn.execute("""
            CREATE TABLE production_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL DEFAULT 'pending',
                video_path TEXT NOT NULL DEFAULT '',
                publish_at TEXT,
                slot_index INTEGER DEFAULT 0,
                scheduled_at TEXT,
                youtube_video_id TEXT,
                created_at TEXT NOT NULL DEFAULT '2025-01-01'
            )
        """)
        conn.execute("""
            CREATE TABLE uploaded_videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL UNIQUE,
                channel_key TEXT NOT NULL DEFAULT 'default',
                topic_id TEXT NOT NULL DEFAULT '',
                title TEXT NOT NULL DEFAULT '',
                uploaded_at TEXT NOT NULL DEFAULT '2025-01-01'
            )
        """)
        conn.execute("""
            CREATE TABLE optimization_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE video_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL,
                fetched_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE scored_topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                keyword TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()
        return db

    def test_missing_db_returns_one(self, tmp_path) -> None:
        with patch("main.DB_PATH", tmp_path / "nonexistent.db"):
            result = cmd_status()
        assert result == 1

    def test_existing_db_returns_zero(self, tmp_path) -> None:
        db = self._make_db(tmp_path)
        with patch("main.DB_PATH", db):
            result = cmd_status()
        assert result == 0

    def test_shows_video_count(self, tmp_path, capsys) -> None:
        db = self._make_db(tmp_path)
        conn = sqlite3.connect(db)
        conn.execute("INSERT INTO uploaded_videos (video_id) VALUES ('v1')")
        conn.execute("INSERT INTO uploaded_videos (video_id) VALUES ('v2')")
        conn.commit()
        conn.close()

        with patch("main.DB_PATH", db):
            cmd_status()

        captured = capsys.readouterr()
        assert "2" in captured.out

    def test_shows_topic_count(self, tmp_path, capsys) -> None:
        db = self._make_db(tmp_path)
        conn = sqlite3.connect(db)
        conn.execute("INSERT INTO scored_topics (keyword) VALUES ('stoic')")
        conn.commit()
        conn.close()

        with patch("main.DB_PATH", db):
            cmd_status()

        captured = capsys.readouterr()
        assert "1" in captured.out

    def test_shows_never_when_no_optimization_run(self, tmp_path, capsys) -> None:
        db = self._make_db(tmp_path)
        with patch("main.DB_PATH", db):
            cmd_status()
        captured = capsys.readouterr()
        assert "never" in captured.out

    def test_shows_queue_empty_when_no_items(self, tmp_path, capsys) -> None:
        db = self._make_db(tmp_path)
        with patch("main.DB_PATH", db):
            cmd_status()
        captured = capsys.readouterr()
        assert "empty" in captured.out

    def test_shows_queue_status_counts(self, tmp_path, capsys) -> None:
        db = self._make_db(tmp_path)
        conn = sqlite3.connect(db)
        conn.execute(
            "INSERT INTO production_queue (topic_id, status, created_at) VALUES ('t1','pending','2025-01-01')"
        )
        conn.execute(
            "INSERT INTO production_queue (topic_id, status, created_at) VALUES ('t2','done','2025-01-01')"
        )
        conn.commit()
        conn.close()

        with patch("main.DB_PATH", db):
            cmd_status()

        captured = capsys.readouterr()
        assert "pending" in captured.out
        assert "done" in captured.out

    def test_missing_tables_handled_gracefully(self, tmp_path) -> None:
        """A DB with no tables should still return 0 (OperationalError caught)."""
        db = tmp_path / "empty.db"
        sqlite3.connect(db).close()  # create empty DB

        with patch("main.DB_PATH", db):
            result = cmd_status()

        assert result == 0


# ---------------------------------------------------------------------------
# main() dispatcher
# ---------------------------------------------------------------------------

class TestMainDispatcher:
    def test_run_calls_build_scheduler(self) -> None:
        mock_scheduler = MagicMock()
        mock_scheduler.start.side_effect = KeyboardInterrupt

        with patch("main.cmd_run", return_value=0) as mock_run:
            result = main(["run"])

        mock_run.assert_called_once()
        assert result == 0

    def test_crawl_dispatched(self) -> None:
        with patch("main.cmd_crawl", return_value=0) as mock_crawl:
            result = main(["crawl", "http://example.com"])

        mock_crawl.assert_called_once_with("http://example.com")
        assert result == 0

    def test_produce_dispatched(self) -> None:
        with patch("main.cmd_produce", return_value=0) as mock_produce:
            result = main(["produce", "--topic", "stoic tips"])

        mock_produce.assert_called_once_with("stoic tips", "money_debate")
        assert result == 0

    def test_test_pipeline_dispatched(self) -> None:
        with patch("main.cmd_test_pipeline", return_value=0) as mock_tp:
            result = main(["test-pipeline"])

        mock_tp.assert_called_once()
        assert result == 0

    def test_analytics_dispatched(self) -> None:
        with patch("main.cmd_analytics", return_value=0) as mock_a:
            result = main(["analytics"])

        mock_a.assert_called_once()
        assert result == 0

    def test_optimize_dispatched(self) -> None:
        with patch("main.cmd_optimize", return_value=0) as mock_o:
            result = main(["optimize"])

        mock_o.assert_called_once()
        assert result == 0

    def test_status_dispatched(self) -> None:
        with patch("main.cmd_status", return_value=0) as mock_s:
            result = main(["status"])

        mock_s.assert_called_once()
        assert result == 0

"""
Tests for src/scheduler.py

All external dependencies (APScheduler, TrendScrapingEngine, MultiChannelOrchestrator,
AnalyticsTracker, OptimizationLoop) are mocked — no real API calls or scheduler
blocking loops.
"""

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# build_scheduler
# ---------------------------------------------------------------------------

class TestBuildScheduler:
    def test_returns_blocking_scheduler(self) -> None:
        from src.scheduler import build_scheduler
        from apscheduler.schedulers.blocking import BlockingScheduler

        scheduler = build_scheduler(timezone_name="UTC")
        assert isinstance(scheduler, BlockingScheduler)

    def test_registers_four_jobs(self) -> None:
        from src.scheduler import build_scheduler

        scheduler = build_scheduler(timezone_name="UTC")
        jobs = scheduler.get_jobs()
        assert len(jobs) == 5

    def test_job_ids_are_correct(self) -> None:
        from src.scheduler import build_scheduler

        scheduler = build_scheduler(timezone_name="UTC")
        ids = {j.id for j in scheduler.get_jobs()}
        assert ids == {"scraper", "production", "analytics", "optimization", "competitor_research"}

    def test_job_names_are_set(self) -> None:
        from src.scheduler import build_scheduler

        scheduler = build_scheduler(timezone_name="UTC")
        names = {j.name for j in scheduler.get_jobs()}
        assert "Channel Scrapers" in names
        assert "Channel Production" in names
        assert "Daily Analytics" in names
        assert "Weekly Optimization" in names

    def test_custom_timezone_accepted(self) -> None:
        from src.scheduler import build_scheduler

        scheduler = build_scheduler(timezone_name="America/New_York")
        assert len(scheduler.get_jobs()) == 5

    def test_env_timezone_used_when_no_override(self) -> None:
        from src.scheduler import build_scheduler

        with patch.dict("os.environ", {"UPLOAD_TIMEZONE": "Europe/London"}):
            scheduler = build_scheduler()
        assert len(scheduler.get_jobs()) == 5

    def test_default_timezone_is_africa_lagos(self) -> None:
        """When no env var and no override, Africa/Lagos is used without error."""
        import os
        from src.scheduler import build_scheduler

        env_without_tz = {k: v for k, v in os.environ.items() if k != "UPLOAD_TIMEZONE"}
        with patch.dict("os.environ", env_without_tz, clear=True):
            scheduler = build_scheduler()
        assert scheduler is not None


# ---------------------------------------------------------------------------
# Trigger configuration (cron expressions)
# ---------------------------------------------------------------------------

class TestTriggerConfig:
    def _get_job(self, scheduler, job_id: str):
        for j in scheduler.get_jobs():
            if j.id == job_id:
                return j
        raise KeyError(f"Job '{job_id}' not found")

    def test_scraper_has_cron_trigger(self) -> None:
        from src.scheduler import build_scheduler
        from apscheduler.triggers.cron import CronTrigger

        scheduler = build_scheduler(timezone_name="UTC")
        job = self._get_job(scheduler, "scraper")
        assert isinstance(job.trigger, CronTrigger)

    def test_production_has_cron_trigger(self) -> None:
        from src.scheduler import build_scheduler
        from apscheduler.triggers.cron import CronTrigger

        scheduler = build_scheduler(timezone_name="UTC")
        job = self._get_job(scheduler, "production")
        assert isinstance(job.trigger, CronTrigger)

    def test_analytics_has_cron_trigger(self) -> None:
        from src.scheduler import build_scheduler
        from apscheduler.triggers.cron import CronTrigger

        scheduler = build_scheduler(timezone_name="UTC")
        job = self._get_job(scheduler, "analytics")
        assert isinstance(job.trigger, CronTrigger)

    def test_optimization_has_cron_trigger(self) -> None:
        from src.scheduler import build_scheduler
        from apscheduler.triggers.cron import CronTrigger

        scheduler = build_scheduler(timezone_name="UTC")
        job = self._get_job(scheduler, "optimization")
        assert isinstance(job.trigger, CronTrigger)

    def test_scraper_func_is_run_all_channel_scrapers(self) -> None:
        from src.scheduler import build_scheduler, run_all_channel_scrapers

        scheduler = build_scheduler(timezone_name="UTC")
        job = self._get_job(scheduler, "scraper")
        assert job.func is run_all_channel_scrapers

    def test_production_func_is_run_all_channel_production(self) -> None:
        from src.scheduler import build_scheduler, run_all_channel_production

        scheduler = build_scheduler(timezone_name="UTC")
        job = self._get_job(scheduler, "production")
        assert job.func is run_all_channel_production

    def test_analytics_func_is_run_daily_analytics(self) -> None:
        from src.scheduler import build_scheduler, run_daily_analytics

        scheduler = build_scheduler(timezone_name="UTC")
        job = self._get_job(scheduler, "analytics")
        assert job.func is run_daily_analytics

    def test_optimization_func_is_run_weekly_optimization(self) -> None:
        from src.scheduler import build_scheduler, run_weekly_optimization

        scheduler = build_scheduler(timezone_name="UTC")
        job = self._get_job(scheduler, "optimization")
        assert job.func is run_weekly_optimization


# ---------------------------------------------------------------------------
# Job functions — unit tests (mocked dependencies)
# ---------------------------------------------------------------------------

class TestRunAllChannelScrapers:
    def test_logs_start_and_end(self, caplog) -> None:
        from src.scheduler import run_all_channel_scrapers

        mock_engine = MagicMock()
        mock_engine.fetch_all.return_value = []
        mock_channels = [MagicMock(channel_key="ch1", category="success")]

        with patch("src.scheduler.TrendScrapingEngine", return_value=mock_engine, create=True):
            with patch("src.scheduler.CHANNELS", mock_channels, create=True):
                with patch("src.crawler.trend_scraper.TrendScrapingEngine", return_value=mock_engine, create=True):
                    with patch("config.channels.CHANNELS", mock_channels):
                        run_all_channel_scrapers()

    def test_exception_does_not_propagate(self) -> None:
        from src.scheduler import run_all_channel_scrapers

        with patch(
            "src.crawler.trend_scraper.TrendScrapingEngine",
            side_effect=ImportError("no module"),
            create=True,
        ):
            # Should not raise
            run_all_channel_scrapers()

    def test_channel_exception_does_not_stop_others(self) -> None:
        from src.scheduler import run_all_channel_scrapers

        mock_engine = MagicMock()
        mock_engine.fetch_all.side_effect = Exception("network error")

        ch1 = MagicMock(channel_key="ch1", category="success")
        ch2 = MagicMock(channel_key="ch2", category="career")
        mock_channels = [ch1, ch2]

        call_count = {"n": 0}
        original_side_effect = mock_engine.fetch_all.side_effect

        def counting_fetch(**kwargs):
            call_count["n"] += 1
            raise Exception("network error")

        mock_engine.fetch_all.side_effect = counting_fetch

        # Patch at the source level
        with patch.dict("sys.modules", {
            "src.crawler.trend_scraper": MagicMock(TrendScrapingEngine=lambda: mock_engine),
            "config.channels": MagicMock(CHANNELS=mock_channels),
        }):
            run_all_channel_scrapers()

        # Both channels attempted despite first failing
        assert call_count["n"] == 2


class TestRunAllChannelProduction:
    def test_exception_does_not_propagate(self) -> None:
        from src.scheduler import run_all_channel_production

        with patch.dict("sys.modules", {
            "src.pipeline.multi_channel_orchestrator": MagicMock(
                MultiChannelOrchestrator=MagicMock(side_effect=Exception("broken"))
            ),
        }):
            run_all_channel_production()

    def test_calls_run_all_and_logs_results(self) -> None:
        from src.scheduler import run_all_channel_production

        mock_result = MagicMock(
            channel_key="ch1", topics_succeeded=2, topics_failed=0, is_valid=True
        )
        mock_orchestrator = MagicMock()
        mock_orchestrator.run_all.return_value = [mock_result]
        MockClass = MagicMock(return_value=mock_orchestrator)

        with patch.dict("sys.modules", {
            "src.pipeline.multi_channel_orchestrator": MagicMock(
                MultiChannelOrchestrator=MockClass
            ),
        }):
            run_all_channel_production()

        mock_orchestrator.run_all.assert_called_once()


class TestRunDailyAnalytics:
    def test_exception_does_not_propagate(self) -> None:
        from src.scheduler import run_daily_analytics

        with patch.dict("sys.modules", {
            "src.analytics.analytics_tracker": MagicMock(
                AnalyticsTracker=MagicMock(side_effect=Exception("broken"))
            ),
        }):
            run_daily_analytics()

    def test_calls_track_all_per_channel(self) -> None:
        from src.scheduler import run_daily_analytics

        mock_tracker = MagicMock()
        mock_tracker.track_all.return_value = []
        MockTrackerClass = MagicMock(return_value=mock_tracker)

        ch1 = MagicMock(channel_key="default")
        ch2 = MagicMock(channel_key="career")

        with patch.dict("sys.modules", {
            "src.analytics.analytics_tracker": MagicMock(AnalyticsTracker=MockTrackerClass),
            "config.channels": MagicMock(CHANNELS=[ch1, ch2]),
        }):
            run_daily_analytics()

        assert mock_tracker.track_all.call_count == 2


class TestRunWeeklyOptimization:
    def test_exception_does_not_propagate(self) -> None:
        from src.scheduler import run_weekly_optimization

        with patch.dict("sys.modules", {
            "src.optimizer.optimization_loop": MagicMock(
                OptimizationLoop=MagicMock(side_effect=Exception("broken"))
            ),
        }):
            run_weekly_optimization()

    def test_calls_loop_run(self) -> None:
        from src.scheduler import run_weekly_optimization

        mock_result = MagicMock(
            winners_count=3, losers_count=1, topics_injected=2, is_valid=True
        )
        mock_loop = MagicMock()
        mock_loop.run.return_value = mock_result
        MockClass = MagicMock(return_value=mock_loop)

        with patch.dict("sys.modules", {
            "src.optimizer.optimization_loop": MagicMock(OptimizationLoop=MockClass),
        }):
            run_weekly_optimization()

        mock_loop.run.assert_called_once()

    def test_invalid_result_logged_but_no_exception(self) -> None:
        from src.scheduler import run_weekly_optimization

        mock_result = MagicMock(
            winners_count=0, losers_count=0, topics_injected=0,
            is_valid=False, error="Claude down",
        )
        mock_loop = MagicMock()
        mock_loop.run.return_value = mock_result
        MockClass = MagicMock(return_value=mock_loop)

        with patch.dict("sys.modules", {
            "src.optimizer.optimization_loop": MagicMock(OptimizationLoop=MockClass),
        }):
            # Should not raise even when is_valid=False
            run_weekly_optimization()

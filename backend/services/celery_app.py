"""
Celery Application Configuration
Background task processing for the orchestrator
"""

import os
import logging
from celery import Celery

from config import settings
from celery.signals import (
    after_setup_logger,
    worker_ready,
    worker_shutdown,
    worker_process_init,
)
from kombu import Queue

logger = logging.getLogger(__name__)

# Celery configuration
CELERY_BROKER_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
CELERY_RESULT_BACKEND = os.getenv("REDIS_URL", "redis://redis:6379/0")

# Create Celery app
celery_app = Celery(
    "codex_orchestrator",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=[
        "services.celery_tasks.agent_tasks",
        "services.celery_tasks.scheduled_agent_tasks",
        "services.celery_tasks.rss_tasks",
        "services.celery_tasks.chat_attachment_tasks",
        "services.celery_tasks.image_cleanup_tasks",
        "services.celery_tasks.document_tasks",
        "services.celery_tasks.connection_health_tasks",
        "services.celery_tasks.team_heartbeat_tasks",
        "services.celery_tasks.chat_directive_task",
        "services.celery_tasks.browser_session_health_tasks",
        "services.celery_tasks.proposal_cleanup_tasks",
        "services.celery_tasks.scraper_tasks",
        "services.celery_tasks.fact_tasks",
        "services.celery_tasks.fact_extraction_task",
        "services.celery_tasks.episode_tasks",
        "services.celery_tasks.session_analysis_task",
        "services.celery_tasks.fact_theme_tasks",
        "services.celery_tasks.document_version_tasks",
        "services.celery_tasks.model_health_tasks",
        "services.celery_tasks.audio_export_tasks",
        "services.celery_tasks.federation_tasks",
        "services.celery_tasks.skill_metrics_tasks",
        "services.celery_tasks.skill_promotion_tasks",
        "services.celery_tasks.code_workspace_tasks",
        "services.celery_tasks.music_cover_warm_tasks",
    ]
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Task routing
    task_routes={
        "services.celery_tasks.agent_tasks.*": {"queue": "agents"},
        "services.celery_tasks.scheduled_agent_tasks.*": {"queue": "agents"},
        "services.celery_tasks.rss_tasks.*": {"queue": "rss"},
        "services.celery_tasks.scraper_tasks.*": {"queue": "scrapers"},
        "services.celery_tasks.document_tasks.bulk_reindex_batch": {"queue": "reindex"},
    },
    
    # Queue configuration
    task_default_queue="default",
    task_queues=(
        Queue("default", routing_key="default"),
        Queue("orchestrator", routing_key="orchestrator"),
        Queue("agents", routing_key="agents"),
        Queue("research", routing_key="research"),
        Queue("coding", routing_key="coding"),
        Queue("rss", routing_key="rss"),
        Queue("scrapers", routing_key="scrapers"),
        Queue("reindex", routing_key="reindex"),
    ),
    
    # Worker settings
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_disable_rate_limits=True,
    
    # Task execution settings
    task_soft_time_limit=600,  # 10 minutes soft limit (RSS crawling can be slow)
    task_time_limit=1200,      # 20 minutes hard limit
    task_track_started=True,
    
    # Result settings
    result_expires=3600,       # Results expire after 1 hour
    result_persistent=True,
    
    # Serialization settings to prevent exception issues
    task_ignore_result=False,
    task_store_errors_even_if_ignored=True,
    
    # Task retry settings
    task_default_retry_delay=60,
    task_max_retries=3,
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,

    # Celery Beat Schedule Configuration
    # Schedule for automatic RSS feed polling and other periodic tasks
    beat_schedule={
        # RSS feed polling - run every 5 minutes
        'poll-rss-feeds': {
            'task': 'services.celery_tasks.rss_tasks.scheduled_rss_poll_task',
            'schedule': 300.0,  # 5 minutes in seconds
        },
        # RSS health check - run every 30 minutes
        'rss-health-check': {
            'task': 'services.celery_tasks.rss_tasks.rss_health_check_task',
            'schedule': 1800.0,  # 30 minutes in seconds
        },
        # Clean up stuck RSS polling feeds - run every 15 minutes
        'cleanup-stuck-rss-feeds': {
            'task': 'services.celery_tasks.rss_tasks.cleanup_stuck_rss_feeds_task',
            'schedule': 900.0,  # 15 minutes in seconds
        },
        # Retry Crawl4AI full-content for articles still missing body (small batch, infrequent)
        'rss-full-content-backfill': {
            'task': 'services.celery_tasks.rss_tasks.scheduled_rss_full_content_backfill_task',
            'schedule': 21600.0,  # 6 hours
        },
        # Clean up old chat attachments - run daily
        'cleanup-old-chat-attachments': {
            'task': 'services.celery_tasks.chat_attachment_tasks.cleanup_old_chat_attachments_task',
            'schedule': 86400.0,  # 24 hours in seconds
        },
        'cleanup-orphaned-generated-images': {
            'task': 'services.celery_tasks.image_cleanup_tasks.cleanup_orphaned_generated_images_task',
            'schedule': 86400.0,  # 24 hours in seconds
        },
        # Agent Factory: check for due scheduled agents every 60 seconds
        'check-agent-schedules': {
            'task': 'services.celery_tasks.scheduled_agent_tasks.check_agent_schedules',
            'schedule': 60.0,
        },
        # Agent Factory: check for due team heartbeats every 60 seconds
        'check-team-heartbeats': {
            'task': 'services.celery_tasks.team_heartbeat_tasks.check_team_heartbeats',
            'schedule': 60.0,
        },
        # Agent Factory: check for teams with pending worker tasks every 60 seconds
        'check-worker-dispatches': {
            'task': 'services.celery_tasks.team_heartbeat_tasks.check_worker_dispatches',
            'schedule': 60.0,
        },
        # Agent Factory: poll watched email accounts every 5 minutes
        'poll-watched-emails': {
            'task': 'services.celery_tasks.agent_tasks.poll_watched_emails',
            'schedule': 300.0,
        },
        # Chat bot connections: check status and re-register stopped bots every 2 minutes
        'sync-chat-bot-connections': {
            'task': 'services.celery_tasks.connection_health_tasks.sync_chat_bot_connections',
            'schedule': 120.0,
        },
        # OAuth (email) connections: refresh tokens expiring within 10 minutes every 30 minutes
        'sync-oauth-connections': {
            'task': 'services.celery_tasks.connection_health_tasks.sync_oauth_connections',
            'schedule': 1800.0,
        },
        # Federation: background pull of remote outboxes + local outbox prune
        'federation-outbox-sync': {
            'task': 'services.celery_tasks.federation_tasks.federation_sync_outbox_beat',
            'schedule': float(getattr(settings, "FEDERATION_POLL_INTERVAL_SECONDS", 5) or 5),
        },
        'federation-presence-sync': {
            'task': 'services.celery_tasks.federation_tasks.federation_presence_sync_beat',
            'schedule': float(
                getattr(settings, "FEDERATION_PRESENCE_SYNC_INTERVAL_SECONDS", 30) or 30
            ),
        },
        # Expired document edit proposals: cleanup hourly
        'cleanup-expired-proposals': {
            'task': 'services.celery_tasks.proposal_cleanup_tasks.cleanup_expired_proposals',
            'schedule': 3600.0,
        },
        # User facts: purge expired facts hourly
        'purge-expired-facts': {
            'task': 'services.celery_tasks.fact_tasks.purge_expired_facts_task',
            'schedule': 3600.0,
        },
        # User facts: cluster embeddings into themes (every 6 hours)
        'cluster-user-fact-themes': {
            'task': 'services.celery_tasks.fact_theme_tasks.cluster_user_fact_themes_task',
            'schedule': 21600.0,
        },
        # Episodic memory: mark episodes older than 48h as aged (every 6 hours)
        'mark-aged-episodes': {
            'task': 'services.celery_tasks.episode_tasks.mark_episodes_aged_task',
            'schedule': 21600.0,
        },
        # Episodic memory: graduate facts from 7d+ episodes then delete (daily)
        'graduate-old-episodes': {
            'task': 'services.celery_tasks.episode_tasks.graduate_and_purge_old_episodes_task',
            'schedule': 86400.0,
        },
        # Session memory: enqueue post_session_analysis for idle conversations (every 5 minutes)
        'detect-idle-session-summaries': {
            'task': 'services.celery_tasks.session_analysis_task.detect_idle_sessions_task',
            'schedule': 300.0,
        },
        # Document versions: prune old versions daily (retention 90 days, keep every 10th, max 200/doc)
        'prune-document-versions': {
            'task': 'services.celery_tasks.document_version_tasks.prune_document_versions_task',
            'schedule': 86400.0,
        },
        # Browser session states: check saved sessions every 30 minutes, invalidate if login page detected
        'check-browser-session-health': {
            'task': 'services.celery_tasks.browser_session_health_tasks.check_browser_session_health',
            'schedule': 1800.0,
        },
        # Admin LLM catalog: refresh and notify admins if enabled/role models are orphaned (6 hours)
        'check-admin-llm-catalog-health': {
            'task': 'services.celery_tasks.model_health_tasks.check_admin_llm_catalog_health_task',
            'schedule': 21600.0,
        },
        # Skill metrics: refresh materialized view and prune old events (daily)
        'refresh-skill-usage-stats': {
            'task': 'services.celery_tasks.skill_metrics_tasks.refresh_skill_usage_stats',
            'schedule': 86400.0,
        },
        # Skill promotion/demotion recommendations (weekly)
        'generate-skill-promotion-recommendations': {
            'task': 'services.celery_tasks.skill_promotion_tasks.generate_skill_promotion_recommendations',
            'schedule': 604800.0,
        },
    },
    # Beat scheduler settings
    beat_max_loop_interval=300,  # Check for new tasks every 5 minutes
)


class _CeleryBeatAndTaskChatterToDebugFilter(logging.Filter):
    """Demote noisy periodic Celery INFO lines to DEBUG (visible when log level is DEBUG)."""

    _LOGGER_NAMES = frozenset(
        {"celery.beat", "celery.worker.strategy", "celery.app.trace"}
    )

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno != logging.INFO or record.name not in self._LOGGER_NAMES:
            return True
        try:
            msg = record.getMessage()
        except (ValueError, TypeError):
            return True
        if "Scheduler: Sending due task" in msg:
            pass
        elif msg.startswith("Task ") and "] received" in msg:
            pass
        elif msg.startswith("Task ") and " succeeded in " in msg:
            pass
        else:
            return True
        record.levelno = logging.DEBUG
        record.levelname = logging.getLevelName(logging.DEBUG)
        return True


def _install_celery_chatter_demote_filters() -> None:
    filt = _CeleryBeatAndTaskChatterToDebugFilter()
    for name in _CeleryBeatAndTaskChatterToDebugFilter._LOGGER_NAMES:
        lg = logging.getLogger(name)
        if not any(type(f) is _CeleryBeatAndTaskChatterToDebugFilter for f in lg.filters):
            lg.addFilter(filt)


@after_setup_logger.connect(weak=False)
def _after_setup_logger_demote_celery_chatter(
    sender=None, logger=None, **kwargs
) -> None:
    if logger is None or logger.name not in _CeleryBeatAndTaskChatterToDebugFilter._LOGGER_NAMES:
        return
    if any(type(f) is _CeleryBeatAndTaskChatterToDebugFilter for f in logger.filters):
        return
    logger.addFilter(_CeleryBeatAndTaskChatterToDebugFilter())


_install_celery_chatter_demote_filters()

# Worker lifecycle events
@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    """Called when Celery worker is ready to receive tasks"""
    # Set environment variable to indicate we're in a Celery worker
    os.environ['CELERY_WORKER_RUNNING'] = 'true'
    raw = os.environ.get("FEDERATION_ENABLED")
    role = "beat" if sender is not None and getattr(sender, "schedule", None) else "worker"
    logger.info(
        "Celery %s ready; FEDERATION_ENABLED=%s (raw env %r). "
        "If federation tasks skip, align this service's env with the API (docker-compose / k8s).",
        role,
        getattr(settings, "FEDERATION_ENABLED", None),
        raw,
    )

@worker_shutdown.connect  
def worker_shutdown_handler(sender=None, **kwargs):
    """Called when Celery worker is shutting down"""
    logger.info("🛑 CELERY WORKER: Shutting down orchestrator worker")

# Task status constants
class TaskStatus:
    PENDING = "PENDING"
    STARTED = "STARTED"
    PROGRESS = "PROGRESS"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RETRY = "RETRY"
    REVOKED = "REVOKED"

# Progress update utility  
def update_task_progress(task, current_step: int, total_steps: int, message: str):
    """Update task progress for real-time monitoring"""
    from datetime import datetime
    
    try:
        progress = {
            "current_step": current_step,
            "total_steps": total_steps,
            "percentage": int((current_step / total_steps) * 100) if total_steps > 0 else 0,
            "message": str(message)[:500],  # Limit message length
            "timestamp": datetime.now().isoformat()
        }
        
        task.update_state(
            state=TaskStatus.PROGRESS,
            meta=progress
        )
        
        logger.info(f"📊 TASK PROGRESS: {message} ({current_step}/{total_steps})")
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to update task progress: {e}")
        # Continue execution even if progress update fails

# Worker initialization hook
@worker_process_init.connect
def on_worker_process_init(sender=None, **kwargs):
    """Create persistent event loop in each worker process."""
    from services.celery_tasks.async_runner import init_worker_loop, run_async
    from services.schema_guards import ensure_user_memory_schema_columns

    init_worker_loop()
    run_async(ensure_user_memory_schema_columns())


@celery_app.task(bind=True, name="worker.warmup")
def warmup_worker_task(self):
    """Task to warm up worker on startup"""
    from services.celery_tasks.async_runner import run_async
    from services.worker_warmup import worker_warmup_service

    logger.info("🔥 WORKER WARMUP TASK: Starting...")

    try:
        result = run_async(worker_warmup_service.warmup_worker())
        logger.info(f"🔥 WORKER WARMUP RESULT: {result}")
        return result
    except Exception as e:
        logger.error(f"❌ WORKER WARMUP FAILED: {e}")
        return {"status": "failed", "error": str(e)}


# Worker ready signal - warm up when worker starts
@worker_ready.connect
def on_worker_ready(sender, **kwargs):
    """Warm up worker when it starts"""
    logger.info("🔥 WORKER READY: Starting warmup process...")

    # Run warmup task
    warmup_worker_task.delay()

# Beat ready signal - log when beat scheduler starts
@worker_ready.connect
def on_beat_ready(sender, **kwargs):
    """Log when Celery Beat scheduler is ready"""
    if hasattr(sender, 'schedule') and sender.schedule:
        logger.info("⏰ CELERY BEAT: Scheduler ready with configured tasks:")
        for task_name, task_config in sender.schedule.items():
            logger.info(f"   📅 {task_name}: {task_config}")
    else:
        logger.info("⏰ CELERY BEAT: Scheduler ready (no tasks configured)")

# Worker shutdown signal - cleanup resources
@worker_shutdown.connect
def on_worker_shutdown(sender, **kwargs):
    """Clean up resources when worker shuts down"""
    logger.info("🛑 WORKER SHUTDOWN: Cleaning up resources...")

    try:
        from services.celery_tasks.async_runner import close_worker_loop
        close_worker_loop()
        logger.info("✅ Worker shutdown cleanup completed")
    except Exception as e:
        logger.error(f"❌ Worker shutdown cleanup failed: {e}")


if __name__ == "__main__":
    # For running Celery worker directly
    celery_app.start()

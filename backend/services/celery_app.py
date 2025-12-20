"""
Celery Application Configuration
Background task processing for the "Big Stick" Orchestrator
"""

import os
import logging
from celery import Celery
from celery.signals import worker_ready, worker_shutdown
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
        "services.celery_tasks.orchestrator_tasks",
        "services.celery_tasks.agent_tasks",
        "services.celery_tasks.rss_tasks",
        "services.celery_tasks.entertainment_sync_tasks"
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
        "services.celery_tasks.orchestrator_tasks.*": {"queue": "orchestrator"},
        "services.celery_tasks.agent_tasks.*": {"queue": "agents"},
        "services.celery_tasks.rss_tasks.*": {"queue": "rss"},
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
        # Purge old synthesized news articles daily
        'purge-old-news-articles': {
            'task': 'services.celery_tasks.rss_tasks.purge_old_news_task',
            'schedule': 86400.0,  # 24 hours in seconds
        },
        # Entertainment content sync - run every hour
        'sync-entertainment-content': {
            'task': 'services.celery_tasks.entertainment_sync_tasks.scheduled_entertainment_sync',
            'schedule': 3600.0,  # 1 hour in seconds
        },
    },
    # Beat scheduler settings
    beat_max_loop_interval=300,  # Check for new tasks every 5 minutes
)

# Worker lifecycle events
@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    """Called when Celery worker is ready to receive tasks"""
    # Set environment variable to indicate we're in a Celery worker
    os.environ['CELERY_WORKER_RUNNING'] = 'true'
    logger.info("🚀 CELERY WORKER: Ready to process orchestrator tasks")

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
@celery_app.task(bind=True, name="worker.warmup")
def warmup_worker_task(self):
    """Task to warm up worker on startup"""
    import asyncio
    from services.worker_warmup import worker_warmup_service
    
    logger.info("🔥 WORKER WARMUP TASK: Starting...")
    
    try:
        result = asyncio.run(worker_warmup_service.warmup_worker())
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
    
    # Close shared database pool
    try:
        import asyncio
        from utils.shared_db_pool import close_shared_db_pool
        
        # Run cleanup in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(close_shared_db_pool())
        finally:
            loop.close()
            
        logger.info("✅ Worker shutdown cleanup completed")
    except Exception as e:
        logger.error(f"❌ Worker shutdown cleanup failed: {e}")


if __name__ == "__main__":
    # For running Celery worker directly
    celery_app.start()

"""
Scheduled Agent Tasks - Celery tasks for Agent Factory schedule execution.

check_agent_schedules: Beat task that runs every 60s, finds due schedules,
  enqueues execute_scheduled_agent with overlap and per-user concurrency checks.
execute_scheduled_agent: Runs a single agent profile via gRPC orchestrator,
  logs to agent_execution_log, handles circuit breaker and Redis locks.
"""

import json
import logging
import os
import random
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from croniter import croniter

from services.celery_app import celery_app, TaskStatus
from services.celery_tasks.celery_error_handling import SOFT_TIME_LIMIT_EXCEEDED_TYPES
from services.celery_tasks.async_runner import run_async
from services.database_manager.celery_database_helpers import (
    celery_execute,
    celery_fetch_all,
    celery_fetch_one,
)

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
MAX_CONCURRENT_PER_USER = 3
LOCK_TTL_SECONDS = 330
USER_COUNTER_TTL_SECONDS = 600


def _get_redis():
    import redis
    return redis.Redis.from_url(REDIS_URL, decode_responses=True)


def _compute_next_run_at(
    schedule_type: str,
    cron_expression: Optional[str],
    interval_seconds: Optional[int],
    tz_str: str = "UTC",
    from_time: Optional[datetime] = None,
) -> Optional[datetime]:
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        ZoneInfo = None
    from_time = from_time or datetime.now(timezone.utc)
    tz = timezone.utc
    if tz_str and tz_str != "UTC" and ZoneInfo is not None:
        try:
            tz = ZoneInfo(tz_str)
        except Exception:
            pass
    if schedule_type == "cron" and cron_expression:
        try:
            now_in_tz = from_time.astimezone(tz)
            now_naive = now_in_tz.replace(tzinfo=None)
            it = croniter(cron_expression, now_naive)
            next_naive = it.get_next(datetime)
            next_in_tz = next_naive.replace(tzinfo=tz)
            next_utc = next_in_tz.astimezone(timezone.utc)
            return next_utc
        except Exception:
            return None
    if schedule_type == "interval" and interval_seconds:
        from datetime import timedelta
        return from_time + timedelta(seconds=interval_seconds)
    return None


async def _fetch_due_schedules() -> List[Dict[str, Any]]:
    query = """
        SELECT s.id, s.agent_profile_id, s.user_id, s.schedule_type,
               s.cron_expression, s.interval_seconds, s.timezone, s.timeout_seconds,
               s.max_consecutive_failures, s.input_context
        FROM agent_schedules s
        JOIN agent_profiles p ON p.id = s.agent_profile_id AND p.is_active = true
        WHERE s.is_active = true AND s.next_run_at IS NOT NULL AND s.next_run_at <= NOW()
        ORDER BY s.next_run_at ASC
    """
    rows = await celery_fetch_all(query)
    return rows


async def _update_schedule_next_run(schedule_id: str, next_run_at: Optional[datetime]) -> None:
    if next_run_at is None:
        await celery_execute(
            "UPDATE agent_schedules SET next_run_at = NULL, updated_at = NOW() WHERE id = $1",
            uuid.UUID(schedule_id),
        )
    else:
        await celery_execute(
            "UPDATE agent_schedules SET next_run_at = $1, updated_at = NOW() WHERE id = $2",
            next_run_at,
            uuid.UUID(schedule_id),
        )


async def _insert_execution_log(
    agent_profile_id: str,
    user_id: str,
    schedule_id: str,
    query: str,
    playbook_id: Optional[str],
) -> Optional[str]:
    row = await celery_fetch_one(
        """
        INSERT INTO agent_execution_log (
            agent_profile_id, user_id, query, trigger_type, schedule_id, playbook_id, status
        )
        VALUES ($1, $2, $3, 'scheduled', $4, $5, 'running')
        RETURNING id
        """,
        uuid.UUID(agent_profile_id),
        user_id,
        query,
        uuid.UUID(schedule_id),
        uuid.UUID(playbook_id) if playbook_id else None,
    )
    return str(row["id"]) if row else None


async def _update_execution_log_complete(
    execution_id: str,
    status: str,
    error_details: Optional[str] = None,
    duration_ms: Optional[int] = None,
) -> None:
    await celery_execute(
        """
        UPDATE agent_execution_log
        SET status = $1, completed_at = NOW(), error_details = $2, duration_ms = $3
        WHERE id = $4
        """,
        status,
        error_details,
        duration_ms,
        uuid.UUID(execution_id),
    )


async def _update_schedule_after_run(
    schedule_id: str,
    last_status: str,
    consecutive_failures: int,
    is_active: bool,
    next_run_at: Optional[datetime],
) -> None:
    await celery_execute(
        """
        UPDATE agent_schedules
        SET last_run_at = NOW(), last_status = $1, run_count = run_count + 1,
            consecutive_failures = $2, is_active = $3, next_run_at = $4, updated_at = NOW()
        WHERE id = $5
        """,
        last_status,
        consecutive_failures,
        is_active,
        next_run_at,
        uuid.UUID(schedule_id),
    )


async def _check_agent_budget(agent_profile_id: str) -> tuple[bool, bool]:
    """
    Check if agent is within budget. Returns (allowed, over_limit).
    If budget row has current_period_start in a past month, resets period and allows run.
    """
    from datetime import date
    today = date.today()
    period_start = today.replace(day=1)
    row = await celery_fetch_one(
        "SELECT monthly_limit_usd, current_period_start, current_period_spend_usd, enforce_hard_limit FROM agent_budgets WHERE agent_profile_id = $1",
        uuid.UUID(agent_profile_id),
    )
    if not row or row.get("monthly_limit_usd") is None:
        return True, False
    limit = float(row["monthly_limit_usd"])
    spend = float(row.get("current_period_spend_usd") or 0)
    existing_start = row.get("current_period_start")
    if existing_start and (getattr(existing_start, "year", None) != today.year or getattr(existing_start, "month", None) != today.month):
        spend = 0
    enforce = row.get("enforce_hard_limit", True)
    if enforce and limit > 0 and spend >= limit:
        return False, True
    return True, False


async def _pause_schedule_and_notify_budget(
    schedule_id: str, user_id: str, agent_name: str, agent_profile_id: Optional[str] = None
) -> None:
    """Pause schedule and send WebSocket notification (budget exceeded)."""
    await celery_execute(
        "UPDATE agent_schedules SET is_active = false, updated_at = NOW() WHERE id = $1",
        uuid.UUID(schedule_id),
    )
    base = os.getenv("BACKEND_URL", "http://backend:8000")
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{base}/api/agent-factory/internal/notify-schedule-paused",
                json={
                    "user_id": user_id,
                    "agent_name": agent_name or "Scheduled Agent",
                    "consecutive": 0,
                    "last_error": "Monthly budget limit reached. Schedule paused.",
                },
            )
            await client.post(
                f"{base}/api/agent-factory/internal/notify-execution-event",
                json={
                    "user_id": user_id,
                    "subtype": "budget_exceeded",
                    "agent_profile_id": agent_profile_id,
                    "agent_name": agent_name or "Scheduled Agent",
                },
            )
    except Exception as e:
        logger.warning("Notify budget paused failed: %s", e)


@celery_app.task(bind=True, name="services.celery_tasks.scheduled_agent_tasks.check_agent_schedules")
def check_agent_schedules(self) -> Dict[str, Any]:
    """Beat task: find due schedules, enforce concurrency, enqueue execute_scheduled_agent."""
    try:
        due = run_async(_fetch_due_schedules())
        if not due:
            return {"enqueued": 0, "skipped": 0}

        redis_client = _get_redis()
        enqueued = 0
        skipped = 0

        for row in due:
            schedule_id = str(row["id"])
            profile_id = str(row["agent_profile_id"])
            user_id = row["user_id"]

            lock_key = f"sched:agent:{profile_id}:lock"
            if redis_client.get(lock_key):
                skipped += 1
                continue

            user_key = f"sched:user:{user_id}:running"
            try:
                current = redis_client.incr(user_key)
                if current == 1:
                    redis_client.expire(user_key, USER_COUNTER_TTL_SECONDS)
                if current > MAX_CONCURRENT_PER_USER:
                    redis_client.decr(user_key)
                    skipped += 1
                    continue
            except Exception:
                skipped += 1
                continue

            next_run = _compute_next_run_at(
                row["schedule_type"],
                row.get("cron_expression"),
                row.get("interval_seconds"),
                row.get("timezone") or "UTC",
            )
            run_async(_update_schedule_next_run(schedule_id, next_run))

            delay = random.randint(0, 10)
            execute_scheduled_agent.apply_async(
                args=[schedule_id, profile_id, user_id],
                countdown=delay,
            )
            enqueued += 1

        return {"enqueued": enqueued, "skipped": skipped}
    except Exception as e:
        logger.exception("check_agent_schedules failed: %s", e)
        return {"enqueued": 0, "skipped": 0, "error": str(e)}


@celery_app.task(
    bind=True,
    name="services.celery_tasks.scheduled_agent_tasks.execute_scheduled_agent",
    soft_time_limit=300,
)
def execute_scheduled_agent(
    self,
    schedule_id: str,
    agent_profile_id: str,
    user_id: str,
) -> Dict[str, Any]:
    """Execute one scheduled agent run via gRPC orchestrator; update log and schedule."""
    redis_client = _get_redis()
    user_key = f"sched:user:{user_id}:running"
    lock_key = f"sched:agent:{agent_profile_id}:lock"

    try:
        schedule_row = run_async(_get_schedule(schedule_id))
        if not schedule_row or not schedule_row.get("is_active"):
            return {"success": False, "error": "Schedule not found or inactive"}
        if not schedule_row.get("profile_is_active", True):
            return {"success": False, "error": "Agent profile is paused"}

        allowed, over_limit = run_async(_check_agent_budget(agent_profile_id))
        if not allowed and over_limit:
            run_async(_pause_schedule_and_notify_budget(
                schedule_id,
                user_id,
                schedule_row.get("agent_name"),
                agent_profile_id,
            ))
            return {"success": False, "error": "Monthly budget limit reached; schedule paused"}

        timeout_sec = schedule_row.get("timeout_seconds") or 300
        self.request.soft_time_limit = timeout_sec

        raw_ctx = schedule_row.get("input_context")
        if isinstance(raw_ctx, str):
            try:
                input_context = json.loads(raw_ctx) if raw_ctx else {}
            except (json.JSONDecodeError, TypeError):
                input_context = {}
        elif isinstance(raw_ctx, dict):
            input_context = raw_ctx
        else:
            input_context = {}
        query = input_context.get("query") or "Run scheduled playbook."

        execution_id = run_async(
            _insert_execution_log(
                agent_profile_id,
                user_id,
                schedule_id,
                query,
                None,
            )
        )
        if not execution_id:
            return {"success": False, "error": "Failed to create execution log"}

        async def _notify_started():
            try:
                import httpx
                base = os.getenv("BACKEND_URL", "http://backend:8000")
                async with httpx.AsyncClient() as client:
                    await client.post(
                        f"{base}/api/agent-factory/internal/notify-execution-event",
                        json={
                            "user_id": user_id,
                            "subtype": "execution_started",
                            "execution_id": execution_id,
                            "agent_profile_id": agent_profile_id,
                            "agent_name": schedule_row.get("agent_name"),
                            "status": "running",
                            "trigger_type": "scheduled",
                            "query": (query or "")[:200],
                        },
                    )
            except Exception as notify_err:
                logger.debug("notify execution_started failed: %s", notify_err)

        run_async(_notify_started())

        if not redis_client.set(lock_key, "1", nx=True, ex=LOCK_TTL_SECONDS):
            run_async(
                _update_execution_log_complete(
                    execution_id,
                    "failed",
                    "Overlap: agent already running",
                )
            )
            return {"success": False, "error": "Overlap"}

        started = datetime.now(timezone.utc)
        try:
            from services.celery_tasks.agent_tasks import _call_grpc_orchestrator_custom_agent
            result = run_async(
                _call_grpc_orchestrator_custom_agent(
                    agent_profile_id=agent_profile_id,
                    query=query,
                    user_id=user_id,
                    conversation_id="",
                    extra_context={
                        "execution_id": execution_id,
                        "trigger_type": "scheduled",
                    },
                )
            )
        except SOFT_TIME_LIMIT_EXCEEDED_TYPES:
            result = {"success": False, "error": "Task time limit exceeded"}
        finally:
            redis_client.delete(lock_key)
            try:
                redis_client.decr(user_key)
            except Exception:
                pass

        duration_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
        approval_parked = result.get("task_status") == "approval_parked"
        success = result.get("success", False) or approval_parked
        error_details = result.get("error")
        if approval_parked:
            last_status = "running"
        else:
            last_status = "success" if success else "failed"

        if not approval_parked:
            run_async(
                _update_execution_log_complete(
                    execution_id,
                    last_status,
                    error_details=error_details,
                    duration_ms=duration_ms,
                )
            )

        schedule_row_after = run_async(_get_schedule(schedule_id))
        if not schedule_row_after:
            return {"success": success, "execution_id": execution_id}

        consecutive = schedule_row_after.get("consecutive_failures", 0)
        if success:
            consecutive = 0
        else:
            consecutive += 1

        max_failures = schedule_row_after.get("max_consecutive_failures") or 5
        is_active = consecutive < max_failures
        next_run = _compute_next_run_at(
            schedule_row_after["schedule_type"],
            schedule_row_after.get("cron_expression"),
            schedule_row_after.get("interval_seconds"),
            schedule_row_after.get("timezone") or "UTC",
            from_time=datetime.now(timezone.utc),
        )

        run_async(
            _update_schedule_after_run(
                schedule_id,
                last_status,
                consecutive,
                is_active,
                next_run,
            )
        )

        if not is_active and consecutive >= max_failures:
            _notify_schedule_paused(user_id, schedule_row_after.get("agent_name"), consecutive, error_details)

        return {"success": success, "execution_id": execution_id}
    except Exception as e:
        logger.exception("execute_scheduled_agent failed: %s", e)
        try:
            redis_client.delete(lock_key)
            redis_client.decr(user_key)
        except Exception:
            pass
        return {"success": False, "error": str(e)}


async def _get_schedule(schedule_id: str) -> Optional[Dict[str, Any]]:
    row = await celery_fetch_one(
        """
        SELECT s.id, s.agent_profile_id, s.user_id, s.is_active, s.timeout_seconds,
               s.consecutive_failures, s.max_consecutive_failures, s.schedule_type,
               s.cron_expression, s.interval_seconds, s.timezone, s.input_context,
               p.name AS agent_name, p.is_active AS profile_is_active
        FROM agent_schedules s
        JOIN agent_profiles p ON p.id = s.agent_profile_id
        WHERE s.id = $1
        """,
        uuid.UUID(schedule_id),
    )
    return row


def _notify_schedule_paused(
    user_id: str,
    agent_name: Optional[str],
    consecutive: int,
    last_error: Optional[str],
) -> None:
    """Notify user that schedule was auto-paused (circuit breaker). Logs and sends WebSocket via internal API."""
    logger.warning(
        "Schedule paused for user=%s agent=%s after %s failures: %s",
        user_id,
        agent_name or "Scheduled Agent",
        consecutive,
        last_error or "Unknown",
    )
    base_url = os.getenv("BACKEND_INTERNAL_URL", "http://backend:8000").rstrip("/")
    try:
        import urllib.request
        import urllib.error
        payload = json.dumps({
            "user_id": user_id,
            "agent_name": agent_name,
            "consecutive": consecutive,
            "last_error": last_error,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/api/agent-factory/internal/notify-schedule-paused",
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status != 200:
                logger.warning("notify_schedule_paused API returned %s", resp.status)
    except Exception as e:
        logger.warning("notify_schedule_paused API call failed: %s", e)

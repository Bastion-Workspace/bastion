"""
Episode Celery tasks: tiered retention (mark aged, graduate facts + purge), legacy per-turn task stub.
"""

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from services.celery_app import celery_app, TaskStatus
from services.celery_tasks.celery_error_handling import SOFT_TIME_LIMIT_EXCEEDED_TYPES
from services.celery_tasks.async_runner import run_async

logger = logging.getLogger(__name__)

EPISODE_TYPES = "chat, research, editing, coding, automation, file_management, general"


@celery_app.task(
    bind=True,
    name="services.celery_tasks.episode_tasks.extract_episode_task",
    soft_time_limit=120,
)
def extract_episode_task(
    self,
    user_id: str,
    conversation_id: Optional[str],
    query: str,
    accumulated_response: str,
    agent_name_used: Optional[str],
    metadata_received: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Deprecated: per-turn episode extraction replaced by post_session_analysis_task.
    Kept so in-flight Celery messages after deploy do not crash the worker.
    """
    logger.warning(
        "extract_episode_task is deprecated and no-op; use post_session_analysis_task (user_id=%s conv=%s)",
        user_id,
        conversation_id,
    )
    return {
        "success": False,
        "deprecated": True,
        "task_id": self.request.id,
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "conversation_id": conversation_id,
        "episode_id": None,
    }


async def _mark_episodes_aged() -> int:
    from services.episode_service import episode_service

    cutoff = datetime.now(timezone.utc) - timedelta(hours=48)
    return await episode_service.mark_episodes_aged_before(cutoff)


@celery_app.task(bind=True, name="services.celery_tasks.episode_tasks.mark_episodes_aged_task")
def mark_episodes_aged_task(self) -> Dict[str, Any]:
    """Mark episodes older than 48h as is_aged for deprioritized retrieval."""
    try:
        updated = run_async(_mark_episodes_aged())
        if updated > 0:
            logger.info("mark_episodes_aged_task: marked %s episode(s) aged", updated)
        return {
            "success": True,
            "task_id": self.request.id,
            "timestamp": datetime.now().isoformat(),
            "updated_count": updated,
        }
    except Exception as e:
        logger.exception("mark_episodes_aged_task failed: %s", e)
        return {"success": False, "error": str(e), "updated_count": 0}


async def _extract_facts_from_episode_summary(user_id: str, summary: str) -> int:
    """Lightweight fact extraction from a single episode summary before purge."""
    from config import settings
    from services.settings_service import settings_service
    from services.user_settings_kv_service import get_user_setting
    from utils.openrouter_client import get_openrouter_client

    summary = (summary or "").strip()
    if len(summary) < 40:
        return 0

    if not await settings_service.get_facts_write_enabled(user_id):
        return 0

    existing = await settings_service.get_user_facts(user_id)
    existing_text = settings_service.format_user_facts_for_prompt(existing)
    if not existing_text.strip():
        existing_text = "(none)"

    prompt = f"""From this past activity summary only, extract DURABLE USER FACTS — things that define who the user IS or what they ALWAYS prefer across unrelated future chats (identity, lasting preferences, recurring tools/workflows).

NEVER extract: debugging/troubleshooting details, one-off task steps, project progress, specific file/server/pod names, incident findings, or anything irrelevant to a completely different topic tomorrow.

KEY DISCIPLINE: If an existing fact covers the same topic, use the SAME fact_key with a refined value rather than inventing a new key. Most summaries produce ZERO facts.

EXISTING FACTS (do not duplicate):
{existing_text}

ACTIVITY SUMMARY:
{summary[:2000]}

JSON only:
{{"facts": [{{"fact_key": "snake_case", "value": "text", "category": "preferences|work|personal|general"}}]}}
If none: {{"facts": []}}"""

    try:
        fast_model = (
            await get_user_setting(user_id, "user_fast_model")
            or await settings_service.get_classification_model()
        )
        model = fast_model or getattr(settings, "FAST_MODEL", "anthropic/claude-3.5-haiku")
        client = get_openrouter_client()
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.2,
        )
        content = (resp.choices[0].message.content or "").strip()
        if content.startswith("```"):
            for marker in ("```json", "```"):
                if content.startswith(marker):
                    content = content[len(marker) :].strip()
                    break
            if content.endswith("```"):
                content = content[:-3].strip()
        m = re.search(r"\{[\s\S]*\}", content)
        if m:
            content = m.group(0)
        data = json.loads(content)
        raw = data.get("facts")
        if not isinstance(raw, list):
            return 0
        extracted = 0
        for item in raw[:2]:
            fact_key = (item.get("fact_key") or "").strip()
            value = (item.get("value") or "").strip()
            category = (item.get("category") or "general").strip().lower()
            if category not in ("preferences", "work", "personal", "general"):
                category = "general"
            if not fact_key or not value:
                continue
            if not re.match(r"^[a-z][a-z0-9_]*$", fact_key):
                fact_key = re.sub(r"[^a-z0-9]+", "_", fact_key.lower()).strip("_") or "preference"
            result = await settings_service.upsert_user_fact(
                user_id=user_id,
                fact_key=fact_key,
                value=value,
                category=category,
                source="auto_extract",
            )
            if result.get("success"):
                extracted += 1
        return extracted
    except Exception as e:
        logger.warning("graduate episode fact extract failed: %s", e)
        return 0


async def _graduate_and_purge() -> Dict[str, int]:
    from services.episode_service import episode_service

    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    rows = await episode_service.fetch_episodes_for_graduation(cutoff, limit=30)
    facts_total = 0
    deleted = 0
    for row in rows:
        uid = str(row["user_id"])
        facts_total += await _extract_facts_from_episode_summary(uid, row.get("summary") or "")
        if await episode_service.delete_episode_by_id(int(row["id"])):
            deleted += 1
    return {"deleted": deleted, "facts_extracted": facts_total}


@celery_app.task(
    bind=True,
    name="services.celery_tasks.episode_tasks.graduate_and_purge_old_episodes_task",
    soft_time_limit=600,
)
def graduate_and_purge_old_episodes_task(self) -> Dict[str, Any]:
    """Episodes older than 7 days: extract durable facts, then delete rows."""
    try:
        stats = run_async(_graduate_and_purge())
        if stats.get("deleted", 0) > 0:
            logger.info(
                "graduate_and_purge_old_episodes_task: deleted %s episode(s), facts=%s",
                stats.get("deleted"),
                stats.get("facts_extracted"),
            )
        return {
            "success": True,
            "task_id": self.request.id,
            "timestamp": datetime.now().isoformat(),
            **stats,
        }
    except Exception as e:
        logger.exception("graduate_and_purge_old_episodes_task failed: %s", e)
        self.update_state(
            state=TaskStatus.FAILURE,
            meta={"error": str(e), "timestamp": datetime.now().isoformat()},
        )
        return {"success": False, "error": str(e), "deleted": 0, "facts_extracted": 0}


@celery_app.task(bind=True, name="services.celery_tasks.episode_tasks.purge_old_episodes_task")
def purge_old_episodes_task(self, hours_retention: int = 48) -> Dict[str, Any]:
    """
    Deprecated: replaced by mark_episodes_aged_task + graduate_and_purge_old_episodes_task.
    """
    logger.warning(
        "purge_old_episodes_task is deprecated (hours_retention=%s); no longer deletes at 48h",
        hours_retention,
    )
    return {
        "success": True,
        "deprecated": True,
        "task_id": self.request.id,
        "timestamp": datetime.now().isoformat(),
        "deleted_count": 0,
    }

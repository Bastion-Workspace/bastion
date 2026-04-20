"""
Post-session memory: one LLM call produces a session episode summary and durable user facts.
Triggered after conversation idle, on conversation delete, on new-conversation boundary, or via beat.
"""

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from services.celery_app import celery_app, TaskStatus
from services.celery_tasks.async_runner import run_async

logger = logging.getLogger(__name__)

SESSION_MESSAGE_LIMIT = 40
IDLE_MINUTES = 15
MAX_FACTS_PER_SESSION = 3
EPISODE_TYPES = "chat, research, editing, coding, automation, file_management, general"


def _strip_code_fence(content: str) -> str:
    content = (content or "").strip()
    if content.startswith("```"):
        for marker in ("```json", "```"):
            if content.startswith(marker):
                content = content[len(marker) :].strip()
                break
        if content.endswith("```"):
            content = content[:-3].strip()
    return content


async def _clear_session_summary_flag(conversation_id: str) -> None:
    from services.database_manager.database_helpers import execute

    await execute(
        "UPDATE conversations SET needs_session_summary = FALSE WHERE conversation_id = $1",
        conversation_id,
    )


async def _run_post_session_analysis(
    user_id: str,
    conversation_id: str,
    force: bool = False,
    transcript: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load recent messages (or use preloaded transcript), single LLM call for session summary + facts.
    When transcript is set (e.g. conversation delete), skips DB message load and does not update conversations row.
    """
    from config import settings
    from services.conversation_service import ConversationService
    from services.database_manager.database_helpers import fetch_one
    from services.episode_service import episode_service
    from services.embedding_service_wrapper import get_embedding_service
    from services.settings_service import settings_service
    from services.user_settings_kv_service import get_user_setting
    from utils.openrouter_client import get_openrouter_client

    if not conversation_id or not user_id:
        return {"ok": False, "skipped": "missing_ids"}

    transcript_final: Optional[str] = None
    skip_flag_update = bool(transcript)

    if transcript:
        transcript_final = transcript.strip()
    else:
        conv_row = await fetch_one(
            "SELECT needs_session_summary FROM conversations WHERE conversation_id = $1 AND user_id = $2",
            conversation_id,
            user_id,
        )
        if not conv_row:
            return {"ok": False, "skipped": "conversation_not_found"}

        if not force and not conv_row.get("needs_session_summary"):
            return {"ok": True, "skipped": "not_dirty"}

        cs = ConversationService()
        cs.set_current_user(user_id)
        msgs_result = await cs.get_conversation_messages(
            conversation_id,
            user_id,
            most_recent=True,
            limit=SESSION_MESSAGE_LIMIT,
        )
        messages: List[Dict[str, Any]] = msgs_result.get("messages") or []
        if not messages:
            await _clear_session_summary_flag(conversation_id)
            return {"ok": True, "skipped": "no_messages"}

        has_assistant = any(m.get("message_type") == "assistant" for m in messages)
        if not has_assistant:
            await _clear_session_summary_flag(conversation_id)
            return {"ok": True, "skipped": "no_assistant_messages"}

        lines: List[str] = []
        for m in messages:
            role = m.get("message_type") or "unknown"
            content = (m.get("content") or "").strip()
            if not content:
                continue
            prefix = "USER" if role == "user" else "ASSISTANT" if role == "assistant" else role.upper()
            lines.append(f"{prefix}: {content[:4000]}")

        transcript_final = "\n\n".join(lines)

    if not transcript_final or len(transcript_final) < 80:
        if not skip_flag_update:
            await _clear_session_summary_flag(conversation_id)
        return {"ok": True, "skipped": "short_transcript"}

    existing_facts = await settings_service.get_user_facts(user_id)
    existing_text = settings_service.format_user_facts_for_prompt(existing_facts)
    if not existing_text.strip():
        existing_text = "(none)"

    episodes_write = await settings_service.get_episodes_inject_enabled(user_id)
    facts_write = await settings_service.get_facts_write_enabled(user_id)

    if not episodes_write and not facts_write:
        if not skip_flag_update:
            await _clear_session_summary_flag(conversation_id)
        return {"ok": True, "skipped": "memory_writes_disabled"}

    prompt = f"""Analyze this chat session (multiple turns). Produce:
1) A concise SESSION SUMMARY (1-3 sentences): what the user wanted and what was accomplished overall.
2) Durable USER FACTS only — things that define who the user IS or what they ALWAYS prefer across unrelated future conversations.
3) episode_type, key_topics, outcome for the session.

EXISTING FACTS (do not duplicate; refine only if clearly contradicted):
{existing_text}

TRANSCRIPT:
{transcript_final[:24000]}

DURABLE FACTS (extract these):
- Stable identity/profile: job title, employer, location, timezone, dietary needs
- Lasting preferences: preferred programming language, writing style, communication style
- Recurring tools/workflows: \"uses Blender for 3D\", \"prefers org-mode for task management\"

NOT DURABLE FACTS (never extract these):
- Details of today's debugging/troubleshooting session (node names, error messages, commands run, stack traces)
- One-off technical tasks (migration steps, deployment specifics, a particular PR or branch)
- Current project state or progress (\"chapter 5 is drafted\", \"PR #42 is open\", \"deployed v2.3 today\")
- Names of specific files, servers, pods, resources, or hosts being worked on right now
- Incident-specific findings (iptables rule conflicts, container runtime errors on a specific node)
- Anything that would be irrelevant if the user started a completely different topic tomorrow

KEY DISCIPLINE:
- If an existing fact covers the same topic, return that SAME fact_key with a refined value rather than inventing a new key.
- Prefer broad keys (e.g. \"infrastructure_preferences\") over narrow variants (\"infrastructure_iptables_issue_detailed\").
- Most sessions produce ZERO durable facts. Return \"facts\": [] unless something clearly permanent was revealed.

Respond with ONLY valid JSON (no markdown):
{{
  "session_summary": "1-3 sentences",
  "episode_type": "one of: {EPISODE_TYPES}",
  "key_topics": ["topic1", "topic2"],
  "outcome": "completed|in_progress|abandoned",
  "facts": [
    {{
      "fact_key": "snake_case_key",
      "value": "concise value",
      "category": "preferences|work|personal|general",
      "reasoning": "why durable across conversations"
    }}
  ]
}}

If no durable facts, use "facts": []. If episodes should not be stored, you would still return session_summary — the caller decides storage."""

    fast_model = (
        await get_user_setting(user_id, "user_fast_model")
        or await settings_service.get_classification_model()
    )
    model = fast_model or getattr(settings, "FAST_MODEL", "anthropic/claude-3.5-haiku")
    use_admin = (await get_user_setting(user_id, "use_admin_models")) or "true"
    client = get_openrouter_client()
    if use_admin.lower() == "false" and model:
        from services.user_llm_provider_service import user_llm_provider_service

        ctx = await user_llm_provider_service.get_llm_context_for_model(user_id, model)
        if ctx:
            client = get_openrouter_client(api_key=ctx["api_key"], base_url=ctx["base_url"])
            model = ctx.get("real_model_id", model)

    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=768,
            temperature=0.2,
        )
        content = _strip_code_fence((resp.choices[0].message.content or "").strip())
        if not content:
            return {"ok": False, "skipped": "empty_llm"}

        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            content = json_match.group(0)
        data = json.loads(content)
    except Exception as e:
        logger.warning("post_session_analysis LLM/parse failed: %s", e)
        return {"ok": False, "skipped": "parse_error", "error": str(e)}

    session_summary = (data.get("session_summary") or "").strip()
    episode_type = (data.get("episode_type") or "general").strip().lower()
    if episode_type not in tuple(EPISODE_TYPES.split(", ")):
        episode_type = "general"
    key_topics = data.get("key_topics")
    if not isinstance(key_topics, list):
        key_topics = []
    key_topics = [str(t) for t in key_topics[:5]]
    outcome = (data.get("outcome") or "completed").strip().lower()[:50]
    raw_facts = data.get("facts")
    if not isinstance(raw_facts, list):
        raw_facts = []
    raw_facts = raw_facts[:MAX_FACTS_PER_SESSION]

    episode_id = None
    if episodes_write and session_summary:
        episode_id = await episode_service.create_episode(
            user_id=user_id,
            summary=session_summary,
            episode_type=episode_type,
            conversation_id=conversation_id,
            agent_used="session_summary",
            tools_used=None,
            key_topics=key_topics or None,
            outcome=outcome,
            is_aged=False,
        )
        if episode_id:
            try:
                emb_svc = await get_embedding_service()
                vectors = await emb_svc.generate_embeddings([session_summary])
                if vectors and len(vectors) > 0:
                    await episode_service.update_episode_embedding(episode_id, vectors[0])
            except Exception as emb_e:
                logger.warning("Session episode embedding failed: %s", emb_e)

    facts_extracted = 0
    if facts_write:
        for item in raw_facts:
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
                facts_extracted += 1

    if not skip_flag_update:
        await _clear_session_summary_flag(conversation_id)
    return {
        "ok": True,
        "episode_id": episode_id,
        "facts_extracted": facts_extracted,
        "skipped": None,
    }


async def _find_idle_conversations(limit: int = 30) -> List[Dict[str, str]]:
    from services.database_manager.database_helpers import fetch_all

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=IDLE_MINUTES)
    rows = await fetch_all(
        """
        SELECT c.conversation_id, c.user_id
        FROM conversations c
        WHERE c.needs_session_summary = TRUE
          AND c.updated_at < $1
          AND EXISTS (
            SELECT 1 FROM conversation_messages cm
            WHERE cm.conversation_id = c.conversation_id
              AND cm.message_type = 'assistant'
              AND COALESCE(cm.is_deleted, FALSE) = FALSE
          )
        ORDER BY c.updated_at ASC
        LIMIT $2
        """,
        cutoff,
        limit,
    )
    return [{"conversation_id": str(r["conversation_id"]), "user_id": str(r["user_id"])} for r in (rows or [])]


@celery_app.task(
    bind=True,
    name="services.celery_tasks.session_analysis_task.post_session_analysis_task",
    soft_time_limit=180,
)
def post_session_analysis_task(
    self,
    user_id: str,
    conversation_id: str,
    force: bool = False,
    transcript: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        result = run_async(
            _run_post_session_analysis(
                user_id, conversation_id, force=force, transcript=transcript
            )
        )
        return {
            "success": result.get("ok", False),
            "task_id": self.request.id,
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "conversation_id": conversation_id,
            **{k: v for k, v in result.items() if k != "ok"},
        }
    except Exception as e:
        logger.exception("post_session_analysis_task failed: %s", e)
        self.update_state(
            state=TaskStatus.FAILURE,
            meta={"error": str(e), "timestamp": datetime.now().isoformat()},
        )
        return {"success": False, "error": str(e), "user_id": user_id, "conversation_id": conversation_id}


@celery_app.task(bind=True, name="services.celery_tasks.session_analysis_task.detect_idle_sessions_task")
def detect_idle_sessions_task(self) -> Dict[str, Any]:
    """Beat: enqueue post_session_analysis for conversations idle long enough."""
    try:
        rows = run_async(_find_idle_conversations(limit=40))
        enqueued = 0
        for row in rows:
            post_session_analysis_task.delay(
                user_id=row["user_id"],
                conversation_id=row["conversation_id"],
                force=False,
            )
            enqueued += 1
        return {
            "success": True,
            "task_id": self.request.id,
            "timestamp": datetime.now().isoformat(),
            "enqueued": enqueued,
        }
    except Exception as e:
        logger.exception("detect_idle_sessions_task failed: %s", e)
        return {"success": False, "error": str(e), "enqueued": 0}


async def _enqueue_pending_summaries_for_user_except(
    user_id: str,
    exclude_conversation_id: Optional[str],
    limit: int = 10,
) -> int:
    from services.database_manager.database_helpers import fetch_all

    if exclude_conversation_id:
        rows = await fetch_all(
            """
            SELECT conversation_id FROM conversations
            WHERE user_id = $1 AND needs_session_summary = TRUE
              AND conversation_id <> $2
            ORDER BY updated_at DESC
            LIMIT $3
            """,
            user_id,
            exclude_conversation_id,
            limit,
        )
    else:
        rows = await fetch_all(
            """
            SELECT conversation_id FROM conversations
            WHERE user_id = $1 AND needs_session_summary = TRUE
            ORDER BY updated_at DESC
            LIMIT $2
            """,
            user_id,
            limit,
        )
    n = 0
    for r in rows or []:
        cid = str(r["conversation_id"])
        post_session_analysis_task.delay(user_id=user_id, conversation_id=cid, force=False)
        n += 1
    return n



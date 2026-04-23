"""
Agent Factory service - shared CRUD and validation for profiles, playbooks, schedules, data sources.

Used by both the REST API (agent_factory_api.py) and gRPC tool handlers (grpc_tool_service.py).
"""

import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

import asyncpg

logger = logging.getLogger(__name__)

VALID_USER_FACTS_POLICY_VALUES = frozenset({"inherit", "no_write", "isolated"})

VALID_STEP_TYPES = frozenset(
    {
        "tool",
        "llm_task",
        "llm_agent",
        "approval",
        "loop",
        "parallel",
        "branch",
        "deep_agent",
        "browser_authenticate",
    }
)


def _rls(user_id: Optional[str], role: str = "user") -> Dict[str, str]:
    """Build RLS context for database helpers. Use role='admin' only for operations that must bypass RLS."""
    return {"user_id": user_id if user_id is not None else "", "user_role": role}


def _ensure_json_obj(val: Any, fallback: Any = None) -> Any:
    if fallback is None:
        fallback = {}
    if val is None:
        return fallback
    if isinstance(val, (dict, list)):
        return val
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, (dict, list)):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    return fallback


def _row_to_profile(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not row:
        return {}
    ownership = row.get("ownership") or "owned"
    if row.get("is_builtin"):
        ownership = "builtin"
    out = {
        "id": str(row["id"]),
        "user_id": row["user_id"],
        "name": row["name"],
        "handle": row.get("handle"),
        "description": row.get("description"),
        "is_active": row.get("is_active", True),
        "model_preference": row.get("model_preference"),
        "model_source": row.get("model_source"),
        "model_provider_type": row.get("model_provider_type"),
        "max_research_rounds": row.get("max_research_rounds", 3),
        "system_prompt_additions": row.get("system_prompt_additions"),
        "knowledge_config": _ensure_json_obj(row.get("knowledge_config"), {}),
        "default_playbook_id": str(row["default_playbook_id"]) if row.get("default_playbook_id") else None,
        "journal_config": _ensure_json_obj(row.get("journal_config"), {}),
        "team_config": _ensure_json_obj(row.get("team_config"), {}),
        "watch_config": _ensure_json_obj(row.get("watch_config"), {}),
        "prompt_history_enabled": row.get("chat_history_enabled", False),
        "chat_history_lookback": row.get("chat_history_lookback", 10),
        "summary_threshold_tokens": row.get("summary_threshold_tokens", 5000),
        "summary_keep_messages": row.get("summary_keep_messages", 10),
        "persona_mode": row.get("persona_mode") or "none",
        "persona_id": str(row["persona_id"]) if row.get("persona_id") else None,
        "include_user_context": row.get("include_user_context", False),
        "include_datetime_context": row.get("include_datetime_context", True),
        "include_user_facts": row.get("include_user_facts", False),
        "include_facts_categories": _ensure_json_obj(row.get("include_facts_categories"), []),
        "use_themed_memory": row.get("use_themed_memory", True),
        "include_agent_memory": row.get("include_agent_memory", False),
        "auto_routable": row.get("auto_routable", False),
        "chat_visible": row.get("chat_visible", True),
        "category": row.get("category"),
        "data_workspace_config": _ensure_json_obj(row.get("data_workspace_config"), {}),
        "allowed_connections": _ensure_json_obj(row.get("allowed_connections"), []),
        "default_run_context": row.get("default_run_context") or "interactive",
        "default_approval_policy": row.get("default_approval_policy") or "require",
        "created_at": row.get("created_at").isoformat() if row.get("created_at") else None,
        "updated_at": row.get("updated_at").isoformat() if row.get("updated_at") else None,
        "is_locked": row.get("is_locked", False),
        "is_builtin": row.get("is_builtin", False),
        "ownership": ownership,
        "owner_username": row.get("owner_username"),
        "owner_display_name": row.get("owner_display_name"),
    }
    if "last_execution_status" in row:
        out["last_execution_status"] = row.get("last_execution_status")
    if "monthly_limit_usd" in row and row.get("monthly_limit_usd") is not None:
        out["budget"] = {
            "monthly_limit_usd": float(row["monthly_limit_usd"]),
            "current_period_start": row["current_period_start"].isoformat() if row.get("current_period_start") else None,
            "current_period_spend_usd": float(row.get("current_period_spend_usd") or 0),
            "warning_threshold_pct": row.get("warning_threshold_pct", 80),
            "enforce_hard_limit": row.get("enforce_hard_limit", True),
        }
    else:
        out["budget"] = None
    return out


def _is_empty_value(val: Any) -> bool:
    """True if value is missing or empty (None, "", [], {}). Used to decide when to copy from old step."""
    if val is None:
        return True
    if isinstance(val, str) and not val.strip():
        return True
    if isinstance(val, (list, dict)) and len(val) == 0:
        return True
    return False


# Step fields where [] is a deliberate save (clearing packs/tools/skills), not "omit for merge".
_MERGE_PRESERVE_EXPLICIT_EMPTY = frozenset(
    {"tool_packs", "available_tools", "skill_ids", "skills", "subagents"}
)


def ensure_step_ids(definition: Dict[str, Any]) -> None:
    """Assign _step_id (UUID) to every step dict missing one, including nested steps. Mutates in place."""
    if not isinstance(definition, dict):
        return
    steps = definition.get("steps")
    if not isinstance(steps, list):
        return
    for step in steps:
        if isinstance(step, dict):
            _ensure_step_ids_on_step(step)


def _ensure_step_ids_on_step(step: Dict[str, Any]) -> None:
    if not isinstance(step, dict):
        return
    sid = (step.get("_step_id") or "").strip()
    if not sid:
        step["_step_id"] = str(uuid.uuid4())
    for key in ("then_steps", "else_steps", "parallel_steps", "steps"):
        children = step.get(key)
        if isinstance(children, list):
            for child in children:
                if isinstance(child, dict):
                    _ensure_step_ids_on_step(child)


def _build_step_merge_lookups(
    old_steps: List[Any],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, int], Dict[str, Dict[str, Any]]]:
    """Index old step dicts by _step_id, output_key, and name (with duplicate-name count)."""
    old_by_step_id: Dict[str, Dict[str, Any]] = {}
    old_by_output_key: Dict[str, Dict[str, Any]] = {}
    old_name_counts: Dict[str, int] = {}
    old_by_name: Dict[str, Dict[str, Any]] = {}
    for old_step in old_steps:
        if not isinstance(old_step, dict):
            continue
        sid = (old_step.get("_step_id") or "").strip()
        ok = (old_step.get("output_key") or "").strip()
        n = (old_step.get("name") or "").strip()
        if sid:
            old_by_step_id[sid] = old_step
        if ok:
            old_by_output_key[ok] = old_step
        if n:
            old_name_counts[n] = old_name_counts.get(n, 0) + 1
            old_by_name[n] = old_step
    return old_by_step_id, old_by_output_key, old_name_counts, old_by_name


def _find_merge_partner_for_new_step(
    new_step: Dict[str, Any],
    *,
    old_by_step_id: Dict[str, Dict[str, Any]],
    old_by_output_key: Dict[str, Dict[str, Any]],
    old_name_counts: Dict[str, int],
    old_by_name: Dict[str, Dict[str, Any]],
    consumed: Set[int],
) -> Optional[Dict[str, Any]]:
    """Pick an unconsumed old step matching new_step by _step_id, then output_key, then unique name."""
    if not isinstance(new_step, dict):
        return None
    sid = (new_step.get("_step_id") or "").strip()
    ok = (new_step.get("output_key") or "").strip()
    n = (new_step.get("name") or "").strip()

    def _try(cand: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not cand or not isinstance(cand, dict):
            return None
        if id(cand) in consumed:
            return None
        return cand

    for cand in (
        old_by_step_id.get(sid) if sid else None,
        old_by_output_key.get(ok) if ok else None,
        old_by_name.get(n) if n and old_name_counts.get(n, 0) == 1 else None,
    ):
        taken = _try(cand)
        if taken is not None:
            return taken
    return None


def _merge_step_child_lists(old_children: List[Any], new_children: List[Any]) -> None:
    """
    Merge old nested step dicts into new list by identity (_step_id, output_key, unique name),
    with positional fallback and legacy tail append. Mutates new_children in place.
    """
    if not isinstance(old_children, list) or not isinstance(new_children, list):
        return
    old_by_step_id, old_by_output_key, old_name_counts, old_by_name = _build_step_merge_lookups(old_children)
    consumed: Set[int] = set()
    for i, new_child in enumerate(new_children):
        if not isinstance(new_child, dict):
            continue
        matched = _find_merge_partner_for_new_step(
            new_child,
            old_by_step_id=old_by_step_id,
            old_by_output_key=old_by_output_key,
            old_name_counts=old_name_counts,
            old_by_name=old_by_name,
            consumed=consumed,
        )
        if matched is None and i < len(old_children) and isinstance(old_children[i], dict):
            cand = old_children[i]
            if id(cand) not in consumed:
                matched = cand
        if matched is not None:
            consumed.add(id(matched))
            _merge_step(matched, new_child)
    for i in range(len(new_children), len(old_children)):
        old_child = old_children[i]
        if isinstance(old_child, dict) and id(old_child) not in consumed:
            new_children.append(old_child)
            consumed.add(id(old_child))


def _merge_phase(old_p: Dict[str, Any], new_p: Dict[str, Any]) -> None:
    """Copy keys from old phase into new when new is missing or empty. Mutates new_p."""
    if not isinstance(old_p, dict) or not isinstance(new_p, dict):
        return
    for k, v in old_p.items():
        if _is_empty_value(new_p.get(k)):
            new_p[k] = v


def _merge_step(old_s: Dict[str, Any], new_s: Dict[str, Any]) -> None:
    """
    Copy keys from old step into new when new is missing or empty. Mutates new_s.
    Recursively merges nested step lists (then_steps, else_steps, parallel_steps, steps) and phases.
    """
    if not isinstance(old_s, dict) or not isinstance(new_s, dict):
        return
    for k, v in old_s.items():
        if k in ("then_steps", "else_steps", "parallel_steps", "steps"):
            if isinstance(v, list) and isinstance(new_s.get(k), list):
                _merge_step_child_lists(v, new_s[k])
            elif _is_empty_value(new_s.get(k)) and isinstance(v, list):
                new_s[k] = list(v)
            continue
        if k == "phases":
            if isinstance(v, list) and isinstance(new_s.get(k), list):
                for i, old_phase in enumerate(v):
                    if i < len(new_s[k]) and isinstance(old_phase, dict) and isinstance(new_s[k][i], dict):
                        _merge_phase(old_phase, new_s[k][i])
                    elif i >= len(new_s[k]):
                        new_s[k].append(dict(old_phase))
            elif _is_empty_value(new_s.get(k)) and isinstance(v, list):
                new_s[k] = [dict(p) for p in v]
            continue
        if k in _MERGE_PRESERVE_EXPLICIT_EMPTY:
            # Client sent the key (including []): do not resurrect cleared tool packs / tools / skills.
            if k in new_s:
                continue
            new_s[k] = v
            continue
        if _is_empty_value(new_s.get(k)):
            new_s[k] = v


def merge_playbook_definition_steps(
    old_definition: Dict[str, Any], new_definition: Dict[str, Any]
) -> None:
    """
    Merge old playbook definition into new so that any field present in old but missing or empty
    in new is copied. Prevents agent/API updates from wiping user-set step fields (prompts,
    params, conditions, nested steps, phases, etc.). Mutates new_definition in place.
    """
    if not isinstance(old_definition, dict) or not isinstance(new_definition, dict):
        return
    ensure_step_ids(new_definition)
    if _is_empty_value(new_definition.get("run_context")) and old_definition.get("run_context"):
        new_definition["run_context"] = old_definition["run_context"]
    old_steps = old_definition.get("steps") or []
    new_steps = new_definition.get("steps") or []
    if not isinstance(new_steps, list):
        return

    old_by_step_id, old_by_output_key, old_name_counts, old_by_name = _build_step_merge_lookups(
        old_steps if isinstance(old_steps, list) else []
    )
    consumed: Set[int] = set()
    for new_step in new_steps:
        if not isinstance(new_step, dict):
            continue
        matched = _find_merge_partner_for_new_step(
            new_step,
            old_by_step_id=old_by_step_id,
            old_by_output_key=old_by_output_key,
            old_name_counts=old_name_counts,
            old_by_name=old_by_name,
            consumed=consumed,
        )
        if matched:
            consumed.add(id(matched))
            _merge_step(matched, new_step)


def _row_to_playbook(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not row:
        return {}
    ownership = row.get("ownership") or "owned"
    if row.get("is_builtin"):
        ownership = "builtin"
    elif row.get("is_template"):
        ownership = "template"
    return {
        "id": str(row["id"]),
        "user_id": row.get("user_id"),
        "name": row["name"],
        "description": row.get("description"),
        "version": row.get("version") or "1.0",
        "definition": _ensure_json_obj(row.get("definition"), {}),
        "triggers": _ensure_json_obj(row.get("triggers"), []),
        "is_template": row.get("is_template", False),
        "category": row.get("category"),
        "tags": list(row["tags"]) if row.get("tags") else [],
        "required_connectors": list(row["required_connectors"]) if row.get("required_connectors") else [],
        "created_at": row.get("created_at").isoformat() if row.get("created_at") else None,
        "updated_at": row.get("updated_at").isoformat() if row.get("updated_at") else None,
        "is_locked": row.get("is_locked", False),
        "is_builtin": row.get("is_builtin", False),
        "ownership": ownership,
        "owner_username": row.get("owner_username"),
        "owner_display_name": row.get("owner_display_name"),
    }


def _row_to_schedule(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not row:
        return {}
    return {
        "id": str(row["id"]),
        "agent_profile_id": str(row["agent_profile_id"]),
        "user_id": row.get("user_id"),
        "schedule_type": row.get("schedule_type"),
        "cron_expression": row.get("cron_expression"),
        "interval_seconds": row.get("interval_seconds"),
        "timezone": row.get("timezone") or "UTC",
        "is_active": row.get("is_active", True),
        "next_run_at": row["next_run_at"].isoformat() if row.get("next_run_at") else None,
        "last_run_at": row["last_run_at"].isoformat() if row.get("last_run_at") else None,
        "last_status": row.get("last_status"),
        "run_count": row.get("run_count", 0),
        "consecutive_failures": row.get("consecutive_failures", 0),
        "max_consecutive_failures": row.get("max_consecutive_failures", 5),
        "timeout_seconds": row.get("timeout_seconds", 300),
        "input_context": row.get("input_context") or {},
        "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
        "updated_at": row["updated_at"].isoformat() if row.get("updated_at") else None,
    }


def _row_to_data_source(
    row: Optional[Dict[str, Any]],
    connector_name: Optional[str] = None,
    connector_definition: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not row:
        return {}
    out = {
        "id": str(row["id"]),
        "agent_profile_id": str(row["agent_profile_id"]),
        "connector_id": str(row["connector_id"]),
        "credentials_encrypted": row.get("credentials_encrypted"),
        "config_overrides": row.get("config_overrides") or {},
        "permissions": row.get("permissions") or {},
        "is_enabled": row.get("is_enabled", True),
        "created_at": row.get("created_at").isoformat() if row.get("created_at") else None,
    }
    if connector_name is not None:
        out["connector_name"] = connector_name
    if connector_definition is not None:
        endpoints = connector_definition.get("endpoints") or {}
        out["connector_endpoints"] = list(endpoints.keys()) if isinstance(endpoints, dict) else []
    return out


def _compute_next_run_at(
    schedule_type: str,
    cron_expression: Optional[str],
    interval_seconds: Optional[int],
    timezone_str: str = "UTC",
) -> Optional[Any]:
    from datetime import datetime, timedelta, timezone
    try:
        from croniter import croniter
    except ImportError:
        return None
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        ZoneInfo = None
    tz = timezone.utc
    if timezone_str and timezone_str != "UTC" and ZoneInfo is not None:
        try:
            tz = ZoneInfo(timezone_str)
        except Exception:
            pass
    now = datetime.now(timezone.utc)
    if schedule_type == "cron" and cron_expression:
        try:
            now_in_tz = now.astimezone(tz)
            now_naive = now_in_tz.replace(tzinfo=None)
            it = croniter(cron_expression, now_naive)
            next_naive = it.get_next(datetime)
            next_in_tz = next_naive.replace(tzinfo=tz)
            next_utc = next_in_tz.astimezone(timezone.utc)
            return next_utc
        except Exception:
            return None
    if schedule_type == "interval" and interval_seconds:
        return now + timedelta(seconds=interval_seconds)
    return None


async def _sync_agent_line_watches(
    agent_profile_id: str,
    user_id: str,
    team_config: Dict[str, Any],
) -> None:
    from services.database_manager.database_helpers import execute
    from services.team_service import TeamService

    await execute(
        "DELETE FROM agent_line_watches WHERE agent_profile_id = $1",
        agent_profile_id,
    )
    memberships = team_config.get("team_memberships") or []
    if not memberships:
        return
    team_svc = TeamService()
    await team_svc.initialize()
    for m in memberships:
        team_id = m.get("team_id")
        if not team_id:
            continue
        try:
            await team_svc.add_member(
                team_id=str(team_id),
                user_id=user_id,
                role="member",
                added_by=user_id,
            )
        except (PermissionError, ValueError):
            pass
        except Exception as e:
            logger.warning("Team member ensure failed for team %s: %s", team_id, e)
        trigger = m.get("trigger_on_new_post", True)
        respond_as = (m.get("respond_as") or "comment")[:20]
        await execute(
            """
            INSERT INTO agent_line_watches (agent_profile_id, line_id, user_id, trigger_on_new_post, respond_as)
            VALUES ($1::uuid, $2::uuid, $3, $4, $5)
            ON CONFLICT (agent_profile_id, line_id) DO UPDATE SET
                trigger_on_new_post = EXCLUDED.trigger_on_new_post,
                respond_as = EXCLUDED.respond_as
            """,
            agent_profile_id,
            team_id,
            user_id,
            trigger,
            respond_as,
        )


async def _sync_agent_email_watches(
    agent_profile_id: str,
    user_id: str,
    watch_config: Dict[str, Any],
) -> None:
    from services.database_manager.database_helpers import execute, fetch_one

    await execute(
        "DELETE FROM agent_email_watches WHERE agent_profile_id = $1",
        agent_profile_id,
    )
    for w in watch_config.get("email_watches") or []:
        connection_id = w.get("connection_id")
        if connection_id is None:
            continue
        conn = await fetch_one(
            "SELECT id FROM external_connections WHERE id = $1 AND user_id = $2 AND connection_type = 'email' AND is_active = true",
            int(connection_id),
            user_id,
        )
        if not conn:
            continue
        subject_pattern = (w.get("subject_pattern") or "")[:500] or None
        sender_pattern = (w.get("sender_pattern") or "")[:500] or None
        folder = (w.get("folder") or "Inbox")[:255]
        await execute(
            """
            INSERT INTO agent_email_watches (agent_profile_id, connection_id, user_id, subject_pattern, sender_pattern, folder)
            VALUES ($1::uuid, $2, $3, $4, $5, $6)
            ON CONFLICT (agent_profile_id, connection_id) DO UPDATE SET
                subject_pattern = EXCLUDED.subject_pattern,
                sender_pattern = EXCLUDED.sender_pattern,
                folder = EXCLUDED.folder,
                is_active = true
            """,
            agent_profile_id,
            int(connection_id),
            user_id,
            subject_pattern,
            sender_pattern,
            folder,
        )


async def _sync_agent_folder_watches(
    agent_profile_id: str,
    user_id: str,
    watch_config: Dict[str, Any],
) -> None:
    from services.database_manager.database_helpers import execute

    await execute(
        "DELETE FROM agent_folder_watches WHERE agent_profile_id = $1",
        agent_profile_id,
    )
    for w in watch_config.get("folder_watches") or []:
        folder_id = w.get("folder_id")
        if not folder_id:
            continue
        folder_id_str = str(folder_id).strip()[:255]
        if not folder_id_str:
            continue
        file_type_filter = (w.get("file_type_filter") or "").strip()[:255] or None
        try:
            await execute(
                """
                INSERT INTO agent_folder_watches (agent_profile_id, folder_id, user_id, file_type_filter)
                VALUES ($1::uuid, $2, $3, $4)
                ON CONFLICT (agent_profile_id, folder_id) DO UPDATE SET
                    file_type_filter = EXCLUDED.file_type_filter,
                    is_active = true
                """,
                agent_profile_id,
                folder_id_str,
                user_id,
                file_type_filter,
            )
        except Exception as e:
            logger.debug("Skip folder watch: folder_id=%s %s", folder_id_str, e)


async def _ensure_agent_bot_user(agent_profile_id: str, user_id: str) -> Optional[str]:
    """Ensure a bot user row exists for an agent profile. Returns bot_user_id."""
    from services.database_manager.database_helpers import execute, fetch_one

    profile = await fetch_one(
        "SELECT id, name, handle, bot_user_id FROM agent_profiles WHERE id = $1::uuid",
        agent_profile_id,
    )
    if not profile:
        return None

    if profile.get("bot_user_id"):
        return profile["bot_user_id"]

    bot_user_id = f"agent-bot-{agent_profile_id}"
    display_name = profile.get("name") or profile.get("handle") or "Agent"
    username = f"bot-{(profile.get('handle') or agent_profile_id)[:40]}"

    existing = await fetch_one("SELECT user_id FROM users WHERE user_id = $1", bot_user_id)
    if not existing:
        await execute(
            """
            INSERT INTO users (user_id, username, display_name, password_hash, salt, role, is_active, is_bot)
            VALUES ($1, $2, $3, 'noauth', 'noauth', 'user', true, true)
            ON CONFLICT (user_id) DO UPDATE SET display_name = $3, is_bot = true
            """,
            bot_user_id,
            username,
            display_name,
        )
    await execute(
        "UPDATE agent_profiles SET bot_user_id = $1 WHERE id = $2::uuid",
        bot_user_id,
        agent_profile_id,
    )
    return bot_user_id


async def _ensure_bot_in_room(bot_user_id: str, room_id: str) -> None:
    """Add bot user as a room participant if not already present."""
    from services.database_manager.database_helpers import execute, fetch_one

    exists = await fetch_one(
        "SELECT 1 FROM room_participants WHERE room_id = $1::uuid AND user_id = $2",
        room_id,
        bot_user_id,
    )
    if not exists:
        try:
            await execute(
                """
                INSERT INTO room_participants (room_id, user_id, last_read_at)
                VALUES ($1::uuid, $2, NOW())
                """,
                room_id,
                bot_user_id,
            )
        except Exception:
            pass


async def _sync_agent_conversation_watches(
    agent_profile_id: str,
    user_id: str,
    watch_config: Dict[str, Any],
) -> None:
    from services.database_manager.database_helpers import execute, fetch_one

    await execute(
        "DELETE FROM agent_conversation_watches WHERE agent_profile_id = $1",
        agent_profile_id,
    )
    has_room_watches = False
    for w in watch_config.get("conversation_watches") or []:
        watch_type = (w.get("watch_type") or "").strip().lower()
        if watch_type == "ai_conversations":
            try:
                await execute(
                    """
                    INSERT INTO agent_conversation_watches (agent_profile_id, user_id, watch_type, room_id)
                    VALUES ($1::uuid, $2, 'ai_conversations', NULL)
                    ON CONFLICT (agent_profile_id, watch_type, room_id) DO UPDATE SET is_active = true
                    """,
                    agent_profile_id,
                    user_id,
                )
            except Exception:
                pass
        elif watch_type == "chat_room":
            room_id = w.get("room_id")
            if not room_id:
                continue
            room_id_str = str(room_id).strip()
            try:
                member = await fetch_one(
                    "SELECT 1 FROM room_participants WHERE room_id = $1::uuid AND user_id = $2",
                    room_id_str,
                    user_id,
                )
                creator = await fetch_one(
                    "SELECT 1 FROM chat_rooms WHERE room_id = $1::uuid AND created_by = $2",
                    room_id_str,
                    user_id,
                )
                if not member and not creator:
                    continue
                await execute(
                    """
                    INSERT INTO agent_conversation_watches (agent_profile_id, user_id, watch_type, room_id)
                    VALUES ($1::uuid, $2, 'chat_room', $3::uuid)
                    ON CONFLICT (agent_profile_id, watch_type, room_id) DO UPDATE SET is_active = true
                    """,
                    agent_profile_id,
                    user_id,
                    room_id_str,
                )
                has_room_watches = True
            except Exception:
                pass

    if has_room_watches:
        bot_user_id = await _ensure_agent_bot_user(agent_profile_id, user_id)
        if bot_user_id:
            for w in watch_config.get("conversation_watches") or []:
                if (w.get("watch_type") or "").strip().lower() == "chat_room" and w.get("room_id"):
                    await _ensure_bot_in_room(bot_user_id, str(w["room_id"]).strip())


def _warn_invalid_user_facts_policy_on_step(step: Dict[str, Any], path: str, warnings: List[str]) -> None:
    """Non-blocking warnings for bad user_facts_policy; recurse into nested step lists."""
    ufp = step.get("user_facts_policy")
    if ufp is not None and str(ufp).strip():
        if str(ufp).strip().lower() not in VALID_USER_FACTS_POLICY_VALUES:
            warnings.append(
                f"{path}: invalid user_facts_policy {ufp!r} (use inherit, no_write, or isolated)"
            )
    for key in ("then_steps", "else_steps", "parallel_steps", "steps"):
        children = step.get(key)
        if not isinstance(children, list):
            continue
        for ci, child in enumerate(children):
            if isinstance(child, dict):
                _warn_invalid_user_facts_policy_on_step(child, f"{path} → {key}[{ci}]", warnings)


def _playbook_step_skill_discovery_off(step: Dict[str, Any]) -> bool:
    """Return True only when skill discovery is explicitly disabled on a step."""
    raw = step.get("discovery_mode") or step.get("skill_discovery_mode")
    if isinstance(raw, str):
        v = raw.strip().lower()
        if v == "on_demand":
            v = "full"
        if v in ("off", "auto", "catalog", "full"):
            return v == "off"
    if step.get("inject_skill_manifest"):
        return False
    auto = step.get("auto_discover_skills")
    dyn = step.get("dynamic_tool_discovery")
    if auto is False and not dyn:
        return True
    return False


def _step_has_nonempty_condition(step: Dict[str, Any]) -> bool:
    c = step.get("condition")
    if c is None:
        return False
    return bool(str(c).strip())


def _step_exclusive_set(step: Dict[str, Any]) -> bool:
    return bool(step.get("exclusive"))


def _step_type_for_exclusive_warn(step: Dict[str, Any]) -> str:
    return str(step.get("step_type") or step.get("type") or "").strip().lower()


def _warn_missing_exclusive(steps: List[Any], warnings: List[str]) -> None:
    """
    Warn when 2+ consecutive steps have a condition but not `exclusive`, followed by a step
    with no condition (catch-all). Without exclusive, the catch-all still runs after a match.
    Skips runs where every step is type branch (different routing semantics).
    """
    if not isinstance(steps, list) or len(steps) < 3:
        return
    n = len(steps)
    i = 0
    while i < n:
        step = steps[i]
        if not isinstance(step, dict) or not _step_has_nonempty_condition(step) or _step_exclusive_set(step):
            i += 1
            continue
        j = i
        run_indices: List[int] = []
        while j < n:
            s = steps[j]
            if not isinstance(s, dict) or not _step_has_nonempty_condition(s) or _step_exclusive_set(s):
                break
            run_indices.append(j)
            j += 1
        if len(run_indices) >= 2 and j < n:
            nxt = steps[j]
            if isinstance(nxt, dict) and not _step_has_nonempty_condition(nxt):
                if not all(_step_type_for_exclusive_warn(steps[k]) == "branch" for k in run_indices):
                    names: List[str] = []
                    for k in run_indices:
                        nm = str(steps[k].get("name") or steps[k].get("output_key") or f"step_{k}").strip()
                        quoted = f'"{nm}"' if nm else f"step_{k}"
                        names.append(quoted)
                    qnames = ", ".join(names)
                    lo, hi = run_indices[0], run_indices[-1]
                    catch = nxt.get("name") or nxt.get("output_key") or f"step_{j}"
                    catch_s = str(catch).strip() or f"step_{j}"
                    warnings.append(
                        f"Steps {qnames} (steps {lo}-{hi}) have conditions but are not marked exclusive; "
                        f'step {j} ("{catch_s}") has no condition and will always run — even after a conditional '
                        f'step matches. Enable "Exclusive (stop after match)" on the conditional steps, or add a '
                        f'condition to "{catch_s}".'
                    )
        i = run_indices[0] + 1


def validate_playbook_definition(definition: Dict[str, Any]) -> List[str]:
    """
    Validate playbook definition structure. Returns list of warning/error messages.
    Does not validate action names against Action I/O Registry (that is done in orchestrator).
    """
    warnings: List[str] = []
    if not definition:
        return ["definition is empty"]
    steps = definition.get("steps")
    if not isinstance(steps, list):
        return ["definition.steps must be a list"]
    seen_step_names: set = set()
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            warnings.append(f"step {i}: must be an object")
            continue
        name = step.get("name")
        name_str = str(name or "").strip()
        if not name_str:
            warnings.append(f"step {i}: missing or empty 'name'")
        step_type = step.get("step_type")
        if not step_type:
            warnings.append(f"step {i} ({name_str or '?'}): missing 'step_type'")
        elif step_type not in VALID_STEP_TYPES:
            warnings.append(f"step {i} ({name_str or '?'}): invalid step_type '{step_type}'")
        output_key = step.get("output_key")
        if not output_key or not str(output_key).strip():
            warnings.append(f"step {i} ({name_str or '?'}): missing or empty 'output_key'")
        if step_type == "tool":
            if not step.get("action"):
                warnings.append(f"step {i} ({name_str or '?'}): tool step requires 'action'")
        _warn_invalid_user_facts_policy_on_step(step, f"step {i} ({name_str or '?'})", warnings)
        if step_type == "deep_agent":
            phases = step.get("phases")
            if not isinstance(phases, list):
                warnings.append(f"step {i} ({name_str or '?'}): deep_agent step requires 'phases' (list)")
            else:
                valid_phase_types = {"reason", "act", "search", "evaluate", "synthesize", "refine"}
                seen_phase_names: set = set()
                phase_name_to_type: Dict[str, str] = {}
                for pj, phase in enumerate(phases):
                    if not isinstance(phase, dict):
                        warnings.append(f"step {i} ({name_str or '?'}): phase {pj} must be an object")
                        continue
                    pname = (phase.get("name") or "").strip()
                    if not pname:
                        warnings.append(f"step {i} ({name_str or '?'}): phase {pj} missing or empty 'name'")
                    ptype = (phase.get("type") or "").strip().lower()
                    if not ptype:
                        warnings.append(f"step {i} ({name_str or '?'}): phase {pj} ({pname or '?'}) missing 'type'")
                    elif ptype not in valid_phase_types:
                        warnings.append(f"step {i} ({name_str or '?'}): phase {pj} ({pname or '?'}) invalid type '{ptype}'")
                    if ptype == "evaluate":
                        if not (phase.get("criteria") or "").strip():
                            warnings.append(f"step {i} ({name_str or '?'}): phase {pj} (evaluate) requires 'criteria'")
                        if phase.get("on_pass") is None and phase.get("on_fail") is None:
                            warnings.append(f"step {i} ({name_str or '?'}): phase {pj} (evaluate) should set 'on_pass' and/or 'on_fail'")
                    if ptype == "refine":
                        if not (phase.get("target") or "").strip():
                            warnings.append(f"step {i} ({name_str or '?'}): phase {pj} (refine) requires 'target' (phase name)")
                    if pname:
                        seen_phase_names.add(pname)
                        phase_name_to_type[pname] = ptype
                _out_ph = step.get("output_phase")
                if _out_ph is not None:
                    if not isinstance(_out_ph, str) or not _out_ph.strip():
                        warnings.append(
                            f"step {i} ({name_str or '?'}): output_phase must be a non-empty string when set "
                            "(use the Composer default to clear it)"
                        )
                    else:
                        _opn = _out_ph.strip()
                        if _opn not in seen_phase_names:
                            warnings.append(
                                f"step {i} ({name_str or '?'}): output_phase {_opn!r} does not match any defined phase name"
                            )
                        elif phase_name_to_type.get(_opn) == "evaluate":
                            warnings.append(
                                f"step {i} ({name_str or '?'}): output_phase points at evaluate phase {_opn!r} — "
                                "that phase's output is usually JSON gate text; prefer synthesize, refine, or reason"
                            )
                _out_tpl = step.get("output_template")
                if _out_tpl is not None and not isinstance(_out_tpl, str):
                    warnings.append(
                        f"step {i} ({name_str or '?'}): output_template must be a string when set"
                    )
                _op_set = isinstance(_out_ph, str) and bool(_out_ph.strip())
                _tpl_set = isinstance(_out_tpl, str) and bool(str(_out_tpl).strip())
                if _op_set and _tpl_set:
                    warnings.append(
                        f"step {i} ({name_str or '?'}): note — output_template takes precedence over output_phase at runtime"
                    )
        skill_refs = step.get("skill_ids") or step.get("skills")
        if skill_refs is not None and not isinstance(skill_refs, list):
            warnings.append(f"step {i} ({name_str or '?'}): skill_ids/skills must be a list")
        elif isinstance(skill_refs, list):
            for ref in skill_refs:
                if not isinstance(ref, str) or not ref.strip():
                    warnings.append(f"step {i} ({name_str or '?'}): skill_ids/skills must be list of non-empty strings")
                    break
        if step_type in ("llm_agent", "deep_agent"):
            _step_tools = list(step.get("available_tools") or [])
            _step_skills = list(skill_refs or [])
            if not _step_tools and not _step_skills and _playbook_step_skill_discovery_off(step):
                warnings.append(
                    f"step {i} ({name_str or '?'}): {step_type} has no available_tools and no pinned skill_ids, "
                    "and skill discovery is off — the agent will have no capabilities at runtime. "
                    "Add tools to available_tools, add skill_ids, or enable skill discovery "
                    "(set discovery_mode to 'auto' or 'catalog')."
                )
            # Best-of-N sampling
            raw_samples = step.get("samples")
            if raw_samples is not None:
                try:
                    sn = int(raw_samples)
                    if sn < 1 or sn > 5:
                        warnings.append(
                            f"step {i} ({name_str or '?'}): samples must be an integer 1–5 (got {raw_samples!r})"
                        )
                    elif sn > 3:
                        warnings.append(
                            f"step {i} ({name_str or '?'}): samples={sn} increases cost and latency — "
                            "consider 2–3 unless quality demands more."
                        )
                except (TypeError, ValueError):
                    warnings.append(
                        f"step {i} ({name_str or '?'}): samples must be an integer 1–5 (got {raw_samples!r})"
                    )
            ss_raw = step.get("selection_strategy")
            if ss_raw is not None and str(ss_raw).strip():
                ss = str(ss_raw).strip().lower()
                if ss not in ("llm_judge", "highest_score"):
                    warnings.append(
                        f"step {i} ({name_str or '?'}): invalid selection_strategy {ss_raw!r} "
                        "(use llm_judge or highest_score)"
                    )
                elif ss == "highest_score" and step_type != "deep_agent":
                    warnings.append(
                        f"step {i} ({name_str or '?'}): selection_strategy highest_score is only supported "
                        "on deep_agent steps (use llm_judge for llm_agent)."
                    )
                elif ss == "highest_score" and step_type == "deep_agent":
                    phases_chk = step.get("phases") if isinstance(step.get("phases"), list) else []
                    has_eval = any(
                        isinstance(p, dict) and str(p.get("type") or "").strip().lower() == "evaluate"
                        for p in phases_chk
                    )
                    if not has_eval:
                        warnings.append(
                            f"step {i} ({name_str or '?'}): selection_strategy highest_score needs at least one "
                            "evaluate phase in phases; otherwise the runtime falls back to llm_judge."
                        )
            sc = step.get("selection_criteria")
            if sc is not None and not isinstance(sc, str):
                warnings.append(
                    f"step {i} ({name_str or '?'}): selection_criteria must be a string if set"
                )
            # Dynamic fan-out
            fo = step.get("fan_out")
            if fo is not None:
                if step_type not in ("llm_agent", "deep_agent"):
                    warnings.append(
                        f"step {i} ({name_str or '?'}): fan_out is only supported on llm_agent and deep_agent steps"
                    )
                elif not isinstance(fo, dict):
                    warnings.append(f"step {i} ({name_str or '?'}): fan_out must be an object")
                else:
                    src = str(fo.get("source") or "").strip()
                    if not src:
                        warnings.append(
                            f"step {i} ({name_str or '?'}): fan_out.source is required (dot-path to a list)"
                        )
                    iv = fo.get("item_variable")
                    if iv is not None and str(iv).strip():
                        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", str(iv).strip()):
                            warnings.append(
                                f"step {i} ({name_str or '?'}): fan_out.item_variable must be a valid identifier"
                            )
                    mi = fo.get("max_items")
                    if mi is not None:
                        try:
                            miv = int(mi)
                            if miv < 1 or miv > 10:
                                warnings.append(
                                    f"step {i} ({name_str or '?'}): fan_out.max_items must be 1–10 (got {mi!r})"
                                )
                        except (TypeError, ValueError):
                            warnings.append(
                                f"step {i} ({name_str or '?'}): fan_out.max_items must be an integer 1–10"
                            )
                    mg = fo.get("merge")
                    if mg is not None and str(mg).strip().lower() not in ("list", "concat", ""):
                        warnings.append(
                            f"step {i} ({name_str or '?'}): fan_out.merge must be list or concat (got {mg!r})"
                        )
        heading_level = step.get("heading_level")
        if heading_level is not None:
            try:
                hl = int(heading_level)
                if hl < 1 or hl > 6:
                    warnings.append(f"step {i} ({name_str or '?'}): heading_level must be 1–6")
            except (TypeError, ValueError):
                warnings.append(f"step {i} ({name_str or '?'}): heading_level must be an integer 1–6")
        inputs = step.get("inputs")
        if isinstance(inputs, dict):
            for _k, v in inputs.items():
                if isinstance(v, str) and v.strip().startswith("{") and v.strip().endswith("}"):
                    ref = v.strip()[1:-1].strip()
                    if "." in ref:
                        step_ref = ref.split(".", 1)[0].strip()
                        if step_ref and step_ref not in seen_step_names:
                            warnings.append(f"step {i} ({name_str or '?'}): input references unknown step '{step_ref}'")
        if name_str:
            seen_step_names.add(name_str)
    _warn_missing_exclusive(steps, warnings)
    return warnings


async def validate_and_remediate_playbook_models_for_user(
    user_id: str,
    definition: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """
    Fix step model_override values that are not on the user's current provider catalog.
    Returns (definition_copy, changed_step_names, human_messages). Skips mutation if definition has no steps.
    """
    import copy

    from services.model_source_resolver import get_available_models, try_soft_retarget

    definition = copy.deepcopy(definition or {})
    steps = definition.get("steps")
    if not isinstance(steps, list):
        return definition, [], []

    available = await get_available_models(user_id)
    avail_ids = {m.id for m in available}
    changed_steps: List[str] = []
    messages: List[str] = []

    for step in steps:
        if not isinstance(step, dict):
            continue
        mo = step.get("model_override")
        if not mo or not str(mo).strip():
            continue
        mid = str(mo).strip()
        if mid in avail_ids:
            continue
        retarget = await try_soft_retarget(user_id, mid)
        name = str(step.get("name") or step.get("output_key") or "?").strip()
        if retarget.get("available"):
            new_id = retarget.get("model_id") or mid
            step["model_override"] = new_id
            changed_steps.append(name)
            messages.append(f"Step «{name}»: model override set to «{new_id}».")
        else:
            step.pop("model_override", None)
            changed_steps.append(name)
            messages.append(f"Step «{name}»: invalid model override removed.")

    return definition, changed_steps, messages


async def notify_playbook_model_remediation(
    user_id: str,
    playbook_id: str,
    changed_steps: List[str],
    messages: List[str],
) -> None:
    if not messages:
        return
    from services.model_configuration_notifier import maybe_notify_model_configuration_issue

    preview = " ".join(messages)[:500]
    await maybe_notify_model_configuration_issue(
        user_id,
        title="Playbook model overrides updated",
        preview=preview,
        dedupe_key=f"playbook:{playbook_id}:models:{','.join(changed_steps[:5])}",
        playbook_id=playbook_id,
        step_names=changed_steps,
    )


async def create_profile(user_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Create an agent profile. Handle is optional; when empty, agent is not @mentionable (schedule/Run-only)."""
    from services.database_manager.database_helpers import fetch_one

    handle_raw = (data.get("handle") or "").strip()
    handle = handle_raw if handle_raw else None
    if handle:
        existing = await fetch_one(
            "SELECT id FROM agent_profiles WHERE user_id = $1 AND handle = $2",
            user_id,
            handle,
            rls_context=_rls(user_id),
        )
        if existing:
            raise ValueError("Handle already in use for this user")

    name = data.get("name") or handle or "Unnamed"
    persona_mode = data.get("persona_mode") or "none"
    if data.get("persona_enabled") is True and persona_mode == "none":
        persona_mode = "default"
    persona_id = data.get("persona_id")
    if persona_mode != "specific":
        persona_id = None
    row = await fetch_one(
        """
        INSERT INTO agent_profiles (
            user_id, name, handle, description, is_active,
            model_preference, model_source, model_provider_type, max_research_rounds, system_prompt_additions,
            knowledge_config, default_playbook_id, default_run_context,
            default_approval_policy, journal_config, team_config, watch_config,
            chat_history_enabled, chat_history_lookback, summary_threshold_tokens, summary_keep_messages,
            persona_mode, persona_id, include_user_context, include_datetime_context, include_user_facts, include_facts_categories, use_themed_memory, auto_routable, chat_visible, data_workspace_config,
            include_agent_memory, allowed_connections, category
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb, $12::uuid, $13, $14, $15::jsonb, $16::jsonb, $17::jsonb, $18, $19, $20, $21, $22, $23::uuid, $24, $25, $26, $27::jsonb, $28, $29, $30, $31::jsonb, $32, $33::jsonb, $34)
        RETURNING *
        """,
        user_id,
        name,
        handle,
        data.get("description"),
        data.get("is_active", True),
        data.get("model_preference"),
        data.get("model_source"),
        data.get("model_provider_type"),
        data.get("max_research_rounds", 3),
        data.get("system_prompt_additions"),
        json.dumps(data.get("knowledge_config") or {}),
        data.get("default_playbook_id"),
        data.get("default_run_context") or "interactive",
        data.get("default_approval_policy") or "require",
        json.dumps(data.get("journal_config") or {}),
        json.dumps(data.get("team_config") or {}),
        json.dumps(data.get("watch_config") or {}),
        data.get("prompt_history_enabled", data.get("chat_history_enabled", False)),
        data.get("chat_history_lookback", 10),
        data.get("summary_threshold_tokens", 5000),
        data.get("summary_keep_messages", 10),
        persona_mode,
        persona_id,
        data.get("include_user_context", False),
        data.get("include_datetime_context", True),
        data.get("include_user_facts", False),
        json.dumps(data.get("include_facts_categories") or []),
        data.get("use_themed_memory", True),
        data.get("auto_routable", False),
        data.get("chat_visible", True),
        json.dumps(data.get("data_workspace_config") or {}),
        data.get("include_agent_memory", False),
        json.dumps(data.get("allowed_connections") or []),
        data.get("category"),
        rls_context=_rls(user_id),
    )
    if not row:
        raise RuntimeError("Profile created but not found")
    profile_id = str(row["id"])
    for sync_fn, payload in [
        (_sync_agent_line_watches, data.get("team_config") or {}),
        (_sync_agent_email_watches, data.get("watch_config") or {}),
        (_sync_agent_folder_watches, data.get("watch_config") or {}),
        (_sync_agent_conversation_watches, data.get("watch_config") or {}),
    ]:
        try:
            await sync_fn(profile_id, user_id, payload)
        except Exception as e:
            logger.warning("Sync after create_profile failed: %s", e)
    return _row_to_profile(row)


async def create_playbook(user_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a custom playbook."""
    from services.database_manager.database_helpers import fetch_one

    definition_in = data.get("definition") or {}
    if not isinstance(definition_in, dict):
        definition_in = {}
    definition_fixed, changed_steps, remediation_msgs = await validate_and_remediate_playbook_models_for_user(
        user_id, definition_in
    )
    ensure_step_ids(definition_fixed)

    row = await fetch_one(
        """
        INSERT INTO custom_playbooks (
            user_id, name, description, version, definition, triggers,
            is_template, category, tags, required_connectors
        ) VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, $7, $8, $9, $10)
        RETURNING *
        """,
        user_id,
        data.get("name") or "Unnamed",
        data.get("description"),
        data.get("version") or "1.0",
        json.dumps(definition_fixed),
        json.dumps(data.get("triggers") or []),
        data.get("is_template", False),
        data.get("category"),
        data.get("tags") or [],
        data.get("required_connectors") or [],
    )
    if remediation_msgs and row and row.get("id"):
        await notify_playbook_model_remediation(
            user_id, str(row["id"]), changed_steps, remediation_msgs
        )
    return _row_to_playbook(row)


async def _ensure_agent_profiles_fact_columns() -> None:
    """Ensure agent_profiles has include_user_facts and include_facts_categories (migrations 065/066). Idempotent."""
    from services.database_manager.database_helpers import execute
    await execute(
        "ALTER TABLE agent_profiles ADD COLUMN IF NOT EXISTS include_user_facts BOOLEAN NOT NULL DEFAULT false"
    )
    await execute(
        "ALTER TABLE agent_profiles ADD COLUMN IF NOT EXISTS include_facts_categories JSONB DEFAULT '[]'::jsonb"
    )
    await execute(
        "ALTER TABLE agent_profiles ADD COLUMN IF NOT EXISTS use_themed_memory BOOLEAN DEFAULT true"
    )
    logger.info(
        "Ensured agent_profiles columns: include_user_facts, include_facts_categories, use_themed_memory"
    )


DEFAULT_PLAYBOOK_ID = "00000000-0001-4000-8000-000000000001"


async def seed_default_agent_profiles(user_id: Optional[str] = None) -> None:
    """
    Ensure every user has the built-in default agent profile (Bastion Assistant).
    Idempotent: only inserts for users that do not have a profile with is_builtin = true.
    When user_id is provided, only seed for that user (e.g. after create_user).
    When user_id is None, seed for all users (startup).
    """
    from services.database_manager.database_helpers import execute, fetch_all

    ctx = _rls(None, "admin") if user_id is None else _rls(user_id)
    try:
        if user_id:
            await execute(
                """
                INSERT INTO agent_profiles (
                    user_id, name, handle, description, is_active, is_locked, is_builtin,
                    default_playbook_id, chat_history_enabled, chat_history_lookback,
                    persona_mode, include_user_facts, include_datetime_context, include_user_context
                )
                SELECT u.user_id, $1, $2, $3, $4, $5, $6, $7::uuid, $8, $9, $10, $11, $12, $13
                FROM users u
                WHERE u.user_id = $14
                AND NOT EXISTS (SELECT 1 FROM agent_profiles p WHERE p.user_id = u.user_id AND p.is_builtin = true)
                """,
                "Bastion Assistant",
                "assistant",
                "Default general-purpose assistant with skill search, research, and conversation",
                True,
                False,
                True,
                DEFAULT_PLAYBOOK_ID,
                True,
                10,
                "default",
                True,
                True,
                True,
                user_id,
                rls_context=ctx,
            )
            logger.info("Seeded default agent profile for user %s", user_id)
        else:
            await execute(
                """
                INSERT INTO agent_profiles (
                    user_id, name, handle, description, is_active, is_locked, is_builtin,
                    default_playbook_id, chat_history_enabled, chat_history_lookback,
                    persona_mode, include_user_facts, include_datetime_context, include_user_context
                )
                SELECT u.user_id, $1, $2, $3, $4, $5, $6, $7::uuid, $8, $9, $10, $11, $12, $13
                FROM users u
                WHERE NOT EXISTS (SELECT 1 FROM agent_profiles p WHERE p.user_id = u.user_id AND p.is_builtin = true)
                """,
                "Bastion Assistant",
                "assistant",
                "Default general-purpose assistant with skill search, research, and conversation",
                True,
                False,
                True,
                DEFAULT_PLAYBOOK_ID,
                True,
                10,
                "default",
                True,
                True,
                True,
                rls_context=ctx,
            )
            logger.info("Seeded default agent profiles for all users")
    except Exception as e:
        if "is_builtin" in str(e) and ("does not exist" in str(e) or "column" in str(e).lower()):
            logger.debug("Default agent profile seed skipped (is_builtin column may not exist yet): %s", e)
        else:
            logger.warning("Default agent profile seed failed: %s", e)


async def update_profile(user_id: str, profile_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Partial update of an agent profile. Raises ValueError if not found or handle conflict."""
    from services.database_manager.database_helpers import fetch_one, execute

    row = await fetch_one(
        "SELECT * FROM agent_profiles WHERE id = $1 AND user_id = $2",
        profile_id,
        user_id,
        rls_context=_rls(user_id),
    )
    if not row:
        raise ValueError("Profile not found")
    if not data:
        return _row_to_profile(row)
    if row.get("is_builtin"):
        allowed_builtin_only = {"category"}
        if set(data.keys()) - allowed_builtin_only:
            raise ValueError(
                "Built-in profile is read-only. Create a custom agent profile for edits, "
                "or use Reset to defaults to restore factory settings."
            )
        if not (set(data.keys()) & allowed_builtin_only):
            return _row_to_profile(row)
    else:
        allowed_when_locked = {"is_active", "is_locked"}
        if row.get("is_locked") and set(data.keys()) - allowed_when_locked:
            raise ValueError("Profile is locked; only pause/resume and lock toggle are allowed")
    if "handle" in data:
        handle_val = (data["handle"] or "").strip() or None
        data["handle"] = handle_val
        if handle_val:
            existing = await fetch_one(
                "SELECT id FROM agent_profiles WHERE user_id = $1 AND handle = $2 AND id != $3",
                user_id,
                handle_val,
                profile_id,
                rls_context=_rls(user_id),
            )
            if existing:
                raise ValueError("Handle already in use for this user")
    jsonb_fields = (
        "knowledge_config",
        "journal_config",
        "team_config",
        "watch_config",
        "include_facts_categories",
        "data_workspace_config",
        "allowed_connections",
    )
    set_clauses = []
    args = []
    idx = 1
    column_map = {"prompt_history_enabled": "chat_history_enabled"}
    for k, v in data.items():
        db_key = column_map.get(k, k)
        if k in jsonb_fields:
            set_clauses.append(f"{db_key} = ${idx}::jsonb")
            args.append(json.dumps(v) if isinstance(v, (dict, list)) else v)
        else:
            set_clauses.append(f"{db_key} = ${idx}")
            args.append(v)
        idx += 1
    set_clauses.append("updated_at = NOW()")
    args.extend([profile_id, user_id])
    update_sql = (
        f"UPDATE agent_profiles SET {', '.join(set_clauses)} WHERE id = ${idx}::uuid AND user_id = ${idx + 1}"
    )
    ctx = _rls(user_id)
    try:
        await execute(update_sql, *args, rls_context=ctx)
    except asyncpg.exceptions.UndefinedColumnError:
        logger.warning("Agent profiles missing fact columns, adding them and retrying")
        await _ensure_agent_profiles_fact_columns()
        await execute(update_sql, *args, rls_context=ctx)
    except Exception as e:
        err_msg = str(e)
        if (
            "include_user_facts" in err_msg
            or "include_facts_categories" in err_msg
            or "use_themed_memory" in err_msg
        ) and "does not exist" in err_msg:
            logger.warning("Agent profiles missing fact columns, adding them and retrying: %s", e)
            await _ensure_agent_profiles_fact_columns()
            await execute(update_sql, *args, rls_context=ctx)
        else:
            raise
    if "team_config" in data:
        try:
            await _sync_agent_line_watches(profile_id, user_id, data.get("team_config") or {})
        except Exception as e:
            logger.warning("Sync agent team watches failed: %s", e)
    if "watch_config" in data:
        wc = data.get("watch_config") or {}
        for fn in (_sync_agent_email_watches, _sync_agent_folder_watches, _sync_agent_conversation_watches):
            try:
                await fn(profile_id, user_id, wc)
            except Exception as e:
                logger.warning("Sync watches failed: %s", e)
    row = await fetch_one("SELECT * FROM agent_profiles WHERE id = $1", profile_id, rls_context=_rls(user_id))
    return _row_to_profile(row)


async def reset_builtin_profile_defaults(user_id: str, profile_id: str) -> Dict[str, Any]:
    """
    Reset a built-in agent profile to seed defaults (name, description, playbook, persona, context flags).
    Raises ValueError if profile not found or not built-in.
    """
    from services.database_manager.database_helpers import execute, fetch_one

    row = await fetch_one(
        "SELECT id, is_builtin FROM agent_profiles WHERE id = $1 AND user_id = $2",
        profile_id,
        user_id,
        rls_context=_rls(user_id),
    )
    if not row:
        raise ValueError("Profile not found")
    if not row.get("is_builtin"):
        raise ValueError("Only built-in profiles can be reset to defaults")

    ctx = _rls(user_id)
    await execute(
        """
        UPDATE agent_profiles SET
            name = $1,
            handle = $2,
            description = $3,
            default_playbook_id = $4::uuid,
            chat_history_enabled = $5,
            chat_history_lookback = $6,
            summary_threshold_tokens = $7,
            summary_keep_messages = $8,
            persona_mode = $9,
            persona_id = NULL,
            include_user_facts = $10,
            include_datetime_context = $11,
            include_user_context = $12,
            model_preference = NULL,
            model_source = NULL,
            model_provider_type = NULL,
            system_prompt_additions = NULL,
            default_approval_policy = COALESCE(default_approval_policy, 'require'),
            updated_at = NOW()
        WHERE id = $13::uuid AND user_id = $14
        """,
        "Bastion Assistant",
        "assistant",
        "Default general-purpose assistant with skill search, research, and conversation",
        DEFAULT_PLAYBOOK_ID,
        True,
        10,
        5000,
        10,
        "default",
        True,
        True,
        True,
        profile_id,
        user_id,
        rls_context=ctx,
    )
    try:
        await execute(
            "UPDATE agent_profiles SET include_facts_categories = '[]'::jsonb WHERE id = $1::uuid AND user_id = $2",
            profile_id,
            user_id,
            rls_context=ctx,
        )
    except Exception:
        pass
    row = await fetch_one("SELECT * FROM agent_profiles WHERE id = $1", profile_id, rls_context=_rls(user_id))
    return _row_to_profile(row)


async def list_profiles(user_id: str) -> List[Dict[str, Any]]:
    """List agent profiles owned by or shared with the user."""
    from services.database_manager.database_helpers import fetch_all

    rows = await fetch_all(
        """
        SELECT p.*, e.status AS last_execution_status,
               b.monthly_limit_usd, b.current_period_start, b.current_period_spend_usd,
               b.warning_threshold_pct, b.enforce_hard_limit,
               CASE
                   WHEN p.user_id = $1 THEN 'owned'
                   ELSE 'shared'
               END AS ownership,
               u_owner.username AS owner_username,
               u_owner.display_name AS owner_display_name
        FROM agent_profiles p
        LEFT JOIN LATERAL (
            SELECT status FROM agent_execution_log
            WHERE agent_profile_id = p.id
            ORDER BY started_at DESC
            LIMIT 1
        ) e ON true
        LEFT JOIN agent_budgets b ON b.agent_profile_id = p.id
        LEFT JOIN users u_owner ON u_owner.user_id = p.user_id
        WHERE p.user_id = $1
           OR EXISTS (
               SELECT 1 FROM agent_artifact_shares _sh
               WHERE _sh.artifact_type = 'agent_profile'
                 AND _sh.artifact_id = p.id
                 AND _sh.shared_with_user_id = $1
           )
        ORDER BY ownership ASC, LOWER(COALESCE(NULLIF(TRIM(p.name), ''), p.handle, '')) ASC
        """,
        user_id,
        rls_context=_rls(user_id),
    )
    return [_row_to_profile(r) for r in rows]


# ---------- Sidebar categories (Agent Factory folder list per section) ----------

VALID_SIDEBAR_SECTIONS = frozenset({"agents", "playbooks", "skills", "connectors"})


async def list_sidebar_categories(
    user_id: str, section: Optional[str] = None
) -> List[Dict[str, Any]]:
    """List sidebar categories for the user, optionally filtered by section."""
    from services.database_manager.database_helpers import fetch_all

    if section and section not in VALID_SIDEBAR_SECTIONS:
        return []
    sql = """
        SELECT id, user_id, section, name, sort_order, created_at
        FROM agent_factory_sidebar_categories
        WHERE user_id = $1
    """
    args: List[Any] = [user_id]
    if section:
        sql += " AND section = $2 ORDER BY sort_order, name"
        args.append(section)
    else:
        sql += " ORDER BY section, sort_order, name"
    rows = await fetch_all(sql, *args, rls_context=_rls(user_id))
    return [
        {
            "id": str(r["id"]),
            "user_id": r["user_id"],
            "section": r["section"],
            "name": r["name"],
            "sort_order": int(r["sort_order"]),
            "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
        }
        for r in rows
    ]


async def create_sidebar_category(
    user_id: str, section: str, name: str
) -> Dict[str, Any]:
    """Create a sidebar category. Name must be unique per user+section."""
    from services.database_manager.database_helpers import fetch_one, execute

    if section not in VALID_SIDEBAR_SECTIONS:
        raise ValueError(f"Invalid section: {section}")
    name = (name or "").strip()
    if not name:
        raise ValueError("Category name is required")
    ctx = _rls(user_id)
    max_order = await fetch_one(
        """
        SELECT COALESCE(MAX(sort_order), -1) + 1 AS next_order
        FROM agent_factory_sidebar_categories
        WHERE user_id = $1 AND section = $2
        """,
        user_id,
        section,
        rls_context=ctx,
    )
    next_order = int(max_order["next_order"]) if max_order else 0
    await execute(
        """
        INSERT INTO agent_factory_sidebar_categories (user_id, section, name, sort_order)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (user_id, section, name) DO NOTHING
        """,
        user_id,
        section,
        name,
        next_order,
        rls_context=ctx,
    )
    row = await fetch_one(
        "SELECT id, user_id, section, name, sort_order, created_at FROM agent_factory_sidebar_categories WHERE user_id = $1 AND section = $2 AND name = $3",
        user_id,
        section,
        name,
        rls_context=ctx,
    )
    if not row:
        raise ValueError("Category already exists with that name")
    return {
        "id": str(row["id"]),
        "user_id": row["user_id"],
        "section": row["section"],
        "name": row["name"],
        "sort_order": int(row["sort_order"]),
        "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
    }


async def update_sidebar_category(
    user_id: str, category_id: str, data: Dict[str, Any]
) -> Dict[str, Any]:
    """Update a sidebar category (name and/or sort_order)."""
    from services.database_manager.database_helpers import fetch_one, execute

    ctx = _rls(user_id)
    row = await fetch_one(
        "SELECT * FROM agent_factory_sidebar_categories WHERE id = $1 AND user_id = $2",
        category_id,
        user_id,
        rls_context=ctx,
    )
    if not row:
        raise ValueError("Category not found")
    set_parts: List[str] = []
    args: List[Any] = []
    pos = 1
    if "name" in data and data["name"] is not None:
        name = (data["name"] or "").strip()
        if name:
            set_parts.append(f"name = ${pos}")
            args.append(name)
            pos += 1
    if "sort_order" in data and data["sort_order"] is not None:
        set_parts.append(f"sort_order = ${pos}")
        args.append(int(data["sort_order"]))
        pos += 1
    if not set_parts:
        return {
            "id": str(row["id"]),
            "user_id": row["user_id"],
            "section": row["section"],
            "name": row["name"],
            "sort_order": int(row["sort_order"]),
            "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
        }
    args.extend([category_id, user_id])
    where_pos = pos
    await execute(
        f"UPDATE agent_factory_sidebar_categories SET {', '.join(set_parts)} WHERE id = ${where_pos}::uuid AND user_id = ${where_pos + 1}",
        *args,
        rls_context=ctx,
    )
    row = await fetch_one(
        "SELECT * FROM agent_factory_sidebar_categories WHERE id = $1", category_id, rls_context=ctx
    )
    return {
        "id": str(row["id"]),
        "user_id": row["user_id"],
        "section": row["section"],
        "name": row["name"],
        "sort_order": int(row["sort_order"]),
        "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
    }


async def delete_sidebar_category(user_id: str, category_id: str) -> None:
    """Delete a sidebar category. Items keep their category string; they still group under that name until moved."""
    from services.database_manager.database_helpers import execute

    await execute(
        "DELETE FROM agent_factory_sidebar_categories WHERE id = $1::uuid AND user_id = $2",
        category_id,
        user_id,
        rls_context=_rls(user_id),
    )


async def get_profile_budget(profile_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """Get budget for an agent profile (if set)."""
    from services.database_manager.database_helpers import fetch_one
    row = await fetch_one(
        """
        SELECT monthly_limit_usd, current_period_start, current_period_spend_usd,
               warning_threshold_pct, enforce_hard_limit
        FROM agent_budgets
        WHERE agent_profile_id = $1 AND user_id = $2
        """,
        profile_id,
        user_id,
        rls_context=_rls(user_id),
    )
    if not row or row.get("monthly_limit_usd") is None:
        return None
    return {
        "monthly_limit_usd": float(row["monthly_limit_usd"]),
        "current_period_start": row["current_period_start"].isoformat() if row.get("current_period_start") else None,
        "current_period_spend_usd": float(row.get("current_period_spend_usd") or 0),
        "warning_threshold_pct": row.get("warning_threshold_pct", 80),
        "enforce_hard_limit": row.get("enforce_hard_limit", True),
    }


async def set_profile_budget(
    profile_id: str,
    user_id: str,
    monthly_limit_usd: Optional[float] = None,
    warning_threshold_pct: int = 80,
    enforce_hard_limit: bool = True,
) -> Dict[str, Any]:
    """Create or update budget for an agent profile. Use null monthly_limit_usd to clear (unlimited)."""
    from datetime import date
    from services.database_manager.database_helpers import fetch_one, execute
    ctx = _rls(user_id)
    profile = await fetch_one(
        "SELECT id FROM agent_profiles WHERE id = $1 AND user_id = $2",
        profile_id,
        user_id,
        rls_context=ctx,
    )
    if not profile:
        raise ValueError("Profile not found")
    period_start = date.today().replace(day=1)
    existing = await fetch_one(
        "SELECT id FROM agent_budgets WHERE agent_profile_id = $1",
        profile_id,
    )
    if existing:
        await execute(
            """
            UPDATE agent_budgets
            SET monthly_limit_usd = $1, warning_threshold_pct = $2, enforce_hard_limit = $3, updated_at = NOW()
            WHERE agent_profile_id = $4
            """,
            monthly_limit_usd,
            warning_threshold_pct,
            enforce_hard_limit,
            profile_id,
        )
    else:
        await execute(
            """
            INSERT INTO agent_budgets (agent_profile_id, user_id, monthly_limit_usd, current_period_start, warning_threshold_pct, enforce_hard_limit)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            profile_id,
            user_id,
            monthly_limit_usd,
            period_start,
            warning_threshold_pct,
            enforce_hard_limit,
        )
    return (await get_profile_budget(profile_id, user_id)) or {}


async def list_playbooks(user_id: str) -> List[Dict[str, Any]]:
    """List playbooks owned by, shared with, or template for the user."""
    from services.database_manager.database_helpers import fetch_all

    rows = await fetch_all(
        """
        SELECT pb.*,
               CASE
                   WHEN pb.user_id = $1 THEN 'owned'
                   WHEN pb.is_template = true THEN 'template'
                   ELSE 'shared'
               END AS ownership,
               u_owner.username AS owner_username,
               u_owner.display_name AS owner_display_name
        FROM custom_playbooks pb
        LEFT JOIN users u_owner ON u_owner.user_id = pb.user_id
        WHERE pb.user_id = $1
           OR pb.is_template = true
           OR EXISTS (
               SELECT 1 FROM agent_artifact_shares _sh
               WHERE _sh.artifact_type = 'playbook'
                 AND _sh.artifact_id = pb.id
                 AND _sh.shared_with_user_id = $1
           )
        ORDER BY ownership ASC, LOWER(pb.name) ASC NULLS LAST
        """,
        user_id,
    )
    return [_row_to_playbook(r) for r in rows]


async def create_schedule(
    user_id: str,
    profile_id: str,
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """Create a schedule for an agent profile. Raises ValueError if profile not found or invalid params."""
    from services.database_manager.database_helpers import fetch_one, execute

    profile = await fetch_one(
        "SELECT id FROM agent_profiles WHERE id = $1 AND user_id = $2",
        profile_id,
        user_id,
        rls_context=_rls(user_id),
    )
    if not profile:
        raise ValueError("Profile not found")
    schedule_type = (data.get("schedule_type") or "cron").strip().lower()
    if schedule_type == "cron" and not data.get("cron_expression"):
        raise ValueError("cron_expression required for schedule_type cron")
    if schedule_type == "interval" and not data.get("interval_seconds"):
        raise ValueError("interval_seconds required for schedule_type interval")
    tz = data.get("timezone")
    if tz is None or tz == "":
        try:
            from services.settings_service import settings_service
            if getattr(settings_service, "_initialized", False):
                tz = await settings_service.get_user_timezone(user_id)
        except Exception:
            pass
        tz = tz or "UTC"
    else:
        tz = tz or "UTC"
    next_run_at = _compute_next_run_at(
        schedule_type,
        data.get("cron_expression"),
        data.get("interval_seconds"),
        tz,
    )
    await execute(
        """
        INSERT INTO agent_schedules (
            agent_profile_id, user_id, schedule_type, cron_expression,
            interval_seconds, timezone, timeout_seconds, max_consecutive_failures,
            input_context, next_run_at, is_active
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10, $11)
        """,
        profile_id,
        user_id,
        schedule_type,
        data.get("cron_expression"),
        data.get("interval_seconds"),
        tz,
        data.get("timeout_seconds", 300),
        data.get("max_consecutive_failures", 5),
        json.dumps(data.get("input_context") or {}),
        next_run_at,
        data.get("is_active", True),
    )
    row = await fetch_one(
        "SELECT * FROM agent_schedules WHERE agent_profile_id = $1 ORDER BY created_at DESC LIMIT 1",
        profile_id,
    )
    return _row_to_schedule(row)


async def update_schedules_timezone_for_user(user_id: str, new_timezone: str) -> int:
    """Update all agent schedules for a user to the new timezone and recompute next_run_at.
    Called when the user changes their timezone in settings so schedules keep the same local time."""
    from services.database_manager.database_helpers import execute, fetch_all

    rows = await fetch_all(
        "SELECT id, schedule_type, cron_expression, interval_seconds FROM agent_schedules WHERE user_id = $1",
        user_id,
    )
    if not rows:
        return 0
    for row in rows:
        next_run_at = _compute_next_run_at(
            row["schedule_type"],
            row.get("cron_expression"),
            row.get("interval_seconds"),
            new_timezone,
        )
        if next_run_at is not None:
            await execute(
                "UPDATE agent_schedules SET timezone = $1, next_run_at = $2, updated_at = NOW() WHERE id = $3",
                new_timezone,
                next_run_at,
                row["id"],
            )
        else:
            await execute(
                "UPDATE agent_schedules SET timezone = $1, updated_at = NOW() WHERE id = $2",
                new_timezone,
                row["id"],
            )
    return len(rows)


async def create_data_source_binding(
    user_id: str,
    profile_id: str,
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """Add a data source binding to an agent profile. Raises ValueError if profile not found."""
    from services.database_manager.database_helpers import fetch_one, execute

    profile = await fetch_one(
        "SELECT id FROM agent_profiles WHERE id = $1 AND user_id = $2",
        profile_id,
        user_id,
        rls_context=_rls(user_id),
    )
    if not profile:
        raise ValueError("Profile not found")
    connector_id = data.get("connector_id")
    if not connector_id:
        raise ValueError("connector_id is required")
    await execute(
        """
        INSERT INTO agent_data_sources (
            agent_profile_id, connector_id, credentials_encrypted,
            config_overrides, permissions, is_enabled
        ) VALUES ($1, $2, $3::jsonb, $4::jsonb, $5::jsonb, $6)
        """,
        profile_id,
        connector_id,
        json.dumps(data.get("credentials_encrypted")) if data.get("credentials_encrypted") else None,
        json.dumps(data.get("config_overrides") or {}),
        json.dumps(data.get("permissions") or {}),
        data.get("is_enabled", True),
    )
    row = await fetch_one(
        "SELECT * FROM agent_data_sources WHERE agent_profile_id = $1 AND connector_id = $2 ORDER BY created_at DESC LIMIT 1",
        profile_id,
        connector_id,
    )
    return _row_to_data_source(row)

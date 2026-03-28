"""
Agent Factory Skills service - CRUD, version management, and built-in skill seeding.

Skills are procedural knowledge injected into LLM steps (profile, playbook, step levels).
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


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


def _row_to_skill(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not row:
        return {}
    return {
        "id": str(row["id"]),
        "user_id": row.get("user_id"),
        "name": row["name"],
        "slug": row["slug"],
        "description": row.get("description"),
        "category": row.get("category"),
        "procedure": row.get("procedure") or "",
        "required_tools": list(row["required_tools"]) if row.get("required_tools") else [],
        "optional_tools": list(row["optional_tools"]) if row.get("optional_tools") else [],
        "inputs_schema": _ensure_json_obj(row.get("inputs_schema"), {}),
        "outputs_schema": _ensure_json_obj(row.get("outputs_schema"), {}),
        "examples": _ensure_json_obj(row.get("examples"), []),
        "tags": list(row["tags"]) if row.get("tags") else [],
        "is_builtin": row.get("is_builtin", False),
        "is_locked": row.get("is_locked", False),
        "version": row.get("version", 1),
        "parent_skill_id": str(row["parent_skill_id"]) if row.get("parent_skill_id") else None,
        "improvement_rationale": row.get("improvement_rationale"),
        "evidence_metadata": _ensure_json_obj(row.get("evidence_metadata"), {}),
        "created_at": row.get("created_at").isoformat() if row.get("created_at") else None,
        "updated_at": row.get("updated_at").isoformat() if row.get("updated_at") else None,
    }


async def list_skills(
    user_id: str,
    category: Optional[str] = None,
    include_builtin: bool = True,
) -> List[Dict[str, Any]]:
    """Return user skills and optionally built-in skills, optionally filtered by category."""
    from services.database_manager.database_helpers import fetch_all

    if include_builtin and category:
        rows = await fetch_all(
            """
            SELECT * FROM agent_skills
            WHERE (user_id = $1 OR is_builtin = true)
              AND (category = $2 OR category IS NULL AND $2 IS NULL)
            ORDER BY is_builtin ASC, name ASC
            """,
            user_id,
            category,
        )
    elif include_builtin:
        rows = await fetch_all(
            """
            SELECT * FROM agent_skills
            WHERE user_id = $1 OR is_builtin = true
            ORDER BY is_builtin ASC, name ASC
            """,
            user_id,
        )
    elif category:
        rows = await fetch_all(
            """
            SELECT * FROM agent_skills
            WHERE user_id = $1 AND (category = $2 OR category IS NULL AND $2 IS NULL)
            ORDER BY name ASC
            """,
            user_id,
            category,
        )
    else:
        rows = await fetch_all(
            "SELECT * FROM agent_skills WHERE user_id = $1 ORDER BY name ASC",
            user_id,
        )
    return [_row_to_skill(r) for r in rows]


async def get_skill(skill_id: str) -> Optional[Dict[str, Any]]:
    """Get a single skill by ID. Returns None if not found."""
    from services.database_manager.database_helpers import fetch_one

    row = await fetch_one(
        "SELECT * FROM agent_skills WHERE id = $1::uuid",
        skill_id,
    )
    return _row_to_skill(row) if row else None


async def get_skill_by_slug(slug: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Get the latest version of a skill by slug. Prefer user's version if user_id given."""
    from services.database_manager.database_helpers import fetch_one

    if user_id:
        row = await fetch_one(
            """
            SELECT * FROM agent_skills
            WHERE slug = $1 AND (user_id = $2 OR is_builtin = true)
            ORDER BY user_id ASC NULLS LAST, version DESC
            LIMIT 1
            """,
            slug,
            user_id,
        )
    else:
        row = await fetch_one(
            """
            SELECT * FROM agent_skills
            WHERE slug = $1 AND is_builtin = true
            ORDER BY version DESC
            LIMIT 1
            """,
            slug,
        )
    return _row_to_skill(row) if row else None


def _is_valid_uuid(s: str) -> bool:
    """Return True if s is a valid UUID string."""
    if not s or not isinstance(s, str):
        return False
    s = s.strip()
    if len(s) != 36:
        return False
    try:
        import uuid
        uuid.UUID(s)
        return True
    except (ValueError, TypeError):
        return False


async def get_skills_by_ids(skill_ids: List[str]) -> List[Dict[str, Any]]:
    """Batch fetch skills by IDs. Returns list in same order as requested; missing IDs omitted. Non-UUID entries are skipped."""
    from services.database_manager.database_helpers import fetch_all

    if not skill_ids:
        return []
    seen = set()
    ordered = []
    for sid in skill_ids:
        sid = (sid or "").strip()
        if not sid or sid in seen or not _is_valid_uuid(sid):
            continue
        seen.add(sid)
        ordered.append(sid)
    if not ordered:
        return []
    placeholders = ", ".join(f"${i + 1}::uuid" for i in range(len(ordered)))
    rows = await fetch_all(
        f"SELECT * FROM agent_skills WHERE id IN ({placeholders})",
        *ordered,
    )
    by_id = {str(r["id"]): _row_to_skill(r) for r in rows}
    return [by_id[sid] for sid in ordered if sid in by_id]


async def create_skill(
    user_id: str,
    name: str,
    slug: str,
    procedure: str,
    required_tools: Optional[List[str]] = None,
    optional_tools: Optional[List[str]] = None,
    description: Optional[str] = None,
    category: Optional[str] = None,
    inputs_schema: Optional[Dict[str, Any]] = None,
    outputs_schema: Optional[Dict[str, Any]] = None,
    examples: Optional[List[Any]] = None,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Create a user skill. Slug must be unique per user."""
    from services.database_manager.database_helpers import fetch_one

    slug_clean = (slug or "").strip().lower().replace(" ", "-")[:100]
    if not slug_clean:
        raise ValueError("slug is required")
    existing = await fetch_one(
        "SELECT id FROM agent_skills WHERE user_id = $1 AND slug = $2",
        user_id,
        slug_clean,
    )
    if existing:
        raise ValueError("Slug already in use for this user")

    row = await fetch_one(
        """
        INSERT INTO agent_skills (
            user_id, name, slug, description, category, procedure,
            required_tools, optional_tools, inputs_schema, outputs_schema,
            examples, tags, is_builtin, is_locked, version
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10::jsonb, $11::jsonb, $12, false, false, 1)
        RETURNING *
        """,
        user_id,
        (name or slug_clean).strip()[:255],
        slug_clean,
        (description or "")[:5000] or None,
        (category or "")[:100] or None,
        (procedure or "").strip(),
        list(required_tools or []),
        list(optional_tools or []),
        json.dumps(inputs_schema or {}),
        json.dumps(outputs_schema or {}),
        json.dumps(examples or []),
        list(tags or []),
    )
    skill_dict = _row_to_skill(row)
    try:
        from services.skill_vector_service import embed_skill
        await embed_skill(skill_dict)
    except Exception as e:
        logger.warning("embed_skill after create_skill failed: %s", e)
    return skill_dict


async def update_skill(
    skill_id: str,
    user_id: str,
    procedure: Optional[str] = None,
    required_tools: Optional[List[str]] = None,
    optional_tools: Optional[List[str]] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    category: Optional[str] = None,
    inputs_schema: Optional[Dict[str, Any]] = None,
    outputs_schema: Optional[Dict[str, Any]] = None,
    examples: Optional[List[Any]] = None,
    tags: Optional[List[str]] = None,
    improvement_rationale: Optional[str] = None,
    evidence_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Update a user skill by creating a new version. Parent skill id set to previous version."""
    from services.database_manager.database_helpers import fetch_one, execute

    row = await fetch_one(
        "SELECT * FROM agent_skills WHERE id = $1::uuid AND user_id = $2 AND NOT is_builtin",
        skill_id,
        user_id,
    )
    if not row:
        raise ValueError("Skill not found or not editable")
    prev_id = str(row["id"])
    version = (row.get("version") or 1) + 1
    name_val = (name or row["name"] or "").strip()[:255]
    procedure_val = (procedure if procedure is not None else row.get("procedure") or "").strip()
    required_tools_val = required_tools if required_tools is not None else list(row.get("required_tools") or [])
    optional_tools_val = optional_tools if optional_tools is not None else list(row.get("optional_tools") or [])
    description_val = description if description is not None else row.get("description")
    category_val = category if category is not None else row.get("category")
    inputs_schema_val = inputs_schema if inputs_schema is not None else _ensure_json_obj(row.get("inputs_schema"), {})
    outputs_schema_val = outputs_schema if outputs_schema is not None else _ensure_json_obj(row.get("outputs_schema"), {})
    examples_val = examples if examples is not None else _ensure_json_obj(row.get("examples"), [])
    tags_val = tags if tags is not None else list(row.get("tags") or [])
    improvement_rationale_val = improvement_rationale or None
    evidence_metadata_val = evidence_metadata if evidence_metadata is not None else _ensure_json_obj(row.get("evidence_metadata"), {})

    new_row = await fetch_one(
        """
        INSERT INTO agent_skills (
            user_id, name, slug, description, category, procedure,
            required_tools, optional_tools, inputs_schema, outputs_schema,
            examples, tags, is_builtin, is_locked, version, parent_skill_id,
            improvement_rationale, evidence_metadata
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10::jsonb, $11::jsonb, $12, false, false, $13, $14::uuid, $15, $16::jsonb)
        RETURNING *
        """,
        user_id,
        name_val,
        row["slug"],
        (description_val or "")[:5000] or None,
        (category_val or "")[:100] or None,
        procedure_val,
        required_tools_val,
        optional_tools_val,
        json.dumps(inputs_schema_val),
        json.dumps(outputs_schema_val),
        json.dumps(examples_val),
        tags_val,
        version,
        prev_id,
        improvement_rationale_val,
        json.dumps(evidence_metadata_val),
    )
    new_skill = _row_to_skill(new_row)
    try:
        from services.skill_vector_service import embed_skill, remove_skill_vector
        await remove_skill_vector(prev_id)
        await embed_skill(new_skill)
    except Exception as e:
        logger.warning("skill vector update failed: %s", e)
    return new_skill


async def delete_skill(skill_id: str, user_id: str) -> None:
    """Delete a user skill. Built-in skills cannot be deleted."""
    from services.database_manager.database_helpers import execute

    try:
        from services.skill_vector_service import remove_skill_vector
        await remove_skill_vector(skill_id)
    except Exception as e:
        logger.warning("remove_skill_vector before delete failed: %s", e)

    await execute(
        "DELETE FROM agent_skills WHERE id = $1::uuid AND user_id = $2 AND NOT is_builtin",
        skill_id,
        user_id,
    )


def resolve_skills_for_step(step_skill_ids: Optional[List[str]] = None) -> List[str]:
    """Dedupe step-level skill IDs. Skills are assigned only at step level."""
    seen = set()
    out = []
    for sid in step_skill_ids or []:
        s = (sid or "").strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


async def list_skill_versions(skill_id: str, user_id: str) -> List[Dict[str, Any]]:
    """Return version history for a skill (current first)."""
    from services.database_manager.database_helpers import fetch_all

    rows = await fetch_all(
        """
        SELECT * FROM agent_skills
        WHERE id = $1::uuid AND (user_id = $2 OR is_builtin = true)
        ORDER BY version DESC
        """,
        skill_id,
        user_id,
    )
    if not rows:
        return []
    slug = rows[0].get("slug")
    all_versions = await fetch_all(
        """
        SELECT * FROM agent_skills
        WHERE slug = $1 AND (user_id = $2 OR is_builtin = true)
        ORDER BY version DESC
        """,
        slug,
        user_id,
    )
    return [_row_to_skill(r) for r in all_versions]


async def revert_skill_to_version(skill_id: str, version_id: str, user_id: str) -> Dict[str, Any]:
    """Create a new version with content from a previous version (rollback)."""
    from services.database_manager.database_helpers import fetch_one

    target = await fetch_one(
        "SELECT * FROM agent_skills WHERE id = $1::uuid",
        version_id,
    )
    if not target:
        raise ValueError("Version not found")
    current = await fetch_one(
        "SELECT * FROM agent_skills WHERE id = $1::uuid AND user_id = $2 AND NOT is_builtin",
        skill_id,
        user_id,
    )
    if not current:
        raise ValueError("Skill not found or not editable")
    if target.get("slug") != current.get("slug"):
        raise ValueError("Version does not belong to this skill")
    return await update_skill(
        skill_id,
        user_id,
        procedure=target.get("procedure"),
        required_tools=list(target.get("required_tools") or []),
        optional_tools=list(target.get("optional_tools") or []),
        name=target.get("name"),
        description=target.get("description"),
        category=target.get("category"),
        inputs_schema=_ensure_json_obj(target.get("inputs_schema"), {}),
        outputs_schema=_ensure_json_obj(target.get("outputs_schema"), {}),
        examples=_ensure_json_obj(target.get("examples"), []),
        tags=list(target.get("tags") or []),
        improvement_rationale="Reverted to version " + str(target.get("version")),
    )


def _get_builtin_skills() -> List[Dict[str, Any]]:
    """Return the full list of built-in skill definitions (Python dict skills + retained procedural)."""
    return [
        # ---- Automation (20) ----
        {
            "slug": "weather",
            "name": "Weather",
            "description": "Get current weather conditions, forecasts, and historical weather data for any location.",
            "category": "automation",
            "procedure": (
                "You are a weather assistant. ALWAYS use get_weather to fetch real weather data - never make up weather information. "
                "Parameters: location (required - city name, ZIP code, or place name like 'New York' or '14532'), "
                "data_types (comma-separated: 'current' for now, 'forecast' for upcoming days, 'history' for past weather; default 'current'), "
                "date_str (for history only: 'YYYY-MM-DD' for a specific day, 'YYYY-MM' for monthly average, or 'YYYY-MM to YYYY-MM' for date ranges up to 24 months). "
                "If the user does not specify a location, ask them for one before calling the tool. "
                "Present temperatures in both Fahrenheit and Celsius. Include practical recommendations based on conditions."
            ),
            "required_tools": ["get_weather"],
            "optional_tools": [],
            "tags": ["weather", "temperature", "forecast", "rain", "snow", "sunny", "cloudy"],
            "evidence_metadata": {"engine_type": "automation"},
        },
        {
            "slug": "email",
            "name": "Email",
            "description": "Read, search, send, and manage email. Reply threading, attachments, and connection routing.",
            "category": "email",
            "procedure": (
                "You are an email assistant. Use the available tools for all operations. "
                "list_emails: folder (inbox/sent/drafts), top (limit), unread_only (bool). search_emails: query, top. "
                "get_email_thread: conversation_id from a previous list. get_email_statistics: inbox/unread counts. "
                "For sending: ALWAYS call send_email with confirmed=False first to show the draft; only call with confirmed=True after the user explicitly approves (e.g. yes, send, approve). "
                "For replying: ALWAYS call reply_to_email with confirmed=False first to show the draft; only call with confirmed=True after the user approves. "
                "send_email: to (comma-separated), subject, body, confirmed (bool, default False). "
                "reply_to_email: message_id from thread/list, body, reply_all (bool), confirmed (bool, default False). "
                "When replying, use the same connection_id as the source message. Attachments require separate upload steps before sending. "
                "For new threads, use the connection_id from the agent's configured email binding. Include clear subject lines and avoid stripping reply threading headers."
            ),
            "required_tools": ["list_emails", "search_emails", "get_email_thread", "get_email_statistics", "send_email", "reply_to_email"],
            "optional_tools": [],
            "tags": ["email", "inbox", "mail", "send email", "reply", "read my email", "check email", "search email", "unread", "draft", "compose"],
            "evidence_metadata": {"engine_type": "automation"},
        },
        {
            "slug": "calendar",
            "name": "Calendar",
            "description": "List calendars, view events in a date range, get event details, create/update/delete events (O365).",
            "category": "calendar",
            "procedure": (
                "You are a calendar assistant. Use the available tools for all operations. "
                "list_calendars: list user's calendars. get_calendar_events: start_datetime and end_datetime required (ISO 8601); optional calendar_id, top. "
                "get_event_by_id: event_id from a previous list. "
                "For create_event: ALWAYS call with confirmed=False first to show the draft; only call with confirmed=True after the user explicitly approves. "
                "For update_event and delete_event: ALWAYS call with confirmed=False first; only call with confirmed=True after the user approves. "
                "create_event: subject, start_datetime, end_datetime (ISO 8601), confirmed (bool), optional location, body, attendee_emails (comma-separated), is_all_day. "
                "update_event: event_id, confirmed (bool), optional subject, start_datetime, end_datetime, location, body, attendee_emails, is_all_day. delete_event: event_id, confirmed (bool)."
            ),
            "required_tools": ["list_calendars", "get_calendar_events", "get_event_by_id", "create_event", "update_event", "delete_event"],
            "optional_tools": [],
            "tags": ["calendar", "schedule", "meeting", "event", "appointment", "create event", "delete event", "update event"],
            "evidence_metadata": {"engine_type": "automation"},
        },
        {
            "slug": "contacts",
            "name": "Contacts",
            "description": "List, get, create, update, and delete contacts (O365 and org-mode unified when include_org).",
            "category": "contacts",
            "procedure": (
                "You are a contacts assistant. Use the available tools for all operations. "
                "list_contacts: list contacts (O365 + org-mode when include_org=True); optional folder_id, top. "
                "get_contact_by_id: get a single contact by contact_id (from a previous list). "
                "create_contact: display_name, given_name, surname, optional company_name, job_title, birthday, notes, email_addresses, phone_numbers. "
                "update_contact: contact_id (required), optional display_name, given_name, surname, company_name, job_title, birthday, notes. delete_contact: contact_id (required)."
            ),
            "required_tools": ["list_contacts", "get_contact_by_id", "create_contact", "update_contact", "delete_contact"],
            "optional_tools": [],
            "tags": ["contact", "contacts", "people", "address book", "look up contact", "create contact", "add contact", "update contact", "delete contact"],
            "evidence_metadata": {"engine_type": "automation"},
        },
        {
            "slug": "navigation",
            "name": "Navigation",
            "description": "Manage saved locations, plan routes, and get directions.",
            "category": "navigation",
            "procedure": (
                "You are a navigation assistant. Multi-step flow: (1) list_locations to see saved locations and their IDs; create_location to add a new one (name, address). "
                "(2) compute_route with from_location_id and to_location_id (from list_locations), or use coordinates string. profile: driving, walking, cycling. "
                "save_route to save a computed route (pass waypoints, geometry, steps, distance_meters, duration_seconds from compute_route). "
                "list_saved_routes to list saved routes. delete_location to remove a location by location_id."
            ),
            "required_tools": ["create_location", "list_locations", "delete_location", "compute_route", "save_route", "list_saved_routes"],
            "optional_tools": [],
            "tags": ["create location", "list locations", "saved locations", "delete location", "route", "directions", "navigate", "map", "turn by turn", "save route"],
            "evidence_metadata": {"engine_type": "automation"},
        },
        {
            "slug": "rss",
            "name": "RSS",
            "description": "Subscribe to and manage RSS feeds; list feeds and fetch recent items from them. Not for finding articles on the web (use research).",
            "category": "rss",
            "procedure": (
                "You are an RSS feed assistant. add_rss_feed: feed_url (required), feed_name, category, is_global (bool). "
                "list_rss_feeds: scope 'user' or 'global'. refresh_rss_feed: feed_name or feed_id from list to trigger refresh."
            ),
            "required_tools": ["add_rss_feed", "list_rss_feeds", "refresh_rss_feed"],
            "optional_tools": [],
            "tags": ["rss", "feed", "feeds", "subscribe", "news", "articles"],
            "evidence_metadata": {"engine_type": "automation"},
        },
        {
            "slug": "org-capture",
            "name": "Org Capture",
            "description": "Quick capture items to the org-mode inbox. Use for 'capture to inbox', 'add to inbox', 'quick capture'.",
            "category": "org",
            "procedure": (
                "You help capture items to the user's org-mode inbox. "
                "ALWAYS call add_org_inbox_item to add each item - never just describe what you would do. "
                "\n\n"
                "SCOPE: Only capture items requested in the CURRENT user message (the last message). "
                "You may see conversation history for context — it helps interpret references like 'capture that' "
                "or 'add what you just said'. NEVER independently scan history for additional items to capture. "
                "If the current message does not ask to capture something, do not capture anything from history."
                "\n\n"
                "PRIOR STEP CONTENT: When the user message includes 'Content to capture (from a previous step):', "
                "that content is the main text to capture. Use it as the inbox item: use a concise title/summary for the heading "
                "if the content is long, and put the full content or a clear summary in the text parameter. "
                "Do not capture only the 'User request' line; capture the prior-step content. "
                "\n\n"
                "ORG FORMATTING: When capturing research, articles, or multi-paragraph content, convert the content to org-mode "
                "syntax so the inbox stays neat and readable. Output a single text value: first line = short title (used as the "
                "headline), then a blank line, then the body in org format. Conversion rules: "
                "Markdown ## H2 → ** H2, ### H3 → *** H3 (subheadings in body). "
                "Markdown **bold** → *bold*, *italic* → /italic/. "
                "Tables: use org table form | col1 | col2 | with a |-| separator row after the header. "
                "Lists: use - for unordered, 1. 2. for ordered; indent continuation lines with 2 spaces. "
                "Links: use [[url][description]] or plain URL. "
                "Keep paragraphs separated by blank lines; do not leave raw markdown (e.g. ## or **) in the captured text. "
                "\n\n"
                "CALL RULES: Call add_org_inbox_item exactly once per distinct inbox entry. "
                "One user-provided item → one tool call. Multiple items in one message → one tool call per item. "
                "Never call the tool multiple times with the same text for the same logical entry. "
                "\n\n"
                "REPLY RULES: After calling the tool, reply with a short confirmation only (e.g. 'Added X to your inbox.'). "
                "Do not mention deduplication, 'you provided it twice', 'I captured it just once', or how many times the tool was called. "
                "\n\n"
                "KIND: Use kind='todo' for explicit tasks or short action items (e.g. 'remind me to X', '* TODO X'). "
                "Use kind='note' for longer-form notes, ideas, or when the user says 'note' or 'capture this' without implying a task. "
                "Use kind='checkbox' for '- [ ] Item' or checklist items; kind='event' for calendar; kind='contact' for contacts. "
                "Org headlines: '* TODO Title' or '* Title' (note); extract title as text. Strip prefixes like 'capture to inbox:', 'add to inbox:'. "
                "\n\n"
                "TAGS: If the user asks to tag the item (e.g. 'tag with work', 'with tag urgent', 'add tag @project'), "
                "pass those tags in the tags parameter (comma-separated string or list). Never put :tag: in the title text — "
                "always use the tags parameter so tags are stored correctly and not duplicated. "
                "\n\n"
                "Parameters: text (required; title only, no tags or priority in the text), kind ('todo', 'note', 'checkbox', 'event', 'contact'; default 'todo'), "
                "schedule (optional org timestamp like '<2026-02-05 Thu>'), tags (optional; use when user specifies tags)."
            ),
            "required_tools": ["add_org_inbox_item"],
            "optional_tools": [],
            "tags": ["capture", "inbox", "capture to inbox", "add to inbox", "quick capture"],
            "evidence_metadata": {"engine_type": "automation", "stateless": True},
        },
        {
            "slug": "org-journal",
            "name": "Org Journal",
            "description": "Read and search the user's org-mode journal (diary entries). Use for 'what did I write', 'read my journal', 'search journal'.",
            "category": "org",
            "procedure": (
                "You help the user read and search their org-mode journal. ALWAYS call the appropriate tool - never make up journal content. "
                "get_journal_entry for a single date (date='today' or YYYY-MM-DD). get_journal_entries for a date range with full content. "
                "list_journal_entries for counts or metadata only. search_journal for keyword search. "
                "When the user does not specify a date, use 'today' for get_journal_entry. For ranges, use start_date and end_date in YYYY-MM-DD format."
            ),
            "required_tools": ["get_journal_entry", "get_journal_entries", "list_journal_entries", "search_journal"],
            "optional_tools": [],
            "tags": ["journal", "journal entry", "read my journal", "search my journal", "diary", "journal entries"],
            "evidence_metadata": {"engine_type": "automation"},
        },
        {
            "slug": "agent-run-history",
            "name": "Agent Run History",
            "description": "Report on this agent's or another agent's recent run history, status, and performance.",
            "category": "agent",
            "procedure": (
                "You report on agent execution (run) history. ALWAYS call get_agent_run_history to fetch real data - never make up run counts, dates, or statuses. "
                "When the user asks about 'your' history or 'this agent', omit agent_profile_id so the tool returns this agent's runs. "
                "Parameters: agent_profile_id (omit for self), limit (default 10, max 50), status ('completed', 'failed', 'running' to filter), start_date and end_date (YYYY-MM-DD). "
                "Summarize patterns: e.g. 'You have run 47 times this month with a 94% success rate.' Use status='failed' when the user asks about failures."
            ),
            "required_tools": ["get_agent_run_history"],
            "optional_tools": [],
            "tags": ["last run", "run history", "when did you last run", "agent log", "execution history", "recent runs"],
            "evidence_metadata": {"engine_type": "automation"},
        },
        {
            "slug": "save-user-fact",
            "name": "Save User Fact",
            "description": "Save a fact about the user to their persistent fact store for future conversations.",
            "category": "memory",
            "procedure": (
                "You help users store facts about themselves for future conversations. When the user shares information they want remembered "
                "(e.g. 'remember I'm a vegetarian', 'save that I prefer Python'), ALWAYS call save_user_fact to store it. "
                "fact_key: short snake_case label (e.g. job_title, preferred_language, dietary_restriction, city). value: the actual information. "
                "category: 'work', 'preferences', 'personal', or 'general'. After saving, confirm naturally (e.g. 'Got it, I'll remember that [value].')."
            ),
            "required_tools": ["save_user_fact"],
            "optional_tools": [],
            "tags": ["remember", "save fact", "note that", "store preference", "remember that", "save that"],
            "evidence_metadata": {"engine_type": "automation"},
        },
        {
            "slug": "org-content",
            "name": "Org Content",
            "description": "Read-only queries about org-mode files, todos, and tasks.",
            "category": "org",
            "procedure": (
                "You help the user query their org-mode file. The full file content is included in the user message between === FILE ... === and === END FILE === markers. READ IT CAREFULLY before answering. "
                "You have tools: parse_org_structure to get the full outline; search_org_headings with search_term to find headings or content; get_org_statistics for counts and completion rate. "
                "When the user says 'today', use the YYYY-MM-DD date from the datetime context as search_term. "
                "Base your answers ONLY on the actual file content provided. Never fabricate data. Read-only; do not modify the file."
            ),
            "required_tools": ["parse_org_structure", "search_org_headings", "get_org_statistics"],
            "optional_tools": [],
            "tags": ["org", "org-mode", "todo", "task", "project", "show", "list", "find", "tagged", "tag", "org file"],
            "evidence_metadata": {"engine_type": "automation", "requires_editor": True, "editor_types": ["org"], "editor_preference": "require", "context_boost": 15},
        },
        {
            "slug": "task-management",
            "name": "Task Management",
            "description": "List, create, update, toggle, delete, or archive todos across any org file or inbox.",
            "category": "org",
            "procedure": (
                "You help manage the user's todos across all org files. Org-mode todos: Put tags in the tags parameter (or add_tags/remove_tags for update_todo). "
                "Put priority in the priority parameter (A, B, or C). Do not embed :tag: or [#A] in the title text. "
                "Apply TODO updates (state, tags, priority) with update_todo or toggle_todo only - do not use propose_document_edit or patch_file; the todo API applies directly. Effort and category are not settable via these tools. "
                "list_todos: scope 'all', 'inbox', or a file path; optional states, tags, query, limit. Results include file_path, line_number (0-based), heading, todo_state, tags, scheduled, deadline. "
                "create_todo: text = title only (required). Use tags parameter for tags and priority parameter for A/B/C. Optional: body, deadline, scheduled, file_path, insert_after_line_number. "
                "update_todo: file_path, line_number (0-based). Optional: new_state, new_text, add_tags/remove_tags, priority, scheduled, deadline, new_body. "
                "toggle_todo: file_path, line_number to toggle TODO <-> DONE. delete_todo: file_path, line_number. "
                "archive_done: single entry (file_path, line_number) or bulk (omit line_number, provide file_path or omit for inbox). Use list_todos first when the user asks to see or manage todos."
            ),
            "required_tools": ["list_todos", "create_todo", "update_todo", "toggle_todo", "delete_todo", "archive_done"],
            "optional_tools": [],
            "tags": ["todo", "todos", "task", "tasks", "list my todos", "create todo", "add todo", "mark done", "toggle todo", "update todo", "delete todo", "archive done", "inbox", "org"],
            "evidence_metadata": {"engine_type": "automation"},
        },
        {
            "slug": "image-generation",
            "name": "Image Generation",
            "description": "Generate images from text descriptions. Use for 'draw', 'create image', 'make a picture'.",
            "category": "image",
            "procedure": (
                "You are an image generation assistant. Use generate_image: prompt (required), size (e.g. 1024x1024, 512x512), num_images (1-4), optional negative_prompt, model. "
                "Set check_reference_first=True to return a reference image from the user's library if one exists for the prompt/object name before generating."
            ),
            "required_tools": ["generate_image"],
            "optional_tools": [],
            "tags": ["create image", "generate image", "draw", "visualize", "image", "picture", "photo", "create a picture", "make an image"],
            "evidence_metadata": {"engine_type": "automation"},
        },
        {
            "slug": "reference",
            "name": "Reference",
            "description": "Query and analyze reference documents, journals, and logs; run calculations and visualizations.",
            "category": "reference",
            "procedure": (
                "You help with reference documents, journals, and logs. The file content is included in the user message between === FILE ... === and === END FILE === markers. "
                "Read it carefully and base your answers on the actual content. You can run calculations (calculate_expression, evaluate_formula), convert units (convert_units), and create visualizations (create_chart). "
                "Use the available tools. Never fabricate data that is not in the file."
            ),
            "required_tools": ["calculate_expression", "evaluate_formula", "convert_units", "create_chart"],
            "optional_tools": [],
            "tags": ["journal", "log", "record", "tracking", "diary", "graph", "chart", "visualize", "calculate", "calculation", "compute", "math", "formula", "convert units", "unit conversion"],
            "evidence_metadata": {"engine_type": "automation", "requires_editor": True, "editor_types": ["reference"], "context_boost": 20},
        },
        {
            "slug": "document-creator",
            "name": "Document Creator",
            "description": "Create new files or documents in a folder (not for editing the current open document; use editor skills for that).",
            "category": "documents",
            "procedure": (
                "You are a document creation assistant. You create files and folders in the user's document tree. "
                "WORKFLOW: (1) If the user specifies a folder by name, call list_folders first to find its folder_id or verify it exists. "
                "(2) If the folder does not exist, call create_user_folder to create it first. "
                "(3) Call create_typed_document with content, filename, and folder_path (e.g. 'Reference'). "
                "PARAMETERS for create_typed_document: filename (with extension), content (document body - clean markdown with a title heading), folder_path (auto-create if missing). "
                "If context from a prior step is available (prior_step_*_response keys), use it as the document body. If the user doesn't specify a filename, generate a descriptive one from the content."
            ),
            "required_tools": ["list_folders", "create_typed_document", "create_user_folder"],
            "optional_tools": [],
            "tags": ["create file", "create document", "save to folder", "new file", "put in folder", "reference file", "save as"],
            "evidence_metadata": {"engine_type": "automation"},
        },
        # ---- Research / Search ----
        {
            "slug": "document-search",
            "name": "Document Search",
            "description": "Search the knowledge base: documents, segments, and conversation cache. Use for factual lookups, 'do we have', and collection searches (comics, photos).",
            "category": "search",
            "procedure": (
                "Comprehensive local document and image search. Start with limit=10, scope=my_docs. If fewer than 3 relevant results, call enhance_query and retry with the enhanced query. "
                "Check search_conversation_cache for prior context on this topic. Escalate to team_docs or global scope if needed. "
                "Use file_types filter when the user asks for a specific document type. Covers factual lookups ('what is X'), 'do we have' queries, and image/comic/photo collection searches."
            ),
            "required_tools": ["search_documents", "enhance_query", "search_conversation_cache"],
            "optional_tools": [],
            "tags": ["search", "find", "document", "look up", "do we have", "find me", "in our collection", "comic", "photo", "image", "knowledge base"],
            "evidence_metadata": {"engine_type": "automation"},
        },
        {
            "slug": "web-search",
            "name": "Web Search",
            "description": "Multi-query web search with synthesis and source citation. Use for online lookups, fact-checking, and research.",
            "category": "search",
            "procedure": (
                "Multi-query web search with synthesis and source citation. Formulate 2-4 focused queries covering different aspects of the topic. "
                "Use search_web with limit 10-15. For key results needing full content, use crawl_web_content with exact URLs. "
                "Synthesize findings across all queries and crawled pages; cite sources (title, URL). Note when sources disagree or when information is uncertain or missing. "
                "Run follow-up searches to fill gaps. For fact-checking and verification, cross-reference multiple sources."
            ),
            "required_tools": ["search_web", "crawl_web_content"],
            "optional_tools": [],
            "tags": ["web search", "search the web", "look up online", "find online", "research", "fact check", "verify", "what is", "how to", "investigate"],
            "evidence_metadata": {"engine_type": "automation"},
        },
        {
            "slug": "security-analysis",
            "name": "Security Analysis",
            "description": "Security and vulnerability scanning for URLs and websites.",
            "category": "research",
            "procedure": (
                "Security and vulnerability scanning for URLs and websites. Use search_web to find security-related information and crawl_web_content to fetch and analyze page content, headers, and exposed resources. "
                "Report on security headers, exposed files, and common vulnerability indicators. Do not perform active exploitation."
            ),
            "required_tools": ["search_web", "crawl_web_content"],
            "optional_tools": [],
            "tags": ["security scan", "vulnerability scan", "security analysis", "check for vulnerabilities", "security audit", "pen test", "security assessment", "website security"],
            "evidence_metadata": {"engine_type": "research"},
        },
        {
            "slug": "web-crawl",
            "name": "Web Crawl",
            "description": "Crawl websites to extract content: single URL or deeper site ingestion with multiple pages.",
            "category": "research",
            "procedure": (
                "Crawl websites to extract content. For a single URL, call crawl_web_content directly. "
                "For deeper site ingestion (multiple pages, recursive crawl), use search_web to discover additional page URLs within the domain, then crawl each. "
                "Use for one-off content extraction, site scraping, or downloading website content for processing or storage."
            ),
            "required_tools": ["crawl_web_content"],
            "optional_tools": ["search_web"],
            "tags": ["crawl", "crawl site", "crawl website", "ingest", "scrape", "download website", "url", "domain crawl", "capture website"],
            "evidence_metadata": {"engine_type": "research"},
        },
        # ---- Knowledge graph ----
        {
            "slug": "knowledge-graph-traversal",
            "name": "Knowledge Graph Traversal",
            "description": "Entity search and relationship patterns",
            "category": "knowledge-graph",
            "procedure": (
                "Search entities by name or type first. Use relationship depth to expand connections. "
                "When resolving identities across sources, prefer entity_search then follow relationship types. "
                "Limit relationship depth to 2 unless the user asks for deep traversal."
            ),
            "required_tools": ["search_knowledge_graph", "get_entity_details"],
            "optional_tools": [],
            "tags": [],
            "evidence_metadata": {"engine_type": "automation"},
        },
        # ---- Data workspace ----
        {
            "slug": "data-workspace",
            "name": "Data Workspace",
            "description": "Query and inspect tabular data in Data Workspaces. List workspaces, get schema, run SQL or natural language queries.",
            "category": "data",
            "procedure": (
                "You help the user query tabular data in Data Workspaces. "
                "First call list_data_workspaces to see available workspaces and their IDs. "
                "Use get_workspace_schema with workspace_id to inspect tables and columns. "
                "Then use query_data_workspace with workspace_id and either a SQL query or a natural language question. "
                "Present results in a clear table or summary. For large result sets, summarize or paginate."
            ),
            "required_tools": ["query_data_workspace", "list_data_workspaces", "get_workspace_schema"],
            "optional_tools": [],
            "tags": ["data workspace", "query data", "sql", "tabular", "dataset", "spreadsheet", "table data"],
            "evidence_metadata": {"engine_type": "automation"},
        },
        {
            "slug": "notifications",
            "name": "Notifications",
            "description": "Send messages to configured channels: in-app, Telegram, Discord, or email.",
            "category": "messaging",
            "procedure": (
                "You help send notifications to the user or their configured channels. "
                "Use send_channel_message: provide channel_id or channel type (e.g. in_app, telegram, discord), subject (optional), and body. "
                "Confirm with the user before sending if the request is ambiguous. After sending, confirm delivery briefly."
            ),
            "required_tools": ["send_channel_message"],
            "optional_tools": [],
            "tags": ["notify", "notification", "send message", "alert", "telegram", "discord", "in-app message"],
            "evidence_metadata": {"engine_type": "automation"},
        },
        {
            "slug": "file-management",
            "name": "File Management",
            "description": "Create files and folders, and patch or append to existing files. Use for saving outputs and organizing content.",
            "category": "documents",
            "procedure": (
                "You help create and organize files. list_folders to see folder structure. create_user_folder to create a folder. "
                "create_typed_document: use for new documents with a filename, content, and folder_path (creates folder if missing). "
                "create_user_file: use for raw files (e.g. plain text, CSV) with filename, content, and folder_path. "
                "patch_file: apply a text patch to an existing file (use when the user wants to edit or replace a section). append_to_file: add content to the end of a file. "
                "Always confirm the target path or folder with the user when it is ambiguous."
            ),
            "required_tools": ["create_typed_document", "create_user_file", "create_user_folder", "list_folders", "patch_file", "append_to_file"],
            "optional_tools": [],
            "tags": ["create file", "create folder", "save file", "append to file", "patch file", "file management", "organize files"],
            "evidence_metadata": {"engine_type": "automation"},
        },
        {
            "slug": "browser-automation",
            "name": "Browser Automation",
            "description": "Control a browser: navigate, click, fill forms, extract content, take screenshots.",
            "category": "automation",
            "procedure": (
                "You help automate browser tasks. browser_navigate: open a URL. browser_click: click an element (selector or coordinates). "
                "browser_fill: fill form fields by selector. browser_extract: extract text or attributes from the page. browser_screenshot: capture a screenshot. "
                "Use browser_inspect when you need to discover selectors or page structure. For multi-step flows, navigate first, then click or fill as needed. "
                "Confirm destructive or high-impact actions with the user when appropriate."
            ),
            "required_tools": ["browser_navigate", "browser_click", "browser_fill", "browser_extract", "browser_screenshot"],
            "optional_tools": [],
            "tags": ["browser", "automation", "navigate", "click", "fill form", "screenshot", "scrape", "extract content"],
            "evidence_metadata": {"engine_type": "automation"},
        },
        {
            "slug": "chart-visualization",
            "name": "Chart Visualization",
            "description": "Create charts (bar, line, pie, scatter) from data. Use for visualizing numbers or comparisons.",
            "category": "reference",
            "procedure": (
                "You help create charts from data. Use create_chart with chart_type (bar, line, pie, scatter), title, and data. "
                "Data format: provide series and values (e.g. labels and numbers). Specify axis labels and units when relevant. "
                "Choose chart_type by task: bar for categories, line for trends over time, pie for proportions, scatter for correlations. "
                "Return the chart image or embed code to the user."
            ),
            "required_tools": ["create_chart"],
            "optional_tools": [],
            "tags": ["chart", "graph", "visualize", "bar chart", "line chart", "pie chart", "scatter", "plot", "visualization"],
            "evidence_metadata": {"engine_type": "automation"},
        },
    ]


async def seed_builtin_skills() -> None:
    """Upsert built-in skills (idempotent, keyed by slug). Call at backend startup."""
    from services.database_manager.database_helpers import fetch_one, execute

    retired_slugs = [
        "org-todo-management",
        "email-composition",
        "multi-source-research",
        "dictionary",
        "entertainment",
        "image-description",
        "agent-factory-builder",
        "fiction-editing",
        "outline-editing",
        "nonfiction-outline-editing",
        "character-development",
        "rules-editing",
        "style-editing",
        "series-editing",
        "electronics",
        "general-project",
        "podcast-script",
        "proofreading",
        "article-writing",
        "content-analysis",
        "research",
        "knowledge-builder",
        "effective-document-search",
        "thorough-web-search",
        "site-crawl",
        "website-crawler",
        "help",
    ]
    for slug in retired_slugs:
        await execute(
            "DELETE FROM agent_skills WHERE slug = $1 AND is_builtin = true",
            slug,
        )

    builtins = _get_builtin_skills()
    for s in builtins:
        existing = await fetch_one(
            "SELECT id, procedure FROM agent_skills WHERE slug = $1 AND is_builtin = true",
            s["slug"],
        )
        tags = s.get("tags") or []
        evidence_metadata = s.get("evidence_metadata") or {}
        if existing:
            await execute(
                """
                UPDATE agent_skills SET
                    name = $2, description = $3, category = $4, procedure = $5,
                    required_tools = $6, optional_tools = $7, tags = $8, evidence_metadata = $9::jsonb, updated_at = NOW()
                WHERE id = $1
                """,
                existing["id"],
                s["name"],
                s["description"],
                s["category"],
                s["procedure"],
                s.get("required_tools", []),
                s.get("optional_tools", []),
                tags,
                json.dumps(evidence_metadata),
            )
            logger.debug("Updated built-in skill: %s", s["slug"])
        else:
            await execute(
                """
                INSERT INTO agent_skills (
                    user_id, name, slug, description, category, procedure,
                    required_tools, optional_tools, tags, evidence_metadata, is_builtin, is_locked, version
                ) VALUES (NULL, $1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, true, true, 1)
                """,
                s["name"],
                s["slug"],
                s["description"],
                s["category"],
                s["procedure"],
                s.get("required_tools", []),
                s.get("optional_tools", []),
                tags,
                json.dumps(evidence_metadata),
            )
            logger.info("Seeded built-in skill: %s", s["slug"])

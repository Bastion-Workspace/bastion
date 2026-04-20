"""Capability manifest text for dynamic skill discovery (LLM system prompt)."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from orchestrator.engines.tool_resolution import ToolResolutionResult

logger = logging.getLogger(__name__)

FLAT_CATALOG_THRESHOLD = 150
CATEGORIZED_FULL_THRESHOLD = 300
TOP_PER_CATEGORY = 5


def _build_skill_catalog_lines(
    summaries: List[Dict[str, Any]],
    active_connection_types: Optional[Set[str]] = None,
) -> List[str]:
    """
    Build compact skill catalog lines from summaries.

    Only ``is_core`` skills are shown in the catalog.  Non-core skills are
    discoverable via ``search_and_acquire_skills`` at runtime.

    Tiers:
      <150 skills  -> flat list, one line per skill
      150-300      -> grouped by category, all skills listed
      300+         -> grouped by category, top N per category + overflow note
    """
    filtered: List[Dict[str, Any]] = []
    for s in summaries:
        if not s.get("is_core", False):
            continue
        req_types = s.get("required_connection_types") or []
        if req_types and active_connection_types is not None:
            if not any(rt in active_connection_types for rt in req_types):
                continue
        filtered.append(s)

    if not filtered:
        return ["No skills available."]

    total = len(filtered)

    if total < FLAT_CATALOG_THRESHOLD:
        return _flat_catalog(filtered)
    elif total < CATEGORIZED_FULL_THRESHOLD:
        return _categorized_catalog(filtered, truncate=False)
    else:
        return _categorized_catalog(filtered, truncate=True)


def _skill_line(s: Dict[str, Any]) -> str:
    slug = s.get("slug") or s.get("name") or "unknown"
    desc = (s.get("description") or "").strip()
    tags = s.get("tags") or []
    tag_str = f" [{', '.join(str(t) for t in tags[:4])}]" if tags else ""
    line = f'- acquire_skill("{slug}")'
    if desc:
        line += f": {desc}{tag_str}"
    return line


def _flat_catalog(skills: List[Dict[str, Any]]) -> List[str]:
    lines = [f"Available skills ({len(skills)}):"]
    for s in skills:
        lines.append(_skill_line(s))
    return lines


def _categorized_catalog(skills: List[Dict[str, Any]], truncate: bool = False) -> List[str]:
    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in skills:
        cat = s.get("category") or "general"
        by_category[cat].append(s)

    lines = [f"Available skills ({len(skills)}, by category):"]
    for cat in sorted(by_category.keys()):
        cat_skills = by_category[cat]
        lines.append(f"\n**{cat}** ({len(cat_skills)}):")
        show = cat_skills if not truncate else cat_skills[:TOP_PER_CATEGORY]
        for s in show:
            lines.append(_skill_line(s))
        if truncate and len(cat_skills) > TOP_PER_CATEGORY:
            remaining = len(cat_skills) - TOP_PER_CATEGORY
            lines.append(
                f"  ... and {remaining} more (use search_and_acquire_skills to explore this category)"
            )
    return lines


async def build_capability_manifest(
    user_id: str,
    metadata: Optional[Dict[str, Any]],
    resolved: Optional["ToolResolutionResult"] = None,
    include_skill_catalog: bool = False,
) -> str:
    """
    Human-readable capability block for dynamic discovery.
    When resolved is provided, includes categories from the step's bound tools.
    When include_skill_catalog is True, fetches and injects a compressed skill catalog.
    """
    from orchestrator.engines.provider_capability_registry import MULTI_ACCOUNT_MANIFEST_TYPES
    from orchestrator.engines.tool_resolution import parse_connections_map

    cmap = parse_connections_map(metadata)
    active_conn_types: Optional[Set[str]] = set(cmap.keys()) if cmap else None

    lines_active: List[str] = []
    if cmap:
        lines_active.append("Active accounts:")
        multi_account_hints: List[str] = []
        for ctype in sorted(cmap.keys()):
            entries = cmap[ctype] or []
            if len(entries) > 1 and ctype in MULTI_ACCOUNT_MANIFEST_TYPES:
                labels = []
                for ent in entries:
                    if not isinstance(ent, dict):
                        continue
                    cid = ent.get("id")
                    label = ent.get("label") or f"connection {cid}"
                    labels.append(f"{label} (connection {cid})")
                if labels:
                    example_tool = {
                        "email": "list_emails",
                        "calendar": "get_calendar_events",
                        "contacts": "list_contacts",
                        "todo": "list_todo_lists",
                        "files": "list_drive_items",
                        "onenote": "list_onenote_notebooks",
                        "planner": "list_planner_plans",
                        "devops": "list_devops_projects",
                    }.get(ctype, "tool_name")
                    multi_account_hints.append(
                        f"You have {len(entries)} {ctype} account(s): {', '.join(labels)}. "
                        f"Use the scoped tool form shown in your tool list (e.g. {example_tool}_<id>) for the correct account."
                    )
            for ent in entries:
                if not isinstance(ent, dict):
                    continue
                cid = ent.get("id")
                label = ent.get("label") or ""
                prov = ent.get("provider")
                if prov:
                    lines_active.append(f"- {ctype}: {label} (connection {cid}, provider {prov})")
                else:
                    lines_active.append(f"- {ctype}: {label} (connection {cid})")
        lines_active.append(
            "Tools for these accounts are often named with a connection id prefix (email:/calendar:/contacts:/todo:/"
            "files:/onenote:/planner:/github:/gitea: plus connection id); the bound tool list uses a sanitized form."
        )
        if multi_account_hints:
            lines_active.append("")
            lines_active.extend(multi_account_hints)
    else:
        lines_active.append(
            "No external accounts (email, calendar, code platform, etc.) are connected for this user."
        )

    ws_line = "Data workspaces: use list_data_workspaces / query_data_workspace when those tools are available."
    try:
        from orchestrator.backend_tool_client import get_backend_tool_client

        client = await get_backend_tool_client()
        dw = await client.list_data_workspaces(user_id)
        n = len(dw.get("workspaces") or [])
        ws_line = f"Data workspaces: {n} workspace(s); use list_data_workspaces and query_data_workspace."
    except Exception:
        pass

    skill_catalog_lines: List[str] = []
    if include_skill_catalog:
        try:
            from orchestrator.backend_tool_client import get_backend_tool_client

            client = await get_backend_tool_client()
            summaries = await client.list_skill_summaries(user_id)
            skill_catalog_lines = _build_skill_catalog_lines(summaries, active_conn_types)
        except Exception as e:
            logger.warning("Skill catalog fetch failed: %s", e)

    if include_skill_catalog and skill_catalog_lines:
        acquire_hint = (
            "These are your **core skills**. Load any by calling `acquire_skill(slug)`.\n"
            "**Important: the slugs listed below are NOT callable tool names — calling them directly will fail. "
            'You MUST call `acquire_skill("document-search")` (or `search_and_acquire_skills(query)`) '
            "first to load the actual tools.**\n"
            "Specialized skills beyond this list are discoverable via `search_and_acquire_skills` with a capability description.\n"
            "**Strategy:** For factual or reference questions, call `acquire_skill(\"document-search\")` first "
            "to check the local knowledge base before falling back to `acquire_skill(\"web-search\")`. "
            "Local results are faster and may already contain the answer.\n"
            "**URLs:** If the user already supplied one or more explicit http(s) page links and wants those pages read or summarized, "
            'call `acquire_skill(\"web-crawl\")` first and crawl those URLs directly; use `web-search` when you still need search to discover pages.'
        )
    else:
        acquire_hint = (
            "To load specialized procedures and extra tools during this step, call **search_and_acquire_skills** with a short query "
            '(for example: "send email", "this week\'s calendar", "list GitHub repos").'
        )

    parts = [
        "## Capability manifest",
        acquire_hint,
        "",
        *lines_active,
        "",
        ws_line,
    ]

    if skill_catalog_lines:
        parts.extend(["", *skill_catalog_lines])

    if resolved and resolved.tool_categories:
        parts.extend(
            [
                "",
                "Categories represented in this step's current tool binding:",
            ]
        )
        parts.extend(f"- {c}" for c in resolved.tool_categories)

    return "\n".join(parts)

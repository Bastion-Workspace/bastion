"""
Step-level tool resolution: merges available_tools, skills, packs, team tools,
scopes connection-bound tool names, deduplicates scoped vs bare, and filters by capability.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union

from orchestrator.utils.line_context import line_id_from_metadata

from orchestrator.engines.capability_manifest import build_capability_manifest
from orchestrator.engines.provider_capability_registry import (
    build_tool_to_prefix_map,
    get_all_scoped_prefixes,
    get_m365_style_sanitize_prefixes,
)

logger = logging.getLogger(__name__)

USER_FACT_TOOL_NAMES = frozenset({"get_user_facts", "save_user_fact"})

AGENT_MEMORY_TOOL_NAMES: Tuple[str, ...] = (
    "get_agent_memory",
    "set_agent_memory",
    "list_agent_memories",
    "delete_agent_memory",
    "append_agent_memory",
)

_TOOL_TO_PREFIX: Dict[str, str] = build_tool_to_prefix_map()
SCOPED_PREFIXES: FrozenSet[str] = frozenset(get_all_scoped_prefixes())
M365_STYLE_SANITIZE_PREFIXES: FrozenSet[str] = get_m365_style_sanitize_prefixes()


def _normalize_connections_map_dict(data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for k, v in data.items():
        if isinstance(v, list):
            out[str(k)] = [x for x in v if isinstance(x, dict)]
    return out


def parse_connections_map(metadata: Optional[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    raw: Union[str, Dict[str, Any], None] = (metadata or {}).get("active_connections_map")
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return _normalize_connections_map_dict(raw)
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {}
        return _normalize_connections_map_dict(data)
    except json.JSONDecodeError:
        return {}


def connection_allow_ids_from_entries(entries: List[Dict[str, Any]]) -> Set[int]:
    """
    Extract connection row IDs from profile/step allowlists.
    Accepts both new shape ``{connection_id: N}`` and legacy shapes that include
    ``connection_type``, ``provider``, or ``id`` as alias for ``connection_id``.
    """
    out: Set[int] = set()
    for e in entries or []:
        if not isinstance(e, dict):
            continue
        cid_raw = e.get("connection_id")
        if cid_raw is None:
            cid_raw = e.get("id")
        if cid_raw is None:
            continue
        try:
            out.add(int(cid_raw))
        except (TypeError, ValueError):
            continue
    return out


def apply_step_connection_policy(
    cmap: Dict[str, List[Dict[str, Any]]],
    step: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    """Restrict connection map by step.connection_policy (inherit | restrict | none)."""
    policy = (step.get("connection_policy") or "inherit").strip().lower()
    if policy == "none":
        return {}
    if policy == "restrict":
        restricted = step.get("restricted_connections") or []
        if not isinstance(restricted, list):
            restricted = []
        allowed_ids = connection_allow_ids_from_entries(
            [e for e in restricted if isinstance(e, dict)]
        )
        out: Dict[str, List[Dict[str, Any]]] = {}
        for ctype, entries in cmap.items():
            kept = [
                e
                for e in entries
                if isinstance(e, dict)
                and e.get("id") is not None
                and int(e["id"]) in allowed_ids
            ]
            if kept:
                out[str(ctype)] = kept
        return out
    return cmap


def build_step_effective_connections_map(
    metadata: Optional[Dict[str, Any]],
    step: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    """Profile-level map in metadata + step connection_policy."""
    return apply_step_connection_policy(parse_connections_map(metadata), step)


def active_connection_types_from_metadata(metadata: Optional[Dict[str, Any]]) -> Optional[List[str]]:
    cmap = parse_connections_map(metadata)
    if not cmap:
        return None
    return sorted(cmap.keys())


def _github_cmap_entries(cmap: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    gh = cmap.get("github") or []
    if gh:
        return [e for e in gh if isinstance(e, dict)]
    return [
        e
        for e in (cmap.get("code_platform") or [])
        if isinstance(e, dict) and (e.get("provider") or "").lower() == "github"
    ]


def _gitea_cmap_entries(cmap: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    gt = cmap.get("gitea") or []
    if gt:
        return [e for e in gt if isinstance(e, dict)]
    return [
        e
        for e in (cmap.get("code_platform") or [])
        if isinstance(e, dict) and (e.get("provider") or "").lower() == "gitea"
    ]


def _code_platform_scope(
    tool_name: str, cmap: Dict[str, List[Dict[str, Any]]]
) -> Optional[Tuple[str, int]]:
    tl = (tool_name or "").lower()
    if tl.startswith("github_"):
        entries = _github_cmap_entries(cmap)
        if not entries:
            return None
        for e in entries:
            if e.get("id") is not None:
                return "github", int(e["id"])
        return None
    if tl.startswith("gitea_"):
        entries = _gitea_cmap_entries(cmap)
        if not entries:
            return None
        for e in entries:
            if e.get("id") is not None:
                return "gitea", int(e["id"])
        return None
    return None


def resolve_default_code_platform_connection_id(
    registry_tool_name: str,
    metadata: Optional[Dict[str, Any]],
) -> int:
    """
    Default connection id for bare github_*/gitea_* tools when the LLM omits connection_id (0).
    Uses the same rules as tool-name scoping (single unambiguous code_platform row, or provider match).
    """
    cmap = parse_connections_map(metadata)
    scoped = _code_platform_scope(registry_tool_name, cmap)
    if not scoped:
        return 0
    return int(scoped[1])


def scope_connection_bound_tool_name(
    name: str, cmap: Dict[str, List[Dict[str, Any]]]
) -> str:
    """
    Prefix connection-bound tool names with connection id when unambiguous.
    Names that already contain ':' are returned unchanged.
    """
    if not name or not isinstance(name, str):
        return name
    if ":" in name:
        return name
    prefix = _TOOL_TO_PREFIX.get(name)
    if prefix:
        arr = cmap.get(prefix) or []
        if len(arr) == 1 and arr[0].get("id") is not None:
            return f"{prefix}:{int(arr[0]['id'])}:{name}"
        return name
    scoped = _code_platform_scope(name, cmap)
    if scoped:
        pfx, cid = scoped
        return f"{pfx}:{cid}:{name}"
    return name


def apply_connection_scoping_to_tool_names(
    tool_names: List[str], cmap: Dict[str, List[Dict[str, Any]]]
) -> List[str]:
    """Scope any bare connection-bound tools (e.g. from available_tools or member_additional_tools)."""
    return [scope_connection_bound_tool_name(n, cmap) if isinstance(n, str) else n for n in tool_names]


def dedup_scoped_tool_names(tool_names: List[str]) -> List[str]:
    """
    Remove bare tool names when a scoped variant exists (email:, calendar:, …, github:, gitea:).
    """
    scoped_tails: Set[str] = set()
    for n in tool_names:
        if not isinstance(n, str) or n.count(":") < 2:
            continue
        for prefix in SCOPED_PREFIXES:
            if n.startswith(f"{prefix}:"):
                tail = n.split(":", 2)[2]
                if tail:
                    scoped_tails.add(tail)
                break
    if not scoped_tails:
        return tool_names

    def _is_scoped_connection_tool(n: str) -> bool:
        return any(n.startswith(f"{p}:") for p in SCOPED_PREFIXES)

    return [
        n
        for n in tool_names
        if not (
            isinstance(n, str)
            and n in scoped_tails
            and not _is_scoped_connection_tool(n)
        )
    ]


def skill_discovery_mode_from_step(step: Dict[str, Any]) -> str:
    """
    Return 'off' | 'auto' | 'catalog' | 'full'.
    discovery_mode (UI) and skill_discovery_mode; legacy auto_discover_skills + dynamic_tool_discovery.
    """
    raw = step.get("discovery_mode") or step.get("skill_discovery_mode")
    if isinstance(raw, str):
        v = raw.strip().lower()
        if v == "on_demand":
            v = "full"
        if v in ("off", "auto", "catalog", "full"):
            return v
    if step.get("inject_skill_manifest"):
        return "catalog"
    auto = step.get("auto_discover_skills")
    dyn = step.get("dynamic_tool_discovery")
    if auto is False and not dyn:
        return "off"
    if dyn:
        return "full"
    if auto is False:
        return "off"
    return "auto"


def auto_discover_skills_effective(step: Dict[str, Any]) -> bool:
    return skill_discovery_mode_from_step(step) in ("auto", "full")


def dynamic_tool_discovery_effective(step: Dict[str, Any]) -> bool:
    return skill_discovery_mode_from_step(step) == "full"


def inject_skill_manifest_effective(step: Dict[str, Any]) -> bool:
    """Whether to inject a compressed skill catalog into the system prompt."""
    return skill_discovery_mode_from_step(step) in ("catalog", "full")


def max_discovered_skills_from_step(step: Dict[str, Any]) -> int:
    v = step.get("max_discovered_skills")
    if v is not None:
        try:
            return max(1, min(10, int(v)))
        except (TypeError, ValueError):
            pass
    if step.get("max_skill_acquisitions") is not None:
        try:
            return max(1, min(10, int(step.get("max_skill_acquisitions"))))
        except (TypeError, ValueError):
            pass
    return max(1, min(10, int(step.get("max_auto_skills") or 3)))


def max_runtime_skill_acquisitions_from_step(step: Dict[str, Any]) -> int:
    """Cap for mid-loop search_and_acquire_skills merges."""
    v = step.get("max_skill_acquisitions")
    if v is not None:
        try:
            return max(1, min(10, int(v)))
        except (TypeError, ValueError):
            pass
    return max_discovered_skills_from_step(step)


def filter_user_fact_tools_by_policy(tool_names: List[str], policy: str) -> List[str]:
    """Remove user-fact tools according to effective policy (vacuum, isolated, no_write, inherit)."""
    if policy in ("vacuum", "isolated"):
        return [n for n in tool_names if n not in USER_FACT_TOOL_NAMES]
    if policy == "no_write":
        return [n for n in tool_names if n != "save_user_fact"]
    return tool_names


def _ensure_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(t).strip() for t in value if t and str(t).strip()]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(t).strip() for t in parsed if t and str(t).strip()]
        except (json.JSONDecodeError, TypeError):
            pass
        return []
    return []


def _ensure_list_like(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        return []
    return []


@dataclass
class ResolvedSkillInfo:
    """Metadata for a single resolved skill, used for execution metrics."""
    skill_id: str
    slug: str
    version: int = 1
    discovery_method: str = "explicit"  # explicit | auto_discover | runtime_acquire


@dataclass
class ToolResolutionResult:
    tool_names: List[str] = field(default_factory=list)
    skill_guidance: str = ""
    """Registry categories present in tool_names (for capability manifest)."""
    tool_categories: List[str] = field(default_factory=list)
    resolved_skills: List[ResolvedSkillInfo] = field(default_factory=list)


async def _apply_candidate_substitution(
    skills: List[Dict[str, Any]],
    client: Any,
    user_id: str,
) -> List[Dict[str, Any]]:
    """For each non-candidate skill, check if a candidate version exists and
    probabilistically substitute it based on candidate_weight (0-100)."""
    result: List[Dict[str, Any]] = []
    for s in skills:
        if s.get("is_candidate"):
            result.append(s)
            continue
        slug = (s.get("slug") or "").strip()
        if not slug:
            result.append(s)
            continue
        try:
            candidate = await client.get_candidate_for_slug(user_id, slug)
        except Exception:
            candidate = None
        if candidate:
            weight = int(candidate.get("candidate_weight") or 0)
            if 0 < weight and random.randint(1, 100) <= weight:
                logger.info(
                    "A/B candidate substitution: slug=%s weight=%d → using candidate %s",
                    slug, weight, candidate.get("id"),
                )
                result.append(candidate)
                continue
        result.append(s)
    return result


async def resolve_and_inject_skills(
    step_skill_ids: Optional[List[str]] = None,
    user_id: str = "system",
    auto_discover_skills: bool = False,
    max_auto_skills: int = 3,
    step_prompt: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    skill_search_query: str = "",
) -> Tuple[str, List[str], List[ResolvedSkillInfo]]:
    """
    Resolve skill IDs; optional semantic auto-discovery; fetch skills.

    Returns (guidance_text, required_tool_names, resolved_skill_infos).
    When skill_search_query is non-empty, it is used for vector search instead of step_prompt (focused user intent).
    """
    from orchestrator.backend_tool_client import get_backend_tool_client

    merged: List[str] = []
    seen: set = set()
    explicit_ids: set = set()
    discovered_ids: set = set()
    for sid in step_skill_ids or []:
        s = (sid or "").strip()
        if s and s not in seen:
            seen.add(s)
            merged.append(s)
            explicit_ids.add(s)
    search_text = (skill_search_query or step_prompt or "").strip()
    if auto_discover_skills and search_text:
        try:
            client = await get_backend_tool_client()
            disc_limit = max(1, min(10, max_auto_skills))
            act_types = active_connection_types_from_metadata(metadata)
            discovered = await client.search_skills(
                user_id=user_id,
                query=search_text,
                limit=disc_limit,
                score_threshold=0.5,
                active_connection_types=act_types,
            )
            _q_preview = search_text[:500] + ("..." if len(search_text) > 500 else "")
            if discovered:
                _hits = [
                    f"{(h.get('slug') or h.get('name') or h.get('id'))}:{float(h.get('similarity_score') or h.get('score') or 0):.3f}"
                    for h in discovered
                ]
                logger.info(
                    "Skill auto-discovery: query_len=%d limit=%s active_conn_types=%s hits=%s",
                    len(search_text),
                    disc_limit,
                    act_types,
                    _hits,
                )
            else:
                logger.info(
                    "Skill auto-discovery: no hits (query_len=%d limit=%s threshold=0.5 active_conn_types=%s) preview=%r",
                    len(search_text),
                    disc_limit,
                    act_types,
                    _q_preview,
                )
            for hit in discovered:
                sid = (hit.get("id") or hit.get("skill_id") or "").strip()
                if sid and sid not in seen:
                    seen.add(sid)
                    merged.append(sid)
                    discovered_ids.add(sid)
        except Exception as e:
            logger.warning("search_skills (auto-discovery) failed: %s", e)
    if not merged:
        return "", [], []
    try:
        client = await get_backend_tool_client()
        skills = await client.get_skills_by_ids(user_id, merged)
    except Exception as e:
        logger.warning("get_skills_by_ids failed: %s", e)
        return "", [], []

    # Candidate A/B testing: probabilistically swap a skill with its candidate
    skills = await _apply_candidate_substitution(skills, client, user_id)

    parts: List[str] = []
    required_tools: List[str] = []
    resolved_skill_infos: List[ResolvedSkillInfo] = []
    cmap = parse_connections_map(metadata)

    # Resolve depends_on recursively (max 3 levels)
    _dep_slugs_needed: List[str] = []
    _visited_slugs: set = set()
    for s in skills:
        _visited_slugs.add(s.get("slug") or "")
        for dep_slug in s.get("depends_on") or []:
            ds = (dep_slug or "").strip()
            if ds and ds not in _visited_slugs:
                _dep_slugs_needed.append(ds)
                _visited_slugs.add(ds)

    dep_skills: List[Dict[str, Any]] = []
    _dep_depth = 0
    _MAX_DEP_DEPTH = 3
    while _dep_slugs_needed and _dep_depth < _MAX_DEP_DEPTH:
        _dep_depth += 1
        try:
            _dep_batch = await client.get_skills_by_slugs(user_id, _dep_slugs_needed)
        except Exception as e:
            logger.warning("get_skills_by_slugs (dependency resolution) failed: %s", e)
            break
        dep_skills.extend(_dep_batch)
        _dep_slugs_needed = []
        for ds in _dep_batch:
            for nested_slug in ds.get("depends_on") or []:
                ns = (nested_slug or "").strip()
                if ns and ns not in _visited_slugs:
                    _dep_slugs_needed.append(ns)
                    _visited_slugs.add(ns)

    for s in skills:
        name = s.get("name") or s.get("slug") or "Skill"
        procedure = (s.get("procedure") or "").strip()
        skill_parts: List[str] = []
        if procedure:
            skill_parts.append(f"## Skill: {name}\n{procedure}")
        # Append dependency procedures as sub-sections
        for dep_slug in s.get("depends_on") or []:
            dep = next((d for d in dep_skills if d.get("slug") == dep_slug), None)
            if dep:
                dep_proc = (dep.get("procedure") or "").strip()
                dep_name = dep.get("name") or dep.get("slug") or "Dependency"
                if dep_proc:
                    skill_parts.append(f"### Included skill: {dep_name}\n{dep_proc}")
                for t in dep.get("required_tools") or []:
                    tn = str(t).strip()
                    if tn:
                        required_tools.append(scope_connection_bound_tool_name(tn, cmap))
        if skill_parts:
            parts.append("\n\n".join(skill_parts))
        for t in s.get("required_tools") or []:
            tn = str(t).strip()
            if tn:
                required_tools.append(scope_connection_bound_tool_name(tn, cmap))
        sid = str(s.get("id") or "")
        method = "auto_discover" if sid in discovered_ids else "explicit"
        resolved_skill_infos.append(ResolvedSkillInfo(
            skill_id=sid,
            slug=s.get("slug") or "",
            version=s.get("version") or 1,
            discovery_method=method,
        ))

    guidance = "\n\n".join(parts) if parts else ""
    required_deduped = list(dict.fromkeys(required_tools))
    if merged:
        logger.info(
            "Skill resolution: skill_ids=%s required_tools_from_skills=%s dep_skills=%d",
            merged,
            required_deduped,
            len(dep_skills),
        )
    return guidance, required_deduped, resolved_skill_infos


class StepToolResolver:
    """Resolves playbook step configuration into concrete tool names for LLM/deep_agent binding."""

    async def resolve(
        self,
        base_tool_names: List[str],
        step: Dict[str, Any],
        metadata: Optional[Dict[str, Any]],
        user_id: str,
        step_prompt: str = "",
        user_facts_policy: str = "inherit",
        skill_search_query: str = "",
    ) -> ToolResolutionResult:
        from orchestrator.utils.action_io_registry import get_action, get_categories_for_tools

        cmap = build_step_effective_connections_map(metadata, step)
        effective_metadata: Dict[str, Any] = {**(metadata or {}), "active_connections_map": json.dumps(cmap)}
        step_skill_ids = list(step.get("skill_ids") or step.get("skills") or [])
        if effective_metadata.get("team_skill_ids"):
            try:
                parsed = json.loads(effective_metadata["team_skill_ids"])
                if isinstance(parsed, list):
                    for s in parsed:
                        if s and s not in step_skill_ids:
                            step_skill_ids.append(s)
            except (TypeError, json.JSONDecodeError):
                pass

        mode = skill_discovery_mode_from_step(step)
        auto_on = mode in ("auto", "full")
        max_auto = max_discovered_skills_from_step(step)

        skill_guidance, skill_required_tools, resolved_skill_infos = await resolve_and_inject_skills(
            step_skill_ids=step_skill_ids,
            user_id=user_id,
            auto_discover_skills=auto_on,
            max_auto_skills=max_auto,
            step_prompt=step_prompt,
            metadata=effective_metadata,
            skill_search_query=skill_search_query,
        )

        tool_names: List[str] = []
        seen: set = set()

        # Skills are the primary source of tools
        for t in skill_required_tools:
            if t and t not in seen:
                seen.add(t)
                tool_names.append(t)

        # Auto-inject team-collaboration skill for team context
        if effective_metadata and line_id_from_metadata(effective_metadata):
            _team_slug_present = any(
                s == "team-collaboration" for s in step_skill_ids
            )
            if not _team_slug_present:
                try:
                    from orchestrator.backend_tool_client import get_backend_tool_client as _get_btc
                    _team_client = await _get_btc()
                    _team_skill = await _team_client.get_skill_by_slug(user_id, "team-collaboration")
                    if _team_skill:
                        _team_proc = (_team_skill.get("procedure") or "").strip()
                        if _team_proc:
                            _team_name = _team_skill.get("name") or "Team Collaboration"
                            _tg = f"## Skill: {_team_name}\n{_team_proc}"
                            skill_guidance = (skill_guidance + "\n\n" + _tg).strip() if skill_guidance else _tg
                        _team_cmap = parse_connections_map(effective_metadata)
                        for t in _team_skill.get("required_tools") or []:
                            tn = scope_connection_bound_tool_name(str(t).strip(), _team_cmap)
                            if tn and tn not in seen:
                                seen.add(tn)
                                tool_names.append(tn)
                except Exception as e:
                    logger.warning("Auto-inject team-collaboration skill failed: %s", e)

        # --- Backward compatibility shim for legacy available_tools ---
        if base_tool_names:
            logger.info(
                "DEPRECATED: Step has available_tools=%s — migrate to skill_ids",
                base_tool_names[:10],
            )
            for t in base_tool_names:
                if t and t not in seen:
                    seen.add(t)
                    tool_names.append(t)

        # --- Backward compatibility shim for legacy tool_packs ---
        pack_entries = _ensure_list_like(step.get("tool_packs"))
        if effective_metadata.get("team_tool_packs"):
            try:
                team_packs = json.loads(effective_metadata["team_tool_packs"])
                if isinstance(team_packs, list):
                    pack_entries.extend(p for p in team_packs if p)
            except (TypeError, json.JSONDecodeError):
                pass
        if pack_entries:
            logger.info(
                "DEPRECATED: Step/team has tool_packs=%s — migrate to skill_ids",
                [e.get("pack") if isinstance(e, dict) else e for e in pack_entries[:10]],
            )
            try:
                from orchestrator.tools.tool_pack_registry import resolve_pack_tools_with_mode
                from orchestrator.tools.external_pack_registry import split_pack_entries, resolve_external_pack_tools

                builtin_entries, external_entries = split_pack_entries(pack_entries)
                for name in resolve_pack_tools_with_mode(builtin_entries):
                    if name not in seen:
                        seen.add(name)
                        tool_names.append(name)
                for name in await resolve_external_pack_tools(
                    external_entries, user_id, allowed_cmap=cmap
                ):
                    if name not in seen:
                        seen.add(name)
                        tool_names.append(name)
            except Exception as e:
                logger.warning("Resolving legacy tool_packs in StepToolResolver failed: %s", e)

        # --- Backward compatibility shim for member_additional_tools ---
        if effective_metadata.get("member_additional_tools"):
            try:
                parsed = json.loads(effective_metadata["member_additional_tools"])
                if isinstance(parsed, list):
                    for t in parsed:
                        if t and isinstance(t, str) and t not in seen:
                            seen.add(t)
                            tool_names.append(t)
            except (TypeError, json.JSONDecodeError):
                pass

        tool_names = apply_connection_scoping_to_tool_names(tool_names, cmap)
        tool_names = dedup_scoped_tool_names(tool_names)
        seen = set(tool_names)

        if "local_device_tools" in tool_names:
            tool_names = [n for n in tool_names if n != "local_device_tools"]
            try:
                from orchestrator.tools.local_proxy_tools import get_available_local_proxy_tools

                available_local = await get_available_local_proxy_tools(user_id)
                for t in available_local:
                    if t not in seen:
                        seen.add(t)
                        tool_names.append(t)
            except Exception as e:
                logger.warning("Expanding local_device_tools in StepToolResolver failed: %s", e)

        try:
            from orchestrator.tools.local_proxy_tools import get_available_local_proxy_tools

            available_local = await get_available_local_proxy_tools(user_id)
            try:
                from orchestrator.backend_tool_client import get_backend_tool_client

                client = await get_backend_tool_client()
                available_caps = set(await client.get_device_capabilities(user_id))
            except Exception:
                available_caps = set()

            code_workspace_required_caps = {
                "code_file_tree": "file_tree",
                "code_search_files": "search_files",
                "code_git_info": "git_info",
            }
            filtered: List[str] = []
            for name in tool_names:
                contract = get_action(name)
                if contract is None and (name or "").endswith("_tool"):
                    contract = get_action((name or "")[:-5])
                if contract and getattr(contract, "category", None) == "local_proxy":
                    name_for_check = name if (name or "").endswith("_tool") else f"{name}_tool"
                    if name_for_check in available_local:
                        filtered.append(name)
                elif contract and getattr(contract, "category", None) == "code_workspace":
                    base_name = (name or "")[:-5] if (name or "").endswith("_tool") else (name or "")
                    required_cap = code_workspace_required_caps.get(base_name)
                    if not required_cap or required_cap in available_caps:
                        filtered.append(name)
                else:
                    filtered.append(name)
            tool_names = filtered
        except Exception as e:
            logger.warning("Filtering local_proxy in StepToolResolver failed: %s", e)

        tool_names = filter_user_fact_tools_by_policy(tool_names, user_facts_policy)

        if effective_metadata.get("include_agent_memory"):
            step_memory_policy = (step.get("agent_memory_policy") or "inherit").strip().lower()
            if step_memory_policy != "off":
                seen_mem = set(tool_names)
                for t in AGENT_MEMORY_TOOL_NAMES:
                    if t not in seen_mem:
                        seen_mem.add(t)
                        tool_names.append(t)

        core_names = [_core_tool_name_for_category(n) for n in tool_names]
        tool_categories = get_categories_for_tools(core_names)

        return ToolResolutionResult(
            tool_names=tool_names,
            skill_guidance=skill_guidance or "",
            tool_categories=tool_categories,
            resolved_skills=resolved_skill_infos,
        )


def _core_tool_name_for_category(action_name: str) -> str:
    prefix = action_name.split(":", 1)[0] if ":" in action_name else ""
    if prefix in SCOPED_PREFIXES and action_name.count(":") >= 2:
        return action_name.split(":", 2)[2]
    return action_name


_resolver_singleton = StepToolResolver()


async def resolve_step_tools(
    base_tool_names: List[str],
    step: Dict[str, Any],
    metadata: Optional[Dict[str, Any]],
    user_id: str,
    step_prompt: str = "",
    user_facts_policy: str = "inherit",
    skill_search_query: str = "",
) -> ToolResolutionResult:
    return await _resolver_singleton.resolve(
        base_tool_names,
        step,
        metadata,
        user_id,
        step_prompt=step_prompt,
        user_facts_policy=user_facts_policy,
        skill_search_query=skill_search_query,
    )


async def build_augmented_tool_names(
    base_tool_names: List[str],
    step: Dict[str, Any],
    metadata: Optional[Dict[str, Any]],
    user_id: str,
    step_prompt: str = "",
    user_facts_policy: str = "inherit",
    skill_search_query: str = "",
) -> Tuple[List[str], str]:
    """Backward-compatible tuple return for callers that only need names + guidance."""
    r = await resolve_step_tools(
        base_tool_names,
        step,
        metadata,
        user_id,
        step_prompt=step_prompt,
        user_facts_policy=user_facts_policy,
        skill_search_query=skill_search_query,
    )
    return r.tool_names, r.skill_guidance

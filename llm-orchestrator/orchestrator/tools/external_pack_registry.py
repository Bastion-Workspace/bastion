"""
External tool packs: email, calendar, GitHub (per-connection OAuth), and MCP servers (dynamic).

Playbook steps use tool_packs entries with optional `connections` and/or pack names like mcp:<id>.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from orchestrator.engines.provider_capability_registry import get_all_capability_keys

@dataclass(frozen=True)
class ExternalPackType:
    """Metadata for a static external pack (email, calendar, github)."""

    name: str
    description: str
    requires_connection: bool
    connection_source: str
    connection_filter: Dict[str, str]


# Declarative catalog (used by docs/UI generators; runtime resolution uses TOOL_PACKS + DB).
EXTERNAL_PACK_TYPES: Dict[str, ExternalPackType] = {
    "email": ExternalPackType(
        name="email",
        description="Email tools scoped to connected accounts",
        requires_connection=True,
        connection_source="external_connections",
        connection_filter={"connection_type": "email"},
    ),
    "calendar": ExternalPackType(
        name="calendar",
        description="Calendar tools scoped to connected calendars",
        requires_connection=True,
        connection_source="external_connections",
        connection_filter={"connection_type": "calendar"},
    ),
    "github": ExternalPackType(
        name="github",
        description="GitHub REST tools scoped to OAuth connections",
        requires_connection=True,
        connection_source="external_connections",
        connection_filter={"connection_type": "code_platform", "provider": "github"},
    ),
    "gitea": ExternalPackType(
        name="gitea",
        description="Gitea REST tools scoped to PAT connections",
        requires_connection=True,
        connection_source="external_connections",
        connection_filter={"connection_type": "code_platform", "provider": "gitea"},
    ),
}


def split_pack_entries(
    pack_entries: List[Any],
) -> Tuple[List[Union[str, Dict[str, Any]]], List[Union[str, Dict[str, Any]]]]:
    """
    Split tool_packs into builtin-only entries vs external (email/calendar/contacts/M365 packs/github/gitea with connections, mcp:*).
    """
    builtin: List[Union[str, Dict[str, Any]]] = []
    external: List[Union[str, Dict[str, Any]]] = []
    for entry in pack_entries or []:
        if isinstance(entry, str):
            e = entry.strip()
            if e.startswith("mcp:"):
                external.append(entry)
            else:
                builtin.append(entry)
            continue
        if isinstance(entry, dict):
            p = entry.get("pack") or entry.get("name")
            p_s = (str(p).strip() if p is not None else "") or ""
            conns = entry.get("connections") or []
            if not isinstance(conns, list):
                conns = []
            if p_s.startswith("mcp:"):
                external.append(entry)
            elif p_s in _EXTERNAL_CAPABILITY_PACK_KEYS and conns:
                external.append(entry)
            else:
                builtin.append(entry)
            continue
        builtin.append(entry)  # type: ignore[arg-type]
    return builtin, external


_EXTERNAL_CAPABILITY_PACK_KEYS: frozenset = get_all_capability_keys()


def _allowed_connection_ids_for_external_pack(
    pack_s: str,
    allowed_cmap: Optional[Dict[str, List[Dict[str, Any]]]],
) -> Optional[Set[int]]:
    """
    When allowed_cmap is set, return connection/server ids permitted for this pack.
    None means do not filter (backward compatible).
    """
    if allowed_cmap is None:
        return None
    if pack_s in _EXTERNAL_CAPABILITY_PACK_KEYS:
        return {
            int(e["id"])
            for e in (allowed_cmap.get(pack_s) or [])
            if isinstance(e, dict) and e.get("id") is not None
        }
    if pack_s.startswith("mcp:"):
        mcp_entries = allowed_cmap.get("mcp")
        if not mcp_entries:
            return None
        return {int(e["id"]) for e in mcp_entries if isinstance(e, dict) and e.get("id") is not None}
    return set()


async def resolve_external_pack_tools(
    pack_entries: List[Union[str, Dict[str, Any]]],
    user_id: str,
    allowed_cmap: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> List[str]:
    """
    Expand external pack entries to concrete tool name strings:
    email:/calendar:/contacts:/todo:/files:/onenote:/planner:<cid>:<registry_tool>, github:/gitea:…, mcp:<sid>:<tool_name>.
    """
    from orchestrator.backend_tool_client import get_backend_tool_client
    from orchestrator.tools.tool_pack_registry import TOOL_PACKS, _tool_name_to_registry_name

    out: List[str] = []
    seen: set = set()
    client = await get_backend_tool_client()

    def add(name: str) -> None:
        if name and name not in seen:
            seen.add(name)
            out.append(name)

    for entry in pack_entries or []:
        if isinstance(entry, str):
            entry_s = entry.strip()
            if entry_s.startswith("mcp:"):
                rest = entry_s[4:].strip()
                try:
                    sid = int(rest)
                except ValueError:
                    continue
                mcp_allowed = _allowed_connection_ids_for_external_pack("mcp:", allowed_cmap)
                if mcp_allowed is not None and sid not in mcp_allowed:
                    continue
                names = await client.get_mcp_server_tool_names(user_id, sid)
                for tn in names:
                    add(f"mcp:{sid}:{tn}")
            continue

        if not isinstance(entry, dict):
            continue
        pack = entry.get("pack") or entry.get("name")
        mode = (entry.get("mode") or "full").lower()
        conns = entry.get("connections") or []
        if not isinstance(conns, list):
            conns = []
        conns_int: List[int] = []
        for c in conns:
            try:
                conns_int.append(int(c))
            except (TypeError, ValueError):
                pass

        pack_s = (str(pack).strip() if pack is not None else "") or ""

        if pack_s.startswith("mcp:"):
            try:
                sid = int(pack_s.split(":", 1)[1])
            except (IndexError, ValueError):
                continue
            mcp_allowed = _allowed_connection_ids_for_external_pack("mcp:", allowed_cmap)
            if mcp_allowed is not None and sid not in mcp_allowed:
                continue
            names = await client.get_mcp_server_tool_names(user_id, sid)
            for tn in names:
                add(f"mcp:{sid}:{tn}")
            continue

        if pack_s in _EXTERNAL_CAPABILITY_PACK_KEYS and conns_int:
            pdef = TOOL_PACKS.get(pack_s)
            if not pdef:
                continue
            if mode == "read" and pdef.read_tools is not None:
                tools_to_add = pdef.read_tools
            else:
                tools_to_add = pdef.tools
            reg_names = [_tool_name_to_registry_name(t) for t in tools_to_add]
            pack_allowed = _allowed_connection_ids_for_external_pack(pack_s, allowed_cmap)
            for cid in conns_int:
                if pack_allowed is not None and cid not in pack_allowed:
                    continue
                for reg in reg_names:
                    add(f"{pack_s}:{cid}:{reg}")

    return out

"""
External tool packs: email, calendar (per-connection), and MCP servers (dynamic).

Playbook steps use tool_packs entries with optional `connections` and/or pack names like mcp:<id>.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

@dataclass(frozen=True)
class ExternalPackType:
    """Metadata for a static external pack (email, calendar)."""

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
}


def split_pack_entries(
    pack_entries: List[Any],
) -> Tuple[List[Union[str, Dict[str, Any]]], List[Union[str, Dict[str, Any]]]]:
    """
    Split tool_packs into builtin-only entries vs external (email/calendar with connections, mcp:*).
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
            elif p_s in ("email", "calendar") and conns:
                external.append(entry)
            else:
                builtin.append(entry)
            continue
        builtin.append(entry)  # type: ignore[arg-type]
    return builtin, external


async def resolve_external_pack_tools(
    pack_entries: List[Union[str, Dict[str, Any]]],
    user_id: str,
) -> List[str]:
    """
    Expand external pack entries to concrete tool name strings:
    email:<cid>:<registry_tool>, calendar:<cid>:<registry_tool>, mcp:<sid>:<tool_name>.
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
            names = await client.get_mcp_server_tool_names(user_id, sid)
            for tn in names:
                add(f"mcp:{sid}:{tn}")
            continue

        if pack_s in ("email", "calendar") and conns_int:
            pdef = TOOL_PACKS.get(pack_s)
            if not pdef:
                continue
            if mode == "read" and pdef.read_tools is not None:
                tools_to_add = pdef.read_tools
            else:
                tools_to_add = pdef.tools
            reg_names = [_tool_name_to_registry_name(t) for t in tools_to_add]
            for cid in conns_int:
                for reg in reg_names:
                    add(f"{pack_s}:{cid}:{reg}")

    return out

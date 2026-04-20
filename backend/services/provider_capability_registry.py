"""
Declarative registry: which external connection rows expand to which capability keys
(active_connections_map / API capabilities). Keeps M365 and code_platform rules in one place.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from services.m365_oauth_utils import (
    DEFAULT_M365_ENABLED_SERVICES,
    M365_ALL_SERVICE_KEYS,
    normalize_m365_services,
)


@dataclass(frozen=True)
class ProviderCapability:
    """One logical capability advertised under a single cmap/API key."""

    key: str
    tool_prefix: str


@dataclass(frozen=True)
class ProviderDefinition:
    provider: str
    connection_type: str
    capabilities: Tuple[ProviderCapability, ...]
    multi_service: bool = False


_REGISTRY: Dict[Tuple[str, str], ProviderDefinition] = {}


def _register(defn: ProviderDefinition) -> None:
    _REGISTRY[(defn.provider.lower(), defn.connection_type.lower())] = defn


_M365_CAPS: Tuple[ProviderCapability, ...] = tuple(
    ProviderCapability(k, k) for k in M365_ALL_SERVICE_KEYS
)

_register(
    ProviderDefinition(
        provider="microsoft",
        connection_type="email",
        capabilities=_M365_CAPS,
        multi_service=True,
    )
)

_register(
    ProviderDefinition(
        provider="github",
        connection_type="code_platform",
        capabilities=(ProviderCapability("github", "github"),),
        multi_service=False,
    )
)

_register(
    ProviderDefinition(
        provider="gitea",
        connection_type="code_platform",
        capabilities=(ProviderCapability("gitea", "gitea"),),
        multi_service=False,
    )
)


def get_provider_definition(provider: str, connection_type: str) -> Optional[ProviderDefinition]:
    p = (provider or "").strip().lower()
    ct = (connection_type or "").strip().lower()
    return _REGISTRY.get((p, ct))


def get_all_capability_keys() -> FrozenSet[str]:
    keys: Set[str] = set()
    for defn in _REGISTRY.values():
        for cap in defn.capabilities:
            keys.add(cap.key)
    return frozenset(keys)


def parse_m365_enabled_services(provider_metadata: Any) -> List[str]:
    meta: Dict[str, Any] = {}
    if isinstance(provider_metadata, dict):
        meta = provider_metadata
    elif isinstance(provider_metadata, str) and provider_metadata.strip():
        try:
            parsed = json.loads(provider_metadata)
            if isinstance(parsed, dict):
                meta = parsed
        except (json.JSONDecodeError, TypeError):
            pass
    raw = meta.get("enabled_services")
    if isinstance(raw, list) and raw:
        return normalize_m365_services([str(x) for x in raw])
    return list(DEFAULT_M365_ENABLED_SERVICES)


def resolve_capability_keys_for_row(
    provider: str,
    connection_type: str,
    provider_metadata: Any,
) -> List[str]:
    """
    Capability keys this connection exposes for API `capabilities` and UI (ordered, stable).
    Unknown providers: single key = connection_type.
    """
    prov = (provider or "").strip()
    ct = (connection_type or "").strip()
    defn = get_provider_definition(prov, ct)
    if not defn:
        return [ct] if ct else []

    if defn.multi_service:
        enabled = set(parse_m365_enabled_services(provider_metadata))
        out: List[str] = []
        for cap in defn.capabilities:
            if cap.key in enabled:
                out.append(cap.key)
        return out

    return [cap.key for cap in defn.capabilities]


def expand_row_into_cmap_keys(
    provider: str,
    connection_type: str,
    provider_metadata: Any,
) -> List[str]:
    """
    Keys to mirror this row into besides the canonical DB connection_type bucket.
    For Microsoft email: extra keys (calendar, …) not equal to connection_type.
    For github/gitea: capability key (github/gitea) in addition to code_platform.
    """
    prov = (provider or "").strip()
    ct = (connection_type or "").strip()
    defn = get_provider_definition(prov, ct)
    if not defn:
        return []

    if defn.multi_service:
        enabled = set(parse_m365_enabled_services(provider_metadata))
        extra: List[str] = []
        for cap in defn.capabilities:
            if cap.key == ct:
                continue
            if cap.key in enabled:
                extra.append(cap.tool_prefix)
        return extra

    out: List[str] = []
    for cap in defn.capabilities:
        if cap.tool_prefix != ct:
            out.append(cap.tool_prefix)
    return out


# Types that get multi-account hints in the capability manifest (keep in sync with orchestrator).
MULTI_ACCOUNT_MANIFEST_TYPES: FrozenSet[str] = frozenset({
    "email",
    "calendar",
    "contacts",
    "todo",
    "files",
    "onenote",
    "planner",
    "devops",
})

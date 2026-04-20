"""
Microsoft 365 OAuth: service keys, scope mapping, and scope string building.
"""

from __future__ import annotations

from typing import FrozenSet, List, Set

M365_BASE_SCOPES: List[str] = [
    "offline_access",
    "openid",
    "profile",
    "email",
    "User.Read",
]

M365_SERVICE_SCOPES: dict[str, List[str]] = {
    "email": ["Mail.Read", "Mail.ReadWrite", "Mail.Send", "MailboxSettings.Read"],
    "calendar": ["Calendars.ReadWrite"],
    "contacts": ["Contacts.ReadWrite"],
    "todo": ["Tasks.ReadWrite"],
    "files": ["Files.ReadWrite"],
    "onenote": ["Notes.ReadWrite"],
    "planner": ["Tasks.ReadWrite"],
    "devops": ["499b84ac-1321-427f-aa17-267ca6975798/user_impersonation"],
}

M365_ALL_SERVICE_KEYS: tuple[str, ...] = tuple(M365_SERVICE_SCOPES.keys())

DEFAULT_M365_ENABLED_SERVICES: List[str] = ["email", "calendar", "contacts"]


def normalize_m365_services(raw: List[str] | None) -> List[str]:
    """Deduplicate and keep only known service keys, stable order."""
    if not raw:
        return list(DEFAULT_M365_ENABLED_SERVICES)
    seen: Set[str] = set()
    out: List[str] = []
    for s in raw:
        key = str(s).strip().lower()
        if key in M365_SERVICE_SCOPES and key not in seen:
            seen.add(key)
            out.append(key)
    return out if out else list(DEFAULT_M365_ENABLED_SERVICES)


def parse_services_query(services_param: str | None) -> List[str]:
    """Parse comma-separated services from query string."""
    if not services_param or not str(services_param).strip():
        return list(DEFAULT_M365_ENABLED_SERVICES)
    parts = [p.strip().lower() for p in str(services_param).split(",") if p.strip()]
    return normalize_m365_services(parts)


def scopes_for_services(services: List[str]) -> List[str]:
    """Union of base scopes and all scopes for the given services."""
    svc = normalize_m365_services(services)
    extra: Set[str] = set()
    for k in svc:
        for sc in M365_SERVICE_SCOPES.get(k, []):
            extra.add(sc)
    ordered = list(M365_BASE_SCOPES)
    for s in sorted(extra):
        if s not in ordered:
            ordered.append(s)
    return ordered


def build_m365_scope_string(services: List[str]) -> str:
    return " ".join(scopes_for_services(services))


def required_scope_set_for_services(services: List[str]) -> FrozenSet[str]:
    """All Graph permission scope strings required for the service set."""
    svc = normalize_m365_services(services)
    req: Set[str] = set()
    for k in svc:
        req.update(M365_SERVICE_SCOPES.get(k, []))
    return frozenset(req)


def granted_scope_set(granted: List[str] | None) -> FrozenSet[str]:
    if not granted:
        return frozenset()
    return frozenset(str(s).strip() for s in granted if str(s).strip())


def missing_scopes_for_services(
    services: List[str], granted_scopes: List[str] | None
) -> List[str]:
    """Scopes needed for `services` that are not in granted_scopes (substring match for .default)."""
    need = required_scope_set_for_services(services)
    have = granted_scope_set(granted_scopes)
    missing: List[str] = []
    for s in sorted(need):
        if s not in have:
            missing.append(s)
    return missing

"""
Route definitions - declarative route definitions per engine.

Load all definition modules and register routes with the global registry.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

from orchestrator.routes.route_registry import get_route_registry
from orchestrator.routes.route_schema import EngineType, Route

from .conversational_routes import CONVERSATIONAL_ROUTES
from .editor_routes import EDITOR_ROUTES
from .research_routes import RESEARCH_ROUTES

logger = logging.getLogger(__name__)

_routes_loaded = False
_loaded_routes_list: List[Route] = []

# Auto-routable custom agents: (user_id, route_name) -> agent_profile_id
_custom_route_profile_map: Dict[Tuple[str, str], str] = {}
# Names of custom routes currently in the registry (so we can unregister when switching users)
_custom_route_names: set = set()
_last_auto_routable_user_id: Optional[str] = None
_auto_routable_cache_ttl_sec = 60
_auto_routable_cache: Dict[str, float] = {}  # user_id -> expiry timestamp


def load_all_routes() -> List[Route]:
    """Load all route definitions into the registry. Idempotent; safe to call multiple times."""
    global _routes_loaded, _loaded_routes_list
    if _routes_loaded:
        return _loaded_routes_list
    registry = get_route_registry()
    all_routes: List[Route] = []
    all_routes.extend(CONVERSATIONAL_ROUTES)
    all_routes.extend(EDITOR_ROUTES)
    all_routes.extend(RESEARCH_ROUTES)
    for r in all_routes:
        registry.register(r)
    _routes_loaded = True
    _loaded_routes_list = all_routes
    logger.info("Loaded %d routes into registry", len(all_routes))
    return all_routes


async def load_auto_routable_agents(user_id: str) -> None:
    """
    Load auto-routable custom agent profiles for the user into the route registry.
    Uses a short TTL cache (60s) to avoid hitting the backend every message.
    Only one user's custom routes are in the registry at a time; switching users
    unregisters the previous user's custom routes and registers the new user's.
    """
    global _custom_route_names, _last_auto_routable_user_id
    now = time.time()
    if user_id in _auto_routable_cache and now < _auto_routable_cache[user_id]:
        return
    registry = get_route_registry()
    if _last_auto_routable_user_id != user_id:
        for name in list(_custom_route_names):
            registry.unregister(name)
        _custom_route_names.clear()
        _last_auto_routable_user_id = None
    try:
        from orchestrator.backend_tool_client import get_backend_tool_client
        client = await get_backend_tool_client()
        profiles = await client.list_auto_routable_profiles(user_id)
    except Exception as e:
        logger.warning("load_auto_routable_agents: failed to fetch profiles: %s", e)
        _auto_routable_cache[user_id] = now + _auto_routable_cache_ttl_sec
        return
    for p in profiles:
        profile_id = p.get("id")
        handle = (p.get("handle") or "").strip()
        if not handle or not profile_id:
            continue
        name = f"custom_{handle}"
        description = (p.get("description") or p.get("name") or name).strip()
        if len(description) > 200:
            description = description[:197] + "..."
        route = Route(
            name=name,
            description=description or f"Custom agent: {p.get('name', handle)}",
            engine=EngineType.CUSTOM_AGENT,
            domains=["general"],
            actions=["query", "generation", "observation"],
            keywords=[],
            priority=40,
            tools=[],
            internal_only=False,
        )
        registry.register(route)
        _custom_route_names.add(name)
        _custom_route_profile_map[(user_id, name)] = profile_id
    _last_auto_routable_user_id = user_id
    _auto_routable_cache[user_id] = now + _auto_routable_cache_ttl_sec
    if profiles:
        logger.info("Registered %d auto-routable custom agent(s) for user %s", len(profiles), user_id)


def resolve_custom_skill_profile_id(user_id: str, skill_name: str) -> Optional[str]:
    """Resolve (user_id, route_name) to agent_profile_id for CUSTOM_AGENT dispatch. skill_name is the route name."""
    return _custom_route_profile_map.get((user_id, skill_name))


__all__ = [
    "load_all_routes",
    "load_auto_routable_agents",
    "resolve_custom_skill_profile_id",
    "CONVERSATIONAL_ROUTES",
    "EDITOR_ROUTES",
    "RESEARCH_ROUTES",
]

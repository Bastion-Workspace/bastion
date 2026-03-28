"""
Routes package - declarative route definitions and registry for LLM-primary routing.

Usage:
    from orchestrator.routes import get_route_registry, load_all_routes
    from orchestrator.routes.route_schema import Route, EngineType
    from orchestrator.routes.route_selector import llm_select_route

    load_all_routes()
    registry = get_route_registry()
    eligible, instant_route = registry.filter_eligible(query, editor_context, conversation_context)
    if instant_route:
        route_name = instant_route
    elif eligible:
        route_name = await llm_select_route(eligible, query, editor_context, conversation_context)
    else:
        route_name = "chat"
    route = registry.get(route_name)
"""

from orchestrator.routes.route_registry import (
    ScoredRoute,
    get_route_registry,
)
from orchestrator.routes.route_schema import (
    EngineType,
    Route,
)
from orchestrator.routes.definitions import load_all_routes

__all__ = [
    "EngineType",
    "Route",
    "ScoredRoute",
    "get_route_registry",
    "load_all_routes",
]

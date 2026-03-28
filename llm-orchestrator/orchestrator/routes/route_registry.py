"""
Route Registry - Central registry and eligibility filter for LLM-primary routing.

Filter layer answers "which routes CAN handle this?" (hard gates only).
LLM selection answers "which route SHOULD handle this?" (intent).
instant_route is only set by the caller (e.g. editor-type pin in grpc_service), never here.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from orchestrator.routes.route_schema import EngineType, Route

logger = logging.getLogger(__name__)


@dataclass
class ScoredRoute:
    """Route with optional score/reason; kept for compatibility with callers that expect it."""

    route: Route
    score: float = 0.0
    reason: str = ""


def _editor_type_from_context(editor_context: Optional[Dict[str, Any]]) -> Optional[str]:
    """Extract normalized editor type from context."""
    if not editor_context:
        return None
    editor_type = (editor_context.get("type") or "").strip().lower()
    if not editor_type:
        fn = (editor_context.get("filename") or "").lower()
        lang = (editor_context.get("language") or "").lower()
        if fn.endswith(".org") or lang == "org":
            return "org"
    return editor_type or None


class RouteRegistry:
    """Singleton registry of routes with eligibility filtering (no scoring)."""

    _instance: Optional["RouteRegistry"] = None

    def __init__(self) -> None:
        self._routes: Dict[str, Route] = {}

    @classmethod
    def get_instance(cls) -> "RouteRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, route: Route) -> None:
        """Register a route by name."""
        self._routes[route.name] = route
        logger.debug("Registered route: %s (engine=%s)", route.name, route.engine)

    def unregister(self, name: str) -> None:
        """Remove a route by name. Used to clear custom agent routes when switching users."""
        if name in self._routes:
            self._routes.pop(name)
            logger.debug("Unregistered route: %s", name)

    def register_many(self, routes: List[Route]) -> None:
        for r in routes:
            self.register(r)

    def get(self, name: str) -> Optional[Route]:
        """Get route by name. Accepts both agent-style (e.g. weather_agent) and short (weather)."""
        if name in self._routes:
            return self._routes[name]
        if name.endswith("_agent") and name[:-6] in self._routes:
            return self._routes[name[:-6]]
        return self._routes.get(name)

    def get_for_engine(self, engine: EngineType) -> List[Route]:
        """Return all routes for an engine."""
        return [r for r in self._routes.values() if r.engine == engine]

    def filter_eligible(
        self,
        query: str,
        editor_context: Optional[Dict[str, Any]] = None,
        conversation_context: Optional[Dict[str, Any]] = None,
        editor_preference: str = "prefer",
        has_image_context: bool = False,
    ) -> Tuple[List[Route], Optional[str]]:
        """
        Return (eligible_routes, instant_route).

        instant_route is always None here. Caller may set instant_route for editor-type pin only.

        Hard gates only: internal_only, requires_editor, editor_type mismatch,
        requires_image_context. When DISABLE_EDITOR_ENGINE_ROUTING is True, editor-type
        CUSTOM_AGENT routes (requires_editor) are excluded so editor-doc requests fall
        through to chat or default Agent Factory profile.
        """
        if not self._routes:
            return [], None

        try:
            from config.settings import settings
            disable_editor_routing = getattr(settings, "DISABLE_EDITOR_ENGINE_ROUTING", False)
        except Exception:
            disable_editor_routing = False

        editor_type = _editor_type_from_context(editor_context)
        eligible: List[Route] = []

        # Document types that have a dedicated editor route (fiction_editing, outline_editing, etc.).
        # When user has such a document open, "generate" means "generate in this document", not "create new file".
        editor_document_types = frozenset(
            et
            for r in self._routes.values()
            if r.editor_types and r.requires_editor
            for et in r.editor_types
        )

        for route in self._routes.values():
            if route.internal_only:
                continue
            if disable_editor_routing and route.engine == EngineType.CUSTOM_AGENT and route.requires_editor:
                continue
            if route.requires_image_context and not has_image_context:
                continue
            if route.requires_editor and not editor_context:
                continue
            if route.editor_types and editor_context is not None:
                if editor_type not in route.editor_types:
                    continue
            # document_creator creates new files/folders; when an editor doc is open, prefer editor routes.
            if route.name == "document_creator" and editor_context and editor_type in editor_document_types:
                continue
            eligible.append(route)

        return eligible, None


def get_route_registry() -> RouteRegistry:
    """Return the singleton route registry (caller must load definitions)."""
    return RouteRegistry.get_instance()

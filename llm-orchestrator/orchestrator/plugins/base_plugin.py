"""
Base class for Agent Factory plugins (Zone 4 - third-party SaaS integrations).

Plugins provide tools that are registered with the Action I/O Registry and can be
assigned to custom agents. Each plugin declares connection requirements and can
validate credentials before use.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel


class PluginToolSpec:
    """Specification for one tool provided by a plugin (for registration)."""

    def __init__(
        self,
        name: str,
        category: str,
        description: str,
        inputs_model: Type[BaseModel],
        outputs_model: Type[BaseModel],
        tool_function: Callable[..., Any],
        params_model: Optional[Type[BaseModel]] = None,
    ):
        self.name = name
        self.category = category
        self.description = description
        self.inputs_model = inputs_model
        self.outputs_model = outputs_model
        self.tool_function = tool_function
        self.params_model = params_model


class BasePlugin(ABC):
    """
    Base class for external integration plugins.

    Plugins are self-contained and provide:
    - get_tools(): Tool specs (name, I/O models, callable) for Action I/O Registry
    - get_connection_requirements(): What credentials the plugin needs
    - configure(): Initialize with user's connection config (called before use)
    - validate_credentials(): Check if current config is valid
    """

    @property
    @abstractmethod
    def plugin_name(self) -> str:
        """Plugin identifier, e.g. 'trello', 'caldav'."""
        ...

    @property
    @abstractmethod
    def plugin_version(self) -> str:
        """Semantic version, e.g. '0.1.0'."""
        ...

    @abstractmethod
    def get_tools(self) -> List[PluginToolSpec]:
        """Return tool specs for all tools this plugin provides."""
        ...

    @abstractmethod
    def get_connection_requirements(self) -> Dict[str, str]:
        """
        Return mapping of config key -> human-readable label.
        E.g. {"api_key": "Trello API Key", "token": "Trello Token"}.
        """
        ...

    def configure(self, connection_config: Dict[str, Any]) -> None:
        """
        Initialize the plugin with the user's connection credentials.
        Called once per agent execution before any tools are invoked.
        """
        self._config = connection_config or {}

    def validate_credentials(self) -> bool:
        """
        Check if the current connection config is valid (e.g. API ping).
        Override to perform real validation; default checks required keys are present.
        """
        req = self.get_connection_requirements()
        config = getattr(self, "_config", None) or {}
        return all(config.get(k) for k in req)

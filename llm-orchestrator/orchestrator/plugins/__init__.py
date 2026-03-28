"""
Agent Factory plugins - third-party SaaS integrations (Zone 4 tools).

Plugins are discovered at startup and register their tools with the Action I/O Registry.
At runtime, agents that use a plugin get tools from the plugin configured with the user's connection.
"""

from orchestrator.plugins.base_plugin import BasePlugin
from orchestrator.plugins.plugin_loader import load_plugins, get_plugin_loader, discover_plugin_names

__all__ = [
    "BasePlugin",
    "load_plugins",
    "get_plugin_loader",
    "discover_plugin_names",
]

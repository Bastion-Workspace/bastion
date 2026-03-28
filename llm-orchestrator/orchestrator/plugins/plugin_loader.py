"""
Plugin loader - discovers and registers Agent Factory plugins at startup.

Discovery: scans orchestrator/plugins/integrations/*.py for classes that inherit BasePlugin.
Registration: for each plugin, registers its tools with the Action I/O Registry so they
appear in GetActions and can be used in playbooks.
"""

import importlib.util
import logging
import pkgutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from orchestrator.plugins.base_plugin import BasePlugin, PluginToolSpec
from orchestrator.utils.action_io_registry import register_action

logger = logging.getLogger(__name__)

_PLUGIN_LOADER: Optional["PluginLoader"] = None


class PluginLoader:
    """
    Discovers and loads tool plugins at startup and on-demand.

    Discovery sources (in order):
    1. Built-in: orchestrator/plugins/integrations/*.py (classes inheriting BasePlugin)
    2. Optional: ENABLED_PLUGINS env var to filter which to load
    """

    def __init__(self) -> None:
        self._plugins: Dict[str, BasePlugin] = {}
        self._discovered: bool = False

    def discover_plugins(self) -> List[str]:
        """Scan integrations package and return available plugin names."""
        if self._discovered:
            return list(self._plugins.keys())
        try:
            import orchestrator.plugins.integrations as integrations  # noqa: F401
            pkg_path = Path(integrations.__file__).parent
            for mod_info in pkgutil.iter_modules([str(pkg_path)]):
                mod_name = f"orchestrator.plugins.integrations.{mod_info.name}"
                try:
                    spec = importlib.util.find_spec(mod_name)
                    if spec is None or spec.loader is None:
                        continue
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    for attr_name in dir(mod):
                        obj = getattr(mod, attr_name)
                        if (
                            isinstance(obj, type)
                            and issubclass(obj, BasePlugin)
                            and obj is not BasePlugin
                        ):
                            instance = obj()
                            name = instance.plugin_name
                            self._plugins[name] = instance
                            logger.info("Discovered plugin: %s (%s)", name, instance.plugin_version)
                except Exception as e:
                    logger.warning("Failed to load plugin module %s: %s", mod_name, e)
            self._discovered = True
        except Exception as e:
            logger.warning("Plugin discovery failed: %s", e)
        return list(self._plugins.keys())

    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """Return plugin instance by name, or None."""
        if not self._discovered:
            self.discover_plugins()
        return self._plugins.get(plugin_name)

    def register_plugin_with_registry(self, plugin: BasePlugin) -> None:
        """Register all of a plugin's tools with the Action I/O Registry."""
        for spec in plugin.get_tools():
            register_action(
                name=spec.name,
                category=f"plugin:{plugin.plugin_name}",
                description=spec.description,
                inputs_model=spec.inputs_model,
                outputs_model=spec.outputs_model,
                tool_function=spec.tool_function,
                params_model=spec.params_model,
            )
        logger.info("Registered %d tools for plugin %s", len(plugin.get_tools()), plugin.plugin_name)

    def load_and_register_all(self) -> List[str]:
        """Discover all plugins and register their tools. Returns list of plugin names."""
        names = self.discover_plugins()
        for name in names:
            plugin = self._plugins.get(name)
            if plugin:
                try:
                    self.register_plugin_with_registry(plugin)
                except Exception as e:
                    logger.exception("Failed to register plugin %s: %s", name, e)
        return names


def get_plugin_loader() -> PluginLoader:
    """Return the singleton plugin loader."""
    global _PLUGIN_LOADER
    if _PLUGIN_LOADER is None:
        _PLUGIN_LOADER = PluginLoader()
    return _PLUGIN_LOADER


def load_plugins() -> List[str]:
    """Discover and register all plugins. Call at orchestrator startup. Returns plugin names."""
    return get_plugin_loader().load_and_register_all()


def discover_plugin_names() -> List[str]:
    """Return list of discovered plugin names without registering."""
    return get_plugin_loader().discover_plugins()

"""gRPC handler mixins for ToolServiceImplementation.

Each mixin provides a domain-specific set of gRPC handler methods.
ToolServiceImplementation composes all mixins via multiple inheritance.
"""

# -- First refactor wave (unchanged) --
from .org_todo_handlers import OrgTodoHandlersMixin
from .web_handlers import WebHandlersMixin
from .email_m365_handlers import EmailM365HandlersMixin
from .agent_messaging_handlers import AgentMessagingHandlersMixin

# -- Phase 1: extracted from main file --
from .rss_handlers import RssHandlersMixin
from .media_handlers import MediaHandlersMixin
from .data_workspace_handlers import DataWorkspaceHandlersMixin
from .navigation_handlers import NavigationHandlersMixin
from .search_utility_handlers import SearchUtilityHandlersMixin
from .analysis_handlers import AnalysisHandlersMixin

# -- Phase 2: split from agent_data_handlers --
from .agent_profile_handlers import AgentProfileHandlersMixin
from .agent_skills_handlers import AgentSkillsHandlersMixin
from .agent_runtime_handlers import AgentRuntimeHandlersMixin
from .agent_execution_trace_handlers import AgentExecutionTraceHandlersMixin
from .connector_mcp_handlers import ConnectorMcpHandlersMixin

# -- Phase 2: split from agent_factory_handlers --
from .data_connector_builder_handlers import DataConnectorBuilderHandlersMixin
from .control_pane_handlers import ControlPaneHandlersMixin
from .agent_factory_crud_handlers import AgentFactoryCrudHandlersMixin

__all__ = [
    "OrgTodoHandlersMixin",
    "WebHandlersMixin",
    "EmailM365HandlersMixin",
    "AgentMessagingHandlersMixin",
    "RssHandlersMixin",
    "MediaHandlersMixin",
    "DataWorkspaceHandlersMixin",
    "NavigationHandlersMixin",
    "SearchUtilityHandlersMixin",
    "AnalysisHandlersMixin",
    "AgentProfileHandlersMixin",
    "AgentSkillsHandlersMixin",
    "AgentRuntimeHandlersMixin",
    "AgentExecutionTraceHandlersMixin",
    "ConnectorMcpHandlersMixin",
    "DataConnectorBuilderHandlersMixin",
    "ControlPaneHandlersMixin",
    "AgentFactoryCrudHandlersMixin",
]

"""
Centralized Tool Registry - Roosevelt's "Tool Command Center"
LangGraph best practice implementation for agent tool access control
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from enum import Enum

try:
    from langchain_core.tools import StructuredTool
    from langgraph.prebuilt import ToolNode
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StructuredTool = None
    ToolNode = None

logger = logging.getLogger(__name__)


class AgentType(str, Enum):
    """Agent types for tool access control"""
    RESEARCH_AGENT = "research_agent"
    CHAT_AGENT = "chat_agent"
    # CODING_AGENT removed - not fully fleshed out
    REPORT_AGENT = "report_agent"
    # DATA_FORMATTING_AGENT removed - migrated to llm-orchestrator gRPC service
    # WEATHER_AGENT removed - migrated to llm-orchestrator gRPC service
    CALCULATE_AGENT = "calculate_agent"  # ROOSEVELT'S COMPUTATION SPECIALIST
    RSS_BACKGROUND_AGENT = "rss_background_agent"
    RSS_AGENT = "rss_agent"
    # ORG_INBOX_AGENT removed - migrated to llm-orchestrator gRPC service
    # ORG_PROJECT_AGENT removed - migrated to llm-orchestrator gRPC service
    # IMAGE_GENERATION_AGENT removed - migrated to llm-orchestrator gRPC service
    # WARGAMING_AGENT removed - not fully fleshed out
    # PROOFREADING_AGENT removed - migrated to llm-orchestrator gRPC service
    # WEBSITE_CRAWLER_AGENT removed - migrated to llm-orchestrator gRPC service
    # Content and Writing Agents
    FICTION_EDITING_AGENT = "fiction_editing_agent"
    OUTLINE_EDITING_AGENT = "outline_editing_agent"
    # CHARACTER_DEVELOPMENT_AGENT removed - migrated to llm-orchestrator gRPC service
    # RULES_EDITING_AGENT removed - migrated to llm-orchestrator gRPC service
    STYLE_EDITING_AGENT = "style_editing_agent"
    # SYSML_AGENT removed - not fully fleshed out
    # STORY_ANALYSIS_AGENT removed - migrated to llm-orchestrator gRPC service
    # CONTENT_ANALYSIS_AGENT removed - migrated to llm-orchestrator gRPC service
    # FACT_CHECKING_AGENT removed - not actively used
    # SITE_CRAWL_AGENT removed - migrated to llm-orchestrator gRPC service
    # Intent and Intelligence Agents
    # DEPRECATED: SIMPLE_INTENT_AGENT removed - intent classification now in llm-orchestrator
    # SIMPLE_INTENT_AGENT = "simple_intent_agent"
    # PERMISSION_INTELLIGENCE_AGENT removed - not used, functionality handled elsewhere
    # Pipeline Agent
    PIPELINE_AGENT = "pipeline_agent"
    # Template Agent
    TEMPLATE_AGENT = "template_agent"
    # EMAIL_AGENT removed - not used, functionality not migrated
    # PODCAST_SCRIPT_AGENT removed - migrated to llm-orchestrator gRPC service
    # SUBSTACK_AGENT removed - migrated to llm-orchestrator gRPC service
    MESSAGING_AGENT = "messaging_agent"  # ROOSEVELT'S MESSAGING CAVALRY!
    # ENTERTAINMENT_AGENT removed - migrated to llm-orchestrator gRPC service



class ToolAccessLevel(str, Enum):
    """Tool access levels for security"""
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    WEB_ACCESS = "web_access"
    SYSTEM_MODIFY = "system_modify"


class ToolDefinition:
    """Definition of a tool with metadata and access control"""
    
    def __init__(
        self,
        name: str,
        function: Callable,
        description: str,
        access_level: ToolAccessLevel,
        parameters: Dict[str, Any],
        timeout_seconds: int = 30,
        categories: List[Any] = None
    ):
        self.name = name
        self.function = function
        self.description = description
        self.access_level = access_level
        self.parameters = parameters
        self.timeout_seconds = timeout_seconds
        self.categories = categories or []


class CentralizedToolRegistry:
    """
    Roosevelt's Centralized Tool Registry
    
    LangGraph best practice: Single source of truth for agent tool access
    """
    
    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._agent_permissions: Dict[AgentType, Dict[str, ToolAccessLevel]] = {}
        self._initialized = False
        
    async def initialize(self):
        """Initialize the tool registry with all available tools"""
        try:
            logger.info("🔧 Initializing Centralized Tool Registry...")
            
            # Register all available tools
            await self._register_search_tools()
            await self._register_document_tools()
            await self._register_web_tools()
            await self._register_analysis_tools()
            await self._register_math_tools()
            await self._register_weather_tools()
            await self._register_aws_pricing_tools()
            await self._register_expansion_tools()
            await self._register_org_inbox_tools()
            await self._register_org_search_tools()  # **BULLY!** Org file search across all .org files!
            await self._register_image_tools()
            await self._register_messaging_tools()  # **BULLY!** Messaging cavalry tools!
            await self._register_file_creation_tools()  # **BULLY!** File and folder creation for agents!
            await self._register_document_editing_tools()  # **BULLY!** Document editing tools for agents!
            
            # Set up agent permissions
            self._configure_agent_permissions()
            
            self._initialized = True
            logger.info(f"✅ Tool Registry initialized with {len(self._tools)} tools")
            
        except Exception as e:
            logger.error(f"❌ Tool Registry initialization failed: {e}")
            raise
    
    async def _register_search_tools(self):
        """Register search and retrieval tools"""
        from services.langgraph_tools.unified_search_tools import (
            unified_local_search, get_document_content, search_conversation_cache
        )
        
        self._tools["search_local"] = ToolDefinition(
            name="search_local",
            function=unified_local_search,
            description="Search local documents and entities",
            access_level=ToolAccessLevel.READ_ONLY,
            parameters={
                "query": {"type": "string", "required": True},
                "limit": {"type": "integer", "default": 50},
                "search_types": {
                    "type": "array", 
                    "items": {"type": "string"}, 
                    "default": ["vector", "entities"],
                    "description": "Types of search to perform"
                }
            }
        )
        
        self._tools["get_document"] = ToolDefinition(
            name="get_document",
            function=get_document_content,
            description="Retrieve full document content by ID",
            access_level=ToolAccessLevel.READ_ONLY,
            parameters={
                "document_id": {"type": "string", "required": True}
            }
        )
        
        self._tools["search_conversation_cache"] = ToolDefinition(
            name="search_conversation_cache", 
            function=search_conversation_cache,
            description="ROOSEVELT'S UNIVERSAL CACHE: Search conversation history for previous research/chat work before doing new searches",
            access_level=ToolAccessLevel.READ_ONLY,
            parameters={
                "query": {"type": "string", "required": True, "description": "Query to search for in conversation cache"},
                "conversation_id": {"type": "string", "required": False, "description": "Conversation ID (auto-detected if not provided)"},
                "freshness_hours": {"type": "integer", "default": 24, "description": "How recent cache should be (hours)"}
            }
        )
    
    async def _register_web_tools(self):
        """Register web search and crawling tools"""
        # Import from correct locations
        from services.langgraph_tools.web_content_tools import (
            search_web
        )
        from services.langgraph_tools.crawl4ai_web_tools import (
            crawl_web_content
        )
        
        self._tools["search_web"] = ToolDefinition(
            name="search_web",
            function=search_web,
            description="Search the web for information",
            access_level=ToolAccessLevel.WEB_ACCESS,
            parameters={
                "query": {"type": "string", "required": True},
                "num_results": {"type": "integer", "default": 15}
            },
            timeout_seconds=60
        )
        
        self._tools["crawl_web_content"] = ToolDefinition(
            name="crawl_web_content",
            function=crawl_web_content,
            description="Crawl and extract web content - Roosevelt's fixed version",
            access_level=ToolAccessLevel.WEB_ACCESS,
            parameters={
                "url": {"type": "string", "required": False},
                "urls": {"type": "array", "items": {"type": "string"}, "required": False, "description": "List of URLs to crawl"}
            },
            timeout_seconds=90
        )
    
    async def _register_document_tools(self):
        """Register document analysis tools"""
        from services.langgraph_tools.content_analysis_tools import (
            summarize_content, analyze_documents
        )
        
        self._tools["summarize_content"] = ToolDefinition(
            name="summarize_content",
            function=summarize_content,
            description="Summarize content for analysis",
            access_level=ToolAccessLevel.READ_ONLY,
            parameters={
                "content": {"type": "string", "required": True},
                "max_length": {"type": "integer", "default": 500}
            }
        )
        
        self._tools["analyze_documents"] = ToolDefinition(
            name="analyze_documents",
            function=analyze_documents,
            description="Analyze document content and structure",
            access_level=ToolAccessLevel.READ_ONLY,
            parameters={
                "documents": {"type": "array", "items": {"type": "object"}, "required": True, "description": "List of documents to analyze"}
            }
        )
    
    async def _register_analysis_tools(self):
        """Register analysis and processing tools"""
        # format_data tool removed - DataFormattingAgent migrated to llm-orchestrator gRPC service
        pass
    
    async def _register_math_tools(self):
        """Register mathematical computation tools"""
        from services.langgraph_tools.math_tools import calculate, convert_units, solve_equation
        
        self._tools["calculate"] = ToolDefinition(
            name="calculate",
            function=calculate,
            description="Perform mathematical calculations",
            access_level=ToolAccessLevel.READ_ONLY,
            parameters={
                "expression": {"type": "string", "required": True}
            }
        )
        
        self._tools["convert_units"] = ToolDefinition(
            name="convert_units",
            function=convert_units,
            description="Convert between different units",
            access_level=ToolAccessLevel.READ_ONLY,
            parameters={
                "value": {"type": "number", "required": True},
                "from_unit": {"type": "string", "required": True},
                "to_unit": {"type": "string", "required": True}
            }
        )
        
        self._tools["solve_equation"] = ToolDefinition(
            name="solve_equation",
            function=solve_equation,
            description="Solve mathematical equations",
            access_level=ToolAccessLevel.READ_ONLY,
            parameters={
                "equation": {"type": "string", "required": True}
            }
        )
    
    async def _register_weather_tools(self):
        """Register weather information tools"""
        from services.langgraph_tools.weather_tools import weather_conditions, weather_forecast
        
        self._tools["weather_conditions"] = ToolDefinition(
            name="weather_conditions",
            function=weather_conditions,
            description="Get current weather conditions",
            access_level=ToolAccessLevel.WEB_ACCESS,
            parameters={
                "location": {"type": "string", "required": True}
            }
        )
        
        self._tools["weather_forecast"] = ToolDefinition(
            name="weather_forecast",
            function=weather_forecast,
            description="Get weather forecast",
            access_level=ToolAccessLevel.WEB_ACCESS,
            parameters={
                "location": {"type": "string", "required": True},
                "days": {"type": "integer", "default": 5}
            }
        )
    
    async def _register_aws_pricing_tools(self):
        """Register AWS pricing calculator tools"""
        from services.langgraph_tools.aws_pricing_tools import (
            estimate_aws_costs, get_aws_service_pricing, compare_aws_regions, estimate_aws_workload
        )
        
        self._tools["estimate_aws_costs"] = ToolDefinition(
            name="estimate_aws_costs",
            function=estimate_aws_costs,
            description="Calculate estimated AWS costs for specific service configurations and usage patterns",
            access_level=ToolAccessLevel.WEB_ACCESS,
            parameters={
                "service_type": {"type": "string", "required": True},
                "configuration": {"type": "object", "required": True},
                "usage_metrics": {"type": "object", "required": True},
                "region": {"type": "string", "default": "us-east-1"}
            }
        )
        
        self._tools["get_aws_service_pricing"] = ToolDefinition(
            name="get_aws_service_pricing",
            function=get_aws_service_pricing,
            description="Get current AWS pricing information for specific services and configurations",
            access_level=ToolAccessLevel.WEB_ACCESS,
            parameters={
                "service_type": {"type": "string", "required": True},
                "region": {"type": "string", "default": "us-east-1"},
                "filters": {"type": "object", "default": {}}
            }
        )
        
        self._tools["compare_aws_regions"] = ToolDefinition(
            name="compare_aws_regions", 
            function=compare_aws_regions,
            description="Compare AWS service costs across different regions for cost optimization",
            access_level=ToolAccessLevel.WEB_ACCESS,
            parameters={
                "service_type": {"type": "string", "required": True},
                "configuration": {"type": "object", "required": True},
                "regions": {"type": "array", "items": {"type": "string"}, "default": ["us-east-1", "us-west-2", "eu-west-1"]},
                "usage_metrics": {"type": "object", "default": {"hours_per_month": 730}}
            }
        )
        
        self._tools["estimate_aws_workload"] = ToolDefinition(
            name="estimate_aws_workload",
            function=estimate_aws_workload,
            description="Estimate total costs for complete AWS workloads with multiple services",
            access_level=ToolAccessLevel.WEB_ACCESS,
            parameters={
                "workload_name": {"type": "string", "required": True},
                "services": {"type": "array", "items": {"type": "object"}, "required": True},
                "region": {"type": "string", "default": "us-east-1"}
            }
        )

    async def _register_expansion_tools(self):
        """Register query expansion tools"""
        from services.langgraph_tools.query_expansion_tool import expand_query_universal
        
        self._tools["expand_query"] = ToolDefinition(
            name="expand_query",
            function=expand_query_universal,
            description="Generate alternative search queries for better recall",
            access_level=ToolAccessLevel.READ_ONLY,
            parameters={
                "original_query": {"type": "string", "required": True},
                "num_expansions": {"type": "integer", "default": 2},
                "expansion_type": {"type": "string", "default": "semantic"}
            }
        )

    async def _register_org_inbox_tools(self):
        """Register org inbox tools"""
        try:
            from services.langgraph_tools.org_inbox_tools import (
                org_inbox_path,
                org_inbox_list_items,
                org_inbox_add_item,
                org_inbox_toggle_done,
                org_inbox_update_line,
                org_inbox_append_text,
                org_inbox_append_block,
                org_inbox_index_tags,
                org_inbox_apply_tags,
                org_inbox_set_state,
                org_inbox_promote_state,
                org_inbox_demote_state,
                org_inbox_set_schedule_and_repeater,
            )

            self._tools["org_inbox_path"] = ToolDefinition(
                name="org_inbox_path",
                function=org_inbox_path,
                description="Get path to inbox.org, creating it at uploads root if missing",
                access_level=ToolAccessLevel.READ_ONLY,
                parameters={},
                timeout_seconds=10,
            )
            self._tools["org_inbox_list_items"] = ToolDefinition(
                name="org_inbox_list_items",
                function=org_inbox_list_items,
                description="List task-like items from inbox.org (checkboxes and TODO headings)",
                access_level=ToolAccessLevel.READ_ONLY,
                parameters={},
                timeout_seconds=10,
            )
            self._tools["org_inbox_add_item"] = ToolDefinition(
                name="org_inbox_add_item",
                function=org_inbox_add_item,
                description="Append a new checkbox or TODO heading to inbox.org",
                access_level=ToolAccessLevel.READ_WRITE,
                parameters={
                    "text": {"type": "string", "required": True},
                    "kind": {"type": "string", "enum": ["checkbox", "todo"], "default": "checkbox"}
                },
                timeout_seconds=10,
            )
            self._tools["org_inbox_toggle_done"] = ToolDefinition(
                name="org_inbox_toggle_done",
                function=org_inbox_toggle_done,
                description="Toggle done state of a task line by index",
                access_level=ToolAccessLevel.READ_WRITE,
                parameters={"line_index": {"type": "integer", "required": True}},
                timeout_seconds=10,
            )
            self._tools["org_inbox_update_line"] = ToolDefinition(
                name="org_inbox_update_line",
                function=org_inbox_update_line,
                description="Update the text of a task line by index",
                access_level=ToolAccessLevel.READ_WRITE,
                parameters={
                    "line_index": {"type": "integer", "required": True},
                    "new_text": {"type": "string", "required": True}
                },
                timeout_seconds=10,
            )
            self._tools["org_inbox_append_text"] = ToolDefinition(
                name="org_inbox_append_text",
                function=org_inbox_append_text,
                description="Append arbitrary org-mode content to inbox.org",
                access_level=ToolAccessLevel.READ_WRITE,
                parameters={
                    "content": {"type": "string", "required": True}
                },
                timeout_seconds=10,
            )
            self._tools["org_inbox_append_block"] = ToolDefinition(
                name="org_inbox_append_block",
                function=org_inbox_append_block,
                description="Append a multi-line Org block to inbox.org and return inserted line range",
                access_level=ToolAccessLevel.READ_WRITE,
                parameters={
                    "block": {"type": "string", "required": True}
                },
                timeout_seconds=10,
            )
            self._tools["org_inbox_index_tags"] = ToolDefinition(
                name="org_inbox_index_tags",
                function=org_inbox_index_tags,
                description="Scan .org files and return a tag frequency map",
                access_level=ToolAccessLevel.READ_ONLY,
                parameters={},
                timeout_seconds=15,
            )
            self._tools["org_inbox_apply_tags"] = ToolDefinition(
                name="org_inbox_apply_tags",
                function=org_inbox_apply_tags,
                description="Apply tags to a specific line in inbox.org",
                access_level=ToolAccessLevel.READ_WRITE,
                parameters={
                    "line_index": {"type": "integer", "required": True},
                    "tags": {"type": "array", "items": {"type": "string"}, "required": True}
                },
                timeout_seconds=10,
            )
            self._tools["org_inbox_set_state"] = ToolDefinition(
                name="org_inbox_set_state",
                function=org_inbox_set_state,
                description="Set the TODO state for a headline line",
                access_level=ToolAccessLevel.READ_WRITE,
                parameters={
                    "line_index": {"type": "integer", "required": True},
                    "state": {"type": "string", "required": True}
                },
                timeout_seconds=10,
            )
            self._tools["org_inbox_promote_state"] = ToolDefinition(
                name="org_inbox_promote_state",
                function=org_inbox_promote_state,
                description="Promote a headline's state using the configured sequence",
                access_level=ToolAccessLevel.READ_WRITE,
                parameters={
                    "line_index": {"type": "integer", "required": True},
                    "sequence": {"type": "array", "items": {"type": "string"}, "required": True}
                },
                timeout_seconds=10,
            )
            self._tools["org_inbox_demote_state"] = ToolDefinition(
                name="org_inbox_demote_state",
                function=org_inbox_demote_state,
                description="Demote a headline's state using the configured sequence",
                access_level=ToolAccessLevel.READ_WRITE,
                parameters={
                    "line_index": {"type": "integer", "required": True},
                    "sequence": {"type": "array", "items": {"type": "string"}, "required": True}
                },
                timeout_seconds=10,
            )
            self._tools["org_inbox_set_schedule_and_repeater"] = ToolDefinition(
                name="org_inbox_set_schedule_and_repeater",
                function=org_inbox_set_schedule_and_repeater,
                description="Set or update SCHEDULED and repeater on a headline",
                access_level=ToolAccessLevel.READ_WRITE,
                parameters={
                    "line_index": {"type": "integer", "required": True},
                    "scheduled": {"type": "string", "required": False},
                    "repeater": {"type": "string", "required": False}
                },
                timeout_seconds=10,
            )
        except Exception as e:
            logger.error(f"❌ Failed to register org inbox tools: {e}")

    async def _register_org_search_tools(self):
        """Register org-mode search tools for searching across all .org files"""
        try:
            from services.langgraph_tools.org_search_tools import (
                search_org_files,
                list_org_todos,
                search_org_by_tag
            )
            
            self._tools["search_org_files"] = ToolDefinition(
                name="search_org_files",
                function=search_org_files,
                description="Search across all user's org-mode files with full-text and metadata support. Can filter by tags and TODO states.",
                access_level=ToolAccessLevel.READ_ONLY,
                parameters={
                    "query": {"type": "string", "required": True, "description": "Search query string"},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Filter by org tags"},
                    "todo_states": {"type": "array", "items": {"type": "string"}, "description": "Filter by TODO states"},
                    "include_content": {"type": "boolean", "default": True},
                    "limit": {"type": "integer", "default": 50}
                },
                timeout_seconds=30
            )
            
            self._tools["list_org_todos"] = ToolDefinition(
                name="list_org_todos",
                function=list_org_todos,
                description="List all TODO items from user's org files. Quick way to see tasks across all org files.",
                access_level=ToolAccessLevel.READ_ONLY,
                parameters={
                    "todo_states": {"type": "array", "items": {"type": "string"}, "description": "Filter by states"},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Filter by tags"},
                    "limit": {"type": "integer", "default": 50}
                },
                timeout_seconds=20
            )
            
            self._tools["search_org_by_tag"] = ToolDefinition(
                name="search_org_by_tag",
                function=search_org_by_tag,
                description="Search org files by a specific tag",
                access_level=ToolAccessLevel.READ_ONLY,
                parameters={
                    "tag": {"type": "string", "required": True},
                    "limit": {"type": "integer", "default": 30}
                },
                timeout_seconds=20
            )
            
            logger.info("✅ Registered org search tools")
        except Exception as e:
            logger.error(f"❌ Failed to register org search tools: {e}")

    async def _register_image_tools(self):
        """Register image generation tools"""
        try:
            from services.langgraph_tools.image_generation_tools import generate_image

            self._tools["generate_image"] = ToolDefinition(
                name="generate_image",
                function=generate_image,
                description="Generate one or more images from a text prompt using the configured OpenRouter image model",
                access_level=ToolAccessLevel.WEB_ACCESS,
                parameters={
                    "prompt": {"type": "string", "required": True, "description": "Text prompt describing the image to generate"},
                    "size": {"type": "string", "default": "1024x1024", "description": "Image size as WIDTHxHEIGHT"},
                    "format": {"type": "string", "default": "png", "enum": ["png", "jpg", "jpeg", "webp"]},
                    "seed": {"type": "integer", "required": False, "description": "Random seed for reproducibility"},
                    "num_images": {"type": "integer", "default": 1, "description": "Number of images to generate (1-4)"},
                    "negative_prompt": {"type": "string", "required": False, "description": "What to avoid in the image"},
                },
                timeout_seconds=120,
            )
        except Exception as e:
            logger.error(f"❌ Failed to register image tools: {e}")
    
    async def _register_messaging_tools(self):
        """Register messaging tools for sending messages to rooms"""
        try:
            from services.langgraph_tools.messaging_tools import (
                get_user_rooms_tool,
                send_room_message_tool
            )
            
            self._tools["get_user_rooms"] = ToolDefinition(
                name="get_user_rooms",
                function=get_user_rooms_tool,
                description="Get list of user's chat rooms with participants and unread counts. Use this to find the correct room to send a message to.",
                access_level=ToolAccessLevel.READ_ONLY,
                parameters={
                    "user_id": {"type": "string", "required": True, "description": "User ID"},
                    "limit": {"type": "integer", "default": 20, "description": "Maximum rooms to return"}
                },
                timeout_seconds=10
            )
            
            self._tools["send_room_message"] = ToolDefinition(
                name="send_room_message",
                function=send_room_message_tool,
                description="Send a message to a specific chat room. Room ID should be obtained from get_user_rooms first.",
                access_level=ToolAccessLevel.READ_WRITE,
                parameters={
                    "user_id": {"type": "string", "required": True, "description": "User ID sending the message"},
                    "room_id": {"type": "string", "required": True, "description": "Room UUID to send message to"},
                    "message_content": {"type": "string", "required": True, "description": "The message text to send"},
                    "message_type": {"type": "string", "default": "text", "enum": ["text", "ai_share", "system"]}
                },
                timeout_seconds=10
            )
            
            logger.info("✅ BULLY! Registered messaging cavalry tools")
        except Exception as e:
            logger.error(f"❌ Failed to register messaging tools: {e}")
    
    async def _register_file_creation_tools(self):
        """Register file and folder creation tools for agents"""
        try:
            from services.langgraph_tools.file_creation_tools import (
                create_user_file,
                create_user_folder
            )
            
            self._tools["create_user_file"] = ToolDefinition(
                name="create_user_file",
                function=create_user_file,
                description="Create a file in the user's My Documents section. Only creates files in user's collection (not global). Can create folders automatically if folder_path is provided.",
                access_level=ToolAccessLevel.READ_WRITE,
                parameters={
                    "filename": {"type": "string", "required": True, "description": "Name of the file to create (e.g., 'sensor_spec.md', 'circuit_diagram.txt')"},
                    "content": {"type": "string", "required": True, "description": "File content as string"},
                    "folder_id": {"type": "string", "required": False, "description": "Optional folder ID to place file in (must be user's folder)"},
                    "folder_path": {"type": "string", "required": False, "description": "Optional folder path (e.g., 'Projects/Electronics') - will create folders if needed"},
                    "title": {"type": "string", "required": False, "description": "Optional document title (defaults to filename)"},
                    "tags": {"type": "array", "items": {"type": "string"}, "required": False, "description": "Optional list of tags for the document"},
                    "category": {"type": "string", "required": False, "description": "Optional category for the document"},
                    "user_id": {"type": "string", "required": True, "description": "User ID (must match the user making the request)"}
                },
                timeout_seconds=30
            )
            
            self._tools["create_user_folder"] = ToolDefinition(
                name="create_user_folder",
                function=create_user_folder,
                description="Create a folder in the user's My Documents section. Only creates folders in user's collection (not global). Can create parent folders automatically if parent_folder_path is provided.",
                access_level=ToolAccessLevel.READ_WRITE,
                parameters={
                    "folder_name": {"type": "string", "required": True, "description": "Name of the folder to create (e.g., 'Electronics Projects', 'Components')"},
                    "parent_folder_id": {"type": "string", "required": False, "description": "Optional parent folder ID (must be user's folder)"},
                    "parent_folder_path": {"type": "string", "required": False, "description": "Optional parent folder path (e.g., 'Projects') - will resolve to folder_id"},
                    "user_id": {"type": "string", "required": True, "description": "User ID (must match the user making the request)"}
                },
                timeout_seconds=30
            )
            
            logger.info("✅ Registered file creation tools")
            
        except Exception as e:
            logger.error(f"❌ Failed to register file creation tools: {e}")
    
    async def _register_document_editing_tools(self):
        """Register document editing tools for agents"""
        try:
            from services.langgraph_tools.document_editing_tools import (
                update_document_metadata_tool
            )
            
            self._tools["update_document_metadata"] = ToolDefinition(
                name="update_document_metadata",
                function=update_document_metadata_tool,
                description="Update document title and/or frontmatter type. Updates both database metadata and file content frontmatter. Only works on user's own documents.",
                access_level=ToolAccessLevel.READ_WRITE,
                parameters={
                    "document_id": {"type": "string", "required": True, "description": "Document ID to update"},
                    "title": {"type": "string", "required": False, "description": "New document title (updates database and frontmatter)"},
                    "frontmatter_type": {"type": "string", "required": False, "description": "Frontmatter type (e.g., 'electronics', 'fiction', 'rules') - updates file content"},
                    "user_id": {"type": "string", "required": True, "description": "User ID (must match document owner)"}
                },
                timeout_seconds=30
            )
            
            logger.info("✅ Registered document editing tools")
            
        except Exception as e:
            logger.error(f"❌ Failed to register document editing tools: {e}")
    
    def _configure_agent_permissions(self):
        """Configure which tools each agent type can access"""
        
        # Research Agent: Full research capabilities - ROOSEVELT'S CRAWL-ONLY STRATEGY
        # WEATHER TOOLS REMOVED: Research Agent should suggest Weather Agent collaboration instead!
        self._agent_permissions[AgentType.RESEARCH_AGENT] = {
            "expand_query": ToolAccessLevel.READ_ONLY,  # ROOSEVELT'S UNIVERSAL QUERY EXPANSION
            "search_conversation_cache": ToolAccessLevel.READ_ONLY,  # ROOSEVELT'S UNIVERSAL CACHE
            "search_local": ToolAccessLevel.READ_ONLY,
            "get_document": ToolAccessLevel.READ_ONLY,
            "search_web": ToolAccessLevel.WEB_ACCESS,  # RESTORED: Needed for fact-checking agent
            "crawl_web_content": ToolAccessLevel.WEB_ACCESS,
            "summarize_content": ToolAccessLevel.READ_ONLY,
            "analyze_documents": ToolAccessLevel.READ_ONLY,
            # format_data removed - DataFormattingAgent migrated to llm-orchestrator gRPC service
            "get_aws_service_pricing": ToolAccessLevel.WEB_ACCESS,  # AWS pricing research
            "compare_aws_regions": ToolAccessLevel.WEB_ACCESS,  # Regional cost analysis
            "estimate_aws_costs": ToolAccessLevel.WEB_ACCESS,  # Cost estimation research
            "estimate_aws_workload": ToolAccessLevel.WEB_ACCESS,  # Workload cost research
            # **BULLY!** ORG-MODE SEARCH TOOLS - Search user's reference org files!
            "search_org_files": ToolAccessLevel.READ_ONLY,  # Search across all .org files
            "list_org_todos": ToolAccessLevel.READ_ONLY,  # List TODO items
            "search_org_by_tag": ToolAccessLevel.READ_ONLY  # Search by org tags
            # CALCULATION TOOLS REMOVED: Use collaboration with Calculate Agent instead!
            # WEATHER TOOLS REMOVED: Use collaboration with Weather Agent instead!
        }
        

        
        # Chat Agent: Pure conversational intelligence + conversation cache access
        self._agent_permissions[AgentType.CHAT_AGENT] = {
            "search_conversation_cache": ToolAccessLevel.READ_ONLY,  # ROOSEVELT'S UNIVERSAL CACHE
            "calculate": ToolAccessLevel.READ_ONLY,
            "convert_units": ToolAccessLevel.READ_ONLY,
            # format_data removed - DataFormattingAgent migrated to llm-orchestrator gRPC service
            "get_aws_service_pricing": ToolAccessLevel.WEB_ACCESS,  # Basic AWS pricing queries
            "estimate_aws_costs": ToolAccessLevel.WEB_ACCESS,  # Simple cost estimates
            # **BULLY!** Quick org-mode TODO queries for casual questions
            "list_org_todos": ToolAccessLevel.READ_ONLY,  # "What's on my TODO list?"
            "search_org_by_tag": ToolAccessLevel.READ_ONLY  # "What's tagged @work?"
            # REMOVED: search_local, get_document - Chat Agent uses cache first, Research Agent handles external searches
        }
        
        # Report Agent: Document processing and formatting
        self._agent_permissions[AgentType.REPORT_AGENT] = {
            "summarize_content": ToolAccessLevel.READ_ONLY,
            "analyze_documents": ToolAccessLevel.READ_ONLY
        }
        
        # DATA_FORMATTING_AGENT removed - migrated to llm-orchestrator gRPC service
        # WEATHER_AGENT removed - migrated to llm-orchestrator gRPC service
        
        # Calculate Agent: Mathematical operations and unit conversions - ROOSEVELT'S COMPUTATION SPECIALIST
        self._agent_permissions[AgentType.CALCULATE_AGENT] = {
            "calculate": ToolAccessLevel.READ_ONLY,
            "convert_units": ToolAccessLevel.READ_ONLY,
            "solve_equation": ToolAccessLevel.READ_ONLY,
            "estimate_aws_costs": ToolAccessLevel.WEB_ACCESS,  # Cost calculations
            "get_aws_service_pricing": ToolAccessLevel.WEB_ACCESS,  # Pricing data for calculations
            "compare_aws_regions": ToolAccessLevel.WEB_ACCESS,  # Regional cost analysis
            "estimate_aws_workload": ToolAccessLevel.WEB_ACCESS  # Complex workload calculations
        }
        
        # CodingAgent removed - not fully fleshed out
        # Removed agent permissions
        
        # RSS agents
        self._agent_permissions[AgentType.RSS_BACKGROUND_AGENT] = {
            "rss_poll_feeds": ToolAccessLevel.WEB_ACCESS,
            "rss_process_articles": ToolAccessLevel.WEB_ACCESS,
            "rss_get_feeds": ToolAccessLevel.WEB_ACCESS,
            "rss_create_feed": ToolAccessLevel.WEB_ACCESS,
            "rss_update_feed": ToolAccessLevel.WEB_ACCESS,
            "rss_delete_feed": ToolAccessLevel.WEB_ACCESS
        }
        self._agent_permissions[AgentType.RSS_AGENT] = {
            "rss_get_feeds": ToolAccessLevel.WEB_ACCESS,
            "rss_create_feed": ToolAccessLevel.WEB_ACCESS,
            "rss_update_feed": ToolAccessLevel.WEB_ACCESS,
            "rss_delete_feed": ToolAccessLevel.WEB_ACCESS,
            "rss_poll_feeds": ToolAccessLevel.WEB_ACCESS
        }
        # ORG_INBOX_AGENT removed - migrated to llm-orchestrator gRPC service

        # ORG_PROJECT_AGENT removed - migrated to llm-orchestrator gRPC service

        # IMAGE_GENERATION_AGENT removed - migrated to llm-orchestrator gRPC service

        # WargamingAgent removed - not fully fleshed out
        # Removed agent permissions

        # PROOFREADING_AGENT removed - migrated to llm-orchestrator gRPC service
        
        # WEBSITE_CRAWLER_AGENT removed - migrated to llm-orchestrator gRPC service
        
        # Content and Writing Agents - Editor-interactive agents (no external tools needed)
        self._agent_permissions[AgentType.FICTION_EDITING_AGENT] = {
            # Editor-interactive: Works with active editor content, no external tools
        }
        
        self._agent_permissions[AgentType.OUTLINE_EDITING_AGENT] = {
            # Editor-interactive: Works with active editor content, no external tools
        }
        
        # CHARACTER_DEVELOPMENT_AGENT removed - migrated to llm-orchestrator gRPC service
        
        self._agent_permissions[AgentType.STYLE_EDITING_AGENT] = {
            "can_use_document_tools": True,
            "can_use_reference_tools": True,
            "can_use_web_tools": False,
        }
        
        # RULES_EDITING_AGENT removed - migrated to llm-orchestrator gRPC service
        
        # PODCAST_SCRIPT_AGENT removed - migrated to llm-orchestrator gRPC service
        
        # Analysis Agents - May need local search for reference material
        # STORY_ANALYSIS_AGENT removed - migrated to llm-orchestrator gRPC service
        # CONTENT_ANALYSIS_AGENT removed - migrated to llm-orchestrator gRPC service
        
        # FactCheckingAgent removed - not actively used
        # Removed agent permissions
        
        # SITE_CRAWL_AGENT removed - migrated to llm-orchestrator gRPC service
        
        # Intent and Intelligence Agents - Pure LLM-based (no external tools)
        # DEPRECATED: SIMPLE_INTENT_AGENT removed - intent classification now in llm-orchestrator
        # self._agent_permissions[AgentType.SIMPLE_INTENT_AGENT] = {
        #     # Pure LLM-based intent classification
        # }
        
        # PERMISSION_INTELLIGENCE_AGENT removed - not used, functionality handled elsewhere
        
        # Pipeline Agent - Design agent (no external tools)
        self._agent_permissions[AgentType.PIPELINE_AGENT] = {
            # Pipeline design and orchestration, no external tools
        }
        
        # Template Agent - Template for new agents (no tools)
        self._agent_permissions[AgentType.TEMPLATE_AGENT] = {
            # Template agent, configure tools as needed
        }
        
        # **BULLY!** MESSAGING AGENT: Send messages to chat rooms via natural language
        self._agent_permissions[AgentType.MESSAGING_AGENT] = {
            "get_user_rooms": ToolAccessLevel.READ_ONLY,
            "send_room_message": ToolAccessLevel.READ_WRITE
        }
    
    def get_tools_for_agent(self, agent_type: AgentType) -> List[str]:
        """Get list of tool names available to an agent type"""
        if not self._initialized:
            logger.warning("⚠️ Tool registry not initialized, returning empty list")
            return []
        
        permissions = self._agent_permissions.get(agent_type, {})
        available_tools = list(permissions.keys())
        
        logger.info(f"🔧 Agent {agent_type.value} has access to {len(available_tools)} tools: {available_tools}")
        return available_tools
    
    def get_tool_function(self, tool_name: str, agent_type: AgentType) -> Optional[Callable]:
        """Get tool function if agent has permission"""
        if not self._initialized:
            logger.warning("⚠️ Tool registry not initialized")
            return None
        
        # Check if tool exists
        if tool_name not in self._tools:
            logger.warning(f"⚠️ Tool {tool_name} not found in registry")
            return None
        
        # Check if agent has permission
        agent_permissions = self._agent_permissions.get(agent_type, {})
        if tool_name not in agent_permissions:
            logger.warning(f"⚠️ Agent {agent_type.value} does not have permission for tool {tool_name}")
            return None
        
        tool_def = self._tools[tool_name]
        logger.debug(f"🔧 Providing tool {tool_name} to agent {agent_type.value}")
        return tool_def.function
    
    def get_tool_objects_for_agent(self, agent_type: AgentType) -> List[Dict[str, Any]]:
        """Get tool objects for LangGraph (with function definitions)"""
        if not self._initialized:
            logger.warning("⚠️ Tool registry not initialized, returning empty list")
            return []
        
        permissions = self._agent_permissions.get(agent_type, {})
        available_tools = []
        
        for tool_name in permissions.keys():
            if tool_name in self._tools:
                tool_def = self._tools[tool_name]
                # Convert to LangGraph tool format with proper JSON Schema structure
                # ROOSEVELT'S SCHEMA FIX: Wrap parameters in proper JSON Schema object
                schema_parameters = self._convert_to_json_schema(tool_def.parameters)
                
                tool_obj = {
                    "type": "function",
                    "function": {
                        "name": tool_def.name,
                        "description": tool_def.description,
                        "parameters": schema_parameters
                    }
                }
                available_tools.append(tool_obj)
        
        logger.info(f"🔧 Agent {agent_type.value} has {len(available_tools)} tool objects: {[tool['function']['name'] for tool in available_tools]}")
        return available_tools
    
    def _convert_to_json_schema(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Convert simplified parameter format to proper JSON Schema object"""
        try:
            properties = {}
            required = []
            
            for param_name, param_config in parameters.items():
                # Extract type and other properties
                param_type = param_config.get("type", "string")
                param_description = param_config.get("description", f"Parameter {param_name}")
                param_default = param_config.get("default")
                is_required = param_config.get("required", False)
                
                # Build property definition
                property_def = {
                    "type": param_type,
                    "description": param_description
                }
                
                # Add default value if specified
                if param_default is not None:
                    property_def["default"] = param_default
                
                # Add enum if specified
                if "enum" in param_config:
                    property_def["enum"] = param_config["enum"]
                
                # Add array items if it's an array type
                if param_type == "array" and "items" in param_config:
                    property_def["items"] = param_config["items"]
                
                properties[param_name] = property_def
                
                # Track required parameters
                if is_required:
                    required.append(param_name)
            
            # Build proper JSON Schema object
            schema = {
                "type": "object",
                "properties": properties
            }
            
            # Add required array if we have required parameters
            if required:
                schema["required"] = required
            
            return schema
            
        except Exception as e:
            logger.error(f"❌ Failed to convert parameters to JSON Schema: {e}")
            # Fallback to minimal valid schema
            return {
                "type": "object",
                "properties": {},
                "required": []
            }

    def get_tool_metadata(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get tool metadata for validation"""
        if tool_name not in self._tools:
            return None
        
        tool_def = self._tools[tool_name]
        return {
            "name": tool_def.name,
            "description": tool_def.description,
            "parameters": tool_def.parameters,
            "access_level": tool_def.access_level.value,
            "timeout_seconds": tool_def.timeout_seconds
        }
    
    def has_web_access(self, agent_type: AgentType) -> bool:
        """Check if agent has any web access tools"""
        permissions = self._agent_permissions.get(agent_type, {})
        return any(
            level in [ToolAccessLevel.WEB_ACCESS, ToolAccessLevel.SYSTEM_MODIFY]
            for level in permissions.values()
        )
    
    def list_all_tools(self) -> Dict[str, Dict[str, Any]]:
        """List all available tools with metadata"""
        return {
            name: {
                "description": tool_def.description,
                "access_level": tool_def.access_level.value,
                "parameters": tool_def.parameters
            }
            for name, tool_def in self._tools.items()
        }
    
    def get_structured_tools_for_agent(self, agent_type: AgentType) -> List[Any]:
        """
        Get StructuredTool objects for an agent (LangGraph best practice)
        
        Returns list of StructuredTool objects that can be used with:
        - llm.bind_tools(tools) for LLM tool binding
        - ToolNode(tools) for automatic tool execution
        
        Args:
            agent_type: Agent type to get tools for
            
        Returns:
            List of StructuredTool objects
        """
        if not LANGGRAPH_AVAILABLE:
            logger.warning("⚠️ LangGraph not available, cannot create StructuredTool objects")
            return []
        
        if not self._initialized:
            logger.warning("⚠️ Tool registry not initialized, returning empty list")
            return []
        
        permissions = self._agent_permissions.get(agent_type, {})
        structured_tools = []
        
        for tool_name in permissions.keys():
            if tool_name in self._tools:
                tool_def = self._tools[tool_name]
                try:
                    structured_tool = self._convert_to_structured_tool(tool_def)
                    if structured_tool:
                        structured_tools.append(structured_tool)
                except Exception as e:
                    logger.error(f"❌ Failed to convert tool {tool_name} to StructuredTool: {e}")
                    continue
        
        logger.info(f"🔧 Agent {agent_type.value} has {len(structured_tools)} StructuredTool objects: {[tool.name for tool in structured_tools]}")
        return structured_tools
    
    def _convert_to_structured_tool(self, tool_def: ToolDefinition) -> Optional[Any]:
        """
        Convert ToolDefinition to StructuredTool object
        
        Args:
            tool_def: ToolDefinition to convert
            
        Returns:
            StructuredTool object or None if conversion fails
        """
        if not LANGGRAPH_AVAILABLE or StructuredTool is None:
            return None
        
        try:
            # Convert parameters to JSON Schema for StructuredTool
            schema_parameters = self._convert_to_json_schema(tool_def.parameters)
            
            # Create StructuredTool from function
            # StructuredTool.from_function handles async functions automatically
            structured_tool = StructuredTool.from_function(
                func=tool_def.function,
                name=tool_def.name,
                description=tool_def.description,
                args_schema=None,  # Let StructuredTool infer from function signature
                return_direct=False
            )
            
            # Override the schema if we have custom parameter definitions
            # This ensures our parameter descriptions and defaults are used
            if schema_parameters and schema_parameters.get("properties"):
                # Update tool's args_schema if needed
                # Note: StructuredTool.from_function may not support direct schema override
                # We rely on function signature + docstring for now
                pass
            
            return structured_tool
            
        except Exception as e:
            logger.error(f"❌ Failed to create StructuredTool for {tool_def.name}: {e}")
            return None
    
    def _get_core_tools_for_agent(self, agent_type: AgentType) -> List[Dict[str, Any]]:
        """
        Get core tools that are always loaded for an agent
        
        Args:
            agent_type: Agent type
            
        Returns:
            List of tool objects (core tools only)
        """
        from services.langgraph_tools.tool_categories import get_core_tools_for_agent
        
        agent_type_str = agent_type.value
        core_tool_names = get_core_tools_for_agent(agent_type_str)
        
        # Get tool objects for core tools
        core_tools = []
        permissions = self._agent_permissions.get(agent_type, {})
        
        for tool_name in core_tool_names:
            if tool_name in permissions and tool_name in self._tools:
                tool_def = self._tools[tool_name]
                schema_parameters = self._convert_to_json_schema(tool_def.parameters)
                
                tool_obj = {
                    "type": "function",
                    "function": {
                        "name": tool_def.name,
                        "description": tool_def.description,
                        "parameters": schema_parameters
                    }
                }
                core_tools.append(tool_obj)
        
        return core_tools
    
    def _get_tools_by_category(
        self,
        agent_type: AgentType,
        category: Any
    ) -> List[Dict[str, Any]]:
        """
        Get tools for a specific category that the agent has permission for
        
        Args:
            agent_type: Agent type
            category: ToolCategory enum value
            
        Returns:
            List of tool objects matching the category
        """
        from services.langgraph_tools.dynamic_tool_analyzer import DynamicToolAnalyzer
        
        analyzer = DynamicToolAnalyzer()
        tool_category_map = analyzer._map_tools_to_categories([])
        
        # Map category to tool names
        category_to_tools = {
            "search_local": ["search_local", "search_conversation_cache"],
            "search_web": ["search_web", "crawl_web_content"],
            "document_ops": ["get_document"],
            "analysis": ["analyze_documents"],
            "math": ["calculate"],
            "weather": ["get_weather"],
            "aws_pricing": ["get_aws_service_pricing", "compare_aws_regions", "estimate_aws_costs", "estimate_aws_workload"],
            "org_files": ["search_org_files", "list_org_todos", "search_org_by_tag"],
            "messaging": ["send_room_message", "get_user_rooms"],
            "file_creation": ["create_file", "create_folder"],
            "expansion": ["expand_query"],
            "image_generation": ["generate_image"],
            "website_crawl": ["crawl_website"],
        }
        
        category_str = category.value if hasattr(category, 'value') else str(category)
        tool_names = category_to_tools.get(category_str, [])
        
        # Get tool objects for these tools
        tools = []
        permissions = self._agent_permissions.get(agent_type, {})
        
        for tool_name in tool_names:
            if tool_name in permissions and tool_name in self._tools:
                tool_def = self._tools[tool_name]
                schema_parameters = self._convert_to_json_schema(tool_def.parameters)
                
                tool_obj = {
                    "type": "function",
                    "function": {
                        "name": tool_def.name,
                        "description": tool_def.description,
                        "parameters": schema_parameters
                    }
                }
                tools.append(tool_obj)
        
        return tools
    
    async def get_tools_for_agent_dynamic(
        self,
        agent_type: AgentType,
        query: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get tools dynamically based on query analysis
        
        Strategy:
        1. Always include "core" tools for agent (e.g., search_local for research)
        2. Analyze query for additional needs
        3. Load conditional tools based on analysis
        4. Return minimal focused tool set
        
        Args:
            agent_type: Agent type
            query: User query string
            metadata: Optional metadata with conversation context
            
        Returns:
            List of tool objects (core + conditional based on query)
        """
        if not self._initialized:
            logger.warning("⚠️ Tool registry not initialized, falling back to static loading")
            return self.get_tool_objects_for_agent(agent_type)
        
        # Check if dynamic loading is enabled
        import os
        enable_dynamic = os.getenv("ENABLE_DYNAMIC_TOOL_LOADING", "true").lower() == "true"
        
        if not enable_dynamic:
            logger.debug("Dynamic tool loading disabled, using static loading")
            return self.get_tool_objects_for_agent(agent_type)
        
        try:
            # Get core tools (always loaded)
            core_tools = self._get_core_tools_for_agent(agent_type)
            
            # Analyze query for additional needs
            from services.langgraph_tools.dynamic_tool_analyzer import get_dynamic_tool_analyzer
            
            analyzer = get_dynamic_tool_analyzer()
            conversation_context = metadata.get("conversation_context", {}) if metadata else {}
            analysis = await analyzer.analyze_tool_needs(query, agent_type, conversation_context)
            
            # Build final tool list
            tools = core_tools.copy()
            
            # Add conditional tools based on analysis
            for category in analysis["conditional_categories"]:
                category_tools = self._get_tools_by_category(agent_type, category)
                # Avoid duplicates
                existing_names = {t["function"]["name"] for t in tools}
                for tool in category_tools:
                    if tool["function"]["name"] not in existing_names:
                        tools.append(tool)
                        existing_names.add(tool["function"]["name"])
            
            logger.info(
                f"🎯 Dynamic loading: {len(tools)} tools for {agent_type.value} "
                f"(core: {len(core_tools)}, conditional: {len(tools) - len(core_tools)})"
            )
            logger.debug(f"🎯 Categories detected: {[c.value for c in analysis['conditional_categories']]}")
            
            return tools
            
        except Exception as e:
            logger.error(f"❌ Dynamic tool loading failed: {e}, falling back to static")
            return self.get_tool_objects_for_agent(agent_type)
    
    async def load_additional_tools(
        self,
        agent_type: AgentType,
        categories: List[Any],
        existing_tool_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Load additional tools by category (progressive loading)
        
        Used when agent realizes mid-workflow it needs more tools.
        
        Args:
            agent_type: Agent type
            categories: List of ToolCategory values to load
            existing_tool_names: Optional list of tool names already loaded (to avoid duplicates)
            
        Returns:
            List of additional tool objects
        """
        if not self._initialized:
            logger.warning("⚠️ Tool registry not initialized")
            return []
        
        existing_names = set(existing_tool_names or [])
        additional_tools = []
        
        for category in categories:
            category_tools = self._get_tools_by_category(agent_type, category)
            for tool in category_tools:
                tool_name = tool["function"]["name"]
                if tool_name not in existing_names:
                    additional_tools.append(tool)
                    existing_names.add(tool_name)
        
        if additional_tools:
            logger.info(
                f"🎯 Progressive loading: Added {len(additional_tools)} tools "
                f"for categories {[c.value if hasattr(c, 'value') else str(c) for c in categories]}"
            )
        
        return additional_tools
    
    def get_tool_node_for_agent(self, agent_type: AgentType) -> Optional[Any]:
        """
        Get LangGraph ToolNode for an agent (LangGraph best practice)
        
        ToolNode provides automatic tool execution in LangGraph workflows.
        Use this instead of manual tool execution for better integration.
        
        Usage in workflow:
            tool_node = registry.get_tool_node_for_agent(AgentType.RESEARCH_AGENT)
            workflow.add_node("tools", tool_node)
            
            # Route to tools when LLM requests them
            workflow.add_conditional_edges(
                "agent",
                should_continue,  # Checks if tool_calls exist
                {
                    "continue": "tools",
                    "end": END
                }
            )
        
        Args:
            agent_type: Agent type to get ToolNode for
            
        Returns:
            ToolNode object or None if LangGraph not available
        """
        if not LANGGRAPH_AVAILABLE:
            logger.warning("⚠️ LangGraph not available, cannot create ToolNode")
            return None
        
        if ToolNode is None:
            logger.warning("⚠️ ToolNode not available, cannot create ToolNode")
            return None
        
        if not self._initialized:
            logger.warning("⚠️ Tool registry not initialized, cannot create ToolNode")
            return None
        
        try:
            # Get StructuredTool objects for this agent
            structured_tools = self.get_structured_tools_for_agent(agent_type)
            
            if not structured_tools:
                logger.warning(f"⚠️ No tools available for agent {agent_type.value}, cannot create ToolNode")
                return None
            
            # Create ToolNode with tools
            tool_node = ToolNode(structured_tools)
            
            logger.info(f"✅ Created ToolNode for {agent_type.value} with {len(structured_tools)} tools")
            return tool_node
            
        except Exception as e:
            logger.error(f"❌ Failed to create ToolNode for {agent_type.value}: {e}")
            return None


# Global registry instance
_tool_registry_instance: Optional[CentralizedToolRegistry] = None


async def get_tool_registry() -> CentralizedToolRegistry:
    """Get the global tool registry instance"""
    global _tool_registry_instance
    if _tool_registry_instance is None:
        _tool_registry_instance = CentralizedToolRegistry()
        await _tool_registry_instance.initialize()
    return _tool_registry_instance


# Convenience functions for agents
async def get_agent_tools(agent_type: AgentType) -> List[str]:
    """Get available tool names for an agent type"""
    registry = await get_tool_registry()
    return registry.get_tools_for_agent(agent_type)


async def get_tool_function(tool_name: str, agent_type: AgentType) -> Optional[Callable]:
    """Get a tool function for an agent"""
    registry = await get_tool_registry()
    return registry.get_tool_function(tool_name, agent_type)


async def get_tool_objects_for_agent(agent_type: AgentType) -> List[Dict[str, Any]]:
    """Get tool objects for an agent (LangGraph format)"""
    registry = await get_tool_registry()
    return registry.get_tool_objects_for_agent(agent_type)


async def get_tools_for_agent_dynamic(
    agent_type: AgentType,
    query: str,
    metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Get tools dynamically based on query analysis (Kiro-style dynamic loading)
    
    Args:
        agent_type: Agent type
        query: User query string
        metadata: Optional metadata with conversation context
        
    Returns:
        List of tool objects (core + conditional based on query)
    """
    registry = await get_tool_registry()
    return await registry.get_tools_for_agent_dynamic(agent_type, query, metadata)


async def get_structured_tools_for_agent(agent_type: AgentType) -> List[Any]:
    """
    Get StructuredTool objects for an agent (LangGraph best practice)
    
    Returns list of StructuredTool objects that can be used with:
    - llm.bind_tools(tools) for LLM tool binding
    - ToolNode(tools) for automatic tool execution
    
    Args:
        agent_type: Agent type to get tools for
        
    Returns:
        List of StructuredTool objects
    """
    registry = await get_tool_registry()
    return registry.get_structured_tools_for_agent(agent_type)


async def get_tool_node_for_agent(agent_type: AgentType) -> Optional[Any]:
    """
    Get LangGraph ToolNode for an agent (LangGraph best practice)
    
    ToolNode provides automatic tool execution in LangGraph workflows.
    Use this instead of manual tool execution for better integration.
    
    Args:
        agent_type: Agent type to get ToolNode for
        
    Returns:
        ToolNode object or None if not available
    """
    registry = await get_tool_registry()
    return registry.get_tool_node_for_agent(agent_type)


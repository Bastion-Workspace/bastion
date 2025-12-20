"""
MCP Server Implementation
Provides Model Context Protocol (MCP) server functionality for the knowledge base
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional
import uuid

from mcp.schemas.tool_schemas import (
    ToolResponse, 
    ToolError, 
    SearchDocumentsInput, 
    GetDocumentInput,
    WebIngestSelectedInput  # Add this import
)
from mcp.tools.search_tool import SearchTool
from mcp.tools.document_tool import DocumentTool
from mcp.tools.metadata_search_tool import MetadataSearchTool, MetadataSearchInput
from mcp.tools.query_expansion_tool import QueryExpansionTool, QueryExpansionInput
from mcp.tools.filename_search_tool import FilenameSearchTool, FilenameSearchInput
from mcp.tools.entity_search_tool import EntitySearchTool, EntitySearchInput
from mcp.tools.entity_extraction_tool import EntityExtractionTool, EntityExtractionInput
from mcp.tools.knowledge_graph_analytics_tool import KnowledgeGraphAnalyticsTool, GraphAnalyticsInput
from mcp.tools.web_search_tool import WebSearchTool, WebSearchInput
from mcp.tools.web_content_tool import WebContentTool, WebContentInput
from mcp.tools.web_search_ingestion_tool import WebSearchIngestionTool, WebSearchIngestionInput
from mcp.tools.web_search_analysis_tool import WebSearchAnalysisTool, WebSearchAnalysisInput
from mcp.tools.web_ingest_selected_tool import WebIngestSelectedTool, WebIngestSelectedInput
# Research planning tool removed - migrated to LangGraph subgraph workflows
from mcp.tools.document_summarization_tool import DocumentSummarizationTool, DocumentSummarizationInput
from services.user_document_service import UserDocumentService

logger = logging.getLogger(__name__)


class MCPServer:
    """MCP Server for knowledge base operations"""
    
    def __init__(self, embedding_manager=None, document_repository=None, chat_service=None, knowledge_graph_service=None, document_service=None, config=None):
        """Initialize MCP server with required services"""
        self.embedding_manager = embedding_manager
        self.document_repository = document_repository
        self.chat_service = chat_service
        self.knowledge_graph_service = knowledge_graph_service
        self.document_service = document_service
        self.user_document_service = None  # Will be initialized in initialize()
        self.config = config or {}
        self.tools = {}
        self.initialized = False
        
    async def initialize(self):
        """Initialize the MCP server and register tools"""
        try:
            logger.info("🔧 Initializing MCP Server...")
            
            # Initialize user document service
            self.user_document_service = UserDocumentService()
            await self.user_document_service.initialize()
            logger.info("✅ UserDocumentService initialized")
            
            # Initialize and register search tool
            search_tool = SearchTool(
                embedding_manager=self.embedding_manager,
                document_repository=self.document_repository,
                chat_service=self.chat_service
            )
            await search_tool.initialize()
            self.tools[search_tool.name] = search_tool
            
            # Initialize and register document tool
            document_tool = DocumentTool(
                document_repository=self.document_repository,
                embedding_manager=self.embedding_manager
            )
            await document_tool.initialize()
            self.tools[document_tool.name] = document_tool
            
            # Initialize and register metadata search tool
            metadata_search_tool = MetadataSearchTool(
                document_repository=self.document_repository
            )
            await metadata_search_tool.initialize()
            self.tools[metadata_search_tool.name] = metadata_search_tool
            
            # Initialize and register query expansion tool
            query_expansion_tool = QueryExpansionTool(
                embedding_manager=self.embedding_manager,
                chat_service=self.chat_service
            )
            await query_expansion_tool.initialize()
            self.tools[query_expansion_tool.name] = query_expansion_tool
            
            # Initialize and register filename search tool
            filename_search_tool = FilenameSearchTool(
                embedding_manager=self.embedding_manager,
                document_repository=self.document_repository
            )
            await filename_search_tool.initialize()
            self.tools[filename_search_tool.name] = filename_search_tool
            
            # Initialize and register entity-related tools only if knowledge graph service is available
            if self.knowledge_graph_service:
                try:
                    # Initialize and register entity search tool
                    entity_search_tool = EntitySearchTool(
                        knowledge_graph_service=self.knowledge_graph_service
                    )
                    await entity_search_tool.initialize()
                    self.tools[entity_search_tool.name] = entity_search_tool
                    
                    # Initialize and register entity extraction tool
                    entity_extraction_tool = EntityExtractionTool(
                        knowledge_graph_service=self.knowledge_graph_service,
                        chat_service=self.chat_service
                    )
                    await entity_extraction_tool.initialize()
                    self.tools[entity_extraction_tool.name] = entity_extraction_tool
                    
                    # Initialize and register knowledge graph analytics tool
                    graph_analytics_tool = KnowledgeGraphAnalyticsTool(
                        knowledge_graph_service=self.knowledge_graph_service
                    )
                    await graph_analytics_tool.initialize()
                    self.tools[graph_analytics_tool.name] = graph_analytics_tool
                    
                    logger.info("✅ Entity and knowledge graph tools initialized")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to initialize entity tools: {e}")
            else:
                logger.info("ℹ️ Knowledge graph service not available - skipping entity tools")
            
            # Initialize and register web search tool
            web_search_tool = WebSearchTool(
                config=self.config
            )
            await web_search_tool.initialize()
            self.tools[web_search_tool.name] = web_search_tool
            
            # Initialize and register web content tool
            web_content_tool = WebContentTool(
                config=self.config
            )
            await web_content_tool.initialize()
            self.tools[web_content_tool.name] = web_content_tool
            
            # Initialize and register web search ingestion tool
            web_search_ingestion_tool = WebSearchIngestionTool(
                web_search_tool=web_search_tool,
                web_content_tool=web_content_tool,
                document_service=self.document_service,
                embedding_manager=self.embedding_manager,
                user_document_service=self.user_document_service
            )
            await web_search_ingestion_tool.initialize()
            self.tools[web_search_ingestion_tool.name] = web_search_ingestion_tool
            
            # Initialize and register web search analysis tool
            web_search_analysis_tool = WebSearchAnalysisTool(
                web_search_tool=web_search_tool
            )
            await web_search_analysis_tool.initialize()
            self.tools[web_search_analysis_tool.name] = web_search_analysis_tool
            
            # Initialize and register web ingest selected tool
            web_ingest_selected_tool = WebIngestSelectedTool(
                web_content_tool=web_content_tool,
                document_service=self.document_service,
                embedding_manager=self.embedding_manager,
                user_document_service=self.user_document_service
            )
            await web_ingest_selected_tool.initialize()
            self.tools[web_ingest_selected_tool.name] = web_ingest_selected_tool
            
            # Research planning tool removed - migrated to LangGraph subgraph workflows
            
            # Initialize and register Document Summarization tool
            document_summarization_tool = DocumentSummarizationTool(
                search_tool=search_tool,
                filename_search_tool=filename_search_tool,
                metadata_search_tool=metadata_search_tool,
                document_repository=self.document_repository,
                embedding_manager=self.embedding_manager
            )
            await document_summarization_tool.initialize()
            self.tools[document_summarization_tool.name] = document_summarization_tool
            
            # Initialize and register Coding Assistant tool
            from mcp.tools.coding_assistant_tool import CodingAssistantTool
            coding_assistant_tool = CodingAssistantTool(
                config=self.config,
                web_search_tool=web_search_tool,
                openrouter_client=getattr(self.chat_service, 'openrouter_client', None)
            )
            await coding_assistant_tool.initialize()
            self.tools[coding_assistant_tool.name] = coding_assistant_tool
            
            self.initialized = True
            logger.info(f"✅ MCP Server initialized with {len(self.tools)} tools")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize MCP Server: {e}")
            raise
    
    def set_current_user(self, user_id: str):
        """Set the current user ID for all tools that support user-specific operations"""
        logger.debug(f"🔐 Setting current user: {user_id} for MCP tools")
        
        # Set user for web ingest selected tool
        if "web_ingest_selected_results" in self.tools:
            self.tools["web_ingest_selected_results"].set_current_user(user_id)
        
        # Set user for web search ingestion tool
        if "web_search_and_ingest" in self.tools:
            self.tools["web_search_and_ingest"].set_current_user(user_id)
        
        # Add other tools here as needed
        logger.debug(f"✅ Current user set to {user_id} for relevant MCP tools")
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools for LLM"""
        if not self.initialized:
            raise RuntimeError("MCP Server not initialized")
        
        tool_definitions = []
        for tool_name, tool in self.tools.items():
            tool_def = tool.get_tool_definition()
            tool_definitions.append(tool_def)
        
        return tool_definitions
    
    async def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> ToolResponse:
        """Execute a specific tool with given input"""
        if not self.initialized:
            raise RuntimeError("MCP Server not initialized")
        
        start_time = time.time()
        
        try:
            if tool_name not in self.tools:
                return ToolResponse(
                    success=False,
                    error=ToolError(
                        error_code="TOOL_NOT_FOUND",
                        error_message=f"Tool '{tool_name}' not found",
                        details={"available_tools": list(self.tools.keys())}
                    ),
                    execution_time=time.time() - start_time
                )
            
            # Get the tool
            tool = self.tools[tool_name]
            
            # Validate and convert input based on tool type
            validated_input = await self._validate_tool_input(tool_name, tool_input)
            
            # Execute the tool
            logger.info(f"🔧 Executing tool: {tool_name}")
            result = await tool.execute(validated_input)
            
            logger.info(f"✅ Tool {tool_name} completed in {result.execution_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Tool execution failed for {tool_name}: {e}")
            return ToolResponse(
                success=False,
                error=ToolError(
                    error_code="EXECUTION_FAILED",
                    error_message=str(e),
                    details={"tool_name": tool_name, "input": tool_input}
                ),
                execution_time=time.time() - start_time
            )
    
    async def _validate_tool_input(self, tool_name: str, tool_input: Dict[str, Any]) -> Any:
        """Validate tool input according to tool schema"""
        try:
            if tool_name == "search_documents":
                return SearchDocumentsInput(**tool_input)
            elif tool_name == "get_document":
                # Handle common field name mistakes
                if 'doc_id' in tool_input and 'document_id' not in tool_input:
                    tool_input['document_id'] = tool_input.pop('doc_id')
                return GetDocumentInput(**tool_input)
            elif tool_name == "search_by_metadata":
                return MetadataSearchInput(**tool_input)
            elif tool_name == "expand_query":
                # Handle common field name mistakes
                if 'query' in tool_input and 'original_query' not in tool_input:
                    tool_input['original_query'] = tool_input.pop('query')
                return QueryExpansionInput(**tool_input)
            elif tool_name == "search_by_filename":
                return FilenameSearchInput(**tool_input)
            elif tool_name == "search_by_entities":
                return EntitySearchInput(**tool_input)
            elif tool_name == "extract_entities":
                return EntityExtractionInput(**tool_input)
            elif tool_name == "analyze_knowledge_graph":
                return GraphAnalyticsInput(**tool_input)
            elif tool_name == "web_search":
                return WebSearchInput(**tool_input)
            elif tool_name == "fetch_web_content":
                return WebContentInput(**tool_input)
            elif tool_name == "web_search_and_ingest":
                return WebSearchIngestionInput(**tool_input)
            elif tool_name == "web_search_and_analyze":
                return WebSearchAnalysisInput(**tool_input)
            elif tool_name == "web_ingest_selected_results":
                return WebIngestSelectedInput(**tool_input)
            elif tool_name == "coding_assistant":
                from mcp.tools.coding_assistant_tool import CodingAssistantInput
                return CodingAssistantInput(**tool_input)
            elif tool_name == "crawl4ai_web_crawler":
                # Crawl4AITool has been removed - use web_content_tool instead
                raise ValueError("Crawl4AITool has been removed. Use web_content_tool instead.")
            elif tool_name == "plan_research_comprehensive":
                return ResearchQuery(**tool_input)
            elif tool_name == "summarize_documents":
                return DocumentSummarizationInput(**tool_input)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
                
        except Exception as e:
            logger.error(f"❌ Input validation failed for {tool_name}: {e}")
            
            # Provide specific helpful error messages for common issues
            error_msg = f"Invalid input for {tool_name}: {e}"
            
            if tool_name == "search_by_entities" and "search_type" in str(e):
                error_msg = f"Invalid input for {tool_name}: search_type is required and must be one of: 'by_entity', 'related_entities', 'co_occurring', 'entity_relationships', 'document_entities', 'entity_importance'. Current input: {tool_input}"
            elif tool_name == "search_by_metadata" and "search_type" in str(e):
                error_msg = f"Invalid input for {tool_name}: search_type is required and must be one of: 'by_author', 'by_category', 'by_tags', 'by_publication_date', 'similar_metadata'. Current input: {tool_input}"
            elif "Field required" in str(e):
                error_msg = f"Invalid input for {tool_name}: Missing required field. Error: {e}. Full input received: {tool_input}"
            
            raise ValueError(error_msg)
    
    async def process_llm_request(self, query: str, max_iterations: int = 30) -> Dict[str, Any]:
        """
        Process an LLM request using dynamic tool selection
        The LLM will decide which tools to use and when
        """
        if not self.initialized:
            raise RuntimeError("MCP Server not initialized")
        
        start_time = time.time()
        conversation_log = []
        iteration = 0
        
        try:
            logger.info(f"🧠 Processing LLM request: '{query[:100]}...'")
            
            # Initial system message with available tools
            tools_description = self._get_tools_description()
            system_message = self._create_system_message(tools_description)
            
            # Start conversation with the query
            conversation_log.append({
                "role": "system",
                "content": system_message
            })
            conversation_log.append({
                "role": "user", 
                "content": query
            })
            
            # Iterative processing until LLM provides final answer
            while iteration < max_iterations:
                iteration += 1
                logger.info(f"🔄 LLM iteration {iteration}/{max_iterations}")
                
                # Get LLM response
                llm_response = await self._get_llm_response(conversation_log)
                conversation_log.append({
                    "role": "assistant",
                    "content": llm_response
                })
                
                # Check if LLM wants to use a tool
                tool_call = self._extract_tool_call(llm_response)
                
                if not tool_call:
                    # LLM provided final answer
                    logger.info(f"✅ LLM provided final answer after {iteration} iterations")
                    return {
                        "success": True,
                        "answer": llm_response,
                        "iterations": iteration,
                        "processing_time": time.time() - start_time,
                        "conversation_log": conversation_log
                    }
                
                # Execute the tool call
                logger.info(f"🔧 LLM requested tool: {tool_call['tool_name']}")
                tool_result = await self.execute_tool(
                    tool_call["tool_name"], 
                    tool_call["tool_input"]
                )
                
                # Add tool result to conversation
                tool_result_message = self._format_tool_result(tool_call["tool_name"], tool_result)
                if tool_result_message.strip():  # Only add non-empty messages
                    conversation_log.append({
                        "role": "user",
                        "content": tool_result_message
                    })
            
            # Max iterations reached
            logger.warning(f"⚠️ Max iterations ({max_iterations}) reached")
            return {
                "success": False,
                "error": "Maximum iterations reached without final answer",
                "iterations": iteration,
                "processing_time": time.time() - start_time,
                "conversation_log": conversation_log
            }
            
        except Exception as e:
            logger.error(f"❌ LLM request processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "iterations": iteration,
                "processing_time": time.time() - start_time,
                "conversation_log": conversation_log
            }
    
    def _get_tools_description(self) -> str:
        """Get description of available tools for LLM"""
        tools_list = []
        for tool_name, tool in self.tools.items():
            tool_def = tool.get_tool_definition()
            tools_list.append(f"- **{tool_name}**: {tool_def['description']}")
        
        return "\n".join(tools_list)
    
    def _create_system_message(self, tools_description: str) -> str:
        """Create system message for LLM with tool instructions"""
        from services.prompt_service import prompt_service, AgentMode, UserPromptSettings
        
        # Use centralized prompt service
        assembled_prompt = prompt_service.assemble_prompt(
            agent_mode=AgentMode.CHAT,  # Default to chat mode for MCP server
            tools_description=tools_description,
            user_settings=None  # Default neutral/professional settings
        )
        
        return assembled_prompt.content
        
        # Legacy fallback (can be removed after testing)
        base_prompt = f"""You are Alex, an intelligent knowledge base assistant with access to tools.

AVAILABLE TOOLS:
{tools_description}

INSTRUCTIONS:
1. Analyze the user's query carefully
2. Decide which tools to use to gather information
3. Use tools by responding in this exact format:
   TOOL_CALL: {{"tool_name": "search_documents", "tool_input": {{"query": "search text", "limit": 100}}}}
   # For comprehensive coverage, use higher limits up to 300:
   TOOL_CALL: {{"tool_name": "search_documents", "tool_input": {{"query": "search text", "limit": 200}}}}
   # For metadata searches, ALWAYS include search_type:
   TOOL_CALL: {{"tool_name": "search_by_metadata", "tool_input": {{"search_type": "by_author", "author": "author name", "limit": 50}}}}
4. After receiving tool results, you can either:
   - Make another tool call if you need more information
   - Provide a final comprehensive answer to the user

IMPORTANT:
- Always use the exact TOOL_CALL format shown above
- Be strategic about which tools to use and in what order
- Provide thorough, well-sourced answers using the gathered information
- If search results are insufficient, try different queries or use other tools"""
        
        # Use user's timezone for date/time context
        from utils.system_prompt_utils import create_system_prompt_with_context_for_user
        
        # Get user_id from chat_service if available
        user_id = None
        if self.chat_service and hasattr(self.chat_service, 'current_user_id'):
            user_id = self.chat_service.current_user_id
        
        return await create_system_prompt_with_context_for_user(base_prompt, user_id=user_id)
    
    async def _get_llm_response(self, conversation_log: List[Dict[str, str]]) -> str:
        """Get response from LLM (placeholder - will integrate with actual LLM)"""
        # TODO: Integrate with actual LLM service (OpenRouter, etc.)
        # For now, return a mock response to test the structure
        
        # This is a placeholder - in real implementation, this would:
        # 1. Send conversation_log to LLM
        # 2. Get response back
        # 3. Return the response text
        
        logger.warning("🚧 Using mock LLM response - integrate with real LLM service")
        
        # Mock response that would search for documents
        last_user_message = conversation_log[-1]["content"]
        return f'TOOL_CALL: {{"tool_name": "search_documents", "tool_input": {{"query": "{last_user_message}", "limit": 10}}}}'
    
    def _extract_tool_call(self, llm_response: str) -> Optional[Dict[str, Any]]:
        """Extract tool call from LLM response"""
        try:
            if "TOOL_CALL:" not in llm_response:
                return None
            
            # Extract JSON from tool call
            start_idx = llm_response.find("TOOL_CALL:") + len("TOOL_CALL:")
            json_str = llm_response[start_idx:].strip()
            
            # Handle potential text after JSON
            if json_str.startswith("{"):
                # Find the end of the JSON object
                brace_count = 0
                end_idx = 0
                for i, char in enumerate(json_str):
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                
                if end_idx > 0:
                    json_str = json_str[:end_idx]
            
            tool_call = json.loads(json_str)
            
            # Validate tool call structure
            if "tool_name" not in tool_call or "tool_input" not in tool_call:
                logger.warning(f"⚠️ Invalid tool call structure: {tool_call}")
                return None
            
            return tool_call
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to extract tool call: {e}")
            return None
    
    def _format_tool_result(self, tool_name: str, tool_result: ToolResponse) -> str:
        """Format tool result for user-friendly display"""
        if tool_result.success:
            # Format successful result cleanly
            if tool_name == "search_documents" and tool_result.data:
                search_output = tool_result.data
                results_text = ""
                
                for i, result in enumerate(search_output.results[:10], 1):
                    results_text += f"**{result.document_title}**\n"
                    results_text += f"{result.content[:400]}...\n\n"
                
                return results_text
            else:
                # For other tools, return data without "TOOL_RESULT:" prefix
                return str(tool_result.data)
        else:
            # Show detailed error information for debugging failures
            return f"⚠️ Tool Error ({tool_name}): {tool_result.error.error_message}"

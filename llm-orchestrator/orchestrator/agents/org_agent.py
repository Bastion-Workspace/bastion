"""
Unified Org Agent
LangGraph agent for comprehensive org-mode management and cross-document synthesis

Consolidates org_inbox_agent and org_project_agent capabilities with advanced
contextual synthesis based on Org-mode file links.
"""

import logging
import re
import json
from typing import Dict, Any, List, Optional, TypedDict
from datetime import datetime
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END
from orchestrator.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class OrgProjectCaptureIntent(BaseModel):
    """Structured intent for capturing a project into Org inbox"""
    title: str = Field(description="Project title")
    description: Optional[str] = Field(default=None, description="Short project description/goal")
    target_date: Optional[str] = Field(default=None, description="Org timestamp like <YYYY-MM-DD Dow>")
    tags: List[str] = Field(default_factory=lambda: ["project"], description="Tag list for the project")
    initial_tasks: List[str] = Field(default_factory=list, description="Up to 5 starter TODO items")


class OrgAgentState(TypedDict):
    """State for unified org agent LangGraph workflow"""
    query: str
    user_id: str
    metadata: Dict[str, Any]
    messages: List[Any]
    shared_memory: Dict[str, Any]
    user_message: str
    operation: str
    intent_type: str  # "management" | "synthesis" | "project_capture"
    payload: Dict[str, Any]
    detected_links: List[Dict[str, Any]]  # List of detected file links
    referenced_context: Dict[str, Any]  # Loaded content from linked files
    pending: Dict[str, Any]  # For HITL project capture
    response: Dict[str, Any]
    task_status: str
    error: str


class OrgAgent(BaseAgent):
    """
    Unified Org Agent for org-mode management and cross-document synthesis
    
    Handles:
    - TODO/Event/Contact capture to inbox.org
    - Structured Project capture with HITL (preview-confirm) flow
    - Task management (toggle, update, schedule, archive)
    - Cross-document synthesis based on Org-mode file links
    """
    
    def __init__(self):
        super().__init__("org_agent")
        self._grpc_client = None
        logger.info("Org Agent ready!")
    
    def _build_workflow(self, checkpointer) -> StateGraph:
        """Build LangGraph workflow for unified org agent"""
        workflow = StateGraph(OrgAgentState)
        
        # Add nodes
        workflow.add_node("prepare_context", self._prepare_context_node)
        workflow.add_node("analyze_intent", self._analyze_intent_node)
        workflow.add_node("resolve_references", self._resolve_references_node)
        workflow.add_node("execute_org_command", self._execute_org_command_node)
        workflow.add_node("synthesize_analysis", self._synthesize_analysis_node)
        workflow.add_node("format_response", self._format_response_node)
        
        # Entry point
        workflow.set_entry_point("prepare_context")
        
        # Flow: prepare_context -> analyze_intent
        workflow.add_edge("prepare_context", "analyze_intent")
        
        # Conditional routing from analyze_intent
        workflow.add_conditional_edges(
            "analyze_intent",
            self._route_from_intent,
            {
                "resolve_references": "resolve_references",
                "execute_command": "execute_org_command"
            }
        )
        
        # From resolve_references -> synthesize_analysis
        workflow.add_edge("resolve_references", "synthesize_analysis")
        
        # From execute_org_command -> format_response
        workflow.add_edge("execute_org_command", "format_response")
        
        # From synthesize_analysis -> format_response
        workflow.add_edge("synthesize_analysis", "format_response")
        
        # format_response is the final node
        workflow.add_edge("format_response", END)
        
        return workflow.compile(checkpointer=checkpointer)
    
    def _route_from_intent(self, state: OrgAgentState) -> str:
        """Route based on intent type"""
        intent_type = state.get("intent_type", "management")
        
        if intent_type == "synthesis":
            return "resolve_references"
        else:
            # Both "management" and "project_capture" go to execute_org_command
            return "execute_command"
    
    async def _prepare_context_node(self, state: OrgAgentState) -> Dict[str, Any]:
        """Prepare context: extract message and detect Org links with context-aware filtering"""
        try:
            logger.info("Preparing context for org agent...")
            
            messages = state.get("messages", [])
            if not messages:
                return {
                    "error": "No user message found for org processing",
                    "task_status": "error"
                }
            
            latest_message = messages[-1]
            user_message = latest_message.content if hasattr(latest_message, 'content') else str(latest_message)
            
            # Get active editor for link detection
            shared_memory = state.get("shared_memory", {})
            active_editor = shared_memory.get("active_editor", {})
            
            # Detect Org-mode file links with context-aware filtering
            detected_links = []
            if active_editor:
                editor_content = active_editor.get("content", "")
                cursor_offset = int(active_editor.get("cursor_offset", -1))
                
                # Detect all links first
                all_links = self._detect_org_links(editor_content)
                logger.info(f"Detected {len(all_links)} Org-mode file links in document")
                
                # Filter links based on cursor position and query context
                detected_links = self._filter_links_by_context(
                    all_links=all_links,
                    content=editor_content,
                    cursor_offset=cursor_offset,
                    user_query=user_message
                )
                logger.info(f"Filtered to {len(detected_links)} relevant links based on context")
            
            return {
                "user_message": user_message,
                "detected_links": detected_links,
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": shared_memory,
                "messages": messages,
                "query": state.get("query", user_message)
            }
            
        except Exception as e:
            logger.error(f"Failed to prepare context: {e}")
            return {
                "user_message": "",
                "detected_links": [],
                "error": str(e),
                "task_status": "error"
            }
    
    def _detect_org_links(self, content: str) -> List[Dict[str, Any]]:
        """
        Detect Org-mode file links in content
        
        Supports:
        - [[file:path/to/file.md][Description]]
        - [[file:other.org]]
        - [[file:path/to/file.org::*Heading]]
        
        Returns links with position information for context filtering
        """
        links = []
        
        if not content:
            return links
        
        # Pattern for Org-mode file links: [[file:path][description]] or [[file:path]]
        # Also supports heading links: [[file:path::*Heading]]
        pattern = r'\[\[file:([^\]]+?)(?:\]\[([^\]]+?)\])?\]\]'
        
        for match in re.finditer(pattern, content):
            full_match = match.group(0)
            file_path = match.group(1)
            description = match.group(2) if match.group(2) else file_path
            match_start = match.start()
            match_end = match.end()
            
            # Check if it's a heading link (contains ::)
            heading = None
            if "::" in file_path:
                parts = file_path.split("::", 1)
                file_path = parts[0]
                heading = parts[1] if len(parts) > 1 else None
            
            links.append({
                "full_match": full_match,
                "file_path": file_path.strip(),
                "description": description.strip(),
                "heading": heading.strip() if heading else None,
                "position": match_start,  # Position in content for context filtering
                "end_position": match_end
            })
        
        return links
    
    def _find_heading_at_cursor(self, content: str, cursor_offset: int) -> Optional[Dict[str, Any]]:
        """
        Find the Org-mode heading that contains the cursor position
        
        Returns heading info with level, text, and position boundaries
        """
        if cursor_offset < 0 or cursor_offset >= len(content):
            return None
        
        # Find the heading that contains the cursor
        # Org headings start with * (one or more) followed by space
        heading_pattern = r'^(\*+)\s+(.+)$'
        
        lines = content.split('\n')
        current_pos = 0
        current_heading = None
        current_heading_start = 0
        current_heading_level = 0
        
        for i, line in enumerate(lines):
            line_start = current_pos
            line_end = current_pos + len(line)
            
            # Check if this line is a heading
            match = re.match(heading_pattern, line)
            if match:
                level = len(match.group(1))
                heading_text = match.group(2).strip()
                
                # This is a heading - update current heading
                current_heading = heading_text
                current_heading_start = line_start
                current_heading_level = level
            
            # Check if cursor is in this line
            if line_start <= cursor_offset <= line_end:
                # Cursor is in this line - return the current heading
                if current_heading:
                    # Find where this heading's subtree ends (next heading of same or higher level)
                    heading_end = len(content)
                    for j in range(i + 1, len(lines)):
                        next_line = lines[j]
                        next_match = re.match(heading_pattern, next_line)
                        if next_match:
                            next_level = len(next_match.group(1))
                            if next_level <= current_heading_level:
                                # Found a heading at same or higher level - this subtree ends here
                                heading_end = current_pos + sum(len(lines[k]) + 1 for k in range(i + 1, j))
                                break
                    
                    return {
                        "heading": current_heading,
                        "level": current_heading_level,
                        "start_position": current_heading_start,
                        "end_position": heading_end,
                        "line_number": i
                    }
                break
            
            current_pos = line_end + 1  # +1 for newline
        
        return current_heading if current_heading else None
    
    def _extract_project_names_from_query(self, query: str) -> List[str]:
        """
        Extract project names from user query
        
        Looks for patterns like:
        - "For my plumbing project..."
        - "In the kitchen renovation..."
        - "Regarding the shed build..."
        """
        project_names = []
        query_lower = query.lower()
        
        # Patterns that indicate project names
        patterns = [
            r'(?:for|in|regarding|about|with|on)\s+(?:my|the|this|that)\s+([^,\.\?\!]+?)(?:\s+project|\s+build|\s+renovation|\s+plan)',
            r'(?:my|the|this|that)\s+([^,\.\?\!]+?)\s+(?:project|build|renovation|plan)',
            r'project[:\s]+([^,\.\?\!]+?)(?:\s|$)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, query_lower, re.IGNORECASE)
            for match in matches:
                project_name = match.group(1).strip()
                if project_name and len(project_name) > 2:  # Filter out very short matches
                    project_names.append(project_name)
        
        return project_names
    
    def _filter_links_by_context(
        self,
        all_links: List[Dict[str, Any]],
        content: str,
        cursor_offset: int,
        user_query: str
    ) -> List[Dict[str, Any]]:
        """
        Filter links based on cursor position and query context
        
        Priority:
        1. Links within the heading subtree containing the cursor (if cursor is valid)
        2. Links in headings that match project names from the query
        3. All links if no context available
        """
        if not all_links:
            return []
        
        # Strategy 1: Filter by cursor position (find heading containing cursor)
        cursor_heading = None
        if cursor_offset >= 0:
            cursor_heading = self._find_heading_at_cursor(content, cursor_offset)
            if cursor_heading:
                logger.info(f"Cursor is in heading: '{cursor_heading['heading']}' (level {cursor_heading['level']})")
        
        # Strategy 2: Extract project names from query
        project_names = self._extract_project_names_from_query(user_query)
        if project_names:
            logger.info(f"Extracted project names from query: {project_names}")
        
        filtered_links = []
        
        # If we have cursor context, prioritize links within that heading's subtree
        if cursor_heading:
            heading_start = cursor_heading["start_position"]
            heading_end = cursor_heading["end_position"]
            
            for link in all_links:
                link_pos = link.get("position", 0)
                # Include links within the heading's subtree
                if heading_start <= link_pos <= heading_end:
                    link["context_reason"] = f"cursor_in_heading_{cursor_heading['heading'][:30]}"
                    filtered_links.append(link)
            
            if filtered_links:
                logger.info(f"Found {len(filtered_links)} links in cursor heading subtree")
                return filtered_links
        
        # If no cursor context or no links found, try query-based matching
        if project_names:
            # Find headings that match project names
            heading_pattern = r'^(\*+)\s+(.+)$'
            lines = content.split('\n')
            matching_headings = []
            
            for i, line in enumerate(lines):
                match = re.match(heading_pattern, line)
                if match:
                    heading_text = match.group(2).strip().lower()
                    # Check if heading contains any project name
                    for project_name in project_names:
                        if project_name.lower() in heading_text or heading_text in project_name.lower():
                            matching_headings.append({
                                "heading": match.group(2).strip(),
                                "level": len(match.group(1)),
                                "line_number": i
                            })
                            break
            
            if matching_headings:
                logger.info(f"Found {len(matching_headings)} headings matching project names")
                # Find links within matching heading subtrees
                for heading_info in matching_headings:
                    heading_line = heading_info["line_number"]
                    heading_level = heading_info["level"]
                    
                    # Find subtree boundaries
                    subtree_start = sum(len(lines[j]) + 1 for j in range(heading_line))
                    subtree_end = len(content)
                    
                    for j in range(heading_line + 1, len(lines)):
                        next_line = lines[j]
                        next_match = re.match(heading_pattern, next_line)
                        if next_match:
                            next_level = len(next_match.group(1))
                            if next_level <= heading_level:
                                subtree_end = sum(len(lines[k]) + 1 for k in range(heading_line + 1, j))
                                break
                    
                    # Include links in this subtree
                    for link in all_links:
                        link_pos = link.get("position", 0)
                        if subtree_start <= link_pos <= subtree_end:
                            if link not in filtered_links:
                                link["context_reason"] = f"query_matches_heading_{heading_info['heading'][:30]}"
                                filtered_links.append(link)
                
                if filtered_links:
                    logger.info(f"Found {len(filtered_links)} links in query-matched headings")
                    return filtered_links
        
        # Fallback: return all links if no context filtering possible
        logger.info("No context filtering possible - returning all links")
        for link in all_links:
            link["context_reason"] = "no_context_available"
        return all_links
    
    async def _analyze_intent_node(self, state: OrgAgentState) -> Dict[str, Any]:
        """Analyze user intent to determine operation type"""
        try:
            logger.info("Analyzing intent...")
            
            user_message = state.get("user_message", "")
            detected_links = state.get("detected_links", [])
            shared_memory = state.get("shared_memory", {})
            
            # Check if this is a synthesis query (references linked files)
            is_synthesis_query = (
                len(detected_links) > 0 and
                any(keyword in user_message.lower() for keyword in [
                    "compare", "synthesize", "analyze", "based on", "using",
                    "from the", "in the linked", "across", "between"
                ])
            )
            
            # Check if this is a project capture request
            is_project_capture = any(k in user_message.lower() for k in [
                "start project", "create project", "new project", "project:"
            ])
            
            if is_synthesis_query:
                intent_type = "synthesis"
                operation = "synthesize"
            elif is_project_capture:
                intent_type = "project_capture"
                operation = "project_capture"
            else:
                # Standard org management operation
                intent_type = "management"
                operation = await self._infer_operation(user_message, shared_memory)
            
            logger.info(f"Org Agent: Intent type: {intent_type}, Operation: {operation}")
            
            return {
                "intent_type": intent_type,
                "operation": operation,
                "payload": shared_memory.get("org_inbox_payload", {}),
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": shared_memory,
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze intent: {e}")
            return {
                "intent_type": "management",
                "operation": "list",
                "payload": {},
                "error": str(e),
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
    
    async def _infer_operation(self, user_message: str, state: Dict[str, Any]) -> str:
        """Infer the operation from user message"""
        lowered = user_message.lower()
        
        # Check for explicit operation in state first
        op = (state.get("org_inbox_operation") or "").lower()
        if op:
            return op
        
        # Pattern matching for operations
        if any(k in lowered for k in ["add ", "capture ", "note ", "todo ", "remember ", "save "]):
            return "add"
        elif any(k in lowered for k in ["list", "show", "review", "inbox", "what's in", "see my"]):
            return "list"
        elif any(k in lowered for k in ["done", "complete", "toggle", "mark as done"]):
            return "toggle"
        elif any(k in lowered for k in ["edit", "update", "change", "modify"]):
            return "update"
        elif any(k in lowered for k in ["schedule", "set schedule", "set date"]):
            return "schedule"
        elif any(k in lowered for k in ["archive", "archive done", "clean up done"]):
            return "archive_done"
        
        # Default to list
        return "list"
    
    async def _resolve_references_node(self, state: OrgAgentState) -> Dict[str, Any]:
        """Resolve Org-mode file links and load referenced content"""
        try:
            logger.info("Resolving references and loading linked files...")
            
            detected_links = state.get("detected_links", [])
            user_id = state.get("user_id", "system")
            shared_memory = state.get("shared_memory", {})
            active_editor = shared_memory.get("active_editor", {})
            
            if not detected_links:
                logger.info("No links detected, skipping reference resolution")
                return {
                    "referenced_context": {},
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": shared_memory,
                    "messages": state.get("messages", []),
                    "query": state.get("query", "")
                }
            
            # Load referenced files using ReferenceFileLoader
            from orchestrator.tools.reference_file_loader import load_file_by_path
            
            referenced_context = {}
            
            for link in detected_links:
                file_path = link.get("file_path", "")
                if not file_path:
                    continue
                
                logger.info(f"Loading referenced file: {file_path}")
                
                loaded_file = await load_file_by_path(
                    ref_path=file_path,
                    user_id=user_id,
                    active_editor=active_editor
                )
                
                if loaded_file:
                    # Determine document type from filename or content
                    doc_type = self._infer_document_type(loaded_file)
                    
                    # If this is a project file, request assessment from General Project Agent
                    if doc_type == "project":
                        logger.info(f"Project file detected in Org link: {file_path}. Requesting assessment from Project Agent.")
                        try:
                            from orchestrator.agents.general_project_agent import get_general_project_agent
                            project_agent = get_general_project_agent()
                            
                            # Pass document ID to Project Agent for assessment
                            assessment_metadata = metadata.copy() if metadata else {}
                            assessment_metadata["project_document_id"] = loaded_file.get("document_id")
                            assessment_metadata["user_id"] = user_id
                            
                            # Request assessment
                            assessment_query = f"Please assess this project file: {loaded_file.get('filename', file_path)}. Identify any missing sections, incomplete information, or areas that need updates. Provide specific recommendations organized by header."
                            
                            assessment_result = await project_agent.process(
                                query=assessment_query,
                                metadata=assessment_metadata,
                                messages=state.get("messages", [])
                            )
                            
                            assessment_text = assessment_result.get("response", "")
                            if isinstance(assessment_text, dict):
                                assessment_text = assessment_text.get("response", "No assessment available.")
                            
                            # Store assessment in loaded_file for synthesis
                            loaded_file["project_assessment"] = assessment_text
                            logger.info(f"Received project assessment from Project Agent ({len(assessment_text)} chars)")
                        except Exception as e:
                            logger.warning(f"Failed to get project assessment: {e}")
                            import traceback
                            logger.debug(traceback.format_exc())
                            loaded_file["project_assessment"] = "Assessment unavailable."
                    
                    if doc_type not in referenced_context:
                        referenced_context[doc_type] = []
                    
                    referenced_context[doc_type].append({
                        "file_path": file_path,
                        "description": link.get("description", file_path),
                        "heading": link.get("heading"),
                        "content": loaded_file.get("content", ""),
                        "filename": loaded_file.get("filename", ""),
                        "document_id": loaded_file.get("document_id"),
                        "project_assessment": loaded_file.get("project_assessment")  # Include assessment if available
                    })
                    logger.info(f"Loaded {doc_type} file: {loaded_file.get('filename')}")
                else:
                    logger.warning(f"Could not load referenced file: {file_path}")
            
            return {
                "referenced_context": referenced_context,
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": shared_memory,
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
            
        except Exception as e:
            logger.error(f"Failed to resolve references: {e}")
            return {
                "referenced_context": {},
                "error": str(e),
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
    
    def _infer_document_type(self, loaded_file: Dict[str, Any]) -> str:
        """Infer document type from filename or content"""
        filename = loaded_file.get("filename", "").lower()
        content = loaded_file.get("content", "")
        
        # Check filename patterns
        if "reference" in filename or "ref" in filename:
            return "reference"
        elif "project" in filename or "plan" in filename:
            return "project"
        elif filename.endswith(".org"):
            return "org"
        elif filename.endswith(".md"):
            # Try to infer from content frontmatter
            if "type:" in content[:500]:
                match = re.search(r'type:\s*(\w+)', content[:500], re.IGNORECASE)
                if match:
                    return match.group(1).lower()
            return "markdown"
        
        return "unknown"
    
    async def _synthesize_analysis_node(self, state: OrgAgentState) -> Dict[str, Any]:
        """Perform LLM synthesis across primary Org file and linked references"""
        try:
            logger.info("Synthesizing analysis across documents...")
            
            user_message = state.get("user_message", "")
            referenced_context = state.get("referenced_context", {})
            shared_memory = state.get("shared_memory", {})
            active_editor = shared_memory.get("active_editor", {})
            
            # Build context from primary Org file
            primary_content = active_editor.get("content", "")
            primary_filename = active_editor.get("filename", "current file")
            
            # Build context from referenced files
            context_parts = []
            context_parts.append(f"=== PRIMARY ORG FILE: {primary_filename} ===\n")
            context_parts.append(primary_content[:2000])  # Limit primary content
            
            for doc_type, files in referenced_context.items():
                for ref_file in files:
                    context_parts.append(f"\n=== REFERENCED {doc_type.upper()} FILE: {ref_file['filename']} ===\n")
                    context_parts.append(ref_file['content'][:2000])  # Limit each reference
                    
                    # Include project assessment if available (from Project Agent)
                    if doc_type == "project" and ref_file.get("project_assessment"):
                        context_parts.append(f"\n=== PROJECT ASSESSMENT (from Project Agent) ===\n")
                        context_parts.append(ref_file['project_assessment'][:1500])
            
            # Use LLM to synthesize
            llm = self._get_llm(temperature=0.7, state=state)
            
            system_prompt = """You are an Org-Mode Analysis Assistant. Analyze the user's query across the primary Org file and all referenced documents, synthesizing information to provide a comprehensive answer.

When analyzing:
- Consider information from ALL provided documents
- Identify relationships and connections between documents
- Provide specific references to source documents when relevant
- Synthesize findings into a coherent response"""
            
            user_prompt = f"""USER QUERY: {user_message}

CONTEXT FROM DOCUMENTS:
{"".join(context_parts)}

Please analyze the query across all provided documents and synthesize a comprehensive response."""
            
            llm_messages = [
                SystemMessage(content=system_prompt),
                SystemMessage(content=self._get_datetime_context()),
                HumanMessage(content=user_prompt)
            ]
            
            response = await llm.ainvoke(llm_messages)
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            return {
                "response": {
                    "messages": [AIMessage(content=response_content)],
                    "agent_results": {
                        "agent_type": "org_agent",
                        "intent_type": "synthesis",
                        "is_complete": True
                    },
                    "is_complete": True
                },
                "task_status": "complete",
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": shared_memory,
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
            
        except Exception as e:
            logger.error(f"Failed to synthesize analysis: {e}")
            return {
                "response": self._create_error_result(f"Synthesis failed: {str(e)}"),
                "task_status": "error",
                "error": str(e),
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
    
    async def _execute_org_command_node(self, state: OrgAgentState) -> Dict[str, Any]:
        """Execute org management commands (inbox operations or project capture)"""
        try:
            logger.info("Executing org command...")
            
            operation = state.get("operation", "list")
            intent_type = state.get("intent_type", "management")
            user_message = state.get("user_message", "")
            user_id = state.get("user_id", "")
            payload = state.get("payload", {})
            shared_memory = state.get("shared_memory", {})
            
            # Handle project capture separately
            if intent_type == "project_capture":
                result = await self._handle_project_capture(state)
            else:
                # Standard inbox operations
                if operation == "add":
                    result = await self._handle_add_operation(user_message, user_id, shared_memory, payload)
                elif operation == "list":
                    result = await self._handle_list_operation(user_id)
                elif operation == "toggle":
                    result = await self._handle_toggle_operation(user_id, payload)
                elif operation == "update":
                    result = await self._handle_update_operation(user_id, payload)
                elif operation == "schedule":
                    result = await self._handle_schedule_operation(user_id, payload)
                elif operation == "archive_done":
                    result = await self._handle_archive_operation(user_id)
                else:
                    result = await self._handle_list_operation(user_id)
            
            logger.info("Org Agent: Completed command execution")
            
            return {
                "response": result,
                "task_status": "complete",
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": shared_memory,
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
            
        except Exception as e:
            logger.error(f"Failed to execute org command: {e}")
            return {
                "response": self._create_error_result(f"Org operation failed: {str(e)}"),
                "task_status": "error",
                "error": str(e),
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
    
    async def _format_response_node(self, state: OrgAgentState) -> Dict[str, Any]:
        """Format final response"""
        try:
            response = state.get("response", {})
            
            # Ensure response has proper structure
            if not response:
                response = {
                    "messages": [AIMessage(content="Org operation completed")],
                    "agent_results": {
                        "agent_type": "org_agent",
                        "is_complete": True
                    },
                    "is_complete": True
                }
            
            return {
                "response": response,
                "task_status": state.get("task_status", "complete"),
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
            
        except Exception as e:
            logger.error(f"Failed to format response: {e}")
            return {
                "response": self._create_error_result(f"Response formatting failed: {str(e)}"),
                "task_status": "error",
                "error": str(e),
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", "")
            }
    
    # ============================================
    # Project Capture Methods (from org_project_agent)
    # ============================================
    
    async def _handle_project_capture(self, state: OrgAgentState) -> Dict[str, Any]:
        """Handle project capture with HITL flow"""
        try:
            messages = state.get("messages", [])
            user_id = state.get("user_id", "")
            shared_memory = state.get("shared_memory", {})
            pending = shared_memory.get("pending_project_capture", {})
            user_message = state.get("user_message", "")
            
            # Reconstruct state dict for existing methods
            state_dict = {
                "messages": messages,
                "user_id": user_id,
                "shared_memory": shared_memory
            }
            
            # Case 0: Pending capture awaiting more details
            if pending and not pending.get("awaiting_confirmation"):
                pending = self._merge_user_details_into_pending(pending, user_message)
                
                intent = OrgProjectCaptureIntent(
                    title=pending.get("title", ""),
                    description=pending.get("description"),
                    target_date=pending.get("target_date"),
                    tags=pending.get("tags", ["project"]),
                    initial_tasks=pending.get("initial_tasks", [])
                )
                
                remaining_missing = self._compute_missing_fields(intent)
                
                if not remaining_missing:
                    preview = self._build_project_block_preview(pending)
                    pending["preview_block"] = preview
                    pending["awaiting_confirmation"] = True
                    shared_memory["pending_project_capture"] = pending
                    response = self._build_preview_message(preview)
                    
                    return {
                        "messages": [AIMessage(content=response)],
                        "shared_memory": shared_memory,
                        "agent_results": {
                            "agent_type": "org_agent",
                            "task_status": "permission_required",
                            "preview": preview
                        },
                        "requires_user_input": True,
                        "is_complete": False
                    }
                else:
                    question = self._build_clarification_question(intent, remaining_missing)
                    pending["missing_fields"] = remaining_missing
                    shared_memory["pending_project_capture"] = pending
                    response = f"To capture this project, please provide: {', '.join(remaining_missing)}.\n{question}"
                    
                    return {
                        "messages": [AIMessage(content=response)],
                        "shared_memory": shared_memory,
                        "agent_results": {
                            "agent_type": "org_agent",
                            "task_status": "incomplete",
                            "missing_fields": remaining_missing
                        },
                        "requires_user_input": True,
                        "is_complete": False
                    }
            
            # Case 1: Awaiting confirmation
            if pending.get("awaiting_confirmation"):
                if self._is_confirmation(user_message):
                    result = await self._commit_project_block(state_dict, pending, user_id)
                    return result
                elif self._is_cancellation(user_message):
                    shared_memory.pop("pending_project_capture", None)
                    response = "Project capture cancelled."
                    return {
                        "messages": [AIMessage(content=response)],
                        "shared_memory": shared_memory,
                        "agent_results": {
                            "agent_type": "org_agent",
                            "task_status": "complete",
                            "cancelled": True
                        },
                        "is_complete": True
                    }
                else:
                    # Treat as edits
                    pending = self._merge_user_details_into_pending(pending, user_message)
                    preview = self._build_project_block_preview(pending)
                    pending["preview_block"] = preview
                    shared_memory["pending_project_capture"] = pending
                    response = self._build_preview_message(preview)
                    
                    return {
                        "messages": [AIMessage(content=response)],
                        "shared_memory": shared_memory,
                        "agent_results": {
                            "agent_type": "org_agent",
                            "task_status": "permission_required",
                            "preview": preview
                        },
                        "requires_user_input": True,
                        "is_complete": False
                    }
            
            # Case 2: Initialize new capture
            intent = self._derive_initial_intent(user_message)
            
            try:
                intent = await self._smart_enrich_intent_from_message(intent, user_message, state_dict)
            except Exception as enrich_err:
                logger.warning(f"Smart enrichment skipped: {enrich_err}")
            
            missing = self._compute_missing_fields(intent)
            
            if missing:
                question = self._build_clarification_question(intent, missing)
                shared_memory["pending_project_capture"] = {
                    "title": intent.title,
                    "description": intent.description,
                    "target_date": intent.target_date,
                    "tags": intent.tags or ["project"],
                    "initial_tasks": intent.initial_tasks,
                    "missing_fields": missing,
                    "awaiting_confirmation": False
                }
                response = f"To capture this project, please provide: {', '.join(missing)}.\n{question}"
                
                return {
                    "messages": [AIMessage(content=response)],
                    "shared_memory": shared_memory,
                    "agent_results": {
                        "agent_type": "org_agent",
                        "task_status": "incomplete",
                        "missing_fields": missing
                    },
                    "requires_user_input": True,
                    "is_complete": False
                }
            
            # Build preview and request confirmation
            preview = self._build_project_block_preview(intent.dict())
            pending = {
                "title": intent.title,
                "description": intent.description,
                "target_date": intent.target_date,
                "tags": intent.tags or ["project"],
                "initial_tasks": intent.initial_tasks,
                "preview_block": preview,
                "awaiting_confirmation": True
            }
            shared_memory["pending_project_capture"] = pending
            response = self._build_preview_message(preview)
            
            return {
                "messages": [AIMessage(content=response)],
                "shared_memory": shared_memory,
                "agent_results": {
                    "agent_type": "org_agent",
                    "task_status": "permission_required",
                    "preview": preview
                },
                "requires_user_input": True,
                "is_complete": False
            }
            
        except Exception as e:
            logger.error(f"Project capture failed: {e}")
            return self._create_error_result(f"Project capture error: {str(e)}")
    
    def _derive_initial_intent(self, user_message: str) -> OrgProjectCaptureIntent:
        """Extract initial project intent from user message"""
        title = user_message.strip()
        
        lowered = title.lower()
        for lead in ["start project", "create project", "new project", "project:", "project "]:
            if lowered.startswith(lead):
                title = title[len(lead):].strip(" -:–—")
                break
        
        return OrgProjectCaptureIntent(title=title, tags=["project"])
    
    async def _smart_enrich_intent_from_message(
        self,
        intent: OrgProjectCaptureIntent,
        user_message: str,
        state: Dict[str, Any]
    ) -> OrgProjectCaptureIntent:
        """Use LLM to extract description and starter tasks from message"""
        if intent.description and intent.initial_tasks:
            return intent
        
        system_prompt = (
            "You are an Org Project Capture Assistant. "
            "Extract a concise project description (1-2 sentences) "
            "and up to 5 concrete starter tasks from the user message. "
            "If insufficient detail exists, leave fields empty. "
            "Respond with VALID JSON only."
        )
        
        schema_instructions = (
            "STRUCTURED OUTPUT REQUIRED:\n"
            "{\n"
            '  "description": "string or empty",\n'
            '  "initial_tasks": ["task 1", "task 2"]  // up to 5, may be empty\n'
            "}"
        )
        
        llm = self._get_llm(temperature=0.2, state=state)
        datetime_context = self._get_datetime_context()
        llm_messages = [
            SystemMessage(content=system_prompt),
            SystemMessage(content=datetime_context),
            SystemMessage(content=schema_instructions),
            HumanMessage(content=f"USER MESSAGE: {user_message}")
        ]
        
        response = await llm.ainvoke(llm_messages)
        content = response.content if hasattr(response, 'content') else str(response)
        content = content.strip()
        
        try:
            data = json.loads(content)
            description = (data.get("description") or "").strip()
            tasks_raw = data.get("initial_tasks") or []
            
            if not isinstance(tasks_raw, list):
                tasks_raw = []
            
            tasks = []
            for t in tasks_raw:
                if isinstance(t, str):
                    t_clean = t.strip()
                    if t_clean:
                        tasks.append(t_clean)
                if len(tasks) >= 5:
                    break
            
            if description and not intent.description:
                intent.description = description
            if tasks and not intent.initial_tasks:
                intent.initial_tasks = tasks
                
        except Exception as e:
            logger.warning(f"Failed to parse smart enrichment JSON: {e}")
        
        return intent
    
    def _compute_missing_fields(self, intent: OrgProjectCaptureIntent) -> List[str]:
        """Determine which required fields are missing"""
        missing = []
        if not intent.description:
            missing.append("description")
        if not intent.initial_tasks:
            missing.append("initial_tasks")
        return missing
    
    def _build_clarification_question(self, intent: OrgProjectCaptureIntent, missing: List[str]) -> str:
        """Build question requesting missing information"""
        return (
            "Please reply with a short description (1-2 sentences), up to 5 starter tasks "
            "(bulleted or comma-separated), and an optional target date as <YYYY-MM-DD Dow>."
        )
    
    def _merge_user_details_into_pending(self, pending: Dict[str, Any], user_message: str) -> Dict[str, Any]:
        """Merge user-provided details into pending capture"""
        text = user_message.strip()
        
        labeled_desc, labeled_tasks = self._parse_labeled_fields(text)
        if labeled_desc and not pending.get("description"):
            pending["description"] = labeled_desc
        if labeled_tasks:
            merged = list(dict.fromkeys((pending.get("initial_tasks") or []) + labeled_tasks))
            pending["initial_tasks"] = merged[:5]
        
        tasks = []
        description_lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith(("description:", "desc:", "tasks:")):
                continue
            if stripped.startswith(("- ", "* ")):
                tasks.append(stripped[2:].strip())
            else:
                description_lines.append(stripped)
        
        if not labeled_tasks and not tasks and "," in text:
            parts = [p.strip() for p in text.split(",") if p.strip()]
            if len(parts) >= 2:
                tasks = parts[:5]
                description_lines = []
        
        if description_lines and not pending.get("description"):
            pending["description"] = " ".join(description_lines).strip()
        if tasks:
            merged = list(dict.fromkeys((pending.get("initial_tasks") or []) + tasks))
            pending["initial_tasks"] = merged[:5]
        
        m = re.search(r"<\d{4}-\d{2}-\d{2}[^>]*>", text)
        if m and not pending.get("target_date"):
            pending["target_date"] = m.group(0)
        
        return pending
    
    def _parse_labeled_fields(self, text: str) -> tuple:
        """Parse Description: and Tasks: labels"""
        description = []
        tasks = []
        current_section = None
        
        for raw in text.splitlines():
            line = raw.strip()
            lower = line.lower()
            
            if lower.startswith("description:") or lower.startswith("desc:"):
                current_section = "description"
                rest = line.split(":", 1)[1].strip()
                if rest:
                    description.append(rest)
                continue
            
            if lower.startswith("tasks:"):
                current_section = "tasks"
                rest = line.split(":", 1)[1].strip()
                if rest:
                    if ";" in rest:
                        tasks.extend([p.strip() for p in rest.split(";") if p.strip()])
                    else:
                        tasks.append(rest)
                continue
            
            if current_section == "description" and line:
                description.append(line)
            elif current_section == "tasks" and line:
                if line.startswith(("- ", "* ")):
                    tasks.append(line[2:].strip())
                else:
                    tasks.append(line)
        
        desc_text = " ".join(description).strip() if description else ""
        tasks_dedup = list(dict.fromkeys([t.strip() for t in tasks if t and t.strip()]))[:5]
        
        return desc_text, tasks_dedup
    
    def _build_project_block_preview(self, data: Dict[str, Any]) -> str:
        """Build org-mode formatted project block"""
        title = (data.get("title") or "Untitled Project").strip()
        tags = data.get("tags") or ["project"]
        tag_suffix = ":" + ":".join(sorted({t.strip(': ') for t in tags if t})) + ":"
        created = datetime.now().strftime("[%Y-%m-%d %a %H:%M]")
        
        lines = []
        lines.append(f"* {title} {tag_suffix}")
        lines.append(":PROPERTIES:")
        lines.append(f":ID:       {datetime.now().strftime('%Y%m%d%H%M%S')}")
        lines.append(f":CREATED:  {created}")
        lines.append(":END:")
        
        desc = (data.get("description") or "").strip()
        if desc:
            lines.append(desc)
        
        td = (data.get("target_date") or "").strip()
        if td:
            lines.append(f"SCHEDULED: {td}")
        
        for t in (data.get("initial_tasks") or []):
            t_clean = t.strip()
            if t_clean:
                lines.append(f"** TODO {t_clean}")
        
        return "\n".join(lines) + "\n"
    
    def _build_preview_message(self, preview_block: str) -> str:
        """Build preview message for user confirmation"""
        return (
            "Here's the project preview. Shall I add it to inbox.org?\n\n" +
            "```org\n" + preview_block.rstrip("\n") + "\n```\n" +
            "Reply 'yes' to proceed, or edit details (description, tasks, date)."
        )
    
    def _is_confirmation(self, user_message: str) -> bool:
        """Check if user message is a confirmation"""
        lm = (user_message or "").strip().lower()
        return any(w in lm for w in ["yes", "y", "ok", "okay", "proceed", "do it", "confirm"])
    
    def _is_cancellation(self, user_message: str) -> bool:
        """Check if user message is a cancellation"""
        lm = (user_message or "").strip().lower()
        return any(w in lm for w in ["no", "cancel", "stop", "abort"])
    
    async def _commit_project_block(self, state: Dict[str, Any], pending: Dict[str, Any], user_id: Optional[str]) -> Dict[str, Any]:
        """Write project block to inbox.org via gRPC"""
        try:
            preview = pending.get("preview_block") or self._build_project_block_preview(pending)
            
            grpc_client = await self._get_grpc_client()
            result = await grpc_client.append_org_inbox_text(
                text=preview,
                user_id=user_id
            )
            
            if not result.get("success"):
                raise Exception(result.get("error", "Unknown error"))
            
            path = result.get("path")
            line_start = result.get("line_start_index")
            line_end = result.get("line_end_index")
            
            shared_memory = state.get("shared_memory", {})
            shared_memory.pop("pending_project_capture", None)
            
            response = f"Project added to inbox: {path} (lines {line_start}-{line_end})."
            
            return {
                "messages": [AIMessage(content=response)],
                "shared_memory": shared_memory,
                "agent_results": {
                    "agent_type": "org_agent",
                    "task_status": "complete",
                    "path": path,
                    "line_start": line_start,
                    "line_end": line_end
                },
                "is_complete": True
            }
            
        except Exception as e:
            logger.error(f"Commit project failed: {e}")
            response = f"Failed to write project to inbox.org: {e}"
            return self._create_error_result(response)
    
    # ============================================
    # Inbox Management Methods (from org_inbox_agent)
    # ============================================
    
    async def _handle_add_operation(
        self, 
        user_message: str, 
        user_id: str, 
        state: Dict[str, Any],
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle adding items to org inbox with LLM interpretation"""
        try:
            interpretation = await self._llm_interpret_add(state, user_message, payload)
            
            if interpretation.get("clarification_needed"):
                return self._create_response(
                    success=True,
                    response=interpretation.get("clarification_question", "I need more information."),
                    is_complete=False
                )
            
            grpc_client = await self._get_grpc_client()
            
            title = interpretation.get("title", "").strip()
            entry_kind = interpretation.get("entry_kind", "todo")
            schedule = interpretation.get("schedule")
            repeater = interpretation.get("repeater")
            suggested_tags = interpretation.get("suggested_tags", [])
            contact_properties = interpretation.get("contact_properties")
            
            result = await grpc_client.add_org_inbox_item(
                user_id=user_id,
                text=title,
                kind=entry_kind,
                schedule=schedule,
                repeater=repeater,
                tags=suggested_tags,
                contact_properties=contact_properties
            )
            
            if not result.get("success"):
                return self._create_error_result(result.get("error", "Failed to add item"))
            
            response_parts = []
            if entry_kind == "contact":
                response_parts.append(f"Added contact '{title}' to inbox.org")
            elif entry_kind == "event":
                response_parts.append(f"Added event '{title}' to inbox.org")
            else:
                response_parts.append(f"Added TODO '{title}' to inbox.org")
            
            if schedule:
                response_parts.append(f"(scheduled {schedule}")
                if repeater:
                    response_parts[-1] += f", repeats {repeater}"
                response_parts[-1] += ")"
            
            if suggested_tags:
                response_parts.append(f"| tags: {':'.join(suggested_tags)}")
            
            if interpretation.get("assistant_confirmation"):
                response_text = interpretation.get("assistant_confirmation")
            else:
                response_text = " ".join(response_parts)
            
            return self._create_response(
                success=True,
                response=response_text,
                org_inbox_result={
                    "action": "add",
                    "line_index": result.get("line_index"),
                    "title": title,
                    "kind": entry_kind
                }
            )
            
        except Exception as e:
            logger.error(f"Add operation failed: {e}")
            return self._create_error_result(f"Failed to add item: {str(e)}")
    
    async def _handle_list_operation(self, user_id: str) -> Dict[str, Any]:
        """Handle listing org inbox items"""
        try:
            grpc_client = await self._get_grpc_client()
            
            result = await grpc_client.list_org_inbox_items(user_id=user_id)
            
            if not result.get("success"):
                return self._create_error_result(result.get("error", "Failed to list items"))
            
            items = result.get("items", [])
            path = result.get("path", "inbox.org")
            
            if not items:
                response_text = f"Your inbox ({path}) is empty."
            else:
                response_parts = [f"Inbox has {len(items)} items:"]
                for i, item in enumerate(items[:10], 1):
                    status_icon = "✅" if item.get("is_done") else "⬜"
                    todo_state = item.get("todo_state", "")
                    text = item.get("text", "")
                    tags = item.get("tags", [])
                    
                    item_line = f"{i}. {status_icon} {todo_state} {text}".strip()
                    if tags:
                        item_line += f" :{':'.join(tags)}:"
                    response_parts.append(item_line)
                
                if len(items) > 10:
                    response_parts.append(f"... and {len(items) - 10} more items")
                
                response_text = "\n".join(response_parts)
            
            return self._create_response(
                success=True,
                response=response_text,
                org_inbox_result={
                    "action": "list",
                    "count": len(items),
                    "path": path
                }
            )
            
        except Exception as e:
            logger.error(f"List operation failed: {e}")
            return self._create_error_result(f"Failed to list items: {str(e)}")
    
    async def _handle_toggle_operation(self, user_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle toggling DONE status"""
        try:
            line_index = int(payload.get("line_index", -1))
            if line_index < 0:
                return self._create_error_result("Line index required for toggle operation")
            
            grpc_client = await self._get_grpc_client()
            
            result = await grpc_client.toggle_org_inbox_item(
                user_id=user_id,
                line_index=line_index
            )
            
            if not result.get("success"):
                return self._create_error_result(result.get("error", "Failed to toggle item"))
            
            response_text = f"Toggled item at line {result.get('updated_index', line_index)}"
            
            return self._create_response(
                success=True,
                response=response_text,
                org_inbox_result={
                    "action": "toggle",
                    "line_index": result.get("updated_index")
                }
            )
            
        except Exception as e:
            logger.error(f"Toggle operation failed: {e}")
            return self._create_error_result(f"Failed to toggle item: {str(e)}")
    
    async def _handle_update_operation(self, user_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle updating item text"""
        try:
            line_index = int(payload.get("line_index", -1))
            new_text = payload.get("new_text", "")
            
            if line_index < 0 or not new_text:
                return self._create_error_result("Line index and new text required for update operation")
            
            grpc_client = await self._get_grpc_client()
            
            result = await grpc_client.update_org_inbox_item(
                user_id=user_id,
                line_index=line_index,
                new_text=new_text
            )
            
            if not result.get("success"):
                return self._create_error_result(result.get("error", "Failed to update item"))
            
            response_text = f"Updated line {result.get('updated_index', line_index)}: {new_text.strip()}"
            
            return self._create_response(
                success=True,
                response=response_text,
                org_inbox_result={
                    "action": "update",
                    "line_index": result.get("updated_index")
                }
            )
            
        except Exception as e:
            logger.error(f"Update operation failed: {e}")
            return self._create_error_result(f"Failed to update item: {str(e)}")
    
    async def _handle_schedule_operation(self, user_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle setting schedule and repeater"""
        try:
            line_index = int(payload.get("line_index", -1))
            scheduled = payload.get("scheduled")
            repeater = payload.get("repeater")
            
            if line_index < 0 or not scheduled:
                return self._create_error_result("Line index and schedule required")
            
            grpc_client = await self._get_grpc_client()
            
            result = await grpc_client.set_org_inbox_schedule(
                user_id=user_id,
                line_index=line_index,
                scheduled=scheduled,
                repeater=repeater
            )
            
            if not result.get("success"):
                return self._create_error_result(result.get("error", "Failed to set schedule"))
            
            response_text = "Updated schedule"
            if repeater:
                response_text += f" with repeater {repeater}"
            
            return self._create_response(
                success=True,
                response=response_text,
                org_inbox_result={
                    "action": "schedule",
                    "line_index": result.get("updated_index")
                }
            )
            
        except Exception as e:
            logger.error(f"Schedule operation failed: {e}")
            return self._create_error_result(f"Failed to set schedule: {str(e)}")
    
    async def _handle_archive_operation(self, user_id: str) -> Dict[str, Any]:
        """Handle archiving DONE items"""
        try:
            grpc_client = await self._get_grpc_client()
            
            result = await grpc_client.archive_org_inbox_done(user_id=user_id)
            
            if not result.get("success"):
                return self._create_error_result(result.get("error", "Failed to archive"))
            
            archived_count = result.get("archived_count", 0)
            response_text = result.get("message", f"Archived {archived_count} DONE items")
            
            return self._create_response(
                success=True,
                response=response_text,
                org_inbox_result={
                    "action": "archive_done",
                    "archived_count": archived_count
                }
            )
            
        except Exception as e:
            logger.error(f"Archive operation failed: {e}")
            return self._create_error_result(f"Failed to archive: {str(e)}")
    
    async def _llm_interpret_add(
        self, 
        state: Dict[str, Any], 
        user_message: str, 
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use LLM to interpret add requests with conversation context"""
        try:
            text = (payload.get("text") or user_message or "").strip()
            
            messages = state.get("messages", [])[-5:]
            context_lines = []
            for m in messages:
                role = getattr(m, "type", None) or getattr(m, "role", "user")
                content = getattr(m, "content", "")
                context_lines.append(f"- {role}: {content}")
            context_block = "\n".join(context_lines) if context_lines else ""
            
            persona = (state.get("persona") or {}).get("persona_style") or "professional"
            
            prompt = f"""You are an Org-Mode Personal Assistant. Analyze the user's request and produce structured JSON for execution.

USER MESSAGE:
{text}

CONTEXT (recent conversation):
{context_block}

REQUIREMENTS:
- Resolve pronouns like "it/this/that" to actionable phrases from the message
- Choose entry_kind: "todo" for tasks, "event" for appointments/meetings/birthdays, "contact" for people, "checkbox" for quick lists
- For contacts, extract properties into contact_properties object:
  * Name fields: FIRST_NAME, MIDDLE_NAME, LAST_NAME
  * Contact info: EMAIL_HOME, EMAIL_WORK, PHONE_MOBILE, PHONE_WORK
  * Organization: COMPANY, TITLE
  * Location: ADDRESS_HOME, ADDRESS_WORK
  * Personal: BIRTHDAY (YYYY-MM-DD), ANNIVERSARY (YYYY-MM-DD), RELATIONSHIP
  * Notes: Additional info in NOTES
- Extract schedule as org timestamp <YYYY-MM-DD Dow> if date/time present; else null
- If repeating cadence (weekly, daily, monthly), return repeater like +1w, .+1m; else null
- Suggest up to 3 tags as simple lowercase slugs
- If ambiguous, set clarification_needed true and propose clarification_question
- Generate assistant_confirmation in {persona} tone describing what was added

OUTPUT JSON SCHEMA:
{{
  "title": "string",
  "entry_kind": "todo" | "event" | "contact" | "checkbox",
  "schedule": "<YYYY-MM-DD Dow>" | null,
  "repeater": "+1w" | ".+1w" | "+1m" | null,
  "suggested_tags": ["string"],
  "contact_properties": {{"EMAIL": "string", "PHONE": "string"}} | null,
  "clarification_needed": true|false,
  "clarification_question": "string|null",
  "assistant_confirmation": "string|null"
}}

Respond with ONLY the JSON."""
            
            llm = self._get_llm(temperature=0.2, state=state)
            llm_messages = [HumanMessage(content=prompt)]
            response = await llm.ainvoke(llm_messages)
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            try:
                data = json.loads(response_content)
            except Exception:
                data = {
                    "title": text.strip().rstrip("?.!"),
                    "entry_kind": "todo",
                    "schedule": None,
                    "repeater": None,
                    "suggested_tags": [],
                    "clarification_needed": False,
                    "clarification_question": None
                }
            
            return data
            
        except Exception as e:
            logger.error(f"LLM interpretation failed: {e}")
            return {
                "title": text.strip() if text else "New entry",
                "entry_kind": "todo",
                "schedule": None,
                "repeater": None,
                "suggested_tags": [],
                "clarification_needed": False,
                "clarification_question": None
            }
    
    def _create_response(
        self, 
        success: bool, 
        response: str, 
        org_inbox_result: Dict[str, Any] = None,
        is_complete: bool = True
    ) -> Dict[str, Any]:
        """Create standardized response"""
        return {
            "messages": [AIMessage(content=response)],
            "agent_results": {
                "agent_type": "org_agent",
                "success": success,
                "org_inbox": org_inbox_result or {},
                "is_complete": is_complete
            },
            "is_complete": is_complete
        }
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result"""
        logger.error(f"Org Agent error: {error_message}")
        return {
            "messages": [AIMessage(content=f"Org operation failed: {error_message}")],
            "agent_results": {
                "agent_type": "org_agent",
                "success": False,
                "error": error_message,
                "is_complete": True
            },
            "is_complete": True
        }
    
    async def process(self, query: str, metadata: Dict[str, Any] = None, messages: List[Any] = None) -> Dict[str, Any]:
        """
        Process org management commands using LangGraph workflow
        
        Args:
            query: User query string
            metadata: Optional metadata dictionary (persona, editor context, etc.)
            messages: Optional conversation history
            
        Returns:
            Dictionary with org operation response and metadata
        """
        try:
            logger.info(f"Org Agent: Starting processing: {query[:100]}...")
            
            metadata = metadata or {}
            user_id = metadata.get("user_id", "system")
            shared_memory = metadata.get("shared_memory", {}) or {}
            
            new_messages = self._prepare_messages_with_query(messages, query)
            
            workflow = await self._get_workflow()
            config = self._get_checkpoint_config(metadata)
            
            conversation_messages = await self._load_and_merge_checkpoint_messages(
                workflow, config, new_messages
            )
            
            checkpoint_state = await workflow.aget_state(config)
            existing_shared_memory = {}
            if checkpoint_state and checkpoint_state.values:
                existing_shared_memory = checkpoint_state.values.get("shared_memory", {})
            
            shared_memory_merged = shared_memory.copy()
            shared_memory_merged.update(existing_shared_memory)
            
            initial_state: OrgAgentState = {
                "query": query,
                "user_id": user_id,
                "metadata": metadata,
                "messages": conversation_messages,
                "shared_memory": shared_memory_merged,
                "user_message": "",
                "operation": "",
                "intent_type": "",
                "payload": {},
                "detected_links": [],
                "referenced_context": {},
                "pending": {},
                "response": {},
                "task_status": "",
                "error": ""
            }
            
            final_state = await workflow.ainvoke(initial_state, config=config)
            
            return final_state.get("response", {
                "messages": [AIMessage(content="Org operation failed")],
                "agent_results": {
                    "agent_type": self.agent_type,
                    "is_complete": False
                },
                "is_complete": False
            })
            
        except Exception as e:
            logger.error(f"Org Agent ERROR: {e}")
            return self._create_error_result(f"Org operation failed: {str(e)}")


# Singleton instance
_org_agent_instance = None


def get_org_agent() -> OrgAgent:
    """Get global org agent instance"""
    global _org_agent_instance
    if _org_agent_instance is None:
        _org_agent_instance = OrgAgent()
    return _org_agent_instance


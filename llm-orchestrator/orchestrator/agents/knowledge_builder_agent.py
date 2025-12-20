"""
Knowledge Builder Agent

Truth-seeking agent that takes URLs or questions, performs multi-round research
with verification, and outputs structured markdown documents for future reference.

Workflow:
1. Classify input (URL vs question)
2. Parse destination from query
3. Process URL if provided (crawl with Crawl4AI)
4. Research subgraph (multi-round research)
5. Verification subgraph (cross-reference, detect contradictions)
6. Markdown synthesis subgraph (structure into document)
7. Save to user-specified or auto-generated path
"""

import logging
import re
from typing import Dict, Any, List, Optional, TypedDict
from datetime import datetime
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

from orchestrator.agents.base_agent import BaseAgent
from orchestrator.tools import crawl_web_content_tool
from orchestrator.subgraphs import (
    build_research_workflow_subgraph,
    build_fact_verification_subgraph,
    build_knowledge_document_subgraph
)

logger = logging.getLogger(__name__)


class KnowledgeBuilderState(TypedDict):
    """State for Knowledge Builder Agent"""
    # Input
    query: str
    user_id: str
    
    # Parsed input
    input_type: str  # "question" | "url" | "mixed"
    url_content: Dict[str, Any]
    output_folder_path: Optional[str]
    output_filename: Optional[str]
    
    # Subgraph results
    research_findings: Dict[str, Any]
    verified_claims: List[Dict[str, Any]]
    contradictions: List[Dict[str, Any]]
    markdown_content: str
    
    # Output
    document_id: str
    file_path: str
    response: Dict[str, Any]
    error: str
    
    # Control
    current_round: int
    needs_more_research: bool
    shared_memory: Dict[str, Any]
    messages: List[Any]


class KnowledgeBuilderAgent(BaseAgent):
    """
    Knowledge Builder Agent for truth investigation and documentation
    
    Accepts URLs or questions, performs research with verification,
    and saves structured markdown documents.
    """
    
    def __init__(self):
        super().__init__("knowledge_builder_agent")
        self._research_subgraph = None
        self._verification_subgraph = None
        self._synthesis_subgraph = None
        logger.info("Knowledge Builder Agent initialized")
    
    def _get_subgraphs(self, checkpointer):
        """Get or build subgraphs"""
        if self._research_subgraph is None:
            self._research_subgraph = build_research_workflow_subgraph(checkpointer)
        if self._verification_subgraph is None:
            self._verification_subgraph = build_fact_verification_subgraph(checkpointer)
        if self._synthesis_subgraph is None:
            self._synthesis_subgraph = build_knowledge_document_subgraph(checkpointer)
        
        return self._research_subgraph, self._verification_subgraph, self._synthesis_subgraph
    
    def _build_workflow(self, checkpointer) -> StateGraph:
        """Build LangGraph workflow for knowledge builder agent"""
        workflow = StateGraph(KnowledgeBuilderState)
        
        # Get subgraphs
        research_sg, verification_sg, synthesis_sg = self._get_subgraphs(checkpointer)
        
        # Add nodes
        workflow.add_node("classify_input", self._classify_input_node)
        workflow.add_node("parse_destination", self._parse_destination_node)
        workflow.add_node("initial_url_processing", self._initial_url_processing_node)
        workflow.add_node("research_subgraph", self._call_research_subgraph)
        workflow.add_node("verification_subgraph", self._call_verification_subgraph)
        workflow.add_node("gap_detection", self._gap_detection_node)
        workflow.add_node("synthesis_subgraph", self._call_synthesis_subgraph)
        workflow.add_node("save_document", self._save_document_node)
        workflow.add_node("format_response", self._format_response_node)
        
        # Entry point
        workflow.set_entry_point("classify_input")
        
        # Flow
        workflow.add_edge("classify_input", "parse_destination")
        workflow.add_edge("parse_destination", "initial_url_processing")
        
        # Route based on URL presence
        workflow.add_conditional_edges(
            "initial_url_processing",
            lambda state: "research_subgraph",
            {
                "research_subgraph": "research_subgraph"
            }
        )
        
        workflow.add_edge("research_subgraph", "verification_subgraph")
        workflow.add_edge("verification_subgraph", "gap_detection")
        
        # Check if more research needed
        workflow.add_conditional_edges(
            "gap_detection",
            self._route_from_gap_detection,
            {
                "research_subgraph": "research_subgraph",
                "synthesis_subgraph": "synthesis_subgraph"
            }
        )
        
        workflow.add_edge("synthesis_subgraph", "save_document")
        workflow.add_edge("save_document", "format_response")
        workflow.add_edge("format_response", END)
        
        return workflow.compile(checkpointer=checkpointer)
    
    def _extract_url_from_message(self, message: str) -> Optional[str]:
        """Extract URL from message"""
        # Pattern for URLs
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        matches = re.findall(url_pattern, message)
        return matches[0] if matches else None
    
    def _parse_file_destination(self, query: str) -> Dict[str, Optional[str]]:
        """Extract folder path and filename from natural language"""
        # Pattern: "to [path]/[filename.md]" or "in [path]/[filename.md]"
        pattern = r'(?:to|in|at|save to)\s+(.+\.(?:md|txt|org))'
        match = re.search(pattern, query, re.IGNORECASE)
        
        if match:
            full_path = match.group(1).strip()
            
            # Split into folder_path and filename
            if '/' in full_path:
                parts = full_path.rsplit('/', 1)
                folder_path = parts[0]
                filename = parts[1]
            else:
                folder_path = None
                filename = full_path
            
            # Clean up "My Documents" prefix (optional - system adds it automatically)
            if folder_path and folder_path.startswith('My Documents/'):
                folder_path = folder_path.replace('My Documents/', '', 1)
            
            return {
                'folder_path': folder_path,
                'filename': filename
            }
        
        return {'folder_path': None, 'filename': None}
    
    async def _classify_input_node(self, state: KnowledgeBuilderState) -> Dict[str, Any]:
        """Classify input as URL, question, or mixed"""
        try:
            query = state.get("query", "")
            logger.info(f"Classifying input: {query[:100]}...")
            
            # Extract URL
            url = self._extract_url_from_message(query)
            
            # Determine input type
            if url and len(query.strip()) > len(url) + 20:
                input_type = "mixed"
            elif url:
                input_type = "url"
            else:
                input_type = "question"
            
            logger.info(f"Input classified as: {input_type}")
            
            return {
                "input_type": input_type,
                "url": url if url else None
            }
            
        except Exception as e:
            logger.error(f"Classify input error: {e}")
            return {
                "input_type": "question",
                "url": None,
                "error": str(e)
            }
    
    async def _parse_destination_node(self, state: KnowledgeBuilderState) -> Dict[str, Any]:
        """Parse where user wants to save the document"""
        try:
            query = state.get("query", "")
            logger.info("Parsing destination from query")
            
            # Check for explicit file destination
            destination = self._parse_file_destination(query)
            
            if destination['filename']:
                # User specified location
                output_folder_path = destination['folder_path']
                output_filename = destination['filename']
                logger.info(f"User specified destination: {output_folder_path}/{output_filename}")
            else:
                # Generate automatic location
                # Extract topic from query
                topic = query[:50].replace(" ", "_").replace("/", "_")
                # Remove special chars
                topic = re.sub(r'[^\w\-_]', '', topic)
                if not topic:
                    topic = "investigation"
                
                timestamp = datetime.now().strftime('%Y%m%d')
                
                output_folder_path = "Knowledge Base/Auto-Research"
                output_filename = f"{topic}_{timestamp}.md"
                logger.info(f"Auto-generated destination: {output_folder_path}/{output_filename}")
            
            return {
                "output_folder_path": output_folder_path,
                "output_filename": output_filename
            }
            
        except Exception as e:
            logger.error(f"Parse destination error: {e}")
            # Fallback
            timestamp = datetime.now().strftime('%Y%m%d')
            return {
                "output_folder_path": "Knowledge Base/Auto-Research",
                "output_filename": f"investigation_{timestamp}.md",
                "error": str(e)
            }
    
    async def _initial_url_processing_node(self, state: KnowledgeBuilderState) -> Dict[str, Any]:
        """Process URL if provided - crawl with Crawl4AI"""
        try:
            url = state.get("url")
            input_type = state.get("input_type", "question")
            
            if input_type in ["url", "mixed"] and url:
                logger.info(f"Processing URL: {url}")
                
                # Track tool usage
                shared_memory = state.get("shared_memory", {})
                previous_tools = shared_memory.get("previous_tools_used", [])
                if "crawl_web_content_tool" not in previous_tools:
                    previous_tools.append("crawl_web_content_tool")
                    shared_memory["previous_tools_used"] = previous_tools
                    state["shared_memory"] = shared_memory
                
                # Crawl URL
                crawl_result = await crawl_web_content_tool(urls=[url])
                logger.info("Tool used: crawl_web_content_tool (URL processing)")
                
                # Extract claims/topics from URL content
                url_content = {
                    "url": url,
                    "content": crawl_result,
                    "processed": True
                }
                
                # Update query to include URL content context
                original_query = state.get("query", "")
                enhanced_query = f"{original_query}\n\nURL Content:\n{crawl_result[:2000]}"
                
                logger.info(f"URL processed: {len(crawl_result)} characters extracted")
                
                return {
                    "url_content": url_content,
                    "query": enhanced_query  # Enhance query with URL content
                }
            
            # No URL to process
            return {
                "url_content": {},
                "url_processed": False
            }
            
        except Exception as e:
            logger.error(f"Initial URL processing error: {e}")
            return {
                "url_content": {},
                "url_processed": False,
                "error": str(e)
            }
    
    async def _call_research_subgraph(self, state: KnowledgeBuilderState) -> Dict[str, Any]:
        """Call research workflow subgraph"""
        try:
            logger.info("Calling research subgraph")
            
            workflow = await self._get_workflow()
            checkpointer = workflow.checkpointer
            
            research_sg, _, _ = self._get_subgraphs(checkpointer)
            
            # Prepare subgraph state
            subgraph_state = {
                "query": state.get("query", ""),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "user_id": state.get("user_id", "system")
            }
            
            # Run subgraph
            config = self._get_checkpoint_config(state.get("metadata", {}))
            result = await research_sg.ainvoke(subgraph_state, config)
            
            logger.info("Research subgraph completed")
            
            return {
                "research_findings": result.get("research_findings", {}),
                "sources_found": result.get("sources_found", []),
                "citations": result.get("citations", []),
                "research_sufficient": result.get("research_sufficient", False)
            }
            
        except Exception as e:
            logger.error(f"Research subgraph error: {e}")
            return {
                "research_findings": {},
                "sources_found": [],
                "citations": [],
                "research_sufficient": False,
                "error": str(e)
            }
    
    async def _call_verification_subgraph(self, state: KnowledgeBuilderState) -> Dict[str, Any]:
        """Call fact verification subgraph"""
        try:
            logger.info("Calling verification subgraph")
            
            workflow = await self._get_workflow()
            checkpointer = workflow.checkpointer
            
            _, verification_sg, _ = self._get_subgraphs(checkpointer)
            
            # Prepare subgraph state
            subgraph_state = {
                "research_findings": state.get("research_findings", {}),
                "sources_found": state.get("sources_found", []),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "user_id": state.get("user_id", "system")
            }
            
            # Run subgraph
            config = self._get_checkpoint_config(state.get("metadata", {}))
            result = await verification_sg.ainvoke(subgraph_state, config)
            
            logger.info("Verification subgraph completed")
            
            return {
                "verified_claims": result.get("verified_claims", []),
                "contradictions": result.get("contradictions", []),
                "uncertainties": result.get("uncertainties", []),
                "consensus_findings": result.get("consensus_findings", {}),
                "independent_sources": result.get("independent_sources", [])
            }
            
        except Exception as e:
            logger.error(f"Verification subgraph error: {e}")
            return {
                "verified_claims": [],
                "contradictions": [],
                "uncertainties": [],
                "consensus_findings": {},
                "independent_sources": [],
                "error": str(e)
            }
    
    async def _gap_detection_node(self, state: KnowledgeBuilderState) -> Dict[str, Any]:
        """Check if more research is needed"""
        try:
            research_sufficient = state.get("research_sufficient", False)
            verified_claims = state.get("verified_claims", [])
            contradictions = state.get("contradictions", [])
            current_round = state.get("current_round", 0)
            
            # Max 2 research rounds to prevent infinite loops
            max_rounds = 2
            
            needs_more_research = (
                not research_sufficient and
                current_round < max_rounds and
                (len(verified_claims) == 0 or len(contradictions) > len(verified_claims))
            )
            
            logger.info(f"Gap detection: needs_more_research={needs_more_research}, round={current_round}")
            
            return {
                "needs_more_research": needs_more_research,
                "current_round": current_round + 1
            }
            
        except Exception as e:
            logger.error(f"Gap detection error: {e}")
            return {
                "needs_more_research": False,
                "error": str(e)
            }
    
    def _route_from_gap_detection(self, state: KnowledgeBuilderState) -> str:
        """Route based on gap detection"""
        if state.get("needs_more_research"):
            return "research_subgraph"
        return "synthesis_subgraph"
    
    async def _call_synthesis_subgraph(self, state: KnowledgeBuilderState) -> Dict[str, Any]:
        """Call markdown synthesis subgraph"""
        try:
            logger.info("Calling synthesis subgraph")
            
            workflow = await self._get_workflow()
            checkpointer = workflow.checkpointer
            
            _, _, synthesis_sg = self._get_subgraphs(checkpointer)
            
            # Prepare subgraph state
            subgraph_state = {
                "verified_claims": state.get("verified_claims", []),
                "contradictions": state.get("contradictions", []),
                "uncertainties": state.get("uncertainties", []),
                "sources_found": state.get("sources_found", []),
                "independent_sources": state.get("independent_sources", []),
                "query": state.get("query", ""),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "user_id": state.get("user_id", "system")
            }
            
            # Run subgraph
            config = self._get_checkpoint_config(state.get("metadata", {}))
            result = await synthesis_sg.ainvoke(subgraph_state, config)
            
            logger.info("Synthesis subgraph completed")
            
            return {
                "markdown_content": result.get("markdown_content", ""),
                "document_metadata": result.get("document_metadata", {})
            }
            
        except Exception as e:
            logger.error(f"Synthesis subgraph error: {e}")
            return {
                "markdown_content": "# Investigation\n\n*Error generating document*\n",
                "document_metadata": {},
                "error": str(e)
            }
    
    async def _save_document_node(self, state: KnowledgeBuilderState) -> Dict[str, Any]:
        """Save markdown document to user-specified or auto-generated path"""
        try:
            markdown_content = state.get("markdown_content", "")
            output_folder_path = state.get("output_folder_path", "Knowledge Base/Auto-Research")
            output_filename = state.get("output_filename", f"investigation_{datetime.now().strftime('%Y%m%d')}.md")
            user_id = state.get("user_id", "system")
            document_metadata = state.get("document_metadata", {})
            
            logger.info(f"Saving document: {output_folder_path}/{output_filename}")
            
            # Import create_user_file_tool
            from orchestrator.tools.file_creation_tools import create_user_file_tool
            
            # Extract title from metadata
            title = document_metadata.get("title", output_filename.replace(".md", ""))
            tags = document_metadata.get("tags", ["truth-seeking", "research"])
            category = "research"
            
            # Create file
            result = await create_user_file_tool(
                filename=output_filename,
                content=markdown_content,
                folder_path=output_folder_path,
                title=title,
                tags=tags,
                category=category,
                user_id=user_id
            )
            
            if result.get("success"):
                logger.info(f"Document saved successfully: {result.get('document_id')}")
                return {
                    "document_id": result.get("document_id", ""),
                    "file_path": f"{output_folder_path}/{output_filename}",
                    "save_success": True
                }
            else:
                logger.error(f"Failed to save document: {result.get('error')}")
                return {
                    "document_id": "",
                    "file_path": "",
                    "save_success": False,
                    "error": result.get("error", "Unknown error")
                }
            
        except Exception as e:
            logger.error(f"Save document error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "document_id": "",
                "file_path": "",
                "save_success": False,
                "error": str(e)
            }
    
    async def _format_response_node(self, state: KnowledgeBuilderState) -> Dict[str, Any]:
        """Format final response"""
        try:
            save_success = state.get("save_success", False)
            document_id = state.get("document_id", "")
            file_path = state.get("file_path", "")
            output_filename = state.get("output_filename", "")
            
            if save_success:
                response_text = f"✅ Knowledge document created successfully!\n\n"
                response_text += f"**File**: {file_path}\n"
                response_text += f"**Document ID**: {document_id}\n\n"
                response_text += f"The investigation has been saved to your Knowledge Base for future reference."
            else:
                error = state.get("error", "Unknown error")
                response_text = f"❌ Failed to save knowledge document: {error}"
            
            return {
                "response": {
                    "task_status": "complete" if save_success else "error",
                    "response": response_text,
                    "document_id": document_id,
                    "file_path": file_path
                }
            }
            
        except Exception as e:
            logger.error(f"Format response error: {e}")
            return {
                "response": {
                    "task_status": "error",
                    "response": f"Error formatting response: {str(e)}"
                }
            }


def get_knowledge_builder_agent() -> KnowledgeBuilderAgent:
    """Get singleton instance of Knowledge Builder Agent"""
    global _knowledge_builder_agent_instance
    if _knowledge_builder_agent_instance is None:
        _knowledge_builder_agent_instance = KnowledgeBuilderAgent()
    return _knowledge_builder_agent_instance


_knowledge_builder_agent_instance: Optional[KnowledgeBuilderAgent] = None


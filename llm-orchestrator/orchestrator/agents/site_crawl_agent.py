"""
Site Crawl Agent - LangGraph Implementation
Given a link and a query, crawl that site and synthesize relevant pages.
Domain-scoped crawling with query-based filtering.
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict
from urllib.parse import urlparse

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from .base_agent import BaseAgent, TaskStatus
from orchestrator.backend_tool_client import get_backend_tool_client

logger = logging.getLogger(__name__)


class SiteCrawlState(TypedDict):
    """State for site crawl agent LangGraph workflow"""
    query: str
    user_id: str
    metadata: Dict[str, Any]
    messages: List[Any]
    shared_memory: Dict[str, Any]
    seed_url: str
    query_criteria: str
    allowed_path_prefix: Optional[str]
    crawl_results: Dict[str, Any]
    filtered_results: List[Dict[str, Any]]
    synthesized_findings: str
    response: Dict[str, Any]
    task_status: str
    error: str


class SiteCrawlAgent(BaseAgent):
    """
    Agent for domain-scoped site crawling with query-based filtering.
    
    Handles:
    - URL extraction from query
    - Domain-scoped crawling
    - Query-based filtering
    - LLM-based synthesis of findings
    Uses LangGraph workflow for explicit state management
    """
    
    def __init__(self):
        super().__init__("site_crawl_agent")
        logger.info("Site Crawl Agent initialized")
    
    def _build_workflow(self, checkpointer) -> StateGraph:
        """Build LangGraph workflow for site crawl agent"""
        workflow = StateGraph(SiteCrawlState)
        
        # Add nodes
        workflow.add_node("extract_url_and_query", self._extract_url_and_query_node)
        workflow.add_node("discover_scope", self._discover_scope_node)
        workflow.add_node("execute_crawl", self._execute_crawl_node)
        workflow.add_node("filter_results", self._filter_results_node)
        workflow.add_node("synthesize_findings", self._synthesize_findings_node)
        workflow.add_node("format_response", self._format_response_node)
        
        # Entry point
        workflow.set_entry_point("extract_url_and_query")
        
        # Linear flow
        workflow.add_edge("extract_url_and_query", "discover_scope")
        workflow.add_edge("discover_scope", "execute_crawl")
        workflow.add_edge("execute_crawl", "filter_results")
        workflow.add_edge("filter_results", "synthesize_findings")
        workflow.add_edge("synthesize_findings", "format_response")
        workflow.add_edge("format_response", END)
        
        return workflow.compile(checkpointer=checkpointer)
    
    async def _extract_url_and_query_node(self, state: SiteCrawlState) -> Dict[str, Any]:
        """Extract seed URL and query from user input"""
        try:
            logger.info("Extracting URL and query...")
            
            shared_memory = state.get("shared_memory", {}) or {}
            seed_url = shared_memory.get("seed_url") or ""
            query = state.get("query", "")
            
            # Extract from messages if not in state
            messages = state.get("messages", [])
            if not query and messages:
                for msg in reversed(messages):
                    if isinstance(msg, HumanMessage):
                        query = msg.content
                        break
                    elif hasattr(msg, "type") and msg.type == "human":
                        query = msg.content
                        break
            
            # Extract URL from query if not in shared_memory
            if not seed_url and query:
                seed_url = self._extract_url_from_message(query)
            
            if not seed_url:
                return {
                    "response": {
                        "task_status": TaskStatus.ERROR.value,
                        "response": "No seed URL provided. Include a link to the site to crawl.",
                        "error": "no_url"
                    },
                    "task_status": "error",
                    "error": "no_url"
                }
            
            if not query:
                return {
                    "response": {
                        "task_status": TaskStatus.ERROR.value,
                        "response": "No query provided. Please specify what you're looking for on the site.",
                        "error": "no_query"
                    },
                    "task_status": "error",
                    "error": "no_query"
                }
            
            return {
                "seed_url": seed_url,
                "query_criteria": query
            }
            
        except Exception as e:
            logger.error(f"Failed to extract URL and query: {e}")
            return {
                "error": str(e),
                "task_status": "error",
                "response": {
                    "task_status": TaskStatus.ERROR.value,
                    "response": f"Failed to extract URL and query: {str(e)}"
                }
            }
    
    async def _discover_scope_node(self, state: SiteCrawlState) -> Dict[str, Any]:
        """Discover crawl scope from seed URL"""
        try:
            logger.info("Discovering crawl scope...")
            
            seed_url = state.get("seed_url", "")
            if not seed_url:
                return {
                    "error": "No seed URL",
                    "task_status": "error"
                }
            
            # Calculate path prefix scope from seed
            parsed = urlparse(seed_url)
            allowed_prefix = parsed.path if isinstance(parsed.path, str) and len(parsed.path) > 1 else None
            
            return {
                "allowed_path_prefix": allowed_prefix
            }
            
        except Exception as e:
            logger.error(f"Failed to discover scope: {e}")
            return {
                "error": str(e),
                "task_status": "error"
            }
    
    async def _execute_crawl_node(self, state: SiteCrawlState) -> Dict[str, Any]:
        """Execute domain-scoped crawl"""
        try:
            logger.info("Executing site crawl...")
            
            seed_url = state.get("seed_url", "")
            query_criteria = state.get("query_criteria", "")
            allowed_prefix = state.get("allowed_path_prefix")
            user_id = state.get("user_id", "system")
            
            if not seed_url:
                return {
                    "crawl_results": {"success": False, "error": "No seed URL"},
                    "task_status": "error"
                }
            
            # Use domain-scoped crawl via gRPC
            tool_client = await get_backend_tool_client()
            
            crawl_result = await tool_client.crawl_site(
                seed_url=seed_url,
                query_criteria=query_criteria,
                max_pages=120,
                max_depth=3,
                allowed_path_prefix=allowed_prefix,
                include_pdfs=False,
                user_id=user_id
            )
            
            # Result is already in expected format from gRPC client
            crawl_results = crawl_result
            
            return {
                "crawl_results": crawl_results
            }
            
        except Exception as e:
            logger.error(f"Failed to execute crawl: {e}")
            return {
                "crawl_results": {"success": False, "error": str(e), "results": []},
                "error": str(e),
                "task_status": "error"
            }
    
    async def _filter_results_node(self, state: SiteCrawlState) -> Dict[str, Any]:
        """Filter crawl results by relevance"""
        try:
            logger.info("Filtering crawl results...")
            
            crawl_results = state.get("crawl_results", {})
            query_criteria = state.get("query_criteria", "").lower()
            
            if not crawl_results.get("success"):
                return {
                    "filtered_results": [],
                    "error": crawl_results.get("error", "Crawl failed")
                }
            
            items = crawl_results.get("results", [])
            
            # Filter valid items
            def looks_valid(item: Dict[str, Any]) -> bool:
                if not item.get("success"):
                    return False
                url = (item.get("url") or "").lower()
                title = ((item.get("metadata") or {}).get("title") or "").lower()
                if "page not found" in title:
                    return False
                return True
            
            filtered = [i for i in items if looks_valid(i)]
            
            # Sort by relevance score if available
            filtered.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
            
            # De-duplicate by URL
            seen = set()
            top: List[Dict[str, Any]] = []
            for item in filtered:
                url = item.get("url")
                if url and url not in seen:
                    seen.add(url)
                    top.append(item)
                if len(top) >= 40:
                    break
            
            return {
                "filtered_results": top
            }
            
        except Exception as e:
            logger.error(f"Failed to filter results: {e}")
            return {
                "filtered_results": [],
                "error": str(e)
            }
    
    async def _synthesize_findings_node(self, state: SiteCrawlState) -> Dict[str, Any]:
        """Synthesize findings using LLM"""
        try:
            logger.info("Synthesizing findings...")
            
            filtered = state.get("filtered_results", [])
            query = state.get("query_criteria", "")
            seed_url = state.get("seed_url", "")
            
            if not filtered:
                return {
                    "synthesized_findings": "No relevant pages found on the site."
                }
            
            # Prepare compact context of top pages
            def compact(item: Dict[str, Any]) -> str:
                title = ((item.get("metadata") or {}).get("title") or "No title").strip() or "No title"
                url = item.get("url", "")
                content = (item.get("full_content") or "")
                snippet = content[:1800]
                return f"TITLE: {title}\nURL: {url}\nCONTENT_SNIPPET:\n{snippet}\n"
            
            top_context_blocks = "\n\n".join(compact(p) for p in filtered[:12])
            
            system_prompt = (
                "You are a Site Crawl Synthesis Officer. Given the user query and content snippets "
                "from pages strictly within the target site, produce a concise, well-structured answer with "
                "clear headings and bullet lists where appropriate. Use ONLY the provided snippets. Do not "
                "invent facts; explicitly note when data is missing. Include inline numeric citations like (1), (2) "
                "mapping to the order of the snippets provided."
            )
            
            user_prompt = f"""
TARGET SITE: {seed_url}
USER QUERY: {query}

PAGE CONTEXT (snippets in order):
{top_context_blocks}

TASK:
1) Answer the USER QUERY strictly using only the content in the snippets above.
2) If the query asks for lists, counts, rankings, or tables, aggregate best-effort numbers from the text and label them as estimates when uncertain.
3) If the query asks for qualitative analysis or synthesis, provide a clear, well-organized summary addressing each requested facet.
4) Always include inline citations like (1), (2) pointing to the relevant snippet(s).
5) If critical data to answer the query is missing in the snippets, state what is missing and suggest what additional on-site pages might be needed.
"""
            
            # Get LLM instance
            llm = self._get_llm(temperature=0.2, state=state)
            
            # Generate synthesis
            datetime_context = self._get_datetime_context()
            messages = [
                SystemMessage(content=system_prompt),
                SystemMessage(content=datetime_context),
                HumanMessage(content=user_prompt)
            ]
            
            response = await llm.ainvoke(messages)
            content = response.content if hasattr(response, 'content') else str(response)
            
            return {
                "synthesized_findings": content.strip()
            }
            
        except Exception as e:
            logger.warning(f"LLM synthesis failed: {e}")
            return {
                "synthesized_findings": ""
            }
    
    async def _format_response_node(self, state: SiteCrawlState) -> Dict[str, Any]:
        """Format final response with citations"""
        try:
            logger.info("Formatting site crawl response...")
            
            synthesized = state.get("synthesized_findings", "")
            filtered = state.get("filtered_results", [])
            seed_url = state.get("seed_url", "")
            
            # Build simple list for context
            simple_list_lines = [f"## Focused findings from {seed_url}\n"]
            for i, item in enumerate(filtered[:10], 1):
                title = ((item.get("metadata") or {}).get("title") or "No title").strip() or "No title"
                url = item.get("url", "")
                score = item.get("relevance_score", 0.0)
                simple_list_lines.append(f"- {i}. **{title}** (relevance {score:.2f}) — {url}")
            simple_list = "\n".join(simple_list_lines)
            
            # Combine synthesized findings with simple list
            findings = f"{synthesized}\n\n---\n\n{simple_list}" if synthesized else simple_list
            
            # Build citations
            citations = []
            for idx, item in enumerate(filtered[:10], 1):
                citations.append({
                    "id": idx,
                    "title": ((item.get("metadata") or {}).get("title") or "Webpage"),
                    "type": "webpage",
                    "url": item.get("url", ""),
                    "author": None,
                    "date": None,
                    "excerpt": None,
                })
            
            # Build structured response
            response_dict = {
                "task_status": TaskStatus.COMPLETE.value,
                "response": findings,
                "structured_response": {
                    "task_status": "complete",
                    "findings": findings,
                    "citations": citations,
                    "sources_searched": ["domain_crawl"],
                    "permission_request": None,
                    "confidence_level": 0.85,
                    "next_steps": None,
                },
                "research_mode": "site_crawl",
                "timestamp": datetime.now().isoformat(),
            }
            
            # Add assistant message to state for checkpointing
            updated_state = self._add_assistant_response_to_messages(state, findings)
            
            return {
                "response": response_dict,
                "task_status": "complete",
                **updated_state
            }
            
        except Exception as e:
            logger.error(f"Failed to format response: {e}")
            return {
                "response": {
                    "task_status": TaskStatus.ERROR.value,
                    "response": f"Failed to format response: {str(e)}"
                },
                "task_status": "error",
                "error": str(e)
            }
    
    def _extract_url_from_message(self, message: str) -> str:
        """Extract URL from message text"""
        try:
            if not message:
                return ""
            m = re.search(r'https?://[^\s)>\"]+', message)
            return m.group(0) if m else ""
        except Exception:
            return ""
    
    async def process(
        self,
        query: str = None,
        metadata: Dict[str, Any] = None,
        messages: List[Any] = None
    ) -> Dict[str, Any]:
        """
        Process site crawl request using LangGraph workflow
        
        Args:
            query: User query containing URL and search criteria
            metadata: Optional metadata dictionary
            messages: Optional conversation history
            
        Returns:
            Dict with structured response and task status
        """
        try:
            metadata = metadata or {}
            messages = messages or []
            
            # Extract query from messages if not provided
            if not query and messages:
                for msg in reversed(messages):
                    if isinstance(msg, HumanMessage):
                        query = msg.content
                        break
                    elif hasattr(msg, "type") and msg.type == "human":
                        query = msg.content
                        break
            
            if not query:
                return {
                    "task_status": TaskStatus.ERROR.value,
                    "response": "No query provided"
                }
            
            user_id = metadata.get("user_id", "system")
            shared_memory = metadata.get("shared_memory", {}) or {}
            
            logger.info(f"Site Crawl Agent processing: {query[:80]}...")
            
            # Build initial state for LangGraph workflow
            initial_state: SiteCrawlState = {
                "query": query,
                "user_id": user_id,
                "metadata": metadata,
                "messages": messages,
                "shared_memory": shared_memory,
                "seed_url": "",
                "query_criteria": "",
                "allowed_path_prefix": None,
                "crawl_results": {},
                "filtered_results": [],
                "synthesized_findings": "",
                "response": {},
                "task_status": "",
                "error": ""
            }
            
            # Get workflow (lazy initialization with checkpointer)
            workflow = await self._get_workflow()
            
            # Get checkpoint config
            config = self._get_checkpoint_config(metadata)
            
            # Load checkpointed messages and merge with new messages
            merged_messages = await self._load_and_merge_checkpoint_messages(
                workflow, config, messages
            )
            initial_state["messages"] = merged_messages
            
            # Invoke LangGraph workflow with checkpointing
            final_state = await workflow.ainvoke(initial_state, config=config)
            
            # Return response from final state
            return final_state.get("response", {
                "task_status": TaskStatus.ERROR.value,
                "response": "Site crawl failed"
            })
            
        except Exception as e:
            logger.error(f"Site Crawl Agent failed: {e}")
            return {
                "task_status": TaskStatus.ERROR.value,
                "response": f"Site crawl failed: {str(e)}"
            }


# Singleton instance
_site_crawl_agent_instance = None


def get_site_crawl_agent() -> SiteCrawlAgent:
    """Get global site crawl agent instance"""
    global _site_crawl_agent_instance
    if _site_crawl_agent_instance is None:
        _site_crawl_agent_instance = SiteCrawlAgent()
    return _site_crawl_agent_instance


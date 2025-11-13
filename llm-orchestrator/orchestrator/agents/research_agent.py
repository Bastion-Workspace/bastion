"""
Research Agent - Document-focused research using backend gRPC tools
Simplified, focused agent for Phase 4
"""

import logging
from typing import Dict, Any, List, TypedDict
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from orchestrator.tools import search_documents_tool, get_document_content_tool
from config.settings import settings

logger = logging.getLogger(__name__)


class ResearchState(TypedDict):
    """State for research agent workflow"""
    query: str
    messages: List[Any]
    documents_found: List[Dict[str, Any]]
    research_complete: bool
    final_response: str
    error: str


class ResearchAgent:
    """
    Research Agent using backend document tools via gRPC
    
    Workflow:
    1. Analyze query
    2. Search documents via gRPC
    3. Synthesize findings
    4. Return formatted response
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.DEFAULT_MODEL,
            openai_api_key=settings.OPENROUTER_API_KEY,
            openai_api_base=settings.OPENROUTER_BASE_URL,
            temperature=0.7
        )
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build LangGraph workflow for research"""
        
        # Create workflow
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("search_documents", self._search_documents_node)
        workflow.add_node("synthesize_answer", self._synthesize_answer_node)
        
        # Add edges
        workflow.set_entry_point("search_documents")
        workflow.add_edge("search_documents", "synthesize_answer")
        workflow.add_edge("synthesize_answer", END)
        
        return workflow.compile()
    
    async def _search_documents_node(self, state: ResearchState) -> Dict[str, Any]:
        """Search documents via gRPC backend"""
        try:
            query = state["query"]
            logger.info(f"📚 Searching documents for: {query}")
            
            # Search documents via gRPC tool
            search_result = await search_documents_tool(
                query=query,
                limit=10
            )
            
            logger.info(f"✅ Document search completed")
            
            return {
                "documents_found": [{"content": search_result}],
                "messages": state.get("messages", []) + [
                    AIMessage(content=f"Found documents: {search_result[:200]}...")
                ]
            }
            
        except Exception as e:
            logger.error(f"❌ Document search failed: {e}")
            return {
                "documents_found": [],
                "error": str(e),
                "messages": state.get("messages", []) + [
                    AIMessage(content=f"Error searching documents: {e}")
                ]
            }
    
    async def _synthesize_answer_node(self, state: ResearchState) -> Dict[str, Any]:
        """Synthesize answer from search results"""
        try:
            query = state["query"]
            documents = state.get("documents_found", [])
            
            if not documents or state.get("error"):
                return {
                    "final_response": f"I couldn't find relevant documents for your query: {query}",
                    "research_complete": True
                }
            
            logger.info(f"🧠 Synthesizing answer from {len(documents)} document results")
            
            # Build synthesis prompt
            doc_content = "\n\n".join([doc.get("content", "") for doc in documents])
            
            synthesis_prompt = f"""You are a research assistant. Based on the document search results below, provide a comprehensive answer to the user's query.

USER QUERY:
{query}

DOCUMENT SEARCH RESULTS:
{doc_content}

Provide a clear, well-organized response that:
1. Directly answers the user's question
2. Cites relevant information from the documents
3. Acknowledges if information is incomplete
4. Uses a professional but conversational tone

Your response:"""
            
            # Get LLM synthesis
            response = await self.llm.ainvoke([
                SystemMessage(content="You are a helpful research assistant."),
                HumanMessage(content=synthesis_prompt)
            ])
            
            final_response = response.content
            
            logger.info(f"✅ Synthesis complete: {len(final_response)} characters")
            
            return {
                "final_response": final_response,
                "research_complete": True,
                "messages": state.get("messages", []) + [
                    AIMessage(content=final_response)
                ]
            }
            
        except Exception as e:
            logger.error(f"❌ Synthesis failed: {e}")
            return {
                "final_response": f"Error synthesizing answer: {e}",
                "research_complete": True,
                "error": str(e)
            }
    
    async def research(self, query: str) -> Dict[str, Any]:
        """
        Execute research workflow
        
        Args:
            query: Research query
            
        Returns:
            Dict with final_response, documents_found, etc.
        """
        try:
            logger.info(f"🔬 Starting research for: {query}")
            
            # Initialize state
            initial_state = {
                "query": query,
                "messages": [HumanMessage(content=query)],
                "documents_found": [],
                "research_complete": False,
                "final_response": "",
                "error": ""
            }
            
            # Run workflow
            result = await self.workflow.ainvoke(initial_state)
            
            logger.info(f"✅ Research complete")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Research failed: {e}")
            return {
                "query": query,
                "final_response": f"Research failed: {str(e)}",
                "research_complete": True,
                "error": str(e)
            }


# Global research agent instance
_research_agent = None


def get_research_agent() -> ResearchAgent:
    """Get or create global research agent instance"""
    global _research_agent
    if _research_agent is None:
        _research_agent = ResearchAgent()
    return _research_agent


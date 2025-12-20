"""
Full Document Analysis Subgraph

Retrieves full documents and runs multiple targeted queries in parallel
for deeper analysis when chunks are insufficient.

Inputs:
- document_ids: List[str] - Document IDs to analyze (top 2-3)
- analysis_queries: List[str] - Specific questions to ask (3-4 queries)
- original_query: str - Original user query
- chunk_context: List[Dict] - Chunk results that triggered this (for context)
- user_id: str - User ID for access control

Outputs:
- full_doc_insights: Dict[str, Any] - Results from parallel queries
- documents_analyzed: List[str] - IDs of successfully analyzed documents
- synthesis: str - Synthesized answer from all parallel queries
- success: bool
"""

import logging
import json
import re
import asyncio
from typing import Dict, Any, List, Optional

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage

from orchestrator.tools import get_document_content_tool
from orchestrator.agents.base_agent import BaseAgent
from orchestrator.backend_tool_client import get_backend_tool_client

logger = logging.getLogger(__name__)

# Configuration constants
MAX_DOC_TOKENS = 100000  # 100k tokens
MAX_DOCS_TO_ANALYZE = 2
MAX_PARALLEL_QUERIES = 4

# Use Dict[str, Any] for compatibility with any agent state
FullDocumentAnalysisSubgraphState = Dict[str, Any]


async def retrieve_full_documents_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieve full content for specified document IDs
    
    Checks document size and skips if > 100k tokens
    """
    try:
        document_ids = state.get("document_ids", [])
        user_id = state.get("user_id", "system")
        
        if not document_ids:
            logger.warning("No document IDs provided for full document retrieval")
            return {
                "full_documents": {},
                "documents_analyzed": [],
                "retrieval_errors": []
            }
        
        logger.info(f"Retrieving full documents: {len(document_ids)} document(s)")
        
        # Limit to max docs
        document_ids = document_ids[:MAX_DOCS_TO_ANALYZE]
        
        # Get backend client for metadata
        client = await get_backend_tool_client()
        
        full_documents = {}
        documents_analyzed = []
        retrieval_errors = []
        
        # Retrieve documents in parallel
        async def retrieve_doc(doc_id: str):
            """Retrieve single document with size check"""
            try:
                # First check document metadata for size
                try:
                    # Try to get document metadata to check size
                    # Note: This may not be available, so we'll proceed with retrieval
                    # and check content size after retrieval
                    pass
                except Exception:
                    pass
                
                # Retrieve full content
                content = await get_document_content_tool(doc_id, user_id)
                
                # Check for errors
                if content.startswith("Document not found:") or content.startswith("Error getting document content:"):
                    logger.warning(f"Failed to retrieve document {doc_id}: {content}")
                    return None, doc_id, content
                
                # Estimate token count (rough: ~4 chars per token)
                estimated_tokens = len(content) // 4
                
                if estimated_tokens > MAX_DOC_TOKENS:
                    logger.warning(f"Document {doc_id} too large ({estimated_tokens} tokens), skipping")
                    return None, doc_id, f"Document too large: {estimated_tokens} tokens"
                
                logger.info(f"Retrieved document {doc_id}: {len(content)} chars (~{estimated_tokens} tokens)")
                return content, doc_id, None
                
            except Exception as e:
                logger.error(f"Error retrieving document {doc_id}: {e}")
                return None, doc_id, str(e)
        
        # Retrieve all documents in parallel
        retrieval_tasks = [retrieve_doc(doc_id) for doc_id in document_ids]
        results = await asyncio.gather(*retrieval_tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Retrieval task exception: {result}")
                continue
            
            content, doc_id, error = result
            
            if error:
                retrieval_errors.append({"document_id": doc_id, "error": error})
            elif content:
                full_documents[doc_id] = content
                documents_analyzed.append(doc_id)
        
        logger.info(f"Successfully retrieved {len(documents_analyzed)}/{len(document_ids)} documents")
        
        return {
            "full_documents": full_documents,
            "documents_analyzed": documents_analyzed,
            "retrieval_errors": retrieval_errors
        }
        
    except Exception as e:
        logger.error(f"Retrieve full documents node error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "full_documents": {},
            "documents_analyzed": [],
            "retrieval_errors": [{"error": str(e)}]
        }


async def parallel_document_query_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run multiple queries in parallel against full documents
    
    For each (document, query) pair, creates async task and runs all in parallel
    """
    try:
        full_documents = state.get("full_documents", {})
        analysis_queries = state.get("analysis_queries", [])
        original_query = state.get("original_query", "")
        
        if not full_documents:
            logger.warning("No full documents available for querying")
            return {
                "query_results": [],
                "parallel_query_errors": []
            }
        
        if not analysis_queries:
            logger.warning("No analysis queries provided")
            return {
                "query_results": [],
                "parallel_query_errors": []
            }
        
        # Limit queries
        analysis_queries = analysis_queries[:MAX_PARALLEL_QUERIES]
        
        logger.info(f"Running {len(analysis_queries)} queries against {len(full_documents)} documents in parallel")
        
        # Get LLM
        base_agent = BaseAgent("full_document_analysis_subgraph")
        llm = base_agent._get_llm(temperature=0.3, state=state)
        datetime_context = base_agent._get_datetime_context()
        
        async def query_document(doc_content: str, doc_id: str, query: str) -> Dict[str, Any]:
            """Query a single document with a specific question"""
            try:
                prompt = f"""You are analyzing a full document to answer a specific question.

ORIGINAL USER QUERY: {original_query}

DOCUMENT CONTENT:
{doc_content}

SPECIFIC QUESTION TO ANSWER:
{query}

Instructions:
1. Answer the specific question based ONLY on this document
2. Be specific and cite relevant sections or passages
3. If the document doesn't contain information to answer this question, state that clearly
4. Focus on providing a detailed, accurate answer based on the full document context

Your answer:"""
                
                messages = [
                    SystemMessage(content="You are a document analyst. Answer questions based on the provided document content."),
                    SystemMessage(content=datetime_context),
                    HumanMessage(content=prompt)
                ]
                
                # Include conversation history if available
                conversation_messages = state.get("messages", [])
                if conversation_messages:
                    messages = messages[:-1] + conversation_messages + messages[-1:]
                
                response = await llm.ainvoke(messages)
                
                return {
                    "document_id": doc_id,
                    "query": query,
                    "answer": response.content,
                    "success": True
                }
                
            except Exception as e:
                logger.error(f"Error querying document {doc_id} with query '{query}': {e}")
                return {
                    "document_id": doc_id,
                    "query": query,
                    "answer": "",
                    "success": False,
                    "error": str(e)
                }
        
        # Create all query tasks
        query_tasks = []
        for doc_id, doc_content in full_documents.items():
            for query in analysis_queries:
                query_tasks.append(query_document(doc_content, doc_id, query))
        
        logger.info(f"Executing {len(query_tasks)} parallel query tasks")
        
        # Run all queries in parallel
        results = await asyncio.gather(*query_tasks, return_exceptions=True)
        
        # Process results
        query_results = []
        parallel_query_errors = []
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Query task exception: {result}")
                parallel_query_errors.append({"error": str(result)})
                continue
            
            if result.get("success"):
                query_results.append(result)
            else:
                parallel_query_errors.append(result)
        
        logger.info(f"Completed {len(query_results)}/{len(query_tasks)} parallel queries successfully")
        
        return {
            "query_results": query_results,
            "parallel_query_errors": parallel_query_errors
        }
        
    except Exception as e:
        logger.error(f"Parallel document query node error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "query_results": [],
            "parallel_query_errors": [{"error": str(e)}]
        }


async def synthesize_insights_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combine all parallel query results into synthesized insights
    """
    try:
        original_query = state.get("original_query", "")
        query_results = state.get("query_results", [])
        documents_analyzed = state.get("documents_analyzed", [])
        
        if not query_results:
            logger.warning("No query results to synthesize")
            return {
                "full_doc_insights": {},
                "synthesis": "",
                "success": False
            }
        
        logger.info(f"Synthesizing insights from {len(query_results)} query results")
        
        # Format query results for synthesis
        formatted_results = []
        for result in query_results:
            doc_id = result.get("document_id", "unknown")
            query = result.get("query", "")
            answer = result.get("answer", "")
            
            formatted_results.append(f"""
QUESTION: {query}
DOCUMENT: {doc_id}
ANSWER:
{answer}
---""")
        
        results_text = "\n".join(formatted_results)
        
        # Get LLM for synthesis
        base_agent = BaseAgent("full_document_analysis_subgraph")
        llm = base_agent._get_llm(temperature=0.3, state=state)
        datetime_context = base_agent._get_datetime_context()
        
        synthesis_prompt = f"""Synthesize the following parallel query results from full document analysis into a comprehensive answer.

ORIGINAL USER QUERY: {original_query}

PARALLEL QUERY RESULTS FROM FULL DOCUMENTS:
{results_text}

Instructions:
1. Synthesize all the answers into a coherent, comprehensive response
2. Address the original user query directly
3. Integrate insights from all parallel queries
4. Maintain accuracy and cite which document(s) provided which information
5. If there are contradictions or gaps, note them
6. Provide a well-organized, detailed answer

Your synthesized answer:"""
        
        synthesis_messages = [
            SystemMessage(content="You are an expert at synthesizing information from multiple document queries into comprehensive answers."),
            SystemMessage(content=datetime_context)
        ]
        
        # Include conversation history if available
        conversation_messages = state.get("messages", [])
        if conversation_messages:
            synthesis_messages.extend(conversation_messages)
        
        synthesis_messages.append(HumanMessage(content=synthesis_prompt))
        
        response = await llm.ainvoke(synthesis_messages)
        synthesis = response.content
        
        # Build structured insights
        full_doc_insights = {
            "original_query": original_query,
            "documents_analyzed": documents_analyzed,
            "query_count": len(query_results),
            "synthesis": synthesis,
            "raw_results": query_results
        }
        
        logger.info(f"Synthesis complete: {len(synthesis)} characters")
        
        return {
            "full_doc_insights": full_doc_insights,
            "synthesis": synthesis,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Synthesize insights node error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "full_doc_insights": {},
            "synthesis": f"Error synthesizing insights: {str(e)}",
            "success": False
        }


def build_full_document_analysis_subgraph(checkpointer) -> StateGraph:
    """
    Build full document analysis subgraph
    
    This subgraph retrieves full documents and runs multiple targeted queries in parallel.
    
    Expected state inputs:
    - document_ids: List[str] - Document IDs to analyze (max 2)
    - analysis_queries: List[str] - Specific questions to ask (max 4)
    - original_query: str - Original user query
    - chunk_context: List[Dict] (optional) - Chunk results for context
    - user_id: str - User ID for access control
    - messages: List (optional) - Conversation history
    
    Returns state with:
    - full_doc_insights: Dict[str, Any] - Results from parallel queries
    - documents_analyzed: List[str] - IDs of successfully analyzed documents
    - synthesis: str - Synthesized answer from all parallel queries
    - success: bool
    """
    subgraph = StateGraph(Dict[str, Any])
    
    # Add nodes
    subgraph.add_node("retrieve_full_documents", retrieve_full_documents_node)
    subgraph.add_node("parallel_document_query", parallel_document_query_node)
    subgraph.add_node("synthesize_insights", synthesize_insights_node)
    
    # Set entry point
    subgraph.set_entry_point("retrieve_full_documents")
    
    # Linear flow: retrieve -> query -> synthesize
    subgraph.add_edge("retrieve_full_documents", "parallel_document_query")
    subgraph.add_edge("parallel_document_query", "synthesize_insights")
    subgraph.add_edge("synthesize_insights", END)
    
    return subgraph.compile(checkpointer=checkpointer)








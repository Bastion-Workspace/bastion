"""
Intelligent Document Retrieval Subgraph

Provides smart document retrieval with:
- Adaptive strategy based on document size
- Multi-chunk retrieval for large documents
- Full content for small documents
- Configurable depth and breadth
"""

import logging
from typing import Dict, Any, List, TypedDict, Literal, Optional
from langgraph.graph import StateGraph, END

from orchestrator.tools import search_documents_structured, get_document_content_tool
from orchestrator.backend_tool_client import get_backend_tool_client

logger = logging.getLogger(__name__)


class DocumentRetrievalState(TypedDict):
    """State for document retrieval subgraph"""
    # Input parameters
    query: str
    user_id: str
    retrieval_mode: Literal["fast", "comprehensive", "targeted"]
    max_results: int  # How many documents to retrieve
    small_doc_threshold: int  # Size threshold for full vs chunk retrieval
    
    # Output results
    retrieved_documents: List[Dict[str, Any]]
    formatted_context: str
    retrieval_metadata: Dict[str, Any]
    error: str


async def _vector_search_node(state: DocumentRetrievalState) -> Dict[str, Any]:
    """Perform initial vector search"""
    try:
        query = state["query"]
        user_id = state.get("user_id", "system")
        max_results = state.get("max_results", 5)
        retrieval_mode = state.get("retrieval_mode", "fast")
        
        # Perform vector search
        search_result = await search_documents_structured(
            query=query,
            limit=max_results,
            user_id=user_id
        )
        
        results = search_result.get('results', [])
        
        # Filter by relevance score - mode-dependent thresholds
        relevance_thresholds = {
            "fast": 0.3,         # More permissive for chat/quick queries
            "comprehensive": 0.4,  # Balanced for research
            "targeted": 0.5      # Precise for targeted searches
        }
        threshold = relevance_thresholds.get(retrieval_mode, 0.4)
        
        relevant_results = [r for r in results if r.get('relevance_score', 0.0) >= threshold]
        
        logger.info(f"📊 Vector search found {len(relevant_results)} relevant docs (threshold: {threshold}, total: {len(results)})")
        
        return {
            "retrieved_documents": relevant_results,
            "retrieval_metadata": {
                "vector_search_count": len(relevant_results),
                "total_candidates": len(results)
            }
        }
        
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return {
            "error": str(e),
            "retrieved_documents": []
        }


async def _intelligent_content_retrieval_node(state: DocumentRetrievalState) -> Dict[str, Any]:
    """Intelligently retrieve content based on document size"""
    try:
        retrieved_documents = state.get("retrieved_documents", [])
        user_id = state.get("user_id", "system")
        small_doc_threshold = state.get("small_doc_threshold", 5000)
        retrieval_mode = state.get("retrieval_mode", "fast")
        
        # Adjust limits based on mode
        mode_configs = {
            "fast": {"max_docs": 3, "max_chunks": 3},
            "comprehensive": {"max_docs": 10, "max_chunks": 5},
            "targeted": {"max_docs": 1, "max_chunks": 10}
        }
        config = mode_configs.get(retrieval_mode, mode_configs["fast"])
        
        client = await get_backend_tool_client()
        enriched_documents = []
        
        for doc in retrieved_documents[:config["max_docs"]]:
            # Handle both flat structure and nested document structure
            if 'document_id' in doc:
                doc_id = doc['document_id']
                title = doc.get('title', 'Unknown')
            elif 'document' in doc and isinstance(doc['document'], dict):
                doc_id = doc['document'].get('document_id')
                title = doc['document'].get('title', 'Unknown')
            else:
                doc_id = None
                title = doc.get('title', 'Unknown')
            
            # Skip documents without valid IDs
            if not doc_id:
                logger.warning(f"📄 Skipping document without ID: {title}")
                continue
            
            # Get document size
            doc_size = await client.get_document_size(doc_id, user_id)
            
            enriched_doc = {
                **doc,
                "size": doc_size,
                "retrieval_strategy": None,
                "full_content": None,
                "chunks": None
            }
            
            if doc_size == 0:
                # Fallback to preview
                enriched_doc["retrieval_strategy"] = "preview_fallback"
                enriched_doc["full_content"] = doc.get('content_preview', '')[:1500]
                doc_id_short = doc_id[:8] if doc_id else "Unknown"
                logger.info(f"📄 Doc {doc_id_short}: size check failed, using preview")
                
            elif doc_size < small_doc_threshold:
                # SMALL DOC: Get full content
                full_content = await get_document_content_tool(doc_id, user_id)
                doc_id_short = doc_id[:8] if doc_id else "Unknown"
                if full_content and not full_content.startswith("Error") and not full_content.startswith("Document not found"):
                    enriched_doc["retrieval_strategy"] = "full_document"
                    enriched_doc["full_content"] = full_content
                    logger.info(f"📄 Doc {doc_id_short}: {doc_size} chars, using full document")
                else:
                    # Fallback to preview
                    enriched_doc["retrieval_strategy"] = "preview_fallback"
                    enriched_doc["full_content"] = doc.get('content_preview', '')[:1500]
                    logger.info(f"📄 Doc {doc_id_short}: content fetch failed, using preview")
                
            else:
                # LARGE DOC: Get multiple chunks
                chunks = await client.get_document_chunks(doc_id, user_id, limit=config["max_chunks"])
                doc_id_short = doc_id[:8] if doc_id else "Unknown"
                if chunks:
                    enriched_doc["retrieval_strategy"] = "multi_chunk"
                    enriched_doc["chunks"] = chunks
                    logger.info(f"📄 Doc {doc_id_short}: {doc_size} chars, using {len(chunks)} chunks")
                else:
                    # Fallback to expanded preview
                    enriched_doc["retrieval_strategy"] = "preview_fallback"
                    enriched_doc["full_content"] = doc.get('content_preview', '')[:1500]
                    logger.info(f"📄 Doc {doc_id_short}: chunk fetch failed, using preview")
            
            enriched_documents.append(enriched_doc)
        
        return {
            "retrieved_documents": enriched_documents,
            "retrieval_metadata": {
                **state.get("retrieval_metadata", {}),
                "content_retrieval_complete": True,
                "documents_processed": len(enriched_documents)
            }
        }
        
    except Exception as e:
        logger.error(f"Content retrieval failed: {e}")
        return {"error": str(e)}


async def _format_context_node(state: DocumentRetrievalState) -> Dict[str, Any]:
    """Format retrieved documents into context for LLM"""
    try:
        retrieved_documents = state.get("retrieved_documents", [])
        
        if not retrieved_documents:
            return {
                "formatted_context": "",
                "retrieval_metadata": {
                    **state.get("retrieval_metadata", {}),
                    "context_formatted": True,
                    "has_content": False
                }
            }
        
        context_parts = []
        context_parts.append("=== RELEVANT LOCAL INFORMATION ===\n")
        
        for i, doc in enumerate(retrieved_documents, 1):
            # Handle both flat structure and nested document structure
            if 'title' in doc:
                title = doc.get('title', 'Unknown')
            elif 'document' in doc and isinstance(doc['document'], dict):
                title = doc['document'].get('title', 'Unknown')
            else:
                title = 'Unknown'

            score = doc.get('similarity_score', doc.get('relevance_score', 0.0))
            strategy = doc.get('retrieval_strategy', 'unknown')
            
            context_parts.append(f"\n{i}. {title} (relevance: {score:.2f})")
            
            if strategy == "full_document" and doc.get('full_content'):
                context_parts.append(f"   [Full Document Content]\n")
                # Preserve document structure - don't indent content
                context_parts.append(doc['full_content'])
                context_parts.append("")  # Empty line separator
                
            elif strategy == "multi_chunk" and doc.get('chunks'):
                context_parts.append(f"   [Multiple Relevant Sections]\n")
                for j, chunk in enumerate(doc['chunks'], 1):
                    chunk_content = chunk.get('content', '')
                    context_parts.append(f"--- Section {j} ---")
                    context_parts.append(chunk_content)  # Full chunk, no truncation
                    context_parts.append("")  # Empty line separator
                    
            else:
                # Fallback to preview
                preview = doc.get('content_preview', '') or doc.get('full_content', '')
                context_parts.append(f"   {preview[:1000]}...\n")
        
        context_parts.append("\nUse this information to answer the user's question if relevant.\n")
        
        formatted_context = "\n".join(context_parts)
        
        return {
            "formatted_context": formatted_context,
            "retrieval_metadata": {
                **state.get("retrieval_metadata", {}),
                "context_formatted": True,
                "has_content": True,
                "total_context_size": len(formatted_context)
            }
        }
        
    except Exception as e:
        logger.error(f"Context formatting failed: {e}")
        return {"error": str(e), "formatted_context": ""}


def build_intelligent_document_retrieval_subgraph() -> StateGraph:
    """Build the intelligent document retrieval subgraph"""
    
    workflow = StateGraph(DocumentRetrievalState)
    
    # Add nodes
    workflow.add_node("vector_search", _vector_search_node)
    workflow.add_node("intelligent_content_retrieval", _intelligent_content_retrieval_node)
    workflow.add_node("format_context", _format_context_node)
    
    # Define flow
    workflow.set_entry_point("vector_search")
    workflow.add_edge("vector_search", "intelligent_content_retrieval")
    workflow.add_edge("intelligent_content_retrieval", "format_context")
    workflow.add_edge("format_context", END)
    
    return workflow.compile()


# Convenience function for agents to use
async def retrieve_documents_intelligently(
    query: str,
    user_id: str = "system",
    mode: Literal["fast", "comprehensive", "targeted"] = "fast",
    max_results: int = 5,
    small_doc_threshold: int = 5000
) -> Dict[str, Any]:
    """
    Convenience function to retrieve documents with intelligent strategy
    
    Args:
        query: Search query
        user_id: User ID for access control
        mode: Retrieval mode (fast/comprehensive/targeted)
        max_results: Maximum documents to retrieve
        small_doc_threshold: Size threshold for full vs chunk retrieval
        
    Returns:
        Dict with formatted_context, retrieved_documents, and metadata
    """
    subgraph = build_intelligent_document_retrieval_subgraph()
    
    initial_state: DocumentRetrievalState = {
        "query": query,
        "user_id": user_id,
        "retrieval_mode": mode,
        "max_results": max_results,
        "small_doc_threshold": small_doc_threshold,
        "retrieved_documents": [],
        "formatted_context": "",
        "retrieval_metadata": {},
        "error": ""
    }
    
    result = await subgraph.ainvoke(initial_state)
    
    return {
        "formatted_context": result.get("formatted_context", ""),
        "retrieved_documents": result.get("retrieved_documents", []),
        "metadata": result.get("retrieval_metadata", {}),
        "success": not bool(result.get("error")),
        "error": result.get("error", "")
    }


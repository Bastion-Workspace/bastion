"""
Intelligent Document Retrieval Subgraph

Provides smart document retrieval with:
- Adaptive strategy based on document size
- Multi-chunk retrieval for large documents
- Full content for small documents
- Configurable depth and breadth
"""

import logging
import json
import re
from typing import Dict, Any, List, TypedDict, Literal, Optional
from langgraph.graph import StateGraph, END

from orchestrator.tools import search_documents_structured, get_document_content_tool
from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.agents.base_agent import BaseAgent
from langchain_core.messages import SystemMessage, HumanMessage

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
    """Perform initial vector search with recency boosting"""
    try:
        query = state["query"]
        user_id = state.get("user_id", "system")
        max_results = state.get("max_results", 5)
        retrieval_mode = state.get("retrieval_mode", "fast")
        
        # Request more candidates to ensure we don't miss relevant documents
        # Use 2-3x the requested results to get a broader pool for ranking
        candidate_multiplier = {
            "fast": 3,           # Request 3x for fast mode (e.g., 3 -> 9 candidates)
            "comprehensive": 4,  # Request 4x for comprehensive (e.g., 10 -> 40 candidates)
            "targeted": 2       # Request 2x for targeted (e.g., 1 -> 2 candidates)
        }
        multiplier = candidate_multiplier.get(retrieval_mode, 3)
        candidate_limit = max(max_results * multiplier, 10)  # At least 10 candidates
        
        # Perform vector search with expanded candidate pool
        search_result = await search_documents_structured(
            query=query,
            limit=candidate_limit,
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
        
        # Filter by threshold first
        thresholded_results = [r for r in results if r.get('relevance_score', 0.0) >= threshold]
        
        # Apply recency boosting to prioritize newer documents
        # Extract dates and boost scores for recent documents
        from datetime import datetime, timezone
        
        boosted_results = []
        for result in thresholded_results:
            # Get document metadata to find creation/update date
            # Check both nested 'document' structure and top-level fields
            doc_metadata = result.get('document', {}) if isinstance(result.get('document'), dict) else {}
            if not doc_metadata:
                doc_metadata = result
            
            # Try to get date from various fields (upload_date, created_at, updated_at, etc.)
            doc_date = None
            for date_field in ['upload_date', 'created_at', 'updated_at', 'date', 'timestamp', 'publication_date']:
                date_value = doc_metadata.get(date_field) or result.get(date_field)
                if date_value:
                    try:
                        if isinstance(date_value, str):
                            # Try parsing ISO format (handle both with and without timezone)
                            if 'T' in date_value:
                                # ISO datetime format
                                date_value_clean = date_value.replace('Z', '+00:00')
                                if '+' in date_value_clean or date_value_clean.endswith('Z'):
                                    doc_date = datetime.fromisoformat(date_value_clean)
                                else:
                                    doc_date = datetime.fromisoformat(date_value_clean)
                                    doc_date = doc_date.replace(tzinfo=timezone.utc)
                            else:
                                # Date only format
                                from datetime import date
                                date_obj = datetime.fromisoformat(date_value).date()
                                doc_date = datetime.combine(date_obj, datetime.min.time())
                                doc_date = doc_date.replace(tzinfo=timezone.utc)
                        elif isinstance(date_value, datetime):
                            doc_date = date_value
                            if not doc_date.tzinfo:
                                doc_date = doc_date.replace(tzinfo=timezone.utc)
                        elif hasattr(date_value, 'isoformat'):
                            # Date object
                            doc_date = datetime.combine(date_value, datetime.min.time())
                            doc_date = doc_date.replace(tzinfo=timezone.utc)
                        if doc_date:
                            break
                    except (ValueError, AttributeError, TypeError) as e:
                        logger.debug(f"Could not parse date field {date_field}: {e}")
                        continue
            
            # Calculate recency boost (boost newer documents)
            recency_boost = 0.0
            if doc_date:
                try:
                    # Calculate days since document was created/updated
                    now = datetime.now(timezone.utc) if doc_date.tzinfo else datetime.utcnow()
                    if doc_date.tzinfo:
                        now = now.replace(tzinfo=timezone.utc)
                    days_ago = (now - doc_date).days
                    
                    # Boost documents from last 30 days (decay over time)
                    if days_ago <= 30:
                        # Linear boost: 0.1 boost for today, decreasing to 0 for 30 days ago
                        recency_boost = 0.1 * (1.0 - (days_ago / 30.0))
                except Exception:
                    pass  # If date calculation fails, skip recency boost
            
            # Apply boost to relevance score
            original_score = result.get('relevance_score', 0.0)
            boosted_score = min(original_score + recency_boost, 1.0)  # Cap at 1.0
            
            # Store boosted result
            boosted_result = {
                **result,
                'relevance_score': boosted_score,
                'original_score': original_score,
                'recency_boost': recency_boost
            }
            boosted_results.append(boosted_result)
        
        # Sort by boosted score (descending) and take top max_results
        boosted_results.sort(key=lambda x: x.get('relevance_score', 0.0), reverse=True)
        final_results = boosted_results[:max_results]
        
        # Log boosting info
        if boosted_results and any(r.get('recency_boost', 0) > 0 for r in final_results):
            boosted_count = sum(1 for r in final_results if r.get('recency_boost', 0) > 0)
            logger.info(f"📊 Applied recency boosting: {boosted_count}/{len(final_results)} documents boosted")
        
        logger.info(f"📊 Vector search found {len(final_results)} relevant docs (threshold: {threshold}, candidates: {len(thresholded_results)}, requested: {max_results})")
        
        return {
            "retrieved_documents": final_results,
            "retrieval_metadata": {
                "vector_search_count": len(final_results),
                "total_candidates": len(results),
                "thresholded_candidates": len(thresholded_results),
                "recency_boosted": sum(1 for r in final_results if r.get('recency_boost', 0) > 0)
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


async def _check_sufficiency_and_retrieve_full_node(state: DocumentRetrievalState) -> Dict[str, Any]:
    """Check if chunked documents need full retrieval based on query analysis"""
    try:
        query = state.get("query", "")
        retrieved_documents = state.get("retrieved_documents", [])
        user_id = state.get("user_id", "system")
        
        # Find documents that were chunked (not full document)
        chunked_documents = []
        for doc in retrieved_documents:
            strategy = doc.get("retrieval_strategy", "")
            if strategy == "multi_chunk":
                # Extract doc_id
                if 'document_id' in doc:
                    doc_id = doc['document_id']
                elif 'document' in doc and isinstance(doc['document'], dict):
                    doc_id = doc['document'].get('document_id')
                else:
                    doc_id = None
                
                if doc_id:
                    title = doc.get('title', 'Unknown')
                    chunks = doc.get('chunks', [])
                    doc_size = doc.get('size', 0)
                    chunked_documents.append({
                        "doc_id": doc_id,
                        "title": title,
                        "chunks": chunks,
                        "size": doc_size,
                        "doc_index": retrieved_documents.index(doc)
                    })
        
        # If no chunked documents, skip this step
        if not chunked_documents:
            logger.info("📊 No chunked documents to check for full retrieval")
            return {
                "retrieved_documents": retrieved_documents,
                "retrieval_metadata": {
                    **state.get("retrieval_metadata", {}),
                    "sufficiency_check_complete": True,
                    "full_retrievals_requested": 0
                }
            }
        
        # Use LLM to determine if full document retrieval is needed
        # Build assessment prompt with query and chunk summaries
        chunk_summaries = []
        for chunked_doc in chunked_documents:
            chunk_texts = [chunk.get('content', '')[:500] for chunk in chunked_doc['chunks'][:3]]  # First 3 chunks, first 500 chars each
            chunk_summaries.append({
                "title": chunked_doc['title'],
                "size": chunked_doc['size'],
                "chunk_count": len(chunked_doc['chunks']),
                "sample_content": "\n\n".join(chunk_texts)
            })
        
        assessment_prompt = f"""Analyze whether the retrieved document chunks are sufficient to answer the user's query, or if the full document is needed.

USER QUERY: {query}

RETRIEVED CHUNKS:
"""
        for i, summary in enumerate(chunk_summaries, 1):
            assessment_prompt += f"""
Document {i}: {summary['title']}
- Document size: {summary['size']:,} characters
- Chunks retrieved: {summary['chunk_count']}
- Sample content from chunks:
{summary['sample_content']}
"""
        
        assessment_prompt += """
Determine if the full document is needed to answer the query. Consider:
1. Does the query ask for comprehensive information that might span the entire document?
2. Does the query ask about specific details that might not be in the retrieved chunks?
3. Does the query ask about the document's structure, organization, or complete content?
4. Are the chunks clearly incomplete or cut off?

STRUCTURED OUTPUT REQUIRED - Respond with ONLY valid JSON matching this exact schema:
{
    "documents_needing_full": [
        {
            "document_index": number (1-based index from the list above),
            "needs_full": boolean,
            "reasoning": "brief explanation"
        }
    ]
}

Return an entry for each document. Set needs_full=true if the full document should be retrieved."""
        
        # Use BaseAgent for LLM access
        base_agent = BaseAgent("sufficiency_checker")
        # Create a minimal state dict for LLM access
        minimal_state = {"metadata": {}, "user_id": user_id}
        llm = base_agent._get_llm(temperature=0.3, state=minimal_state)
        
        messages = [
            SystemMessage(content="You are a document retrieval analyst. Analyze whether retrieved chunks are sufficient or if full documents are needed. Return ONLY valid JSON."),
            HumanMessage(content=assessment_prompt)
        ]
        
        try:
            response = await llm.ainvoke(messages)
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON response
            json_text = response_content.strip()
            if '```json' in json_text:
                match = re.search(r'```json\s*\n(.*?)\n```', json_text, re.DOTALL)
                if match:
                    json_text = match.group(1).strip()
            elif '```' in json_text:
                match = re.search(r'```\s*\n(.*?)\n```', json_text, re.DOTALL)
                if match:
                    json_text = match.group(1).strip()
            
            assessment_result = json.loads(json_text)
            documents_needing_full = assessment_result.get("documents_needing_full", [])
            
            # Retrieve full documents for those that need it
            full_retrievals = 0
            for assessment in documents_needing_full:
                doc_index = assessment.get("document_index", 0) - 1  # Convert to 0-based
                needs_full = assessment.get("needs_full", False)
                
                if needs_full and 0 <= doc_index < len(chunked_documents):
                    chunked_doc = chunked_documents[doc_index]
                    doc_id = chunked_doc["doc_id"]
                    doc_title = chunked_doc["title"]
                    doc_index_in_retrieved = chunked_doc["doc_index"]
                    
                    logger.info(f"📄 Retrieving full document for {doc_title[:50]} (LLM determined chunks insufficient)")
                    
                    # Retrieve full document
                    full_content = await get_document_content_tool(doc_id, user_id)
                    
                    if full_content and not full_content.startswith("Error") and not full_content.startswith("Document not found"):
                        # Update the document in retrieved_documents
                        retrieved_documents[doc_index_in_retrieved]["retrieval_strategy"] = "full_document"
                        retrieved_documents[doc_index_in_retrieved]["full_content"] = full_content
                        retrieved_documents[doc_index_in_retrieved]["chunks"] = None  # Clear chunks
                        full_retrievals += 1
                        logger.info(f"✅ Retrieved full document for {doc_title[:50]} ({len(full_content):,} chars)")
                    else:
                        logger.warning(f"⚠️ Failed to retrieve full document for {doc_title[:50]}, keeping chunks")
            
            logger.info(f"📊 Sufficiency check complete: {full_retrievals} full document(s) retrieved")
            
            return {
                "retrieved_documents": retrieved_documents,
                "retrieval_metadata": {
                    **state.get("retrieval_metadata", {}),
                    "sufficiency_check_complete": True,
                    "full_retrievals_requested": full_retrievals
                }
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Sufficiency check failed: {e} - continuing with chunks")
            # On error, continue with existing chunks
            return {
                "retrieved_documents": retrieved_documents,
                "retrieval_metadata": {
                    **state.get("retrieval_metadata", {}),
                    "sufficiency_check_complete": True,
                    "full_retrievals_requested": 0,
                    "sufficiency_check_error": str(e)
                }
            }
        
    except Exception as e:
        logger.error(f"Sufficiency check node failed: {e}")
        # On error, return documents as-is
        return {
            "retrieved_documents": state.get("retrieved_documents", []),
            "retrieval_metadata": {
                **state.get("retrieval_metadata", {}),
                "sufficiency_check_complete": True,
                "full_retrievals_requested": 0,
                "error": str(e)
            }
        }


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
    workflow.add_node("check_sufficiency", _check_sufficiency_and_retrieve_full_node)
    workflow.add_node("format_context", _format_context_node)
    
    # Define flow
    workflow.set_entry_point("vector_search")
    workflow.add_edge("vector_search", "intelligent_content_retrieval")
    workflow.add_edge("intelligent_content_retrieval", "check_sufficiency")
    workflow.add_edge("check_sufficiency", "format_context")
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


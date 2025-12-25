"""
Entity Relationship Subgraph
Discovers documents through Neo4j knowledge graph entity relationships
"""

import logging
import asyncio
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.tools import get_document_content_tool
from orchestrator.utils.entity_extraction import (
    extract_entities_from_text,
    extract_entities_from_search_results
)

logger = logging.getLogger(__name__)


# ===== Node Functions =====

async def extract_entities_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract entities from query and initial vector results
    
    CRITICAL: Preserves all 5 critical state keys
    """
    try:
        query = state.get("query", "")
        vector_results = state.get("vector_results", [])
        user_id = state.get("user_id", "system")
        
        logger.info(f"Extracting entities from query and {len(vector_results)} vector results")
        
        # Extract from query
        client = await get_backend_tool_client()
        query_entities = await extract_entities_from_text(query, client)
        
        # Extract from vector results
        result_entity_names = extract_entities_from_search_results(vector_results, max_results=3)
        
        # Combine unique entity names
        all_entity_names = list(set(
            [e["name"] for e in query_entities] + result_entity_names
        ))
        
        logger.info(f"Extracted {len(all_entity_names)} unique entities: {all_entity_names[:5]}...")
        
        return {
            "extracted_entities": all_entity_names,
            "entity_count": len(all_entity_names),
            # CRITICAL 5 - ALWAYS preserve
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }
        
    except Exception as e:
        logger.error(f"Entity extraction node failed: {e}")
        return {
            "extracted_entities": [],
            "entity_count": 0,
            "error": str(e),
            # CRITICAL 5 - preserve even on error
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }


async def query_knowledge_graph_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Query Neo4j for entity-related documents and co-occurring entities
    
    CRITICAL: Preserves all 5 critical state keys + user_id for RLS
    """
    try:
        entities = state.get("extracted_entities", [])
        user_id = state.get("user_id", "system")  # CRITICAL for RLS
        
        if not entities:
            logger.info("No entities to query - skipping knowledge graph")
            return {
                "kg_document_ids": [],
                "related_document_ids": [],
                "co_occurring_entities": [],
                "kg_total_docs": 0,
                # CRITICAL 5
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
            }
        
        logger.info(f"Querying knowledge graph for {len(entities)} entities with user_id={user_id}")
        
        client = await get_backend_tool_client()
        
        # Run queries in parallel
        direct_docs_task = client.find_documents_by_entities(
            entity_names=entities[:10],  # Limit to top 10 entities
            user_id=user_id  # RLS filtering
        )
        
        related_docs_task = client.find_related_documents_by_entities(
            entity_names=entities[:5],  # Top 5 for relationship traversal
            max_hops=1,
            user_id=user_id  # RLS filtering
        )
        
        co_occurring_task = client.find_co_occurring_entities(
            entity_names=entities[:5],
            min_co_occurrences=2,
            user_id=user_id
        )
        
        # Execute in parallel
        direct_docs, related_docs, co_occurring = await asyncio.gather(
            direct_docs_task,
            related_docs_task,
            co_occurring_task,
            return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(direct_docs, Exception):
            logger.error(f"Direct docs query failed: {direct_docs}")
            direct_docs = []
        if isinstance(related_docs, Exception):
            logger.error(f"Related docs query failed: {related_docs}")
            related_docs = []
        if isinstance(co_occurring, Exception):
            logger.error(f"Co-occurring query failed: {co_occurring}")
            co_occurring = []
        
        # Deduplicate document IDs
        all_doc_ids = list(set(direct_docs + related_docs))
        
        logger.info(f"Knowledge graph found: {len(direct_docs)} direct, {len(related_docs)} related, {len(co_occurring)} co-occurring entities")
        logger.info(f"Total unique documents: {len(all_doc_ids)}")
        
        return {
            "kg_document_ids": all_doc_ids,
            "direct_document_count": len(direct_docs),
            "related_document_count": len(related_docs),
            "co_occurring_entities": co_occurring,
            "kg_total_docs": len(all_doc_ids),
            # CRITICAL 5
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }
        
    except Exception as e:
        logger.error(f"Knowledge graph query node failed: {e}")
        return {
            "kg_document_ids": [],
            "related_document_ids": [],
            "co_occurring_entities": [],
            "kg_total_docs": 0,
            "error": str(e),
            # CRITICAL 5
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }


async def retrieve_documents_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieve document content for knowledge graph discovered documents
    
    CRITICAL: Preserves all 5 critical state keys + uses user_id for RLS
    """
    try:
        doc_ids = state.get("kg_document_ids", [])
        user_id = state.get("user_id", "system")  # CRITICAL for RLS
        max_docs = 10
        
        if not doc_ids:
            logger.info("No documents to retrieve from knowledge graph")
            return {
                "kg_documents": [],
                "kg_document_count": 0,
                # CRITICAL 5
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
            }
        
        logger.info(f"Retrieving {min(len(doc_ids), max_docs)} documents (RLS filtered for user_id={user_id})")
        
        # Retrieve documents in parallel (top N)
        retrieve_tasks = []
        for doc_id in doc_ids[:max_docs]:
            task = get_document_content_tool(doc_id, user_id)  # RLS enforced here
            retrieve_tasks.append(task)
        
        contents = await asyncio.gather(*retrieve_tasks, return_exceptions=True)
        
        # Format results
        documents = []
        for i, (doc_id, content) in enumerate(zip(doc_ids[:max_docs], contents)):
            if isinstance(content, Exception):
                logger.warning(f"Failed to retrieve doc {doc_id}: {content}")
                continue
            
            if content and not content.startswith("Error"):
                documents.append({
                    "document_id": doc_id,
                    "content": content[:5000],  # Limit to 5000 chars per doc
                    "source": "knowledge_graph"
                })
        
        logger.info(f"Successfully retrieved {len(documents)} documents via knowledge graph")
        
        return {
            "kg_documents": documents,
            "kg_document_count": len(documents),
            # CRITICAL 5
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }
        
    except Exception as e:
        logger.error(f"Document retrieval node failed: {e}")
        return {
            "kg_documents": [],
            "kg_document_count": 0,
            "error": str(e),
            # CRITICAL 5
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }


async def format_results_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format knowledge graph results for synthesis
    
    CRITICAL: Preserves all 5 critical state keys
    """
    try:
        documents = state.get("kg_documents", [])
        co_occurring = state.get("co_occurring_entities", [])
        extracted_entities = state.get("extracted_entities", [])
        
        if not documents:
            logger.info("No knowledge graph documents to format")
            return {
                "kg_formatted_results": "",
                "kg_success": False,
                # CRITICAL 5
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
            }
        
        # Format as markdown
        formatted_parts = []
        
        # Entity context header
        formatted_parts.append("# Knowledge Graph Results\n")
        formatted_parts.append(f"**Entities Discovered**: {', '.join(extracted_entities[:10])}\n")
        
        if co_occurring:
            related_names = [e["name"] for e in co_occurring[:5]]
            formatted_parts.append(f"**Related Entities**: {', '.join(related_names)}\n")
        
        formatted_parts.append(f"\n**Documents Found via Entity Relationships**: {len(documents)}\n\n")
        
        # Document contents
        for i, doc in enumerate(documents, 1):
            formatted_parts.append(f"## Document {i} (via Knowledge Graph)\n")
            formatted_parts.append(f"**Document ID**: {doc['document_id']}\n\n")
            formatted_parts.append(doc["content"][:3000])  # Limit per doc
            formatted_parts.append("\n\n---\n\n")
        
        formatted_results = "".join(formatted_parts)
        
        logger.info(f"Formatted {len(documents)} knowledge graph documents")
        
        return {
            "kg_formatted_results": formatted_results,
            "kg_success": True,
            # CRITICAL 5
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }
        
    except Exception as e:
        logger.error(f"Format results node failed: {e}")
        return {
            "kg_formatted_results": "",
            "kg_success": False,
            "error": str(e),
            # CRITICAL 5
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }


# ===== Subgraph Builder =====

def build_entity_relationship_subgraph(checkpointer=None) -> StateGraph:
    """
    Build entity relationship discovery subgraph
    
    Flow:
    1. Extract entities from query + vector results
    2. Query knowledge graph for related documents
    3. Retrieve document content (RLS filtered)
    4. Format results with entity context
    
    Args:
        checkpointer: Optional LangGraph checkpointer for state persistence
    
    Returns:
        Compiled StateGraph with checkpointer (if provided)
    """
    subgraph = StateGraph(Dict[str, Any])
    
    # Add nodes
    subgraph.add_node("extract_entities", extract_entities_node)
    subgraph.add_node("query_knowledge_graph", query_knowledge_graph_node)
    subgraph.add_node("retrieve_documents", retrieve_documents_node)
    subgraph.add_node("format_results", format_results_node)
    
    # Entry point
    subgraph.set_entry_point("extract_entities")
    
    # Linear flow
    subgraph.add_edge("extract_entities", "query_knowledge_graph")
    subgraph.add_edge("query_knowledge_graph", "retrieve_documents")
    subgraph.add_edge("retrieve_documents", "format_results")
    subgraph.add_edge("format_results", END)
    
    if checkpointer:
        return subgraph.compile(checkpointer=checkpointer)
    else:
        return subgraph.compile()


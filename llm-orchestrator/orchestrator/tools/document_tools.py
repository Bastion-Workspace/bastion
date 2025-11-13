"""
Document Tools - LangGraph tools using backend gRPC service
"""

import logging
from typing import Optional, List, Dict, Any

from orchestrator.backend_tool_client import get_backend_tool_client

logger = logging.getLogger(__name__)


async def search_documents_tool(
    query: str,
    limit: int = 5,
    user_id: str = "system"
) -> str:
    """
    Search documents in the knowledge base
    
    Args:
        query: Search query
        limit: Maximum number of results (default: 5)
        user_id: User ID for access control
        
    Returns:
        Formatted search results as string
    """
    try:
        logger.info(f"Searching documents: query='{query[:100]}', limit={limit}")
        
        # Get backend client
        client = await get_backend_tool_client()
        
        # Perform search via gRPC
        result = await client.search_documents(
            query=query,
            user_id=user_id,
            limit=limit
        )
        
        # Format results
        if 'error' in result:
            return f"Error searching documents: {result['error']}"
        
        if result['total_count'] == 0:
            return "No documents found matching your query."
        
        # Build formatted response
        response_parts = [f"Found {result['total_count']} document(s):\n"]
        
        for i, doc in enumerate(result['results'][:limit], 1):
            response_parts.append(f"\n{i}. **{doc['title']}** (ID: {doc['document_id']})")
            response_parts.append(f"   File: {doc['filename']}")
            response_parts.append(f"   Relevance: {doc['relevance_score']:.3f}")
            
            if doc['content_preview']:
                preview = doc['content_preview'][:200].replace('\n', ' ')
                response_parts.append(f"   Preview: {preview}...")
        
        logger.info(f"Search completed: {result['total_count']} results")
        return '\n'.join(response_parts)
        
    except Exception as e:
        logger.error(f"Document search tool error: {e}")
        return f"Error searching documents: {str(e)}"


async def get_document_content_tool(
    document_id: str,
    user_id: str = "system"
) -> str:
    """
    Get full content of a document
    
    Args:
        document_id: Document ID
        user_id: User ID for access control
        
    Returns:
        Document content or error message
    """
    try:
        logger.info(f"Getting document content: {document_id}")
        
        # Get backend client
        client = await get_backend_tool_client()
        
        # Get document content via gRPC
        content = await client.get_document_content(
            document_id=document_id,
            user_id=user_id
        )
        
        if content is None:
            return f"Document not found: {document_id}"
        
        logger.info(f"Retrieved document content: {len(content)} characters")
        return content
        
    except Exception as e:
        logger.error(f"Get document content tool error: {e}")
        return f"Error getting document content: {str(e)}"


# Tool registry for LangGraph
DOCUMENT_TOOLS = {
    'search_documents': search_documents_tool,
    'get_document_content': get_document_content_tool
}


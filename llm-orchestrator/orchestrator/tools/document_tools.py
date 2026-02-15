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


async def search_by_tags_tool(
    tags: List[str],
    categories: List[str] = None,
    query: str = "",
    limit: int = 20,
    user_id: str = "system"
) -> str:
    """
    Search documents by tags and/or categories (metadata search, not vector search)
    
    Args:
        tags: List of tags to filter by (e.g., ["entertainment", "movie"])
        categories: List of categories to filter by (e.g., ["entertainment"])
        query: Optional search query for additional filtering
        limit: Maximum number of results (default: 20)
        user_id: User ID for access control
        
    Returns:
        Formatted search results as string
    """
    try:
        logger.info(f"Searching by tags: {tags}, categories: {categories}, query: '{query[:50]}'")
        
        # Get backend client
        client = await get_backend_tool_client()
        
        # Build filters for gRPC
        filters = []
        for tag in tags:
            filters.append(f"tag:{tag}")
        if categories:
            for category in categories:
                filters.append(f"category:{category}")
        
        # Perform search via gRPC with filters
        result = await client.search_documents(
            query=query or "entertainment",  # Use generic query if none provided
            user_id=user_id,
            limit=limit,
            filters=filters
        )
        
        # Format results
        if 'error' in result:
            return f"Error searching by tags: {result['error']}"
        
        if result['total_count'] == 0:
            return f"No documents found with tags {tags} and categories {categories}."
        
        # Build formatted response
        response_parts = [f"Found {result['total_count']} document(s) with tags {tags}:\n"]
        
        for i, doc in enumerate(result['results'][:limit], 1):
            response_parts.append(f"\n{i}. **{doc['title']}** (ID: {doc['document_id']})")
            response_parts.append(f"   File: {doc['filename']}")
            
            if doc['content_preview']:
                preview = doc['content_preview'][:300].replace('\n', ' ')
                response_parts.append(f"   Preview: {preview}...")
        
        logger.info(f"Tag search completed: {result['total_count']} results")
        return '\n'.join(response_parts)
        
    except Exception as e:
        logger.error(f"Tag search tool error: {e}")
        return f"Error searching by tags: {str(e)}"


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


async def search_documents_structured(
    query: str,
    limit: int = 10,
    user_id: str = "system"
) -> Dict[str, Any]:
    """
    Search documents in the knowledge base - returns structured data
    
    Args:
        query: Search query
        limit: Maximum number of results (default: 10)
        user_id: User ID for access control
        
    Returns:
        Dict with 'results' (list of documents) and 'total_count'
    """
    try:
        logger.info(f"Searching documents (structured): query='{query[:100]}', limit={limit}")
        
        # Get backend client
        client = await get_backend_tool_client()
        
        # Perform search via gRPC
        result = await client.search_documents(
            query=query,
            user_id=user_id,
            limit=limit
        )
        
        logger.info(f"Search completed: {result.get('total_count', 0)} results")
        return result
        
    except Exception as e:
        logger.error(f"Document search tool error: {e}")
        return {
            'results': [],
            'total_count': 0,
            'error': str(e)
        }


async def find_documents_by_tags_tool(
    required_tags: List[str],
    user_id: str = "system",
    collection_type: str = "",
    limit: int = 20
) -> List[Dict[str, Any]]:
    """
    Find documents that contain ALL of the specified tags using database query

    Args:
        required_tags: List of tags that ALL must be present
        user_id: User ID for access control
        collection_type: Collection type filter ("user", "global", or empty for both)
        limit: Maximum number of results

    Returns:
        List of document dictionaries with metadata
    """
    try:
        logger.info(f"Finding documents by tags: {required_tags}, limit={limit}")

        # Get backend client
        client = await get_backend_tool_client()

        # Call the new gRPC method
        from protos import tool_service_pb2
        request = tool_service_pb2.FindDocumentsByTagsRequest(
            user_id=user_id,
            required_tags=required_tags,
            collection_type=collection_type,
            limit=limit
        )

        response = await client.find_documents_by_tags(request)

        # response is already a list of document dictionaries
        documents = response

        logger.info(f"Found {len(documents)} documents with tags {required_tags}")
        return documents

    except Exception as e:
        logger.error(f"Find documents by tags tool error: {e}")
        return []


async def search_within_document_tool(
    document_id: str,
    query: str,
    search_type: str = "exact",
    context_window: int = 200,
    case_sensitive: bool = False,
    user_id: str = "system"
) -> Dict[str, Any]:
    """
    Search for terms/concepts within a specific document.
    
    This tool allows agents to find exact locations of text within a document
    before loading the full content, enabling efficient pre-filtering of sections.
    
    Args:
        document_id: Document ID to search within
        query: Search term(s) - can be single term or space-separated terms
        search_type: Type of search - "exact" (default), "fuzzy", or "all_terms"
            - "exact": Find exact phrase match
            - "fuzzy": Find approximate matches (case-insensitive substring)
            - "all_terms": Find sections containing all terms (AND logic)
        context_window: Number of characters of context around each match (default: 200)
        case_sensitive: Whether search should be case-sensitive (default: False)
        user_id: User ID for access control
        
    Returns:
        Dict with:
        - matches: List of match dictionaries, each containing:
            - start: Character offset of match start
            - end: Character offset of match end
            - match_text: The matched text
            - context_before: Text before the match (up to context_window chars)
            - context_after: Text after the match (up to context_window chars)
            - section_name: Name of section containing the match (if found)
            - section_start: Character offset of section start
            - section_end: Character offset of section end
        - total_matches: Total number of matches found
        - document_id: Document ID that was searched
        - search_query: The query that was searched
    """
    import re
    
    try:
        logger.info(f"Searching within document {document_id}: query='{query[:100]}', type={search_type}")
        
        # Get document content
        content = await get_document_content_tool(document_id, user_id)
        if content.startswith("Error"):
            return {
                "matches": [],
                "total_matches": 0,
                "document_id": document_id,
                "search_query": query,
                "error": content
            }
        
        matches = []
        content_lower = content.lower() if not case_sensitive else content
        query_lower = query.lower() if not case_sensitive else query
        
        # Parse search terms
        if search_type == "all_terms":
            # Split query into individual terms
            terms = query_lower.split()
            search_patterns = [re.escape(term) for term in terms if term.strip()]
        elif search_type == "fuzzy":
            # Fuzzy search: find any occurrence of any word in the query
            terms = query_lower.split()
            search_patterns = [re.escape(term) for term in terms if term.strip()]
        else:
            # Exact search: find exact phrase
            search_patterns = [re.escape(query_lower)]
        
        # Find all sections in the document
        section_pattern = r'^(##+\s+[^\n]+)$'
        sections = []
        for section_match in re.finditer(section_pattern, content, re.MULTILINE):
            section_header = section_match.group(1)
            section_start = section_match.start()
            
            # Find section end
            section_end_match = re.search(r'\n##+\s+', content[section_start + 1:], re.MULTILINE)
            if section_end_match:
                section_end = section_start + 1 + section_end_match.start()
            else:
                section_end = len(content)
            
            section_name = re.sub(r'^##+\s+', '', section_header).strip()
            sections.append({
                "name": section_name,
                "start": section_start,
                "end": section_end
            })
        
        # Search for matches
        if search_type == "all_terms":
            # Find sections containing all terms
            for section in sections:
                section_content = content[section["start"]:section["end"]]
                section_content_lower = section_content.lower() if not case_sensitive else section_content
                
                # Check if all terms are present
                all_terms_present = all(term in section_content_lower for term in terms)
                
                if all_terms_present:
                    # Find first occurrence of each term for context
                    for term in terms:
                        term_escaped = re.escape(term)
                        for match in re.finditer(term_escaped, section_content_lower, re.IGNORECASE if not case_sensitive else 0):
                            match_start = section["start"] + match.start()
                            match_end = section["start"] + match.end()
                            
                            # Get context
                            context_start = max(0, match_start - context_window)
                            context_end = min(len(content), match_end + context_window)
                            context_before = content[context_start:match_start]
                            context_after = content[match_end:context_end]
                            match_text = content[match_start:match_end]
                            
                            matches.append({
                                "start": match_start,
                                "end": match_end,
                                "match_text": match_text,
                                "context_before": context_before,
                                "context_after": context_after,
                                "section_name": section["name"],
                                "section_start": section["start"],
                                "section_end": section["end"]
                            })
        else:
            # Find all occurrences of the search pattern(s)
            for pattern in search_patterns:
                flags = re.IGNORECASE if not case_sensitive else 0
                for match in re.finditer(pattern, content_lower, flags):
                    match_start = match.start()
                    match_end = match.end()
                    
                    # Get context
                    context_start = max(0, match_start - context_window)
                    context_end = min(len(content), match_end + context_window)
                    context_before = content[context_start:match_start]
                    context_after = content[match_end:context_end]
                    match_text = content[match_start:match_end]
                    
                    # Find which section contains this match
                    section_name = None
                    section_start = None
                    section_end = None
                    for section in sections:
                        if section["start"] <= match_start < section["end"]:
                            section_name = section["name"]
                            section_start = section["start"]
                            section_end = section["end"]
                            break
                    
                    matches.append({
                        "start": match_start,
                        "end": match_end,
                        "match_text": match_text,
                        "context_before": context_before,
                        "context_after": context_after,
                        "section_name": section_name,
                        "section_start": section_start,
                        "section_end": section_end
                    })
        
        # Remove duplicates (same start/end positions)
        seen = set()
        unique_matches = []
        for match in matches:
            key = (match["start"], match["end"])
            if key not in seen:
                seen.add(key)
                unique_matches.append(match)
        
        logger.info(f"Found {len(unique_matches)} matches in document {document_id}")
        
        return {
            "matches": unique_matches,
            "total_matches": len(unique_matches),
            "document_id": document_id,
            "search_query": query,
            "search_type": search_type
        }
        
    except Exception as e:
        logger.error(f"Search within document tool error: {e}")
        return {
            "matches": [],
            "total_matches": 0,
            "document_id": document_id,
            "search_query": query,
            "error": str(e)
        }


async def append_to_document_tool(
    document_id: str,
    content: str,
    user_id: str = "system"
) -> str:
    """
    Append content to an existing document without replacing existing content.

    Args:
        document_id: Document ID to append to
        content: Content to append
        user_id: User ID for access control

    Returns:
        Human-readable success or error message
    """
    try:
        logger.info("Appending to document: %s", document_id)
        client = await get_backend_tool_client()
        result = await client.update_document_content(
            document_id=document_id,
            content=content,
            user_id=user_id,
            append=True
        )
        if result.get("success"):
            return f"Appended {result.get('content_length', len(content))} characters to document {document_id}."
        return f"Error appending to document: {result.get('error', result.get('message', 'Unknown error'))}"
    except Exception as e:
        logger.error("Append to document tool error: %s", e)
        return f"Error appending to document: {str(e)}"


async def get_document_metadata_tool(
    document_id: str,
    user_id: str = "system"
) -> str:
    """
    Get document metadata (title, filename, content type, tags) by ID.

    Args:
        document_id: Document ID
        user_id: User ID for access control

    Returns:
        Human-readable metadata summary or error message
    """
    try:
        logger.info("Getting document metadata: %s", document_id)
        client = await get_backend_tool_client()
        doc = await client.get_document(document_id=document_id, user_id=user_id)
        if doc is None:
            return f"Document not found: {document_id}"
        parts = [
            f"Document ID: {doc.get('document_id', '')}",
            f"Title: {doc.get('title', '')}",
            f"Filename: {doc.get('filename', '')}",
            f"Content type: {doc.get('content_type', '')}",
        ]
        meta = doc.get("metadata") or {}
        if meta:
            parts.append("Metadata: " + ", ".join(f"{k}={v}" for k, v in meta.items()))
        return "\n".join(parts)
    except Exception as e:
        logger.error("Get document metadata tool error: %s", e)
        return f"Error getting document metadata: {str(e)}"


def _slugify_title(title: str) -> str:
    """Convert title to a safe filename (lowercase, spaces to underscores, strip non-alnum)."""
    import re
    s = title.strip().lower().replace(" ", "_")
    return re.sub(r"[^a-z0-9_.-]", "", s) or "document"


async def create_document_tool(
    title: str,
    content: str,
    user_id: str = "system",
    filename: str = "",
    folder_path: str = "",
    tags: Optional[List[str]] = None
) -> str:
    """
    Create a new document in the user's space. LLM-friendly wrapper around create_user_file.

    Args:
        title: Document title (used for display and, if filename empty, to generate filename)
        content: Full document content
        user_id: User ID for access control
        filename: Optional filename (default: derived from title with .md)
        folder_path: Optional folder path (e.g. "Projects/Notes"). Empty = root.
        tags: Optional list of tags

    Returns:
        Human-readable success message with document_id or error message
    """
    try:
        logger.info("Creating document: title=%s", title[:80] if title else "")
        client = await get_backend_tool_client()
        fname = filename.strip() if filename else (_slugify_title(title) + ".md")
        result = await client.create_user_file(
            filename=fname,
            content=content,
            user_id=user_id,
            folder_path=folder_path if folder_path else None,
            title=title or fname,
            tags=tags,
        )
        if result.get("success"):
            return f"Created document '{title or fname}' (ID: {result.get('document_id', '')})."
        return f"Error creating document: {result.get('error', result.get('message', 'Unknown error'))}"
    except Exception as e:
        logger.error("Create document tool error: %s", e)
        return f"Error creating document: {str(e)}"


# Tool registry for LangGraph
DOCUMENT_TOOLS = {
    'search_documents': search_documents_tool,
    'search_documents_structured': search_documents_structured,
    'search_by_tags': search_by_tags_tool,
    'find_documents_by_tags': find_documents_by_tags_tool,
    'get_document_content': get_document_content_tool,
    'search_within_document': search_within_document_tool,
    'append_to_document': append_to_document_tool,
    'get_document_metadata': get_document_metadata_tool,
    'create_document': create_document_tool,
}


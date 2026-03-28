"""
Document Tools - LangGraph tools using backend gRPC service
"""

import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.utils.action_io_registry import register_action
from orchestrator.utils.tool_type_models import DocumentRef

logger = logging.getLogger(__name__)


# ── I/O models for search_documents_tool ───────────────────────────────────

class SearchDocumentsInputs(BaseModel):
    """Required inputs for document search."""
    query: str = Field(description="Search query (natural language or keywords)")


class SearchDocumentsParams(BaseModel):
    """Optional configuration."""
    limit: int = Field(default=5, description="Max results to return")
    scope: str = Field(
        default="all",
        description="Scope: my_docs, team_docs, global_docs, or all",
    )
    folder_id: str = Field(
        default="",
        description="Restrict to a specific folder ID (from list_folders_tool)",
    )
    file_types: List[str] = Field(
        default_factory=list,
        description="Filter by extensions e.g. ['md', 'org']",
    )
    min_score: float = Field(
        default=0.0,
        description="Minimum relevance score (0.0–1.0)",
    )


class SearchDocumentsOutputs(BaseModel):
    """Typed outputs for search_documents_tool."""
    documents: List[DocumentRef] = Field(description="Matching documents")
    count: int = Field(description="Number of results found")
    query_used: str = Field(description="The query that was executed")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class GetDocumentContentInputs(BaseModel):
    """Required inputs for get_document_content_tool."""
    document_id: str = Field(description="Document ID")


class GetDocumentContentOutputs(BaseModel):
    """Typed outputs for get_document_content_tool."""
    content: str = Field(description="Full document content")
    document_id: str = Field(description="Document ID")
    word_count: int = Field(description="Word count")
    formatted: str = Field(description="Human-readable content for LLM/chat")


class SearchByTagsInputs(BaseModel):
    """Required inputs for search_by_tags_tool."""
    tags: List[str] = Field(description="List of tags to filter by")


class SearchByTagsOutputs(BaseModel):
    """Outputs for search_by_tags_tool."""
    documents: List[Dict[str, Any]] = Field(description="Matching documents")
    count: int = Field(description="Number of results")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class GetDocumentMetadataInputs(BaseModel):
    """Required inputs for get_document_metadata_tool."""
    document_id: str = Field(description="Document ID")


class GetDocumentMetadataOutputs(BaseModel):
    """Outputs for get_document_metadata_tool."""
    document_id: str = Field(description="Document ID")
    title: str = Field(description="Document title")
    filename: str = Field(description="Filename")
    content_type: str = Field(description="Content type")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class PickRandomFileInputs(BaseModel):
    """Required inputs for pick_random_file_tool."""
    folder_id: str = Field(description="Folder ID to pick a random file from (e.g. from list_folders)")


class PickRandomFileParams(BaseModel):
    """Optional parameters for pick_random_file_tool."""
    file_extension: Optional[str] = Field(default=None, description="Filter by extension, e.g. png, jpg (case-insensitive)")


class PickRandomFileOutputs(BaseModel):
    """Outputs for pick_random_file_tool."""
    found: bool = Field(description="Whether a document was found")
    document_id: str = Field(description="Document ID of the picked file")
    filename: str = Field(description="Filename")
    title: Optional[str] = Field(default=None, description="Document title if set")
    folder_id: Optional[str] = Field(default=None, description="Folder ID")
    doc_type: Optional[str] = Field(default=None, description="Document type")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class FindDocumentByPathInputs(BaseModel):
    """Required inputs for find_document_by_path_tool."""
    file_path: str = Field(
        description="Path (e.g. ./outline.md, Projects/Novel/chapter-1.md) or bare filename (e.g. monthly.org, inbox.org)"
    )


class FindDocumentByPathParams(BaseModel):
    """Optional parameters for find_document_by_path_tool."""
    base_path: str = Field(
        default="",
        description="Base directory for resolving relative paths",
    )
    load_content: bool = Field(
        default=False,
        description="If true, also return full file content in one call",
    )


class FindDocumentByPathOutputs(BaseModel):
    """Outputs for find_document_by_path_tool."""
    found: bool = Field(description="Whether a document was found at the path")
    document_id: str = Field(description="Document ID if found")
    filename: str = Field(description="Filename")
    resolved_path: str = Field(description="Resolved path used for lookup")
    content: Optional[str] = Field(default=None, description="Full file content when load_content=True")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class SearchWithinDocumentInputs(BaseModel):
    """Required inputs for search_within_document_tool."""
    document_id: str = Field(description="Document ID to search within")
    query: str = Field(description="Search term(s) - single term or space-separated terms")


class SearchWithinDocumentParams(BaseModel):
    """Optional configuration for search_within_document_tool."""
    search_type: str = Field(default="exact", description="exact, fuzzy, or all_terms")
    context_window: int = Field(default=200, description="Characters of context around each match")
    case_sensitive: bool = Field(default=False, description="Whether search is case-sensitive")


class SearchWithinDocumentOutputs(BaseModel):
    """Outputs for search_within_document_tool."""
    matches: List[Dict[str, Any]] = Field(description="List of match objects with start, end, match_text, context")
    total_matches: int = Field(description="Number of matches found")
    document_id: str = Field(description="Document ID searched")
    search_query: str = Field(description="Query that was searched")
    search_type: Optional[str] = Field(default=None, description="Search type used: exact, fuzzy, or all_terms")
    error: Optional[str] = Field(default=None, description="Error message if search failed")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


async def search_documents_tool(
    query: str,
    limit: int = 5,
    user_id: str = "system",
    scope: str = "all",
    folder_id: str = "",
    file_types: List[str] = None,
    min_score: float = 0.0,
) -> Dict[str, Any]:
    """
    Search TEXT documents (markdown, org-mode, PDFs, notes) in the knowledge base.
    NOT for images, comics, photos, artwork, or visual content — use search_images_tool for those.
    Returns structured dict with documents, count, query_used, and formatted.
    """
    try:
        logger.info(f"Searching documents: query='{query[:100]}', limit={limit}")

        filters = []
        if scope and scope != "all":
            filters.append(f"scope:{scope}")
        if folder_id and folder_id.strip():
            filters.append(f"folder_id:{folder_id.strip()}")
        for ft in file_types or []:
            if ft and str(ft).strip():
                filters.append(f"file_type:{str(ft).strip()}")
        if min_score > 0:
            filters.append(f"min_score:{min_score}")

        client = await get_backend_tool_client()
        result = await client.search_documents(
            query=query,
            user_id=user_id,
            limit=limit,
            filters=filters if filters else None,
        )

        if "error" in result:
            formatted = f"Error searching documents: {result['error']}"
            return {
                "documents": [],
                "count": 0,
                "query_used": query,
                "formatted": formatted,
            }

        raw_results = result.get("results", [])[:limit]
        total_count = result.get("total_count", 0)

        documents = []
        for doc in raw_results:
            meta = doc.get("metadata") or {}
            documents.append({
                "document_id": doc.get("document_id", ""),
                "title": doc.get("title", ""),
                "filename": doc.get("filename", ""),
                "file_type": meta.get("file_type") or meta.get("doc_type") or doc.get("file_type") or "md",
                "folder_path": meta.get("folder_path", doc.get("folder_path", "")),
                "relevance_score": float(doc.get("relevance_score", 0)),
                "content_preview": (doc.get("content_preview") or "")[:1500],
            })

        response_parts = [f"Found {total_count} document(s):\n"]
        for i, doc in enumerate(documents, 1):
            response_parts.append(f"\n{i}. **{doc['title']}** (ID: {doc['document_id']})")
            path_info = doc.get("folder_path", "").strip()
            file_type = doc.get("file_type", "md")
            if path_info:
                response_parts.append(f"   Path: {path_info} | Type: {file_type}")
            else:
                response_parts.append(f"   File: {doc['filename']} | Type: {file_type}")
            response_parts.append(f"   Relevance: {doc['relevance_score']:.3f}")
            if doc["content_preview"]:
                response_parts.append("   Preview:")
                response_parts.append("   " + doc["content_preview"].replace("\n", "\n   "))
                if len((doc.get("content_preview") or "")) >= 1500:
                    response_parts.append("   ...")
        formatted = "\n".join(response_parts)

        logger.info(f"Search completed: {total_count} results")
        return {
            "documents": documents,
            "count": len(documents),
            "query_used": query,
            "formatted": formatted,
        }

    except Exception as e:
        logger.error(f"Document search tool error: {e}")
        return {
            "documents": [],
            "count": 0,
            "query_used": query,
            "formatted": f"Error searching documents: {str(e)}",
        }


register_action(
    name="search_documents",
    category="search",
    description="Search TEXT documents (markdown, org, PDFs, notes) in the knowledge base. NOT for images, comics, photos, artwork, or visual content — use search_images for those.",
    short_description="Search text documents in the knowledge base",
    inputs_model=SearchDocumentsInputs,
    params_model=SearchDocumentsParams,
    outputs_model=SearchDocumentsOutputs,
    tool_function=search_documents_tool,
)


async def search_by_tags_tool(
    tags: List[str],
    categories: List[str] = None,
    query: str = "",
    limit: int = 20,
    user_id: str = "system"
) -> Dict[str, Any]:
    """
    Search documents by tags and/or categories (metadata search, not vector search).
    Returns structured dict with documents, count, and formatted.
    """
    try:
        logger.info(f"Searching by tags: {tags}, categories: {categories}, query: '{query[:50]}'")
        client = await get_backend_tool_client()
        filters = []
        for tag in tags:
            filters.append(f"tag:{tag}")
        if categories:
            for category in categories:
                filters.append(f"category:{category}")
        result = await client.search_documents(
            query=query or "",
            user_id=user_id,
            limit=limit,
            filters=filters
        )
        if "error" in result:
            return {"documents": [], "count": 0, "formatted": f"Error searching by tags: {result['error']}"}
        if result["total_count"] == 0:
            return {"documents": [], "count": 0, "formatted": f"No documents found with tags {tags} and categories {categories}."}
        raw_docs = result["results"][:limit]
        docs = []
        for doc in raw_docs:
            meta = doc.get("metadata") or {}
            docs.append({
                "document_id": doc.get("document_id", ""),
                "title": doc.get("title", ""),
                "filename": doc.get("filename", ""),
                "file_type": meta.get("file_type") or meta.get("doc_type") or doc.get("file_type") or "md",
                "folder_path": meta.get("folder_path", doc.get("folder_path", "")),
                "relevance_score": float(doc.get("relevance_score", 1.0)),
                "content_preview": (doc.get("content_preview") or "")[:800],
            })
        response_parts = [f"Found {result['total_count']} document(s) with tags {tags}:\n"]
        for i, doc in enumerate(docs, 1):
            response_parts.append(f"\n{i}. **{doc['title']}** (ID: {doc['document_id']})")
            path_info = doc.get("folder_path", "").strip()
            file_type = doc.get("file_type", "md")
            if path_info:
                response_parts.append(f"   Path: {path_info} | Type: {file_type}")
            else:
                response_parts.append(f"   File: {doc['filename']} | Type: {file_type}")
            response_parts.append(f"   Relevance: {doc['relevance_score']:.3f}")
            if doc.get("content_preview"):
                response_parts.append("   Preview:")
                response_parts.append("   " + doc["content_preview"].replace("\n", "\n   "))
                if len(doc.get("content_preview", "")) >= 800:
                    response_parts.append("   ...")
        logger.info(f"Tag search completed: {result['total_count']} results")
        return {"documents": docs, "count": len(docs), "formatted": "\n".join(response_parts)}
    except Exception as e:
        logger.error(f"Tag search tool error: {e}")
        return {"documents": [], "count": 0, "formatted": f"Error searching by tags: {str(e)}"}


async def get_document_content_tool(
    document_id: str,
    user_id: str = "system",
) -> Dict[str, Any]:
    """
    Get full content of a document.
    Returns structured dict with content, document_id, word_count, and formatted.
    No truncation — full content is returned. Use prompt variables (e.g. editor_refs_*_toc,
    editor_refs_*_adjacent) or search_documents to scope context when needed.
    """
    try:
        logger.info(f"Getting document content: {document_id}")

        client = await get_backend_tool_client()
        content = await client.get_document_content(
            document_id=document_id,
            user_id=user_id,
        )

        if content is None:
            formatted = f"Document not found: {document_id}"
            return {
                "content": "",
                "document_id": document_id,
                "word_count": 0,
                "formatted": formatted,
            }

        total_chars = len(content)
        word_count = len(content.split()) if content else 0
        logger.info(f"Retrieved document content: {total_chars} characters")

        return {
            "content": content,
            "document_id": document_id,
            "word_count": word_count,
            "formatted": content,
        }

    except Exception as e:
        logger.error(f"Get document content tool error: {e}")
        formatted = f"Error getting document content: {str(e)}"
        return {
            "content": "",
            "document_id": document_id,
            "word_count": 0,
            "formatted": formatted,
        }


register_action(
    name="get_document_content",
    category="knowledge",
    description="Get full content of a document by ID",
    inputs_model=GetDocumentContentInputs,
    outputs_model=GetDocumentContentOutputs,
    tool_function=get_document_content_tool,
)


async def pick_random_file_tool(
    folder_id: str,
    user_id: str = "system",
    file_extension: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Pick a random file (document) from a folder. Optional file_extension filter (e.g. png, jpg).
    Use case: random comic strip from a folder, random image, etc.
    Returns structured dict with document_id, filename, title, folder_id, doc_type, and formatted.
    """
    try:
        client = await get_backend_tool_client()
        result = await client.pick_random_document_from_folder(
            folder_id=folder_id,
            user_id=user_id,
            file_extension=file_extension,
        )
        found = result.get("found", False)
        document_id = result.get("document_id", "") or ""
        filename = result.get("filename", "") or ""
        title = result.get("title")
        folder_id_out = result.get("folder_id")
        doc_type = result.get("doc_type")
        message = result.get("message", "")
        if found:
            formatted = f"Random file: **{filename}** (ID: {document_id})" + (f" — {title}" if title else "")
        else:
            formatted = message or "No document found in folder."
        return {
            "found": found,
            "document_id": document_id,
            "filename": filename,
            "title": title,
            "folder_id": folder_id_out,
            "doc_type": doc_type,
            "formatted": formatted,
        }
    except Exception as e:
        logger.error(f"Pick random file tool error: {e}")
        return {
            "found": False,
            "document_id": "",
            "filename": "",
            "title": None,
            "folder_id": None,
            "doc_type": None,
            "formatted": f"Error picking random file: {str(e)}",
        }


register_action(
    name="pick_random_file",
    category="knowledge",
    description="Pick a random file from a folder; optional extension filter.",
    inputs_model=PickRandomFileInputs,
    params_model=PickRandomFileParams,
    outputs_model=PickRandomFileOutputs,
    tool_function=pick_random_file_tool,
)


def _is_bare_filename(path_or_name: str) -> bool:
    """True if input looks like a bare filename (no path separators, not relative)."""
    s = (path_or_name or "").strip()
    if not s:
        return False
    if s.startswith("./") or s.startswith(".\\"):
        return False
    return "/" not in s and "\\" not in s


async def find_document_by_path_tool(
    file_path: str,
    user_id: str = "system",
    base_path: str = "",
    load_content: bool = False,
) -> Dict[str, Any]:
    """
    Resolve a document by path (e.g. ./outline.md) or bare filename (e.g. monthly.org) to its document_id.
    Set load_content=True to also return full file content in one call.
    Use a bare filename when the path is unknown; use a path when the location is known.
    """
    def _empty_result(resolved: str = "", msg: str = "Error: file_path is required.") -> Dict[str, Any]:
        out = {
            "found": False,
            "document_id": "",
            "filename": "",
            "resolved_path": resolved,
            "formatted": msg,
        }
        if load_content:
            out["content"] = None
        return out

    try:
        if not file_path or not file_path.strip():
            return _empty_result("", "Error: file_path is required.")
        query = file_path.strip()

        if _is_bare_filename(query):
            search_result = await search_documents_tool(query=query, limit=20, user_id=user_id)
            documents = search_result.get("documents", [])
            if not documents:
                return _empty_result(
                    query,
                    f"No document found for '{query}'. Try the exact filename (e.g. monthly.org).",
                )
            target_basename = query.split("/")[-1].split("\\")[-1].lower()
            match = None
            for doc in documents:
                doc_filename = (doc.get("filename") or "").strip()
                doc_path = (doc.get("folder_path") or "").strip()
                if doc_filename.lower() == target_basename:
                    match = doc
                    break
                if doc_path and (doc_path + "/" + doc_filename).endswith(query):
                    match = doc
                    break
            if not match:
                match = documents[0]
            document_id = match.get("document_id")
            if not document_id:
                return _empty_result(query, f"Document found but no document_id for '{query}'.")
            filename = match.get("filename") or query
            resolved_path = (match.get("folder_path") or "") + "/" + filename if match.get("folder_path") else filename
        else:
            client = await get_backend_tool_client()
            result = await client.find_document_by_path(
                file_path=query,
                user_id=user_id,
                base_path=base_path or "",
            )
            if result is None:
                return _empty_result(query, f"No document found at path: {query}")
            document_id = result.get("document_id", "")
            filename = result.get("filename", "")
            resolved_path = result.get("resolved_path", query)

        content = None
        if load_content and document_id:
            content_result = await get_document_content_tool(document_id=document_id, user_id=user_id)
            content = content_result.get("content", "") if isinstance(content_result, dict) else ""
        formatted = f"Found: **{filename}** (ID: {document_id}) at {resolved_path}"
        if load_content and content is not None:
            formatted += f". Content length: {len(content)} chars."
        out = {
            "found": True,
            "document_id": document_id,
            "filename": filename,
            "resolved_path": resolved_path,
            "formatted": formatted,
        }
        if load_content:
            out["content"] = content
        return out
    except Exception as e:
        logger.error("Find document by path tool error: %s", e)
        return _empty_result(file_path or "", f"Error resolving path: {str(e)}")


register_action(
    name="find_document_by_path",
    category="search",
    description="Resolve a document by path or filename to document_id; optional load_content.",
    short_description="Find document by path or filename",
    inputs_model=FindDocumentByPathInputs,
    params_model=FindDocumentByPathParams,
    outputs_model=FindDocumentByPathOutputs,
    tool_function=find_document_by_path_tool,
)


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
        content_result = await get_document_content_tool(document_id, user_id)
        content = content_result.get("content", content_result) if isinstance(content_result, dict) else content_result
        if content.startswith("Error"):
            return {
                "matches": [],
                "total_matches": 0,
                "document_id": document_id,
                "search_query": query,
                "error": content,
                "formatted": f"No matches: {content}",
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
        formatted = f"Found {len(unique_matches)} match(es) in document {document_id} for query '{query}' (search: {search_type})."
        if unique_matches:
            for i, m in enumerate(unique_matches[:10], 1):
                section_name = m.get("section_name")
                section_label = f" [Section: {section_name}]" if section_name else ""
                match_text = (m.get("match_text", "") or "")[:200]
                formatted += f"\n{i}.{section_label} At {m.get('start', 0)}-{m.get('end', 0)}:"
                context_before = (m.get("context_before") or "").strip()
                if context_before:
                    formatted += f"\n   ...{context_before[-100:]}" if len(context_before) > 100 else f"\n   ...{context_before}"
                formatted += f"\n   Match: {match_text}"
                if len((m.get("match_text") or "")) >= 200:
                    formatted += "..."
                context_after = (m.get("context_after") or "").strip()
                if context_after:
                    formatted += f"\n   {context_after[:100]}..." if len(context_after) > 100 else f"\n   {context_after}"
            if len(unique_matches) > 10:
                formatted += f"\n... and {len(unique_matches) - 10} more."

        return {
            "matches": unique_matches,
            "total_matches": len(unique_matches),
            "document_id": document_id,
            "search_query": query,
            "search_type": search_type,
            "formatted": formatted,
        }

    except Exception as e:
        logger.error(f"Search within document tool error: {e}")
        return {
            "matches": [],
            "total_matches": 0,
            "document_id": document_id,
            "search_query": query,
            "error": str(e),
            "formatted": f"Error searching within document: {str(e)}",
        }


async def get_document_metadata_tool(
    document_id: str,
    user_id: str = "system"
) -> Dict[str, Any]:
    """
    Get document metadata by ID. Returns structured dict with document_id, title, filename, content_type, metadata, and formatted.
    """
    try:
        logger.info("Getting document metadata: %s", document_id)
        client = await get_backend_tool_client()
        doc = await client.get_document(document_id=document_id, user_id=user_id)
        if doc is None:
            return {
                "document_id": document_id,
                "title": "",
                "filename": "",
                "content_type": "",
                "metadata": None,
                "formatted": f"Document not found: {document_id}"
            }
        parts = [
            f"Document ID: {doc.get('document_id', '')}",
            f"Title: {doc.get('title', '')}",
            f"Filename: {doc.get('filename', '')}",
            f"Content type: {doc.get('content_type', '')}",
        ]
        meta = doc.get("metadata") or {}
        if meta:
            parts.append("Metadata: " + ", ".join(f"{k}={v}" for k, v in meta.items()))
        formatted = "\n".join(parts)
        return {
            "document_id": doc.get("document_id", document_id),
            "title": doc.get("title", ""),
            "filename": doc.get("filename", ""),
            "content_type": doc.get("content_type", ""),
            "metadata": meta or None,
            "formatted": formatted
        }
    except Exception as e:
        logger.error("Get document metadata tool error: %s", e)
        err = str(e)
        return {
            "document_id": document_id,
            "title": "",
            "filename": "",
            "content_type": "",
            "metadata": None,
            "formatted": f"Error getting document metadata: {err}"
        }


register_action(
    name="search_by_tags",
    category="search",
    description="Search documents by tags and/or categories",
    inputs_model=SearchByTagsInputs,
    outputs_model=SearchByTagsOutputs,
    tool_function=search_by_tags_tool,
)
register_action(
    name="get_document_metadata",
    category="knowledge",
    description="Get document metadata by ID.",
    inputs_model=GetDocumentMetadataInputs,
    outputs_model=GetDocumentMetadataOutputs,
    tool_function=get_document_metadata_tool,
)
register_action(
    name="search_within_document",
    category="search",
    description="Search for terms/concepts within a specific document",
    inputs_model=SearchWithinDocumentInputs,
    params_model=SearchWithinDocumentParams,
    outputs_model=SearchWithinDocumentOutputs,
    tool_function=search_within_document_tool,
)
# Tool registry for LangGraph
DOCUMENT_TOOLS = {
    'search_documents': search_documents_tool,
    'search_by_tags': search_by_tags_tool,
    'find_document_by_path': find_document_by_path_tool,
    'get_document_content': get_document_content_tool,
    'search_within_document': search_within_document_tool,
    'get_document_metadata': get_document_metadata_tool,
}

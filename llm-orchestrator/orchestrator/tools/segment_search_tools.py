"""
Segment Search Tools - Multi-document segment search for finding relevant sections

This tool searches for relevant SEGMENTS within documents, not just documents.
It prioritizes project documents over library documents and returns structured
segment data with section context for precise updates.
"""

import logging
import re
from typing import List, Dict, Any, Optional

from orchestrator.tools.document_tools import (
    search_within_document_tool,
    search_documents_structured,
    get_document_content_tool
)

logger = logging.getLogger(__name__)


def extract_relevant_content_section(
    full_content: str,
    query: str,
    max_length: int = 2000,
    domain_keywords: Optional[List[str]] = None
) -> str:
    """
    Extract relevant content sections from a document based on query.
    
    Uses semantic matching to find the most relevant sections within a document.
    This is a generic version that can work for any domain.
    
    Args:
        full_content: Full document content
        query: Search query to match against
        max_length: Maximum length of extracted content
        domain_keywords: Optional list of domain-specific keywords to boost
        
    Returns:
        Extracted relevant content section
    """
    try:
        content_lower = full_content.lower()
        query_lower = query.lower()
        
        # Extract key terms from query
        query_words = set(query_lower.split())
        
        # Domain-specific keyword matching (if provided)
        domain_matches = []
        if domain_keywords:
            for keyword in domain_keywords:
                if keyword.lower() in query_lower:
                    domain_matches.append(keyword.lower())
        
        # Split content into sections (markdown headers, paragraphs, etc.)
        sections = re.split(r'\n(#{1,6}\s+|\n\n+)', full_content)
        
        # Find sections with highest relevance
        section_scores = []
        
        for section in sections:
            if not section.strip():
                continue
            
            section_lower = section.lower()
            score = 0
            
            # Domain keyword matches get highest score
            for keyword in domain_matches:
                if keyword in section_lower:
                    score += 10
            
            # Query word matches
            for word in query_words:
                if word in section_lower:
                    score += 5
            
            # General relevance (any query terms present)
            if any(word in section_lower for word in query_words):
                score += 2
            
            if score > 0:
                section_scores.append((score, section))
        
        # Sort by relevance score
        section_scores.sort(reverse=True, key=lambda x: x[0])
        
        # Combine top sections
        combined_content = ""
        for score, section in section_scores[:4]:  # Take top 4 sections
            if len(combined_content) + len(section) > max_length:
                remaining_space = max_length - len(combined_content)
                if remaining_space > 100:
                    combined_content += section[:remaining_space] + "...\n\n"
                break
            else:
                combined_content += section + "\n\n"
        
        # If we found relevant content, return it
        if combined_content.strip():
            return combined_content.strip()
        
        # Fallback: find paragraphs containing query terms
        paragraphs = full_content.split('\n\n')
        relevant_paragraphs = []
        
        for para in paragraphs:
            para_lower = para.lower()
            if any(word in para_lower for word in query_words):
                relevant_paragraphs.append(para)
        
        if relevant_paragraphs:
            combined_content = '\n\n'.join(relevant_paragraphs[:3])
            return combined_content[:max_length]
        
        # Final fallback: return beginning of document
        return full_content[:max_length]
        
    except Exception as e:
        logger.warning(f"Failed to extract relevant content: {e}")
        return full_content[:max_length]


async def search_segments_across_documents_tool(
    queries: List[str],
    project_documents: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    user_id: str = "system",
    limit_per_query: int = 5,
    max_queries: int = 3,
    prioritize_project_docs: bool = True,
    context_window: int = 500,
    domain_keywords: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Search for relevant segments across multiple documents.
    
    This tool searches for specific SEGMENTS within documents, not just documents.
    It prioritizes project documents (from referenced_context) over library documents,
    and returns structured segment data with section context for precise updates.
    
    **Use Cases:**
    - Finding specific sections within project files that need updates
    - Searching for relevant content across both project and library documents
    - Getting precise segment locations for targeted edits
    - Prioritizing project context over general library content
    
    **Workflow:**
    1. First searches within project documents (if provided) using `search_within_document_tool`
    2. Then performs semantic search across library documents using `search_documents_structured`
    3. Extracts relevant segments from library documents
    4. Deduplicates and ranks segments (project docs prioritized)
    
    Args:
        queries: List of search queries (can be strings or dicts with "query" key)
        project_documents: Optional dict of project documents organized by category
            Format: {"components": [{"document_id": "...", "filename": "..."}, ...], ...}
        user_id: User ID for access control
        limit_per_query: Maximum matches per query per document (default: 5)
        max_queries: Maximum number of queries to process (default: 3)
        prioritize_project_docs: If True, project documents get higher relevance scores
        context_window: Character context around matches (default: 500)
        domain_keywords: Optional list of domain-specific keywords for content extraction
        
    Returns:
        Dict with:
        - segments: List of segment dicts, each containing:
            - document_id: Document ID
            - filename: Document filename
            - section_name: Name of section containing the segment
            - content: Segment content with context
            - section_start: Character offset of section start
            - section_end: Character offset of section end
            - match_start: Character offset of match start (if from project doc)
            - match_end: Character offset of match end (if from project doc)
            - source: "project_document" or "library_document"
            - relevance_score: Relevance score (0.0-1.0)
            - search_query: Query that found this segment
        - documents_found_count: Number of unique documents found
        - project_doc_count: Number of segments from project documents
        - library_doc_count: Number of segments from library documents
        
    Example:
        ```python
        from orchestrator.tools.segment_search_tools import search_segments_across_documents_tool
        
        # Project documents from referenced_context
        project_docs = {
            "components": [
                {"document_id": "doc1", "filename": "component_spec.md"},
                {"document_id": "doc2", "filename": "circuit_design.md"}
            ]
        }
        
        # Search queries (can be from generate_project_aware_queries)
        queries = [
            "ESP32 keyboard matrix scanning",
            "keyboard scanning circuit design"
        ]
        
        result = await search_segments_across_documents_tool(
            queries=queries,
            project_documents=project_docs,
            user_id=user_id,
            limit_per_query=5
        )
        
        segments = result["segments"]
        for segment in segments:
            print(f"Found in {segment['filename']}: {segment['section_name']}")
            print(f"Content: {segment['content'][:200]}...")
        ```
    """
    try:
        # Normalize queries (handle both strings and dicts)
        normalized_queries = []
        for q in queries[:max_queries]:
            if isinstance(q, dict):
                normalized_queries.append(q.get("query", ""))
            else:
                normalized_queries.append(str(q))
        
        normalized_queries = [q for q in normalized_queries if q.strip()]
        
        if not normalized_queries:
            logger.warning("No valid queries provided")
            return {
                "segments": [],
                "documents_found_count": 0,
                "project_doc_count": 0,
                "library_doc_count": 0
            }
        
        logger.info(f"🔍 Searching for segments with {len(normalized_queries)} queries")
        
        # Collect all document segments found
        all_segments = []
        documents_found = set()
        
        # Phase 1: Search within project documents (if provided)
        if project_documents and prioritize_project_docs:
            logger.info("🔍 Searching within project documents for relevant segments")
            for category, files in project_documents.items():
                if isinstance(files, list):
                    for file_doc in files:
                        if isinstance(file_doc, dict):
                            doc_id = file_doc.get("document_id")
                            if not doc_id:
                                continue
                            
                            # Search within this document for each query
                            for search_query in normalized_queries:
                                try:
                                    matches = await search_within_document_tool(
                                        document_id=doc_id,
                                        query=search_query,
                                        search_type="all_terms",
                                        context_window=context_window,
                                        user_id=user_id
                                    )
                                    
                                    if matches.get("total_matches", 0) > 0:
                                        for match in matches.get("matches", [])[:limit_per_query]:
                                            segment = {
                                                "document_id": doc_id,
                                                "filename": file_doc.get("filename", ""),
                                                "section_name": match.get("section_name"),
                                                "content": f"{match.get('context_before', '')}{match.get('match_text', '')}{match.get('context_after', '')}",
                                                "section_start": match.get("section_start"),
                                                "section_end": match.get("section_end"),
                                                "match_start": match.get("start"),
                                                "match_end": match.get("end"),
                                                "source": "project_document",
                                                "relevance_score": 1.0,  # High relevance for project docs
                                                "search_query": search_query
                                            }
                                            all_segments.append(segment)
                                            documents_found.add(doc_id)
                                except Exception as e:
                                    logger.warning(f"⚠️ Failed to search within document {doc_id}: {e}")
                                    continue
        
        # Phase 2: Perform semantic search across library documents
        for search_query in normalized_queries:
            try:
                search_result = await search_documents_structured(
                    query=search_query,
                    limit=10,
                    user_id=user_id
                )
                
                if search_result.get('total_count', 0) > 0:
                    for doc in search_result.get('results', []):
                        doc_id = doc.get('document_id')
                        if not doc_id or doc_id in documents_found:
                            continue  # Already processed or in project docs
                        
                        # Get document content to extract relevant segments
                        try:
                            content = await get_document_content_tool(doc_id, user_id)
                            
                            if content and not content.startswith("Error"):
                                # Extract relevant section using semantic matching
                                relevant_section = extract_relevant_content_section(
                                    content,
                                    search_query,
                                    max_length=2000,
                                    domain_keywords=domain_keywords
                                )
                                
                                if relevant_section and len(relevant_section.strip()) > 100:
                                    # Find which section this belongs to
                                    section_name = None
                                    section_match = re.search(r'^(##+\s+[^\n]+)', relevant_section, re.MULTILINE)
                                    if section_match:
                                        section_name = re.sub(r'^##+\s+', '', section_match.group(1)).strip()
                                    
                                    segment = {
                                        "document_id": doc_id,
                                        "filename": doc.get('filename', ''),
                                        "section_name": section_name,
                                        "content": relevant_section,
                                        "source": "library_document",
                                        "relevance_score": doc.get('relevance_score', 0.5),
                                        "search_query": search_query
                                    }
                                    all_segments.append(segment)
                                    documents_found.add(doc_id)
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to get content for document {doc_id}: {e}")
                            continue
            
            except Exception as e:
                logger.warning(f"⚠️ Semantic search failed for query '{search_query[:50]}': {e}")
                continue
        
        # Deduplicate segments (same document + section)
        unique_segments = {}
        for segment in all_segments:
            key = (segment.get("document_id"), segment.get("section_name"))
            if key not in unique_segments or segment.get("relevance_score", 0) > unique_segments[key].get("relevance_score", 0):
                unique_segments[key] = segment
        
        segments_list = list(unique_segments.values())
        
        # Sort by relevance (project docs first, then by score)
        segments_list.sort(key=lambda x: (
            0 if x.get("source") == "project_document" else 1,
            -x.get("relevance_score", 0)
        ))
        
        # Count by source
        project_doc_count = sum(1 for seg in segments_list if seg.get("source") == "project_document")
        library_doc_count = sum(1 for seg in segments_list if seg.get("source") == "library_document")
        
        logger.info(f"✅ Found {len(segments_list)} relevant segments across {len(documents_found)} documents "
                   f"({project_doc_count} project, {library_doc_count} library)")
        
        return {
            "segments": segments_list,
            "documents_found_count": len(documents_found),
            "project_doc_count": project_doc_count,
            "library_doc_count": library_doc_count
        }
        
    except Exception as e:
        logger.error(f"❌ Segment search failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "segments": [],
            "documents_found_count": 0,
            "project_doc_count": 0,
            "library_doc_count": 0,
            "error": str(e)
        }


# Tool registry
SEGMENT_SEARCH_TOOLS = {
    'search_segments_across_documents': search_segments_across_documents_tool,
    'extract_relevant_content_section': extract_relevant_content_section
}


"""
Direct Search Service - Provides semantic and full-text search without LLM processing.
Supports hybrid search (vector + PostgreSQL full-text) with RLS for user isolation.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from services.embedding_service_wrapper import get_embedding_service
from repositories.document_repository import DocumentRepository
from config import settings

logger = logging.getLogger(__name__)

# RRF constant for hybrid merge (higher = less influence from rank)
RRF_K = 60


class DirectSearchService:
    """Service for direct semantic search without LLM processing"""
    
    def __init__(self):
        self.embedding_manager = None
        self.document_repository = DocumentRepository()
        self._initialized = False
    
    async def _ensure_initialized(self):
        """Lazy initialization of embedding service wrapper"""
        if not self._initialized:
            self.embedding_manager = await get_embedding_service()
            self._initialized = True
    
    async def search_documents(
        self,
        query: str,
        limit: int = 20,
        similarity_threshold: float = 0.3,  # Lowered from 0.7 to 0.3 for better recall
        search_mode: str = "hybrid",  # "hybrid", "semantic", "fulltext"
        user_id: Optional[str] = None,
        team_ids: Optional[List[str]] = None,
        document_types: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        include_metadata: bool = True,
        exclude_document_ids: Optional[List[str]] = None,
        folder_id: Optional[str] = None,
        file_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Perform search on documents. Supports hybrid (vector + full-text), semantic-only, or fulltext-only.

        Args:
            query: Search query text
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity score for vector search (0.0 to 1.0)
            search_mode: "hybrid" (default), "semantic", or "fulltext"
            user_id: Optional user ID (triggers hybrid scope: user + team + global)
            team_ids: Optional list of team IDs to include
            document_types: Filter by document types
            categories: Filter by document categories
            tags: Filter by document tags
            date_from: Filter documents from this date
            date_to: Filter documents to this date
            include_metadata: Include document metadata in results
            exclude_document_ids: Exclude these document IDs from results
            folder_id: Filter by folder ID
            file_types: Filter by file types

        Returns:
            Dict containing search results and metadata
        """
        try:
            if tags or categories:
                logger.info(f"Direct search with TAG FILTERING: query='{query}', tags={tags}, categories={categories}, limit={limit}")
            else:
                logger.info(f"Direct search query: '{query}' mode={search_mode} limit={limit}")
            await self._ensure_initialized()
            effective_user_id = user_id if user_id and user_id != "system" else None
            if search_mode == "fulltext":
                search_results = await self._fulltext_search(
                    query=query,
                    limit=limit * 2 if exclude_document_ids else limit,
                    user_id=effective_user_id,
                    folder_id=folder_id,
                    file_types=file_types,
                    categories=categories,
                    tags=tags,
                )
            elif search_mode == "hybrid":
                search_results = await self._hybrid_search(
                    query=query,
                    limit=limit,
                    similarity_threshold=similarity_threshold,
                    user_id=effective_user_id,
                    team_ids=team_ids,
                    categories=categories,
                    tags=tags,
                    folder_id=folder_id,
                    file_types=file_types,
                    exclude_document_ids=exclude_document_ids,
                )
            else:
                search_results = await self._vector_search(
                    query=query,
                    limit=limit * 2 if exclude_document_ids else limit,
                    similarity_threshold=similarity_threshold,
                    user_id=effective_user_id,
                    team_ids=team_ids,
                    categories=categories,
                    tags=tags,
                    exclude_document_ids=exclude_document_ids,
                )
                search_results = search_results[:limit]
            filtered_results = []
            for result in search_results:
                formatted = await self._format_search_result(
                    result,
                    query,
                    include_metadata,
                    user_id,
                    folder_id_filter=folder_id,
                    file_types_filter=file_types,
                    is_image_sidecar=result.get("is_image_sidecar"),
                )
                if formatted:
                    filtered_results.append(formatted)
            if exclude_document_ids:
                exclude_set = set(exclude_document_ids)
                filtered_results = [r for r in filtered_results if r.get("document_id") not in exclude_set][:limit]
            else:
                filtered_results = filtered_results[:limit]
            logger.info(f"Direct search completed: {len(filtered_results)} results")
            return {
                "success": True,
                "query": query,
                "results": filtered_results,
                "total_results": len(filtered_results),
                "similarity_threshold": similarity_threshold,
                "search_mode": search_mode,
                "search_metadata": {
                    "query_length": len(query),
                    "search_timestamp": datetime.utcnow().isoformat(),
                },
            }
        except Exception as e:
            logger.error(f"Direct search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "total_results": 0,
            }

    async def _vector_search(
        self,
        query: str,
        limit: int,
        similarity_threshold: float,
        user_id: Optional[str],
        team_ids: Optional[List[str]],
        categories: Optional[List[str]],
        tags: Optional[List[str]],
        exclude_document_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Vector-only search via Qdrant. Returns raw result dicts for merge or format."""
        query_embeddings = await self.embedding_manager.generate_embeddings([query])
        if not query_embeddings:
            return []
        search_results = await self.embedding_manager.search_similar(
            query_embedding=query_embeddings[0],
            limit=limit * 2 if exclude_document_ids else limit,
            score_threshold=similarity_threshold,
            user_id=user_id,
            team_ids=team_ids,
            filter_category=categories[0] if categories else None,
            filter_tags=tags,
        )
        if exclude_document_ids:
            exclude_set = set(exclude_document_ids)
            search_results = [r for r in search_results if r.get("document_id") not in exclude_set]
        return search_results[:limit]

    async def _fulltext_search(
        self,
        query: str,
        limit: int,
        user_id: Optional[str],
        folder_id: Optional[str] = None,
        file_types: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Full-text search on document_chunks and document_metadata (title/tags).
        Always JOINs document_metadata so title/tag edits are searchable without re-embedding.
        """
        try:
            from services.database_manager.database_helpers import fetch_all
            query_clean = (query or "").strip()
            if not query_clean:
                return []
            exact_phrase = query_clean.startswith('"') and query_clean.endswith('"')
            if exact_phrase:
                phrase = query_clean[1:-1].strip()
                tsquery_sql = "phraseto_tsquery('english', $1)"
                query_arg = phrase
            else:
                query_arg = query_clean
                tsquery_sql = "plainto_tsquery('english', $1)"
            rls_context = {"user_id": user_id or "", "user_role": "user"}
            args = [query_arg]
            rank_expr = f"GREATEST(ts_rank(c.content_tsv, {tsquery_sql}), COALESCE(ts_rank(d.meta_tsv, {tsquery_sql}), 0) * 0.7)"
            where_match = f"(c.content_tsv @@ {tsquery_sql} OR d.meta_tsv @@ {tsquery_sql})"
            headline_expr = (
                f"ts_headline('english', c.content, {tsquery_sql}, "
                "'MaxWords=30, MinWords=15, StartSel=<mark>, StopSel=</mark>, HighlightAll=false') AS highlighted_snippet"
            )
            if folder_id or file_types or categories or tags:
                sql = f"""
                    SELECT c.chunk_id, c.document_id, c.content, c.chunk_index, c.is_image_sidecar,
                           c.page_start, c.page_end,
                           {headline_expr},
                           {rank_expr} AS rank
                    FROM document_chunks c
                    JOIN document_metadata d ON d.document_id = c.document_id
                    WHERE {where_match}
                """
                pos = len(args) + 1
                if folder_id:
                    sql += f" AND d.folder_id = ${pos}"
                    args.append(folder_id)
                    pos += 1
                if file_types:
                    ft_list = [t.strip().lower() for t in file_types if t]
                    if ft_list:
                        placeholders = ", ".join(f"${pos + i}" for i in range(len(ft_list)))
                        sql += f" AND LOWER(d.doc_type) IN ({placeholders})"
                        args.extend(ft_list)
                        pos += len(ft_list)
                if categories:
                    placeholders = ", ".join(f"${pos + i}" for i in range(len(categories)))
                    sql += f" AND d.category IN ({placeholders})"
                    args.extend(categories)
                    pos += len(categories)
                if tags:
                    sql += f" AND d.tags && ${pos}"
                    args.append(tags)
                    pos += 1
                sql += f" ORDER BY rank DESC LIMIT ${pos}"
                args.append(limit)
            else:
                sql = f"""
                    SELECT c.chunk_id, c.document_id, c.content, c.chunk_index, c.is_image_sidecar,
                           c.page_start, c.page_end,
                           {headline_expr},
                           {rank_expr} AS rank
                    FROM document_chunks c
                    JOIN document_metadata d ON d.document_id = c.document_id
                    WHERE {where_match}
                    ORDER BY rank DESC
                    LIMIT $2
                """
                args.append(limit)
            rows = await fetch_all(sql, *args, rls_context=rls_context)
            return self._fulltext_rows_to_results(rows, limit)
        except Exception as e:
            logger.error(f"Full-text search failed: {e}")
            return []

    def _fulltext_rows_to_results(self, rows: List[Dict], limit: int) -> List[Dict]:
        """Convert document_chunks rows to same shape as vector search results for _format_search_result."""
        out = []
        for row in rows[:limit]:
            raw_sidecar = row.get("is_image_sidecar")
            is_image_sidecar = raw_sidecar is True or (
                isinstance(raw_sidecar, str) and raw_sidecar.lower() in ("true", "t", "1")
            )
            out.append({
                "chunk_id": row.get("chunk_id"),
                "document_id": row.get("document_id"),
                "content": row.get("content", ""),
                "chunk_index": row.get("chunk_index", 0),
                "score": float(row.get("rank") or 0),
                "is_image_sidecar": is_image_sidecar,
                "page_start": row.get("page_start"),
                "page_end": row.get("page_end"),
                "highlighted_snippet": row.get("highlighted_snippet"),
            })
        return out

    def _merge_rrf(
        self,
        vector_results: List[Dict],
        fts_results: List[Dict],
        k: int = RRF_K,
    ) -> List[Dict]:
        """Merge two ranked lists by Reciprocal Rank Fusion. Deduplicates by chunk_id."""
        scores = {}
        for rank, r in enumerate(vector_results):
            cid = r.get("chunk_id")
            if cid:
                scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
        for rank, r in enumerate(fts_results):
            cid = r.get("chunk_id")
            if cid:
                scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
        seen = set()
        merged = []
        for r in vector_results:
            cid = r.get("chunk_id")
            if cid and cid not in seen:
                seen.add(cid)
                merged.append((scores[cid], r))
        for r in fts_results:
            cid = r.get("chunk_id")
            if cid and cid not in seen:
                seen.add(cid)
                merged.append((scores[cid], r))
        merged.sort(key=lambda x: -x[0])
        return [r for _, r in merged]

    async def _hybrid_search(
        self,
        query: str,
        limit: int,
        similarity_threshold: float,
        user_id: Optional[str],
        team_ids: Optional[List[str]],
        categories: Optional[List[str]],
        tags: Optional[List[str]],
        folder_id: Optional[str],
        file_types: Optional[List[str]],
        exclude_document_ids: Optional[List[str]],
    ) -> List[Dict]:
        """Run vector and full-text in parallel, merge with RRF."""
        request_limit = limit * 2
        vec_task = self._vector_search(
            query=query,
            limit=request_limit,
            similarity_threshold=similarity_threshold,
            user_id=user_id,
            team_ids=team_ids,
            categories=categories,
            tags=tags,
            exclude_document_ids=exclude_document_ids,
        )
        fts_task = self._fulltext_search(
            query=query,
            limit=request_limit,
            user_id=user_id,
            folder_id=folder_id,
            file_types=file_types,
            categories=categories,
            tags=tags,
        )
        vector_results, fts_results = await asyncio.gather(vec_task, fts_task)
        merged = self._merge_rrf(vector_results, fts_results, k=RRF_K)
        return merged[:limit]
    
    async def _format_search_result(
        self,
        result: Dict,
        query: str,
        include_metadata: bool,
        user_id: Optional[str] = None,
        folder_id_filter: Optional[str] = None,
        file_types_filter: Optional[List[str]] = None,
        is_image_sidecar: Optional[bool] = None,
    ) -> Optional[Dict]:
        """Format a single search result for display. Returns None if folder_id or file_types filter excludes this document."""
        try:
            chunk_id = result.get("chunk_id")
            document_id = result.get("document_id")
            similarity_score = result.get("score", result.get("similarity_score", 0.0))
            chunk_text_raw = result.get("content", result.get("chunk_text", ""))
            if isinstance(chunk_text_raw, dict):
                chunk_text = chunk_text_raw.get("text", chunk_text_raw.get("content", str(chunk_text_raw)))
            elif isinstance(chunk_text_raw, str):
                chunk_text = chunk_text_raw
            else:
                chunk_text = str(chunk_text_raw) if chunk_text_raw else ""
            if is_image_sidecar is None:
                raw = result.get("is_image_sidecar")
                is_image_sidecar = raw is True or (isinstance(raw, str) and raw.lower() == "true")

            if not chunk_id or not document_id:
                return None

            document_metadata = {}
            doc_info = None
            if include_metadata or folder_id_filter or file_types_filter:
                doc_info = await self.document_repository.get_document_by_id(document_id, user_id)
                if doc_info:
                    document_metadata = {
                        "document_id": document_id,
                        "filename": doc_info.get("filename", ""),
                        "title": doc_info.get("title", ""),
                        "doc_type": doc_info.get("doc_type", ""),
                        "category": doc_info.get("category", ""),
                        "tags": doc_info.get("tags", []),
                        "upload_date": doc_info.get("upload_date", ""),
                        "file_size": doc_info.get("file_size", 0),
                        "page_count": doc_info.get("page_count", 0),
                        "author": doc_info.get("author", ""),
                        "description": doc_info.get("description", "")
                    }
                if folder_id_filter or file_types_filter:
                    if not doc_info:
                        return None
                    if folder_id_filter and (doc_info.get("folder_id") or "") != folder_id_filter:
                        return None
                    if file_types_filter:
                        doc_type = (doc_info.get("doc_type") or "").strip().lower()
                        ext = (doc_info.get("filename") or "").split(".")[-1].lower() if doc_info.get("filename") else ""
                        allowed = [x.strip().lower().lstrip(".") for x in file_types_filter if x]
                        if allowed and doc_type not in allowed and ext not in allowed:
                            return None

            highlighted_text = self._highlight_query_terms(chunk_text, query)
            context = self._extract_context(chunk_text, query)
            backend_snippet = result.get("highlighted_snippet")

            formatted_result = {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "similarity_score": round(similarity_score, 4),
                "text": chunk_text,
                "highlighted_text": highlighted_text,
                "highlighted_snippet": backend_snippet if backend_snippet else None,
                "context": context,
                "result_type": "image" if is_image_sidecar else "document",
                "chunk_metadata": {
                    "chunk_index": result.get("chunk_index", 0),
                    "page_number": result.get("page_start") or (result.get("metadata") or {}).get("page_start") or result.get("page_number"),
                    "section_title": result.get("section_title", ""),
                    "text_length": len(chunk_text),
                    "word_count": len(chunk_text.split())
                }
            }
            if is_image_sidecar and doc_info:
                filename = doc_info.get("filename") or ""
                if filename:
                    formatted_result["image_filename"] = filename
                if include_metadata:
                    formatted_result["document"] = document_metadata
            elif include_metadata:
                formatted_result["document"] = document_metadata
            return formatted_result
        except Exception as e:
            logger.error(f"Failed to format search result: {e}")
            return None
    
    def _highlight_query_terms(self, text: str, query: str) -> str:
        """Highlight query terms in the text"""
        try:
            import re
            
            # Split query into individual terms
            query_terms = [term.strip().lower() for term in query.split() if len(term.strip()) > 2]
            
            highlighted_text = text
            for term in query_terms:
                # Use word boundaries to match whole words
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                highlighted_text = pattern.sub(f"**{term}**", highlighted_text)
            
            return highlighted_text
            
        except Exception as e:
            logger.error(f"❌ Failed to highlight query terms: {e}")
            return text
    
    def _extract_context(self, text: str, query: str, context_length: int = 200) -> Dict:
        """Extract context around query matches"""
        try:
            import re
            
            query_terms = [term.strip().lower() for term in query.split() if len(term.strip()) > 2]
            
            # Find the best match position
            best_match_pos = 0
            best_match_score = 0
            
            for term in query_terms:
                match = re.search(re.escape(term), text, re.IGNORECASE)
                if match:
                    # Score based on term length and position
                    score = len(term) * (1.0 - match.start() / len(text))
                    if score > best_match_score:
                        best_match_score = score
                        best_match_pos = match.start()
            
            # Extract context around the best match
            start_pos = max(0, best_match_pos - context_length // 2)
            end_pos = min(len(text), best_match_pos + context_length // 2)
            
            context_text = text[start_pos:end_pos]
            
            # Add ellipsis if truncated
            if start_pos > 0:
                context_text = "..." + context_text
            if end_pos < len(text):
                context_text = context_text + "..."
            
            return {
                "text": context_text,
                "start_position": start_pos,
                "end_position": end_pos,
                "match_position": best_match_pos - start_pos if best_match_pos >= start_pos else 0
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to extract context: {e}")
            return {
                "text": text[:400] + "..." if len(text) > 400 else text,
                "start_position": 0,
                "end_position": min(400, len(text)),
                "match_position": 0
            }
    
    async def get_search_suggestions(self, partial_query: str, limit: int = 10) -> List[str]:
        """Get search suggestions based on partial query"""
        try:
            # This could be enhanced with a proper suggestion system
            # For now, return some basic suggestions based on document content
            
            if len(partial_query) < 2:
                return []
            
            # Get common terms from document chunks
            suggestions = await self.embedding_manager.get_common_terms(
                prefix=partial_query,
                limit=limit
            )
            
            return suggestions
            
        except Exception as e:
            logger.error(f"❌ Failed to get search suggestions: {e}")
            return []
    
    async def get_search_filters(self) -> Dict[str, List[str]]:
        """Get available filter options for search"""
        try:
            # Get available document types, categories, and tags
            filter_options = await self.document_repository.get_filter_options()
            
            return {
                "document_types": filter_options.get("doc_types", []),
                "categories": filter_options.get("categories", []),
                "tags": filter_options.get("tags", []),
                "date_range": {
                    "earliest": filter_options.get("earliest_date"),
                    "latest": filter_options.get("latest_date")
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get search filters: {e}")
            return {
                "document_types": [],
                "categories": [],
                "tags": [],
                "date_range": {"earliest": None, "latest": None}
            }
    
    async def export_search_results(
        self, 
        results: List[Dict], 
        format_type: str = "json"
    ) -> Dict[str, Any]:
        """Export search results in various formats"""
        try:
            if format_type.lower() == "csv":
                import csv
                import io
                
                output = io.StringIO()
                writer = csv.writer(output)
                
                # Write header
                writer.writerow([
                    "Similarity Score", "Document Title", "Document Type", 
                    "Category", "Text Preview", "Page Number"
                ])
                
                # Write data
                for result in results:
                    doc = result.get("document", {})
                    chunk = result.get("chunk_metadata", {})
                    
                    writer.writerow([
                        result.get("similarity_score", 0),
                        doc.get("title", doc.get("filename", "")),
                        doc.get("doc_type", ""),
                        doc.get("category", ""),
                        result.get("text", "")[:200] + "...",
                        chunk.get("page_number", "")
                    ])
                
                return {
                    "success": True,
                    "format": "csv",
                    "data": output.getvalue(),
                    "filename": f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                }
            
            else:  # JSON format
                return {
                    "success": True,
                    "format": "json",
                    "data": results,
                    "filename": f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to export search results: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def search_web(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search the web for information using SearXNG"""
        try:
            logger.info(f"🌐 Web search query: '{query}' with {limit} results")
            
            # Use the same SearXNG implementation as the LangGraph tools
            from services.langgraph_tools.web_content_tools import WebContentTools
            web_tools = WebContentTools()
            results = await web_tools._search_searxng(query, limit)
            
            return {
                "success": True,
                "results": results,
                "count": len(results),
                "query": query,
                "message": f"Found {len(results)} results via SearXNG"
            }
            
        except Exception as e:
            logger.error(f"❌ Web search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "count": 0
            }

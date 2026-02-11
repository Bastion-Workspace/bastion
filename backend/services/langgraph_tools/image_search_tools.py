"""
Image Search Tools Module
Search functionality for images with metadata sidecars, supporting multiple image types
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from services.embedding_service_wrapper import get_embedding_service
from models.api_models import DocumentCategory, DocumentFilterRequest
from repositories.document_repository import DocumentRepository

logger = logging.getLogger(__name__)


class ImageSearchTools:
    """Search tools for images with metadata sidecars"""
    
    # Map image type to DocumentCategory for filtering
    TYPE_TO_CATEGORY = {
        "comic": DocumentCategory.COMIC,
        "artwork": DocumentCategory.ENTERTAINMENT,
        "meme": DocumentCategory.ENTERTAINMENT,
        "screenshot": DocumentCategory.TECHNICAL,
        "medical": DocumentCategory.MEDICAL,
        "documentation": DocumentCategory.TECHNICAL,
        "maps": DocumentCategory.REFERENCE,
        "photo": DocumentCategory.ENTERTAINMENT,
        "other": DocumentCategory.OTHER
    }
    
    def __init__(self):
        self._embedding_manager = None
        self._document_repository = None
    
    async def _get_embedding_manager(self):
        """Get embedding service wrapper with lazy initialization"""
        if self._embedding_manager is None:
            try:
                self._embedding_manager = await get_embedding_service()
                logger.info("Embedding service wrapper initialized for image search")
            except Exception as e:
                logger.error(f"Failed to initialize embedding service wrapper: {e}")
        return self._embedding_manager
    
    async def _get_document_repository(self):
        """Get document repository with lazy initialization"""
        if self._document_repository is None:
            try:
                self._document_repository = DocumentRepository()
                await self._document_repository.initialize()
                logger.info("Document repository initialized for image search")
            except Exception as e:
                logger.error(f"Failed to initialize document repository: {e}")
        return self._document_repository
    
    def get_tools(self) -> Dict[str, Any]:
        """Get all image search tools"""
        return {
            "search_images": self.search_images,
        }
    
    def get_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all image search tools"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_images",
                    "description": "Search for images with metadata by content, type, date, or author. Returns markdown-formatted results with image URLs.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query (e.g., 'office politics', 'medical diagnosis', 'architecture diagram')"},
                            "image_type": {"type": "string", "description": "OPTIONAL: Filter by image type (comic, artwork, meme, screenshot, medical, documentation, maps, photo, other)"},
                            "date": {"type": "string", "description": "OPTIONAL: Filter by specific date (YYYY-MM-DD format, e.g., '2001-09-11')"},
                            "author": {"type": "string", "description": "OPTIONAL: Filter by author/creator name"},
                            "identity": {"type": "string", "description": "OPTIONAL: Filter by detected face identity (e.g., 'Steve McQueen', 'Snoopy')"},
                            "limit": {"type": "integer", "description": "Maximum number of results to return", "default": 10}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
    
    async def search_images(
        self,
        query: str,
        image_type: Optional[str] = None,
        date: Optional[str] = None,
        author: Optional[str] = None,
        series: Optional[str] = None,
        identity: Optional[str] = None,
        limit: int = 10,
        user_id: Optional[str] = None,
        is_random: bool = False,
        exclude_document_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Search for images with metadata by content, type, date, or author.
        When user_id is provided, searches both the user's collection and the global collection (hybrid search).
        
        Args:
            query: Search query
            image_type: Optional filter by image type (comic, artwork, meme, screenshot, medical, documentation, maps, photo, other)
            date: Optional date filter (YYYY-MM-DD)
            author: Optional author/creator filter
            series: Optional series filter (e.g., "Dilbert")
            limit: Maximum results
            user_id: When set, run hybrid search (user + global) and apply user RLS for metadata access
            is_random: If True, return random images instead of semantic search
            
        Returns:
            Dict with 'images_markdown' (base64 embedded images) and 'metadata' (list of metadata dicts)
        """
        rls_context = (
            {"user_id": user_id, "user_role": "user"}
            if user_id
            else {"user_id": "", "user_role": "admin"}
        )
        try:
            if is_random:
                logger.info(f"🎲 RANDOM IMAGE SEARCH: type={image_type}, author={author}, series={series}, identity={identity}, limit={limit}")
                return await self._random_image_search(
                    image_type=image_type,
                    author=author,
                    series=series,
                    identity=identity,
                    limit=limit,
                    rls_context=rls_context
                )

            # When user asks for "photos with X" / "photos of X", try object-name search first (detected/annotated objects)
            if user_id and query and not (series or author or date):
                q_stripped = query.strip()
                q_lower = q_stripped.lower()
                object_candidates = []
                if q_stripped:
                    object_candidates.append(q_stripped)
                words = [w for w in q_stripped.split() if len(w) > 1]
                stop = {"find", "me", "some", "the", "a", "an", "with", "in", "them", "photos", "pictures", "images", "of", "containing", "that", "have", "has"}
                for w in words:
                    if w.lower() not in stop and w not in object_candidates:
                        object_candidates.append(w)
                if "with" in q_lower or "of" in q_lower or "photos" in q_lower or "pictures" in q_lower:
                    for candidate in object_candidates[:5]:
                        try:
                            from services.langgraph_tools.object_detection_tools import search_images_by_object
                            obj_result = await search_images_by_object(object_name=candidate, user_id=user_id, limit=limit)
                            if obj_result.get("success") and obj_result.get("document_ids"):
                                doc_ids = obj_result["document_ids"]
                                logger.info(f"🔍 OBJECT SEARCH hit for '{candidate}': {len(doc_ids)} document(s)")
                                built = await self._build_image_results_from_document_ids(
                                    document_ids=doc_ids,
                                    limit=limit,
                                    rls_context=rls_context,
                                    label=f"Photos containing '{candidate}'"
                                )
                                if built.get("metadata"):
                                    return built
                        except Exception as e:
                            logger.debug("Object search try %s: %s", candidate, e)
                            continue

            logger.info(f"🔍 IMAGE SEARCH START: query='{query}', type={image_type}, date={date}, author={author}, series={series}, identity={identity}, limit={limit}")
            
            embedding_manager = await self._get_embedding_manager()
            if not embedding_manager:
                logger.error("❌ Image search unavailable - embedding service not initialized")
                return "❌ Image search unavailable - embedding service not initialized"
            
            # Generate query embedding
            query_embeddings = await embedding_manager.generate_embeddings([query])
            if not query_embeddings or len(query_embeddings) == 0:
                return {
                    "images_markdown": "❌ Failed to generate query embedding",
                    "metadata": []
                }
            
            # Build filters and normalize image_type (accept plurals: comics -> comic, photos -> photo)
            filter_category = None
            image_type_lower = None
            if image_type:
                image_type_lower = image_type.lower().strip()
                plural_to_singular = {
                    "comics": "comic",
                    "photos": "photo",
                    "screenshots": "screenshot",
                    "memes": "meme",
                    "artworks": "artwork",
                }
                if image_type_lower in plural_to_singular:
                    image_type_lower = plural_to_singular[image_type_lower]
                    logger.debug(f"Normalized image type to singular: {image_type_lower}")
                if image_type_lower in self.TYPE_TO_CATEGORY:
                    filter_category = self.TYPE_TO_CATEGORY[image_type_lower].value
                else:
                    logger.warning(f"Unknown image type: {image_type}, ignoring type filter")
            
            # When image_type is specified: vector-first for scale (avoid loading all doc_ids of that type).
            # We only type-check the small set of vector result IDs later.
            image_type_requested = bool(image_type and image_type_lower)
            candidate_doc_ids = None  # Set after vector search when image_type_requested (small set only)
            if image_type_requested:
                try:
                    from services.database_manager.database_helpers import fetch_one
                    # Cheap existence check: any doc of this type? (index-friendly, no large result set)
                    exists_row = await fetch_one(
                        """
                        SELECT 1 FROM document_metadata
                        WHERE (metadata_json->>'image_type' = $1 OR metadata_json->>'type' = $1)
                        AND (metadata_json->>'has_searchable_metadata' = 'true' OR metadata_json->>'image_type' IS NOT NULL OR metadata_json->>'type' IS NOT NULL)
                        LIMIT 1
                        """,
                        image_type_lower,
                        rls_context=rls_context
                    )
                    if not exists_row and filter_category:
                        exists_row = await fetch_one(
                            "SELECT 1 FROM document_metadata WHERE category = $1 LIMIT 1",
                            filter_category,
                            rls_context=rls_context
                        )
                    if not exists_row:
                        type_msg = f" of type '{image_type}'" if image_type else ""
                        return {
                            "images_markdown": f"🔍 No images found{type_msg} matching '{query}'",
                            "metadata": []
                        }
                except Exception as e:
                    logger.warning(f"Type existence check failed: {e}, continuing")
            
            # Hybrid search: Run vector search FIRST, then apply filters as post-filters
            results = []
            
            # If series, author, or date is provided, run hybrid search with filters
            if series or author or date:
                logger.info(f"🔍 HYBRID SEARCH: Running vector search first, then applying filters (series={series}, author={author}, date={date})")
                try:
                    import asyncio
                    
                    # STEP 1: Run vector search FIRST with full semantic query (hybrid when user_id set)
                    vector_results = await embedding_manager.search_similar(
                        query_embedding=query_embeddings[0],
                        limit=limit * 5,  # Get more candidates for filtering
                        score_threshold=0.0,  # Lower threshold to get all candidates
                        user_id=user_id,
                        filter_category=None,  # Don't filter - let post-filters handle it
                        filter_tags=None
                    )
                    
                    logger.info(f"🔍 VECTOR SEARCH RESULT: Found {len(vector_results)} results for query='{query}'")
                    if vector_results:
                        top_3_scores = [f"{vr.get('score', 0):.3f}" for vr in vector_results[:3]]
                        top_3_titles = [vr.get('metadata', {}).get('title', 'N/A') for vr in vector_results[:3]]
                        logger.info(f"   Top 3 scores: {top_3_scores}")
                        logger.info(f"   Top 3 titles: {top_3_titles}")

                    
                    # STEP 2: Apply filters as post-filters on vector results
                    filtered_vector_results = []
                    for vr in vector_results:
                        metadata = vr.get("metadata", {})
                        
                        # Apply series filter if provided
                        if series:
                            vr_series = (metadata.get("series") or "").lower()
                            if series.lower() not in vr_series:
                                continue  # Skip if doesn't match series filter
                        
                        # Apply author filter if provided
                        if author:
                            vr_author = (metadata.get("author") or "").lower()
                            if author.lower() not in vr_author:
                                continue  # Skip if doesn't match author filter
                        
                        # Date filter is applied later in the code (lines 470-487)
                        # So we don't filter by date here
                        
                        filtered_vector_results.append(vr)
                    
                    if exclude_document_ids:
                        exclude_set = set(exclude_document_ids)
                        filtered_vector_results = [
                            vr for vr in filtered_vector_results
                            if (vr.get("metadata", {}).get("document_id") or vr.get("document_id")) not in exclude_set
                        ]
                        logger.info(f"🔍 After excluding {len(exclude_set)} already-shown: {len(filtered_vector_results)} results")
                    
                    logger.info(f"🔍 After filtering: {len(filtered_vector_results)} results match filters")
                    
                    # STEP 3: Check if vector search found meaningful semantic matches
                    # If all scores are very close (within 0.05), vector search didn't find relevant semantic content
                    vector_scores = [vr.get("score", 0) for vr in vector_results]
                    vector_has_meaningful_results = False
                    if len(vector_scores) > 1:
                        max_score = max(vector_scores)
                        min_score = min(vector_scores)
                        score_spread = max_score - min_score
                        vector_has_meaningful_results = score_spread > 0.05  # Significant semantic variation
                        logger.info(f"🔍 Vector score spread (all results): {score_spread:.3f} (max: {max_score:.3f}, min: {min_score:.3f}), meaningful: {vector_has_meaningful_results}")
                    
                    # Also check score spread of filtered results
                    if filtered_vector_results and len(filtered_vector_results) > 1:
                        filtered_scores = [vr.get("score", 0) for vr in filtered_vector_results]
                        filtered_max = max(filtered_scores)
                        filtered_min = min(filtered_scores)
                        filtered_spread = filtered_max - filtered_min
                        logger.info(f"🔍 Filtered results score spread: {filtered_spread:.3f} (max: {filtered_max:.3f}, min: {filtered_min:.3f})")
                    
                    # STEP 4: If no filtered results, try SQL text search fallback
                    if not filtered_vector_results:
                        # Strategy 1: If we have date filter, use exact match (existing logic)
                        # Strategy 2: If vector found NOTHING, use SQL text search on content/tags
                        should_use_exact_fallback = (len(vector_results) == 0 and date is not None)
                        should_use_text_search = (len(vector_results) == 0 and date is None and query)
                        
                        if should_use_exact_fallback:
                            logger.info(f"⚠️ No vector results but have date filter, trying exact match fallback")
                            exact_match_docs = await self._exact_match_search(
                                series=series,
                                date=date,
                                image_type=image_type,
                                limit=limit,
                                rls_context=rls_context
                            )
                            
                            if exact_match_docs:
                                # Convert exact matches to result format (fallback only)
                                from services.database_manager.database_helpers import fetch_one
                                for doc in exact_match_docs[:limit]:
                                    row = await fetch_one(
                                        "SELECT metadata_json FROM document_metadata WHERE document_id = $1",
                                        doc.document_id,
                                        rls_context=rls_context
                                    )
                                    
                                    metadata_json = {}
                                    if row and row.get("metadata_json"):
                                        metadata_json_raw = row["metadata_json"]
                                        if isinstance(metadata_json_raw, str):
                                            import json
                                            try:
                                                metadata_json = json.loads(metadata_json_raw)
                                            except:
                                                metadata_json = {}
                                    
                                    filtered_vector_results.append({
                                        "metadata": {
                                            "document_id": doc.document_id,
                                            "title": doc.title,
                                            "date": metadata_json.get("date"),
                                            "author": doc.author or metadata_json.get("author"),
                                            "series": metadata_json.get("series"),
                                            "image_url": metadata_json.get("image_url"),
                                            "image_type": metadata_json.get("image_type") or metadata_json.get("type"),
                                            "content": metadata_json.get("content"),
                                            "tags": doc.tags or []
                                        },
                                        "score": 0.6,  # Lower score for exact match fallback
                                        "matched_both": False
                                    })
                                    logger.info(f"📋 Exact match fallback: {doc.title}")
                        
                        elif should_use_text_search:
                            logger.info(f"⚠️ Vector search returned 0 results, trying SQL text search fallback for '{query}'")
                            text_search_docs = await self._text_search_fallback(
                                query=query,
                                series=series,
                                author=author,
                                date=date,
                                image_type=image_type,
                                limit=limit,
                                rls_context=rls_context
                            )
                            
                            if text_search_docs:
                                logger.info(f"✅ SQL text search found {len(text_search_docs)} results")
                                filtered_vector_results.extend(text_search_docs)
                            else:
                                # No text search results either - no results found
                                filter_desc = []
                                if series:
                                    filter_desc.append(f"series='{series}'")
                                if author:
                                    filter_desc.append(f"author='{author}'")
                                if date:
                                    filter_desc.append(f"date='{date}'")
                                filter_str = ", ".join(filter_desc) if filter_desc else "no filters"
                                
                                logger.info(f"⚠️ No results found: vector search found no semantic matches, and exact match fallback also found nothing (filters: {filter_str})")
                                return {
                                    "images_markdown": "",
                                    "metadata": [],
                                    "message": f"No comics found matching '{query}' with the specified filters ({filter_str}). The metadata for available comics doesn't contain relevant keywords."
                                }
                        else:
                            # Vector search found meaningful results, but filters excluded them all
                            logger.info(f"⚠️ Vector search found {len(vector_results)} semantically relevant matches, but none match the filters (series={series}, author={author})")
                            return {
                                "images_markdown": "",
                                "metadata": [],
                                "message": f"I found {len(vector_results)} comics about '{query}', but none of them match the specified filters (series={series or 'any'}, author={author or 'any'})."
                            }
                    
                    # STEP 5: Boost results that match both vector search AND exact match (if series/author provided)
                    if series or author:
                        # Run exact match in parallel for boosting
                        exact_match_docs = await self._exact_match_search(
                            series=series,
                            date=date,
                            image_type=image_type,
                            limit=limit,
                            rls_context=rls_context
                        )
                        
                        # Apply type pre-filter if candidate_doc_ids is set
                        if candidate_doc_ids is not None and exact_match_docs:
                            type_filtered_exact = []
                            for doc in exact_match_docs:
                                if doc.document_id in candidate_doc_ids:
                                    type_filtered_exact.append(doc)
                            exact_match_docs = type_filtered_exact
                            logger.info(f"🔍 After type pre-filter on exact match: {len(exact_match_docs)} results match type '{image_type}'")
                        
                        matching_doc_ids = {doc.document_id for doc in exact_match_docs}
                        
                        # Boost scores for items that matched both
                        boost_factor = 1.2
                        merged_results = []
                        
                        for vr in filtered_vector_results:
                            doc_id = vr.get("metadata", {}).get("document_id") or vr.get("document_id")
                            if doc_id in matching_doc_ids:
                                # Boost the score for items that matched both
                                boosted_score = min(vr.get("score", 0) * boost_factor, 1.0)
                                merged_results.append({
                                    **vr,
                                    "score": boosted_score,
                                    "matched_both": True
                                })
                                logger.info(f"✅ Boosted result: {vr.get('metadata', {}).get('title', 'N/A')} (exact + vector match, score: {vr.get('score', 0):.3f} -> {boosted_score:.3f})")
                            else:
                                # Unboosted, but still relevant semantically
                                merged_results.append({
                                    **vr,
                                    "matched_both": False
                                })
                                logger.info(f"📊 Semantic match: {vr.get('metadata', {}).get('title', 'N/A')} (vector only, score: {vr.get('score', 0):.3f})")
                        
                        # Sort merged results by score (boosted items will be at top)
                        merged_results.sort(key=lambda x: x.get("score", 0), reverse=True)
                    else:
                        # No exact match to boost with, just use filtered vector results
                        merged_results = filtered_vector_results
                        merged_results.sort(key=lambda x: x.get("score", 0), reverse=True)
                    
                    # Type-filter (vector-first scale): only check this small set of doc_ids
                    if image_type_requested and merged_results:
                        from services.database_manager.database_helpers import fetch_all
                        doc_ids_hybrid = [(vr.get("metadata", {}).get("document_id") or vr.get("document_id")) for vr in merged_results]
                        doc_ids_hybrid = [did for did in doc_ids_hybrid if did]
                        if doc_ids_hybrid:
                            type_check_rows = await fetch_all(
                                """SELECT document_id FROM document_metadata
                                WHERE document_id = ANY($1) AND (metadata_json->>'image_type' = $2 OR metadata_json->>'type' = $2)""",
                                doc_ids_hybrid,
                                image_type_lower,
                                rls_context=rls_context
                            )
                            allowed_ids = {row["document_id"] for row in type_check_rows}
                            merged_results = [vr for vr in merged_results if (vr.get("metadata", {}).get("document_id") or vr.get("document_id")) in allowed_ids]
                            logger.info(f"🔍 Hybrid type-check: {len(merged_results)} of {len(doc_ids_hybrid)} match type '{image_type}'")
                    
                    # STEP 6: Apply smart filtering based on score gaps
                    if len(merged_results) > 1:
                        top_score = merged_results[0].get("score", 0)
                        second_score = merged_results[1].get("score", 0)
                        score_gap = top_score - second_score
                        
                        query_lower = query.lower()
                        has_semantic_content = True  # If we got here, vector search found something
                        if series:
                            series_lower = series.lower()
                            has_semantic_content = len(query_lower.replace(series_lower, '').strip()) > 5
                        
                        if has_semantic_content and score_gap > 0.1:
                            results = [merged_results[0]]
                            logger.info(f"✅ Returning only top result (significant score gap: {score_gap:.3f})")
                        else:
                            results = merged_results[:limit]
                            logger.info(f"✅ Returning top {len(results)} merged results (hybrid search)")
                    else:
                        results = merged_results[:limit]
                    
                    if not results:
                        logger.info(f"No results from hybrid search after filtering")
                except Exception as e:
                    logger.warning(f"Hybrid search failed, falling back to vector search: {e}")
                    import traceback
                    logger.error(f"Hybrid search traceback: {traceback.format_exc()}")
            
            # If author is provided and we don't have results yet, search by author
            if author and not results:
                # Step 1: PostgreSQL lookup for exact author match
                document_repo = await self._get_document_repository()
                if document_repo:
                    try:
                        filter_request = DocumentFilterRequest(
                            author=author,
                            category=DocumentCategory(filter_category) if filter_category else None,
                            limit=limit * 3  # Get more candidates for filtering
                        )
                        
                        # Get all image metadata documents matching author (user RLS: global + own)
                        postgres_results, total = await document_repo.filter_documents(filter_request, rls_context=rls_context)
                        
                        # Filter to only image metadata sidecars
                        image_categories = [
                            DocumentCategory.COMIC,
                            DocumentCategory.ENTERTAINMENT,
                            DocumentCategory.TECHNICAL,
                            DocumentCategory.MEDICAL,
                            DocumentCategory.REFERENCE,
                            DocumentCategory.OTHER
                        ]
                        
                        image_metadata_docs = []
                        for doc in postgres_results:
                            # Query metadata_json from database
                            from services.database_manager.database_helpers import fetch_one
                            row = await fetch_one(
                                "SELECT metadata_json FROM document_metadata WHERE document_id = $1",
                                doc.document_id,
                                rls_context=rls_context
                            )
                            
                            if not row or not row.get("metadata_json"):
                                continue
                            
                            metadata_json = row["metadata_json"]
                            if isinstance(metadata_json, str):
                                import json
                                try:
                                    metadata_json = json.loads(metadata_json)
                                except:
                                    continue
                            
                            if metadata_json and metadata_json.get("has_searchable_metadata"):
                                if doc.category in image_categories:
                                    # Check both author and series fields
                                    # For comics, "Dilbert" is the series, not the author
                                    doc_author = (doc.author or metadata_json.get("author", "")).lower()
                                    series = metadata_json.get("series", "").lower() if metadata_json else ""
                                    author_lower = author.lower()
                                    
                                    # Include if author matches OR series matches
                                    if author_lower in doc_author or author_lower in series:
                                        image_metadata_docs.append(doc)
                        
                        if image_metadata_docs:
                            # Step 2: Do vector search to get similarity scores (hybrid when user_id set)
                            vector_results = await embedding_manager.search_similar(
                                query_embedding=query_embeddings[0],
                                limit=limit * 5,  # Get more results to match against
                                score_threshold=0.0,  # Lower threshold to get all candidates
                                user_id=user_id,
                                filter_category=filter_category,
                                filter_tags=None
                            )
                            
                            # Step 3: Filter vector results to only include PostgreSQL matches
                            # Create set of matching document IDs
                            matching_doc_ids = {doc.document_id for doc in image_metadata_docs}
                            
                            # Filter and rank vector results by document_id match
                            filtered_vector_results = []
                            for vr in vector_results:
                                doc_id = vr.get("metadata", {}).get("document_id") or vr.get("document_id")
                                if doc_id in matching_doc_ids:
                                    filtered_vector_results.append(vr)
                            
                            # If we have vector-ranked results, use those (they're already sorted by similarity)
                            if filtered_vector_results:
                                results = filtered_vector_results[:limit]
                            else:
                                # No vector matches, but we have PostgreSQL matches - create results from PostgreSQL
                                for doc in image_metadata_docs[:limit]:
                                    # Query metadata_json from database
                                    from services.database_manager.database_helpers import fetch_one
                                    row = await fetch_one(
                                        "SELECT metadata_json FROM document_metadata WHERE document_id = $1",
                                        doc.document_id,
                                        rls_context=rls_context
                                    )
                                    
                                    metadata_json = {}
                                    if row and row.get("metadata_json"):
                                        metadata_json_raw = row["metadata_json"]
                                        if isinstance(metadata_json_raw, str):
                                            import json
                                            try:
                                                metadata_json = json.loads(metadata_json_raw)
                                            except:
                                                metadata_json = {}
                                    
                                    results.append({
                                        "metadata": {
                                            "document_id": doc.document_id,
                                            "title": doc.title,
                                            "date": metadata_json.get("date"),
                                            "author": doc.author or metadata_json.get("author"),
                                            "series": metadata_json.get("series"),
                                            "image_url": metadata_json.get("image_url"),
                                            "image_type": metadata_json.get("image_type") or metadata_json.get("type"),
                                            "content": metadata_json.get("content"),
                                            "tags": doc.tags or []
                                        },
                                        "score": 0.8  # High score for exact author match
                                    })
                        else:
                            # No PostgreSQL matches
                            type_msg = f" of type '{image_type}'" if image_type else ""
                            return {
                                "images_markdown": f"🔍 No images found by author '{author}'{type_msg}",
                                "metadata": []
                            }
                    except Exception as e:
                        logger.warning(f"PostgreSQL lookup failed, falling back to vector search: {e}")
                        # Fall through to vector-only search
            
            # If no author filter or PostgreSQL lookup didn't work, use pure vector search (hybrid when user_id set)
            if not results:
                filter_tags = []
                # Vector-first for scale: fetch more candidates, then type-check only that small set (no 500K-ID load)
                vector_limit = limit * 2
                if image_type_requested:
                    vector_limit = min(500, limit * 30)  # Enough candidates to type-filter without loading all type docs
                results = await embedding_manager.search_similar(
                    query_embedding=query_embeddings[0],
                    limit=vector_limit,
                    score_threshold=0.3,
                    user_id=user_id,
                    filter_category=None,
                    filter_tags=filter_tags if filter_tags else None
                )
                
                # Type-filter: only check the small set of vector result IDs (scale-friendly: one small IN query)
                if image_type_requested and results:
                    from services.database_manager.database_helpers import fetch_all
                    doc_ids_from_vector = [
                        (vr.get("metadata", {}).get("document_id") or vr.get("document_id"))
                        for vr in results
                    ]
                    doc_ids_from_vector = [did for did in doc_ids_from_vector if did]
                    if doc_ids_from_vector:
                        type_check_sql = """
                            SELECT document_id FROM document_metadata
                            WHERE document_id = ANY($1)
                            AND (metadata_json->>'image_type' = $2 OR metadata_json->>'type' = $2)
                            """
                        type_check_params = [doc_ids_from_vector, image_type_lower]
                        if series and series.strip():
                            type_check_sql += " AND metadata_json->>'series' ILIKE $3"
                            type_check_params.append(f"%{series.strip()}%")
                        type_check_rows = await fetch_all(
                            type_check_sql,
                            *type_check_params,
                            rls_context=rls_context
                        )
                        candidate_doc_ids = {row["document_id"] for row in type_check_rows}
                        series_msg = f" and series '{series}'" if series and series.strip() else ""
                        logger.info(f"🔍 Type-check (vector-first): {len(candidate_doc_ids)} of {len(doc_ids_from_vector)} vector results match type '{image_type}'{series_msg}")
                    else:
                        candidate_doc_ids = set()
                    type_filtered_results = [vr for vr in results if (vr.get("metadata", {}).get("document_id") or vr.get("document_id")) in candidate_doc_ids]
                    if type_filtered_results:
                        results = type_filtered_results
                    else:
                        # NO FALLBACK: Vector search found no semantic matches for this query among comics
                        # Better to return 0 results than random comics!
                        logger.info(f"🔍 Type-check: 0 of {len(doc_ids_from_vector)} vector results match type '{image_type}' - no semantic matches found")
                        results = []
            
            if not results:
                type_msg = f" of type '{image_type}'" if image_type else ""
                logger.warning(f"⚠️ IMAGE SEARCH: No results found for query='{query}', series={series}, author={author}, date={date}, type={image_type}")
                return {
                    "images_markdown": f"🔍 No images found matching '{query}'{type_msg}",
                    "metadata": []
                }
            
            # Filter by identity if provided (face detection)
            if identity:
                try:
                    from services.database_manager.database_helpers import fetch_all
                    # Get document IDs that have faces with this identity
                    identity_docs = await fetch_all(
                        """
                        SELECT DISTINCT df.document_id 
                        FROM detected_faces df
                        WHERE LOWER(df.identity_name) = LOWER($1)
                        AND df.identity_confirmed = true
                        """,
                        identity,
                        rls_context=rls_context
                    )
                    
                    identity_doc_ids = {row["document_id"] for row in identity_docs}
                    
                    if identity_doc_ids:
                        logger.info(f"🔍 Identity filter '{identity}': Found {len(identity_doc_ids)} documents with this face")
                        # Filter results to only include documents with this identity
                        filtered_results = []
                        for result in results:
                            doc_id = result.get("metadata", {}).get("document_id") or result.get("document_id")
                            if doc_id in identity_doc_ids:
                                filtered_results.append(result)
                        
                        if filtered_results:
                            results = filtered_results
                            logger.info(f"✅ Identity filter: {len(results)} results match identity '{identity}'")
                        else:
                            logger.warning(f"⚠️ Identity filter: No vector search results match identity '{identity}'")
                            return {
                                "images_markdown": f"🔍 No images found with face identity '{identity}' matching query '{query}'",
                                "metadata": []
                            }
                    else:
                        logger.warning(f"⚠️ Identity filter: No documents found with identity '{identity}'")
                        return {
                            "images_markdown": f"🔍 No images found with face identity '{identity}'",
                            "metadata": []
                        }
                except Exception as e:
                    logger.warning(f"Identity filter failed: {e}, continuing without identity filter")
                    # Continue without identity filter if there's an error
            
            # Filter by date if provided
            if date:
                try:
                    target_date = datetime.strptime(date, "%Y-%m-%d").date()
                    filtered_results = []
                    for result in results:
                        metadata = result.get("metadata", {})
                        result_date_str = metadata.get("date")
                        if result_date_str:
                            try:
                                result_date = datetime.strptime(result_date_str, "%Y-%m-%d").date()
                                if result_date == target_date:
                                    filtered_results.append(result)
                            except ValueError:
                                continue
                    results = filtered_results[:limit]
                except ValueError:
                    logger.warning(f"Invalid date format: {date}")
            
            # Limit results
            results = results[:limit]
            
            # Log what Qdrant actually returned
            logger.info(f"🔍 Qdrant returned {len(results)} results (before filtering):")
            for idx, r in enumerate(results, 1):
                doc_id = r.get("metadata", {}).get("document_id") or r.get("document_id")
                title = r.get("metadata", {}).get("title", "N/A")
                score = r.get("score", 0)
                logger.info(f"  {idx}. doc_id={doc_id}, title={title}, score={score:.3f}")
            
            # Smart filtering: Check relevance scores to filter out weak matches.
            # Use lower threshold when series/author/date filters are applied, since results
            # have already passed those filters and are semantically relevant within that set.
            MIN_SCORE_THRESHOLD = 0.25 if (series or author or date) else 0.35
            results = [r for r in results if r.get("score", 0) >= MIN_SCORE_THRESHOLD]
            
            if not results:
                type_msg = f" of type '{image_type}'" if image_type else ""
                date_msg = f" on {date}" if date else ""
                logger.warning(f"⚠️ IMAGE SEARCH: No results with score >= {MIN_SCORE_THRESHOLD} for query='{query}'")
                return {
                    "images_markdown": f"🔍 No highly relevant images found matching '{query}'{type_msg}{date_msg}",
                    "metadata": []
                }
            
            # When multiple results have similar scores (e.g. Garfield lasagna comics), return top 2-3+
            # Only trim to a single result when there is a very clear winner (gap > 0.25)
            CLEAR_WINNER_GAP = 0.25  # Only return 1 result when top score leads second by this much
            if len(results) > 1:
                top_score = results[0].get("score", 0)
                second_score = results[1].get("score", 0)
                score_gap = top_score - second_score
                if score_gap > CLEAR_WINNER_GAP:
                    logger.info(f"🎯 Top result clearly better: {top_score:.3f} vs {second_score:.3f} (gap: {score_gap:.3f})")
                    logger.info(f"✅ Returning only top result (clear winner)")
                    results = [results[0]]
                else:
                    logger.info(f"🎯 Multiple results with close scores: top={top_score:.3f}, second={second_score:.3f} (gap: {score_gap:.3f})")
                    logger.info(f"✅ Returning top {len(results)} results")
            
            # Enrich results with PostgreSQL metadata (vector search doesn't have image_url!)
            # The image_url is stored in PostgreSQL metadata_json, not in Qdrant vector metadata
            # document_id can be in metadata (merged by vector_store) or at top level
            enriched_results = []
            for idx, result in enumerate(results, 1):
                metadata = result.get("metadata", {})
                document_id = metadata.get("document_id") or result.get("document_id")
                title = metadata.get("title", "")
                
                logger.info(f"🔍 Enriching result {idx}: doc_id={document_id}, title={title}")
                
                # If document_id is None, try to look it up by title
                if not document_id and title:
                    from services.database_manager.database_helpers import fetch_one
                    logger.info(f"⚠️ Result {idx} has no document_id, looking up by title: {title}")
                    
                    # Look up document by title to get document_id
                    doc_row = await fetch_one(
                        """
                        SELECT dm.document_id 
                        FROM document_metadata dm 
                        WHERE dm.title = $1 
                        LIMIT 1
                        """,
                        title,
                        rls_context=rls_context
                    )
                    
                    if doc_row:
                        document_id = doc_row.get("document_id")
                        logger.info(f"✅ Found document_id by title: {document_id}")
                    else:
                        logger.warning(f"⚠️ Could not find document_id for title: {title}")
                
                if document_id:
                    # Fetch full metadata from PostgreSQL (user RLS: global + own)
                    from services.database_manager.database_helpers import fetch_one
                    row = await fetch_one(
                        "SELECT metadata_json FROM document_metadata WHERE document_id = $1",
                        document_id,
                        rls_context=rls_context
                    )
                    
                    if row and row.get("metadata_json"):
                        # Merge PostgreSQL metadata with Qdrant metadata
                        postgres_metadata = row["metadata_json"]
                        if isinstance(postgres_metadata, str):
                            import json
                            try:
                                postgres_metadata = json.loads(postgres_metadata)
                            except:
                                postgres_metadata = {}
                        
                        # Get the actual image filename from metadata (source of truth!)
                        image_filename = postgres_metadata.get("image_filename", "")
                        logger.info(f"✅ Result {idx}: image_filename from metadata = {image_filename}")
                        
                        # Merge - PostgreSQL metadata takes precedence for image-specific fields
                        enriched_metadata = {**metadata, **postgres_metadata, "document_id": document_id}
                        enriched_results.append({**result, "metadata": enriched_metadata})
                    else:
                        logger.warning(f"⚠️ No metadata_json found for document_id: {document_id}")
                        enriched_results.append(result)
                else:
                    logger.warning(f"⚠️ Result {idx} has no document_id, cannot enrich")
                    enriched_results.append(result)
            
            results = enriched_results
            
            # Format results as markdown
            # Limit base64 embedding to top 3 images to avoid huge responses
            # Vector search already ranks by relevance, so top 3 are the most relevant
            max_images_to_embed = 3
            # CRITICAL: Limit results to max_images_to_embed to ensure 1:1 correspondence
            # Metadata is the source of truth - each metadata entry must have a corresponding image
            results = results[:max_images_to_embed]
            total_results = len(results)
            
            type_label = image_type.title() if image_type else "Image"
            # Start with clean results - no header text, just images
            formatted_results = []
            metadata_list = []  # NEW: Collect metadata for LLM
            structured_images = []  # NEW: Collect structured image data for AgentResponse contract
            
            for i, result in enumerate(results, 1):
                metadata = result.get("metadata", {})
                logger.info(f"🔍 Result {i} enriched metadata keys: {list(metadata.keys())}")
                
                title = metadata.get("title", "Untitled Image")
                date_str = metadata.get("date", "")
                image_url = metadata.get("image_url", "")
                image_filename = metadata.get("image_filename", "")  # Source of truth!
                author_str = metadata.get("author", "")
                series = metadata.get("series", "")
                content = metadata.get("content", "")
                image_type_str = metadata.get("image_type") or metadata.get("type") or "other"
                tags = metadata.get("tags", [])
                if isinstance(tags, str):
                    # Handle tags as comma-separated string
                    tags = [t.strip() for t in tags.split(",") if t.strip()]
                elif not isinstance(tags, list):
                    tags = []
                
                logger.info(f"🔍 Result {i}: title={title}, image_url={image_url}, image_filename={image_filename}, image_type={image_type_str}")
                
                # NEW: Collect metadata for LLM (full information)
                metadata_list.append({
                    "title": title,
                    "date": date_str,
                    "series": series,
                    "author": author_str,
                    "content": content,  # Full description
                    "tags": tags,
                    "image_type": image_type_str
                })
                
                # Add image directly - use document API endpoint instead of constructing file paths
                # This ensures images are served from their actual location (same folder as metadata)
                # regardless of directory structure
                should_embed = i <= max_images_to_embed
                
                # Always use /api/documents/{doc_id}/file endpoint
                # This handles auth, folder resolution, and file location automatically
                document_id = metadata.get("document_id")
                if document_id and image_filename:
                    # Use document API endpoint - it handles folder structure internally
                    image_url = f"/api/documents/{document_id}/file"
                    logger.info(f"✅ Using document API endpoint for image: {image_url}")
                
                if image_url:
                    # Always use URL - let the frontend handle loading via document API
                    formatted_results.append(f"![]({image_url})\n")
                    
                    # Add to structured images list with URL
                    structured_images.append({
                        "url": image_url,
                        "alt_text": title or f"{type_label} {i}",
                        "type": "search_result",
                        "metadata": {
                            "title": title,
                            "date": date_str,
                            "series": series,
                            "author": author_str,
                            "image_type": image_type_str,
                            "tags": tags,
                            "document_id": metadata.get("document_id"),
                            "image_filename": image_filename
                        }
                    })
                    logger.info(f"✅ Added image URL: {image_url}")
                elif image_url:
                    # For images beyond the embedding limit, just include URL reference
                    formatted_results.append(f"![]({image_url})\n")
                    
                    # Add to structured images list with URL
                    structured_images.append({
                        "url": image_url,
                        "alt_text": title or f"{type_label} {i}",
                        "type": "search_result",
                        "metadata": {
                            "title": title,
                            "date": date_str,
                            "series": series,
                            "author": author_str,
                            "image_type": image_type_str,
                            "tags": tags,
                            "content": content,  # Full description for modal
                            "document_id": metadata.get("document_id"),
                            "image_filename": image_filename
                        }
                    })
                    logger.debug(f"📋 Image {i} metadata included (beyond embedding limit of {max_images_to_embed})")
                
                # No content snippet or metadata - keep it clean with just images
            
            # Return structured response: images markdown + metadata list + structured images
            images_markdown = "".join(formatted_results)
            
            # Return dict with images markdown, metadata, and structured images
            return {
                "images_markdown": images_markdown,
                "metadata": metadata_list,
                "images": structured_images  # Structured image data for AgentResponse contract
            }
            
        except Exception as e:
            logger.error(f"❌ IMAGE SEARCH FAILED: {e}")
            import traceback
            logger.error(f"❌ IMAGE SEARCH TRACEBACK: {traceback.format_exc()}")
            return {
                "images_markdown": f"❌ Image search failed: {str(e)}",
                "metadata": []
            }
    
    async def _exact_match_search(
        self,
        series: Optional[str] = None,
        author: Optional[str] = None,
        date: Optional[str] = None,
        image_type: Optional[str] = None,
        limit: int = 10,
        rls_context: Optional[Dict[str, str]] = None
    ) -> List[Any]:
        """
        Helper method: Exact match search by series/author/date.
        Uses passed rls_context for user+global visibility when provided.
        """
        if rls_context is None:
            rls_context = {"user_id": "", "user_role": "admin"}
        document_repo = await self._get_document_repository()
        if not document_repo:
            return []
        
        image_metadata_docs = []
        
        if series:
            series_lower = series.lower()
            image_categories = [
                DocumentCategory.COMIC,
                DocumentCategory.ENTERTAINMENT,
                DocumentCategory.TECHNICAL,
                DocumentCategory.MEDICAL,
                DocumentCategory.REFERENCE,
                DocumentCategory.OTHER
            ]
            
            for cat in image_categories:
                filter_request = DocumentFilterRequest(
                    category=cat,
                    limit=limit * 20
                )
                postgres_results, _ = await document_repo.filter_documents(filter_request, rls_context=rls_context)
                
                for doc in postgres_results:
                    from services.database_manager.database_helpers import fetch_one
                    row = await fetch_one(
                        "SELECT metadata_json FROM document_metadata WHERE document_id = $1",
                        doc.document_id,
                        rls_context=rls_context
                    )
                    
                    if not row or not row.get("metadata_json"):
                        continue
                    
                    metadata_json = row["metadata_json"]
                    if isinstance(metadata_json, str):
                        import json
                        try:
                            metadata_json = json.loads(metadata_json)
                        except:
                            continue
                    
                    if not metadata_json.get("has_searchable_metadata"):
                        continue
                    
                    doc_series = (metadata_json.get("series") or "").lower()
                    if doc_series and series_lower in doc_series:
                        image_metadata_docs.append(doc)
        
        # Apply date filter if provided
        if date and image_metadata_docs:
            try:
                target_date = datetime.strptime(date, "%Y-%m-%d").date()
                date_filtered_docs = []
                for doc in image_metadata_docs:
                    from services.database_manager.database_helpers import fetch_one
                    row = await fetch_one(
                        "SELECT metadata_json FROM document_metadata WHERE document_id = $1",
                        doc.document_id,
                        rls_context=rls_context
                    )
                    
                    if not row or not row.get("metadata_json"):
                        continue
                    
                    metadata_json = row["metadata_json"]
                    if isinstance(metadata_json, str):
                        import json
                        try:
                            metadata_json = json.loads(metadata_json)
                        except:
                            continue
                    
                    doc_date_str = metadata_json.get("date")
                    if doc_date_str:
                        try:
                            doc_date = datetime.strptime(doc_date_str, "%Y-%m-%d").date()
                            if doc_date == target_date:
                                date_filtered_docs.append(doc)
                        except ValueError:
                            continue
                    else:
                        date_filtered_docs.append(doc)
                
                image_metadata_docs = date_filtered_docs
            except ValueError:
                pass
        
        return image_metadata_docs[:limit * 5]  # Return more candidates for merging
    
    async def _text_search_fallback(
        self,
        query: str,
        series: Optional[str] = None,
        author: Optional[str] = None,
        date: Optional[str] = None,
        image_type: Optional[str] = None,
        limit: int = 10,
        rls_context: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        SQL text search fallback when vector search returns 0 results.
        Searches content and tags fields using PostgreSQL text matching.
        Uses passed rls_context for user+global visibility when provided.
        """
        if rls_context is None:
            rls_context = {"user_id": "", "user_role": "admin"}
        document_repo = await self._get_document_repository()
        if not document_repo:
            return []
        
        image_categories = [
            DocumentCategory.COMIC,
            DocumentCategory.ENTERTAINMENT,
            DocumentCategory.TECHNICAL,
            DocumentCategory.MEDICAL,
            DocumentCategory.REFERENCE,
            DocumentCategory.OTHER
        ]
        
        results = []
        logger.info(f"🔍 TEXT SEARCH FALLBACK: Starting search for '{query}' with filters (series={series}, author={author}, date={date}, type={image_type})")
        
        for cat in image_categories:
            filter_request = DocumentFilterRequest(
                category=cat,
                limit=limit * 50  # Get many candidates for text filtering
            )
            
            postgres_results, _ = await document_repo.filter_documents(filter_request, rls_context=rls_context)
            logger.info(f"🔍 TEXT SEARCH: Category {cat.value} returned {len(postgres_results)} candidates")
            
            for idx, doc in enumerate(postgres_results):
                if idx < 3:  # Log first 3 docs for debugging
                    logger.info(f"  📄 Document {idx}: {doc.title}, id={doc.document_id}")
                if idx < 3:  # Log first 3 docs for debugging
                    logger.info(f"  📄 Document {idx}: {doc.title}, id={doc.document_id}")
                
                from services.database_manager.database_helpers import fetch_one
                try:
                    row = await fetch_one(
                        "SELECT metadata_json FROM document_metadata WHERE document_id = $1",
                        doc.document_id,
                        rls_context=rls_context
                    )
                    if idx < 3:
                        logger.info(f"  🔍 Metadata fetch for {idx}: row={'exists' if row else 'NULL'}, has_metadata_json={bool(row and row.get('metadata_json')) if row else False}")
                    
                except Exception as e:
                    logger.error(f"  ❌ Metadata fetch FAILED for {doc.title}: {e}")
                    continue
                
                if not row or not row.get("metadata_json"):
                    if idx < 3:
                        logger.info(f"  ⏭️ SKIP {idx}: No metadata row")
                    continue
                
                metadata_json = row["metadata_json"]
                if isinstance(metadata_json, str):
                    import json
                    try:
                        metadata_json = json.loads(metadata_json)
                    except:
                        if idx < 3:
                            logger.info(f"  ⏭️ SKIP {idx}: JSON parse failed")
                        continue
                
                if not metadata_json.get("has_searchable_metadata"):
                    if idx < 3:
                        logger.info(f"  ⏭️ SKIP {idx}: has_searchable_metadata=False")
                    continue
                
                if idx < 3:
                    logger.info(f"  ✅ Doc {idx} PASSED has_searchable_metadata check")
                
                # Apply series filter if provided
                if series:
                    doc_series = (metadata_json.get("series") or "").lower()
                    if series.lower() not in doc_series:
                        if idx < 3:
                            logger.info(f"  ⏭️ SKIP {idx}: series mismatch (want='{series}', got='{doc_series}')")
                        continue
                    if idx < 3:
                        logger.info(f"  ✅ Doc {idx} PASSED series check (series='{doc_series}')")
                
                # Apply author filter if provided
                if author:
                    doc_author = (metadata_json.get("author") or doc.author or "").lower()
                    if author.lower() not in doc_author:
                        continue
                
                # Apply date filter if provided
                if date:
                    doc_date = metadata_json.get("date")
                    if doc_date != date:
                        continue
                
                # Apply image type filter if provided
                # ONLY filter if BOTH query wants a type AND document HAS a type
                # Skip docs with empty type field (database not synced yet)
                if image_type:
                    doc_type = (metadata_json.get("type") or "").lower()
                    # Only apply filter if doc has a type - empty type means "not indexed yet"
                    if doc_type and image_type.lower() != doc_type:
                        if idx < 3:
                            logger.info(f"  ⏭️ SKIP {idx}: type mismatch (want='{image_type}', got='{doc_type}')")
                        continue
                    if idx < 3:
                        if not doc_type:
                            logger.info(f"  ⚠️ Doc {idx} has empty type field (skipping type filter)")
                        else:
                            logger.info(f"  ✅ Doc {idx} PASSED type check (type='{doc_type}')")
                
                # Apply text search on content and tags
                # Check if query terms appear in content or tags
                content = (metadata_json.get("content") or "").lower()
                tags_list = doc.tags or []
                tags_text = " ".join(tags_list).lower()
                
                if idx < 3:
                    logger.info(f"  🔎 Doc {idx} checking text: content_len={len(content)}, tags={tags_list[:3]}")
                
                # Split query into terms and check if ANY term matches
                # This handles plurals somewhat (e.g., "brains" will match if "brain" in content)
                query_terms = query.lower().split()
                matches = False
                
                for term in query_terms:
                    # Check both exact and stemmed versions
                    # Simple stemming: remove common endings
                    stem = term.rstrip('s').rstrip('es').rstrip('ing').rstrip('ed')
                    
                    if (term in content or term in tags_text or 
                        stem in content or stem in tags_text):
                        matches = True
                        if idx < 3:
                            logger.info(f"  ✅ MATCH in Doc {idx}: term='{term}', stem='{stem}'")
                        break
                
                if not matches:
                    if idx < 3:
                        logger.info(f"  ⏭️ SKIP {idx}: no text match for query='{query}'")
                    continue
                
                if matches:
                    results.append({
                        "metadata": {
                            "document_id": doc.document_id,
                            "title": doc.title,
                            "date": metadata_json.get("date"),
                            "author": doc.author or metadata_json.get("author"),
                            "series": metadata_json.get("series"),
                            "image_url": metadata_json.get("image_url"),
                            "image_type": metadata_json.get("image_type") or metadata_json.get("type"),
                            "content": metadata_json.get("content"),
                            "tags": doc.tags or []
                        },
                        "score": 0.5,  # Lower score for text search fallback
                        "matched_both": False
                    })
                    
                    if len(results) >= limit * 2:
                        break
            
            if len(results) >= limit * 2:
                break
        
        logger.info(f"📝 Text search fallback found {len(results)} results for query '{query}'")
        return results[:limit * 2]
    
    async def _random_image_search(
        self,
        image_type: Optional[str] = None,
        author: Optional[str] = None,
        series: Optional[str] = None,
        identity: Optional[str] = None,
        limit: int = 1,
        rls_context: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Get random image(s) with optional filters.
        Uses passed rls_context for user+global visibility when provided.
        """
        if rls_context is None:
            rls_context = {"user_id": "", "user_role": "admin"}
        try:
            logger.info(f"🎲 RANDOM IMAGE SEARCH: type={image_type}, author={author}, series={series}, limit={limit}")
            
            from services.database_manager.database_helpers import fetch_all
            
            # Build SQL query with filters
            query_parts = ["SELECT dm.document_id, dm.title, dm.author, dm.metadata_json"]
            query_parts.append("FROM document_metadata dm")
            query_parts.append("WHERE dm.metadata_json IS NOT NULL")
            query_parts.append("AND dm.metadata_json->>'has_searchable_metadata' = 'true'")
            
            params = []
            param_counter = 1
            
            # Filter by image type (category)
            if image_type:
                image_type_lower = image_type.lower().strip()
                if image_type_lower in self.TYPE_TO_CATEGORY:
                    filter_category = self.TYPE_TO_CATEGORY[image_type_lower].value
                    query_parts.append(f"AND dm.category = ${param_counter}")
                    params.append(filter_category)
                    param_counter += 1
            
            # Filter by series (e.g., "Dilbert")
            if series:
                query_parts.append(f"AND dm.metadata_json->>'series' ILIKE ${param_counter}")
                params.append(f"%{series}%")
                param_counter += 1
            
            # Filter by author
            if author:
                query_parts.append(f"AND (dm.author ILIKE ${param_counter} OR dm.metadata_json->>'author' ILIKE ${param_counter})")
                params.append(f"%{author}%")
                param_counter += 1
            
            # Filter by identity (face detection)
            if identity:
                query_parts.append(f"""
                    AND EXISTS (
                        SELECT 1 FROM detected_faces df
                        WHERE df.document_id = dm.document_id
                        AND LOWER(df.identity_name) = LOWER(${param_counter})
                        AND df.identity_confirmed = true
                    )
                """)
                params.append(identity)
                param_counter += 1
            
            # Random selection
            query_parts.append("ORDER BY RANDOM()")
            query_parts.append(f"LIMIT {limit}")
            
            sql = " ".join(query_parts)
            logger.info(f"🎲 Random query SQL: {sql}")
            logger.info(f"🎲 Parameters: {params}")
            
            rows = await fetch_all(sql, *params, rls_context=rls_context)
            
            if not rows:
                logger.info(f"🎲 No random images found with filters: type={image_type}, author={author}, series={series}")
                return {
                    "images_markdown": "No images found matching criteria.",
                    "metadata": []
                }
            
            logger.info(f"🎲 Found {len(rows)} random image(s)")
            
            # Format results using the same formatting logic as regular search
            formatted_results = []
            metadata_list = []  # NEW: Collect metadata for LLM
            
            for idx, row in enumerate(rows, 1):
                document_id = row.get("document_id")
                title = row.get("title", "Untitled")
                metadata_json = row.get("metadata_json", {})
                
                if isinstance(metadata_json, str):
                    import json
                    try:
                        metadata_json = json.loads(metadata_json)
                    except:
                        metadata_json = {}
                
                # Get image details
                image_filename = metadata_json.get("image_filename", "")
                image_url = metadata_json.get("image_url", "")
                date_str = metadata_json.get("date", "Unknown date")
                author_name = row.get("author") or metadata_json.get("author", "Unknown author")
                series_name = metadata_json.get("series", "")
                content = metadata_json.get("content", "")
                tags = metadata_json.get("tags", [])
                if isinstance(tags, str):
                    tags = [t.strip() for t in tags.split(",") if t.strip()]
                elif not isinstance(tags, list):
                    tags = []
                image_type_str = metadata_json.get("image_type", "other")
                
                # NEW: Collect metadata for LLM
                metadata_list.append({
                    "title": title,
                    "date": date_str if date_str != "Unknown date" else "",
                    "series": series_name,
                    "author": author_name if author_name != "Unknown author" else "",
                    "content": content,
                    "tags": tags,
                    "image_type": image_type_str
                })
                
                # Construct image_url if not present
                if not image_url and image_filename and series_name and date_str and date_str != "Unknown date":
                    try:
                        from datetime import datetime
                        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                        year = date_obj.year
                        month = date_obj.month
                        # Assume /api/comics path structure for comics
                        image_url = f"/api/comics/{series_name}/{year}/{month:02d}/{image_filename}"
                        logger.info(f"Constructed image_url: {image_url}")
                    except:
                        pass
                
                # Embed image if URL available (use same file path logic as regular search)
                if image_url:
                    try:
                        # Resolve image file path from URL (same logic as regular search)
                        from pathlib import Path
                        from config import settings
                        import mimetypes
                        import base64
                        
                        image_file_path = None
                        
                        # Handle /api/comics/ URLs
                        if image_url.startswith('/api/comics/'):
                            relative_path = image_url.replace('/api/comics/', '')
                            # Handle duplicate "Comics" in path (legacy URLs)
                            if relative_path.startswith('Comics/'):
                                relative_path = relative_path.replace('Comics/', '', 1)
                            comics_root = Path(settings.UPLOAD_DIR) / "Global" / "Comics"
                            image_file_path = comics_root / relative_path
                            logger.info(f"🔍 Comics path: {image_file_path} (exists: {image_file_path.exists()})")
                        
                        # Handle /api/images/ URLs
                        elif image_url.startswith('/api/images/'):
                            relative_path = image_url.replace('/api/images/', '')
                            # Try Global directory first
                            global_root = Path(settings.UPLOAD_DIR) / "Global"
                            image_file_path = global_root / relative_path
                            if not image_file_path.exists():
                                # Fallback to web_sources/images
                                web_images = Path(settings.UPLOAD_DIR) / "web_sources" / "images"
                                image_file_path = web_images / relative_path
                        
                        # If we have a valid file path, embed as base64
                        if image_file_path and image_file_path.exists():
                            with open(image_file_path, 'rb') as f:
                                image_data = base64.b64encode(f.read()).decode('utf-8')
                            
                            # Detect MIME type
                            mime_type, _ = mimetypes.guess_type(str(image_file_path))
                            if not mime_type:
                                # Fallback based on extension
                                ext = image_file_path.suffix.lower()
                                mime_map = {
                                    '.gif': 'image/gif',
                                    '.png': 'image/png',
                                    '.jpg': 'image/jpeg',
                                    '.jpeg': 'image/jpeg',
                                    '.webp': 'image/webp'
                                }
                                mime_type = mime_map.get(ext, 'image/png')
                            
                            # Embed as data URI with empty alt text
                            data_uri = f"data:{mime_type};base64,{image_data}"
                            formatted_results.append(f"![]({data_uri})\n")
                            logger.info(f"✅ Embedded image as base64: {image_file_path.name} ({mime_type})")
                        else:
                            # Fallback to URL if file not found
                            formatted_results.append(f"![]({image_url})\n")
                            logger.warning(f"⚠️ Image file not found, using URL: {image_url}")
                    except Exception as e:
                        # If embedding fails, fallback to URL
                        logger.error(f"Failed to embed random image {title}: {e}")
                        formatted_results.append(f"![]({image_url})\n")
                else:
                    # No image URL available
                    logger.warning(f"⚠️ No image_url for random image: {title}")
            
            if not formatted_results:
                return {
                    "images_markdown": "No images could be loaded.",
                    "metadata": []
                }
            
            images_markdown = "\n".join(formatted_results)
            return {
                "images_markdown": images_markdown,
                "metadata": metadata_list
            }
            
        except Exception as e:
            logger.error(f"❌ Random image search failed: {e}")
            import traceback
            logger.error(f"❌ TRACEBACK: {traceback.format_exc()}")
            return {
                "images_markdown": f"❌ Random image search failed: {str(e)}",
                "metadata": []
            }


# Global instance for use by tool registry
_image_search_instance = None


async def _get_image_search():
    """Get global image search instance"""
    global _image_search_instance
    if _image_search_instance is None:
        _image_search_instance = ImageSearchTools()
    return _image_search_instance


async def search_images(
    query: str,
    image_type: Optional[str] = None,
    date: Optional[str] = None,
    author: Optional[str] = None,
    series: Optional[str] = None,
    limit: int = 10,
    user_id: Optional[str] = None,
    is_random: bool = False
) -> Dict[str, Any]:
    """
    LangGraph tool function: Search for images with metadata.
    When user_id is provided, searches both the user's collection and global (hybrid search).
    
    Args:
        query: Search query for vector search
        image_type: Optional filter by type
        date: Optional date filter (YYYY-MM-DD)
        author: Optional author filter
        series: Optional series filter
        limit: Maximum results
        user_id: When set, run hybrid search (user + global) and apply user RLS
        is_random: If True, return random images instead of semantic search
        
    Returns:
        Dict with 'images_markdown' (base64 embedded images) and 'metadata' (list of metadata dicts)
    """
    instance = await _get_image_search()
    return await instance.search_images(
        query=query,
        image_type=image_type,
        date=date,
        author=author,
        series=series,
        limit=limit,
        user_id=user_id,
        is_random=is_random
    )

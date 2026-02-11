"""
Vector Store Service - Manages vector database operations (Qdrant)

Handles all vector storage, retrieval, and collection management operations.
Routes all operations through Vector Service gRPC for centralized Qdrant access.
"""

import asyncio
import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from qdrant_client.models import (
    PointStruct, Distance, VectorParams, Filter,
    FieldCondition, MatchValue, ScrollRequest
)

from config import settings
from clients.vector_service_client import get_vector_service_client

logger = logging.getLogger(__name__)


class VectorStoreService:
    """
    Vector database service for storage and retrieval operations
    
    Responsibilities:
    - Collection management (create, delete, list)
    - Point insertion and deletion
    - Vector similarity search
    - Hybrid search across collections
    - User-specific collection isolation
    """
    
    def __init__(self):
        self.vector_service_client = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize Vector Service client"""
        if self._initialized:
            return
            
        logger.info("Initializing Vector Store Service (via Vector Service)...")
        
        self.vector_service_client = await get_vector_service_client(required=False)
        
        # Mark as initialized before ensuring collection to prevent recursion
        self._initialized = True
        
        # Ensure default collection exists
        await self.ensure_collection_exists(settings.VECTOR_COLLECTION_NAME)
        
        logger.debug("Vector Store Service initialized (routing through Vector Service)")
    
    def _get_user_collection_name(self, user_id: str) -> str:
        """Generate collection name for a specific user"""
        return f"user_{user_id}_documents"
    
    def _get_team_collection_name(self, team_id: str) -> str:
        """Generate collection name for a specific team"""
        return f"team_{team_id}"
    
    async def ensure_collection_exists(self, collection_name: str) -> bool:
        """Ensure a collection exists, create if it doesn't"""
        try:
            # Ensure client is available (but don't call initialize() if already initialized to avoid recursion)
            if not self.vector_service_client:
                if not self._initialized:
                    # Only initialize if truly not initialized
                    await self.initialize()
                    # After initialize(), _initialized is True, so we won't recurse
                else:
                    # If initialized but client is None, get it directly
                    self.vector_service_client = await get_vector_service_client(required=False)
            
            if not self.vector_service_client:
                raise RuntimeError("Vector Service client not available")
            
            # Check if collection exists
            collections_result = await self.vector_service_client.list_collections()
            if not collections_result.get("success"):
                logger.warning(f"Failed to list collections: {collections_result.get('error')}")
                return False
            
            collection_names = [col["name"] for col in collections_result.get("collections", [])]
            
            if collection_name not in collection_names:
                # Create collection via Vector Service
                create_result = await self.vector_service_client.create_collection(
                    collection_name=collection_name,
                    vector_size=settings.EMBEDDING_DIMENSIONS,
                    distance="COSINE"
                )
                if create_result.get("success"):
                    logger.info(f"Created vector collection: {collection_name}")
                    return True
                else:
                    error = create_result.get("error", "Unknown error")
                    logger.error(f"Failed to create collection {collection_name}: {error}")
                    return False
            # Collection already exists
            return True
        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {e}")
            raise
    
    async def ensure_user_collection_exists(self, user_id: str) -> bool:
        """Ensure user-specific collection exists"""
        collection_name = self._get_user_collection_name(user_id)
        return await self.ensure_collection_exists(collection_name)
    
    async def ensure_team_collection_exists(self, team_id: str) -> bool:
        """Ensure team-specific collection exists"""
        collection_name = self._get_team_collection_name(team_id)
        return await self.ensure_collection_exists(collection_name)
    
    async def insert_points(
        self,
        points: List[PointStruct],
        collection_name: Optional[str] = None,
        max_retries: int = 3
    ) -> bool:
        """
        Insert points into vector collection with retry logic
        
        Args:
            points: List of PointStruct objects to insert
            collection_name: Target collection (defaults to global collection)
            max_retries: Maximum number of retry attempts
            
        Returns:
            True if successful, False otherwise
        """
        if not self._initialized:
            await self.initialize()
        
        if not collection_name:
            collection_name = settings.VECTOR_COLLECTION_NAME
        
        if not self._initialized:
            await self.initialize()
        
        if not points:
            logger.warning("No points provided for insertion")
            return False
        
        # Convert PointStruct to dict format for Vector Service
        points_dict = []
        for point in points:
            # Convert payload to dict (handle any complex types)
            payload_dict = {}
            for key, value in point.payload.items():
                payload_dict[key] = value
            
            points_dict.append({
                "id": point.id,
                "vector": point.vector,
                "payload": payload_dict
            })
        
        # Batch insertion for large point sets
        batch_size = 100
        total_points = len(points_dict)
        
        for attempt in range(max_retries):
            try:
                if total_points <= batch_size:
                    # Single batch
                    result = await self.vector_service_client.upsert_vectors(
                        collection_name=collection_name,
                        points=points_dict
                    )
                    if result.get("success"):
                        logger.info(f"Inserted {total_points} points into {collection_name}")
                        return True
                    else:
                        error = result.get("error", "Unknown error")
                        raise Exception(f"Vector Service upsert failed: {error}")
                else:
                    # Multiple batches
                    for i in range(0, total_points, batch_size):
                        batch = points_dict[i:i + batch_size]
                        result = await self.vector_service_client.upsert_vectors(
                            collection_name=collection_name,
                            points=batch
                        )
                        if not result.get("success"):
                            error = result.get("error", "Unknown error")
                            raise Exception(f"Vector Service upsert failed for batch {i//batch_size + 1}: {error}")
                        logger.info(f"Inserted batch {i//batch_size + 1}: {len(batch)} points")
                
                return True
                
            except Exception as e:
                logger.error(f"Insert attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"Will retry in {wait_time}s (attempt {attempt + 2}/{max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Failed to insert points after {max_retries} attempts")
                    return False
        
        return False
    
    async def search_similar(
        self,
        query_embedding: List[float],
        limit: int = 50,
        score_threshold: float = 0.7,
        user_id: Optional[str] = None,
        team_ids: Optional[List[str]] = None,
        collection_name: Optional[str] = None,
        filter_category: Optional[str] = None,
        filter_tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in collection(s)
        
        Args:
            query_embedding: Query vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            user_id: If provided, searches user collection
            team_ids: List of team IDs to search (searches team collections)
            collection_name: Specific collection to search (overrides user_id/team_ids)
            filter_category: Filter by document category
            filter_tags: Filter by document tags
            
        Returns:
            List of search results with scores and metadata
        """
        try:
            if collection_name:
                # Search specific collection
                return await self._search_collection(
                    collection_name, query_embedding, limit, score_threshold,
                    filter_category, filter_tags
                )
            elif user_id or team_ids:
                # Hybrid search: user + team + global collections
                return await self._hybrid_search(
                    query_embedding, user_id, team_ids, limit, score_threshold,
                    filter_category, filter_tags
                )
            else:
                # Search global collection only
                return await self._search_collection(
                    settings.VECTOR_COLLECTION_NAME, query_embedding, 
                    limit, score_threshold, filter_category, filter_tags
                )
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    async def _search_collection(
        self,
        collection_name: str,
        query_embedding: List[float],
        limit: int,
        score_threshold: float,
        filter_category: Optional[str] = None,
        filter_tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search a specific collection via Vector Service"""
        try:
            if not self._initialized:
                await self.initialize()
            
            # Build filters for Vector Service
            filters = []
            if filter_category:
                filters.append({
                    "field": "document_category",
                    "value": filter_category,
                    "operator": "equals"
                })
            if filter_tags:
                for tag in filter_tags:
                    filters.append({
                        "field": "document_tags",
                        "value": tag,
                        "operator": "equals"
                    })
            
            # Execute search via Vector Service
            search_results = await self.vector_service_client.search_vectors(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                filters=filters if filters else None
            )
            
            # Get collection info for diagnostics
            try:
                col_info = await self.vector_service_client.get_collection_info(collection_name)
                if col_info.get("success") and col_info.get("collection"):
                    col = col_info["collection"]
                    logger.info(f"Collection {collection_name}: {col.get('points_count', 0)} points, vector size: {col.get('vector_size', 0)}")
            except Exception as e:
                logger.warning(f"Could not get collection info: {e}")
            
            # Format results to match expected format
            results = []
            for hit in search_results:
                payload = hit.get("payload", {})
                # Handle content - ensure it's always a string (may be dict from JSON parsing)
                content_raw = payload.get('content', '')
                if isinstance(content_raw, dict):
                    # If content is a dict, try to extract text or convert to string
                    content = content_raw.get("text", content_raw.get("content", str(content_raw)))
                elif isinstance(content_raw, str):
                    content = content_raw
                else:
                    content = str(content_raw) if content_raw else ""
                
                # Merge top-level payload into metadata so consumers (e.g. image search)
                # that expect result["metadata"]["document_id"] and result["metadata"]["title"] get them
                chunk_metadata = payload.get('metadata', {}) or {}
                merged_metadata = dict(chunk_metadata)
                if payload.get('document_id') is not None:
                    merged_metadata['document_id'] = payload.get('document_id')
                if payload.get('document_title') is not None:
                    merged_metadata['title'] = payload.get('document_title')
                if payload.get('document_author') is not None:
                    merged_metadata['author'] = payload.get('document_author')
                if payload.get('document_category') is not None:
                    merged_metadata['document_category'] = payload.get('document_category')
                result = {
                    'id': hit.get("id"),
                    'score': hit.get("score", 0.0),
                    'chunk_id': payload.get('chunk_id'),
                    'document_id': payload.get('document_id'),
                    'content': content,
                    'chunk_index': payload.get('chunk_index', 0),
                    'metadata': merged_metadata,
                    'collection': collection_name
                }
                results.append(result)
            
            logger.info(f"Found {len(results)} results in {collection_name}")
            return results
            
        except Exception as e:
            logger.error(f"Collection search failed for {collection_name}: {e}")
            return []
    
    async def _hybrid_search(
        self,
        query_embedding: List[float],
        user_id: Optional[str],
        team_ids: Optional[List[str]],
        limit: int,
        score_threshold: float,
        filter_category: Optional[str] = None,
        filter_tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search global, user, and team collections, combining results"""
        try:
            logger.info(f"Hybrid search for user {user_id}, teams {team_ids}")
            
            # Calculate per-collection limit
            num_collections = 1  # global
            if user_id:
                num_collections += 1
            if team_ids:
                num_collections += len(team_ids)
            
            per_collection_limit = max(limit // num_collections, 10)
            
            # Create search tasks
            tasks = []
            
            # Global collection
            global_task = asyncio.create_task(
                self._search_collection(
                    settings.VECTOR_COLLECTION_NAME,
                    query_embedding,
                    per_collection_limit,
                    score_threshold,
                    filter_category,
                    filter_tags
                )
            )
            tasks.append(("global", global_task))
            
            # User collection
            if user_id:
                user_collection_name = self._get_user_collection_name(user_id)
                user_task = asyncio.create_task(
                    self._search_collection(
                        user_collection_name,
                        query_embedding,
                        per_collection_limit,
                        score_threshold,
                        filter_category,
                        filter_tags
                    )
                )
                tasks.append(("user", user_task))
            
            # Team collections
            if team_ids:
                for team_id in team_ids:
                    team_collection_name = self._get_team_collection_name(team_id)
                    team_task = asyncio.create_task(
                        self._search_collection(
                            team_collection_name,
                            query_embedding,
                            per_collection_limit,
                            score_threshold,
                            filter_category,
                            filter_tags
                        )
                    )
                    tasks.append((f"team_{team_id}", team_task))
            
            # Wait for all searches
            results_dict = {}
            for source, task in tasks:
                try:
                    results = await task
                    results_dict[source] = results
                except Exception as e:
                    logger.warning(f"{source} search failed: {e}")
                    results_dict[source] = []
            
            # Combine and deduplicate results
            combined_results = []
            seen_chunks = set()
            
            # Add results with source annotation
            for source, results in results_dict.items():
                for result in results:
                    if result['chunk_id'] not in seen_chunks:
                        result['source_collection'] = source
                        combined_results.append(result)
                        seen_chunks.add(result['chunk_id'])
            
            # Sort by score (descending)
            combined_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            # Limit final results
            final_results = combined_results[:limit]
            
            # Log search results summary
            global_count = len(results_dict.get("global", []))
            user_count = len(results_dict.get("user", []))
            team_counts = {k: len(v) for k, v in results_dict.items() if k.startswith("team_")}
            team_total = sum(team_counts.values())
            
            logger.info(
                f"Hybrid search: {global_count} global + "
                f"{user_count} user + {team_total} team = {len(final_results)} total"
            )
            return final_results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            # Fallback to global search only
            return await self._search_collection(
                settings.VECTOR_COLLECTION_NAME,
                query_embedding,
                limit,
                score_threshold,
                filter_category,
                filter_tags
            )
    
    async def delete_points_by_filter(
        self,
        document_id: str,
        user_id: Optional[str] = None,
        collection_name: Optional[str] = None
    ) -> bool:
        """
        Delete all points for a document via Vector Service
        
        Args:
            document_id: Document ID to delete
            user_id: User ID for user-specific collection
            collection_name: Specific collection (overrides user_id)
            
        Returns:
            True if successful
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            if collection_name:
                target_collection = collection_name
            elif user_id:
                target_collection = self._get_user_collection_name(user_id)
            else:
                target_collection = settings.VECTOR_COLLECTION_NAME
            
            # Delete points matching document_id via Vector Service
            filters = [{
                "field": "document_id",
                "value": document_id,
                "operator": "equals"
            }]
            
            logger.info(f"Deleting vectors for document {document_id} from {target_collection}")
            
            result = await self.vector_service_client.delete_vectors(
                collection_name=target_collection,
                filters=filters
            )
            
            if result.get("success"):
                points_deleted = result.get("points_deleted", 0)
                logger.info(f"✅ Deleted {points_deleted} point(s) for document {document_id} from {target_collection}")
                return True
            else:
                error = result.get("error", "Unknown error")
                logger.error(f"❌ Failed to delete points for document {document_id}: {error}")
                return False
            
        except Exception as e:
            logger.error(f"❌ Failed to delete points for document {document_id}: {e}")
            return False
    
    async def delete_collection(self, collection_name: str) -> bool:
        """Delete an entire collection via Vector Service"""
        try:
            if not self._initialized:
                await self.initialize()
            
            result = await self.vector_service_client.delete_collection(collection_name)
            
            if result.get("success"):
                logger.info(f"Deleted collection: {collection_name}")
                return True
            else:
                error = result.get("error", "Unknown error")
                logger.warning(f"Failed to delete collection {collection_name}: {error}")
                return False
            
        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {e}")
            return False
    
    async def delete_user_collection(self, user_id: str) -> bool:
        """Delete a user's entire collection"""
        collection_name = self._get_user_collection_name(user_id)
        return await self.delete_collection(collection_name)
    
    async def get_collection_stats(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about a collection via Vector Service"""
        if not self._initialized:
            await self.initialize()
        
        if not collection_name:
            collection_name = settings.VECTOR_COLLECTION_NAME
        
        try:
            result = await self.vector_service_client.get_collection_info(collection_name)
            
            if not result.get("success") or not result.get("collection"):
                return {
                    "exists": False,
                    "collection_name": collection_name,
                    "error": result.get("error", "Collection not found")
                }
            
            col = result["collection"]
            return {
                "exists": True,
                "collection_name": collection_name,
                "points_count": col.get("points_count", 0),
                "vectors_count": col.get("points_count", 0),  # Same as points_count
                "indexed_vectors_count": col.get("points_count", 0),  # Qdrant doesn't expose this separately
                "status": col.get("status", "unknown")
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats for {collection_name}: {e}")
            return {
                "exists": False,
                "collection_name": collection_name,
                "error": str(e)
            }
    
    async def get_user_collection_stats(self, user_id: str) -> Dict[str, Any]:
        """Get statistics about a user's collection"""
        collection_name = self._get_user_collection_name(user_id)
        return await self.get_collection_stats(collection_name)
    
    async def list_all_collections(self) -> List[Dict[str, Any]]:
        """List all collections with basic info via Vector Service"""
        try:
            if not self._initialized:
                await self.initialize()
            
            result = await self.vector_service_client.list_collections()
            
            if not result.get("success"):
                logger.error(f"Failed to list collections: {result.get('error')}")
                return []
            
            collection_list = []
            for col in result.get("collections", []):
                collection_list.append({
                    "name": col.get("name", ""),
                    "is_user_collection": col.get("name", "").startswith("user_")
                })
            
            return collection_list
            
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []
    
    async def list_user_collections(self) -> List[Dict[str, Any]]:
        """List all user collections"""
        all_collections = await self.list_all_collections()
        return [col for col in all_collections if col['is_user_collection']]


# Singleton instance
_vector_store_instance: Optional[VectorStoreService] = None


async def get_vector_store() -> VectorStoreService:
    """Get singleton vector store service instance"""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStoreService()
        await _vector_store_instance.initialize()
    return _vector_store_instance


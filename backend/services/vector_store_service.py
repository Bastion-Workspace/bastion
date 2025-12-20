"""
Vector Store Service - Manages vector database operations (Qdrant)

Handles all vector storage, retrieval, and collection management operations.
Separated from embedding generation for clean architecture and multi-backend support.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct, Distance, VectorParams, Filter,
    FieldCondition, MatchValue, ScrollRequest
)

from config import settings

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
        self.client: Optional[QdrantClient] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize vector database client"""
        if self._initialized:
            return
            
        logger.info("Initializing Vector Store Service...")
        
        if not settings.QDRANT_URL:
            raise ValueError("QDRANT_URL is not configured")
        
        self.client = QdrantClient(url=settings.QDRANT_URL)
        
        # Ensure default collection exists
        await self.ensure_collection_exists(settings.VECTOR_COLLECTION_NAME)
        
        self._initialized = True
        logger.info("Vector Store Service initialized")
    
    def _get_user_collection_name(self, user_id: str) -> str:
        """Generate collection name for a specific user"""
        return f"user_{user_id}_documents"
    
    def _get_team_collection_name(self, team_id: str) -> str:
        """Generate collection name for a specific team"""
        return f"team_{team_id}"
    
    async def ensure_collection_exists(self, collection_name: str) -> bool:
        """Ensure a collection exists, create if it doesn't"""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=settings.EMBEDDING_DIMENSIONS,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created vector collection: {collection_name}")
                return True
            return False
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
        if not collection_name:
            collection_name = settings.VECTOR_COLLECTION_NAME
        
        if not points:
            logger.warning("No points provided for insertion")
            return False
        
        # Batch insertion for large point sets
        batch_size = 100
        total_points = len(points)
        
        for attempt in range(max_retries):
            try:
                if total_points <= batch_size:
                    # Single batch
                    await self._insert_batch_sync(points, collection_name)
                    logger.info(f"Inserted {total_points} points into {collection_name}")
                else:
                    # Multiple batches
                    for i in range(0, total_points, batch_size):
                        batch = points[i:i + batch_size]
                        await self._insert_batch_sync(batch, collection_name)
                        logger.info(f"Inserted batch {i//batch_size + 1}: {len(batch)} points")
                
                return True
                
            except Exception as e:
                logger.error(f"Insert attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    logger.error(f"Failed to insert points after {max_retries} attempts")
                    return False
        
        return False
    
    async def _insert_batch_sync(self, batch_points: List[PointStruct], collection_name: str):
        """Insert a single batch synchronously with proper async handling"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.client.upsert(
                collection_name=collection_name,
                points=batch_points,
                wait=True
            )
        )
    
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
        """Search a specific collection"""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            if collection_name not in collection_names:
                logger.warning(f"Collection {collection_name} does not exist")
                return []
            
            # Build filter if category or tags specified
            search_filter = None
            if filter_category or filter_tags:
                conditions = []
                if filter_category:
                    conditions.append(
                        FieldCondition(
                            key="document_category",
                            match=MatchValue(value=filter_category)
                        )
                    )
                if filter_tags:
                    for tag in filter_tags:
                        conditions.append(
                            FieldCondition(
                                key="document_tags",
                                match=MatchValue(value=tag)
                            )
                        )
                if conditions:
                    search_filter = Filter(must=conditions)
            
            # Get collection info for diagnostics
            try:
                collection_info = self.client.get_collection(collection_name)
                logger.info(f"Collection {collection_name}: {collection_info.points_count} points, vector size: {collection_info.config.params.vectors.size}")
            except Exception as e:
                logger.warning(f"Could not get collection info: {e}")
            
            # Execute search
            loop = asyncio.get_event_loop()
            search_results = await loop.run_in_executor(
                None,
                lambda: self.client.search(
                    collection_name=collection_name,
                    query_vector=query_embedding,
                    limit=limit,
                    score_threshold=score_threshold,
                    query_filter=search_filter,
                    with_payload=True
                )
            )
            
            # Log raw result count for diagnostics
            logger.info(f"Qdrant returned {len(search_results)} results with threshold {score_threshold}")
            
            # Format results
            results = []
            for hit in search_results:
                result = {
                    'id': hit.id,
                    'score': hit.score,
                    'chunk_id': hit.payload.get('chunk_id'),
                    'document_id': hit.payload.get('document_id'),
                    'content': hit.payload.get('content', ''),
                    'chunk_index': hit.payload.get('chunk_index', 0),
                    'metadata': hit.payload.get('metadata', {}),
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
        Delete all points for a document
        
        Args:
            document_id: Document ID to delete
            user_id: User ID for user-specific collection
            collection_name: Specific collection (overrides user_id)
            
        Returns:
            True if successful
        """
        try:
            if collection_name:
                target_collection = collection_name
            elif user_id:
                target_collection = self._get_user_collection_name(user_id)
            else:
                target_collection = settings.VECTOR_COLLECTION_NAME
            
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            if target_collection not in collection_names:
                logger.warning(f"Collection {target_collection} does not exist")
                return False
            
            # Delete points matching document_id
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.client.delete(
                    collection_name=target_collection,
                    points_selector=Filter(
                        must=[
                            FieldCondition(
                                key="document_id",
                                match=MatchValue(value=document_id)
                            )
                        ]
                    )
                )
            )
            
            logger.info(f"Deleted points for document {document_id} from {target_collection}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete points for document {document_id}: {e}")
            return False
    
    async def delete_collection(self, collection_name: str) -> bool:
        """Delete an entire collection"""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if collection_name not in collection_names:
                logger.warning(f"Collection {collection_name} does not exist")
                return False
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.client.delete_collection(collection_name=collection_name)
            )
            
            logger.info(f"Deleted collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {e}")
            return False
    
    async def delete_user_collection(self, user_id: str) -> bool:
        """Delete a user's entire collection"""
        collection_name = self._get_user_collection_name(user_id)
        return await self.delete_collection(collection_name)
    
    async def get_collection_stats(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about a collection"""
        if not collection_name:
            collection_name = settings.VECTOR_COLLECTION_NAME
        
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if collection_name not in collection_names:
                return {
                    "exists": False,
                    "collection_name": collection_name
                }
            
            loop = asyncio.get_event_loop()
            collection_info = await loop.run_in_executor(
                None,
                lambda: self.client.get_collection(collection_name=collection_name)
            )
            
            return {
                "exists": True,
                "collection_name": collection_name,
                "points_count": collection_info.points_count,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "status": collection_info.status
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
        """List all collections with basic info"""
        try:
            collections = self.client.get_collections()
            
            collection_list = []
            for col in collections.collections:
                collection_list.append({
                    "name": col.name,
                    "is_user_collection": col.name.startswith("user_")
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


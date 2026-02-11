"""
Object Encoding Service - Manages object embeddings in Qdrant vector database

Stores CLIP embeddings for user-annotated objects (combined visual + semantic).
Used for similarity search when detecting user-defined objects in new images.

Routes all operations through Vector Service gRPC for centralized Qdrant access.
"""

import logging
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

from config import settings
from clients.vector_service_client import get_vector_service_client

logger = logging.getLogger(__name__)


class ObjectEncodingService:
    """
    Vector database service for object embedding storage and matching.

    Uses Qdrant to store 512-dimensional CLIP combined embeddings per user annotation.
    Maintains a global collection (object_features) and per-user collections
    (user_{user_id}_object_features) so searches query both.
    """

    COLLECTION_NAME = "object_features"
    VECTOR_SIZE = 512
    DISTANCE = "COSINE"

    def __init__(self):
        self.vector_service_client = None
        self._initialized = False
        self._user_collections_ensured: set = set()

    @staticmethod
    def _user_collection_name(user_id: str) -> str:
        return f"user_{user_id}_object_features"

    async def initialize(self):
        """Initialize Vector Service client and ensure global collection exists."""
        if self._initialized:
            return

        logger.info("Initializing Object Encoding Service (via Vector Service)...")

        self.vector_service_client = await get_vector_service_client(required=False)

        await self._ensure_collection_exists(self.COLLECTION_NAME)

        self._initialized = True
        logger.info("Object Encoding Service initialized (routing through Vector Service)")

    async def _ensure_collection_exists(self, collection_name: str) -> None:
        """Create collection if it doesn't exist via Vector Service."""
        try:
            collections_result = await self.vector_service_client.list_collections()
            if not collections_result.get("success"):
                logger.warning("Failed to list collections: %s", collections_result.get("error"))
                return

            collection_names = [col["name"] for col in collections_result.get("collections", [])]

            if collection_name not in collection_names:
                create_result = await self.vector_service_client.create_collection(
                    collection_name=collection_name,
                    vector_size=self.VECTOR_SIZE,
                    distance=self.DISTANCE,
                )
                if create_result.get("success"):
                    logger.info("Created object features collection: %s", collection_name)
                else:
                    error = create_result.get("error", "Unknown error")
                    logger.error("Failed to create object features collection: %s", error)
                    raise Exception(f"Failed to create collection: {error}")
            else:
                logger.debug("Object features collection already exists: %s", collection_name)
        except Exception as e:
            logger.error("Failed to ensure collection exists: %s", e)
            raise

    async def _ensure_user_collection_exists(self, user_id: str) -> None:
        """Ensure per-user object_features collection exists (lazy)."""
        if user_id in self._user_collections_ensured:
            return
        name = self._user_collection_name(user_id)
        await self._ensure_collection_exists(name)
        self._user_collections_ensured.add(user_id)

    async def store_object_annotation(
        self,
        annotation_id: str,
        user_id: str,
        object_name: str,
        combined_embedding: List[float],
        visual_embedding: Optional[List[float]] = None,
        semantic_embedding: Optional[List[float]] = None,
        source_document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store object annotation embeddings in Qdrant.

        Args:
            annotation_id: UUID of the annotation record.
            user_id: User who created the annotation.
            object_name: User-defined name for the object.
            combined_embedding: 512-dim combined CLIP embedding (primary search vector).
            visual_embedding: Optional visual-only embedding.
            semantic_embedding: Optional text-only embedding.
            source_document_id: Document where annotation was drawn.
            metadata: Additional metadata.

        Returns:
            Point ID (UUID string).
        """
        try:
            if not self._initialized:
                await self.initialize()

            point_id = str(uuid.uuid4())

            payload = {
                "annotation_id": annotation_id,
                "user_id": user_id,
                "object_name": object_name,
                "source_document_id": source_document_id or "",
                "tagged_at": datetime.utcnow().isoformat(),
                **(metadata or {}),
            }

            await self._ensure_user_collection_exists(user_id)

            for coll_name in (self.COLLECTION_NAME, self._user_collection_name(user_id)):
                result = await self.vector_service_client.upsert_vectors(
                    collection_name=coll_name,
                    points=[{
                        "id": point_id,
                        "vector": combined_embedding,
                        "payload": payload,
                    }],
                )
                if not result.get("success"):
                    error = result.get("error", "Unknown error")
                    raise Exception(f"Vector Service upsert failed ({coll_name}): {error}")

            logger.info("Stored object annotation '%s' (point: %s)", object_name, point_id)
            return point_id

        except Exception as e:
            logger.error(f"Failed to store object annotation: {e}")
            raise

    async def search_similar_objects(
        self,
        query_embedding: List[float],
        user_id: str,
        top_k: int = 5,
        similarity_threshold: float = 0.75,
    ) -> List[Dict[str, Any]]:
        """
        Find user-defined objects similar to query embedding.

        Args:
            query_embedding: 512-dim embedding from candidate region.
            user_id: Only search this user's defined objects.
            top_k: Return top K matches.
            similarity_threshold: Minimum cosine similarity (0.0-1.0).

        Returns:
            List of matches with annotation_id, object_name, similarity_score, payload.
        """
        try:
            if not self._initialized:
                await self.initialize()

            await self._ensure_user_collection_exists(user_id)

            user_coll = self._user_collection_name(user_id)
            user_results = await self.vector_service_client.search_vectors(
                collection_name=user_coll,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=similarity_threshold,
            )
            global_results = await self.vector_service_client.search_vectors(
                collection_name=self.COLLECTION_NAME,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=similarity_threshold,
                filters=[{
                    "field": "user_id",
                    "value": user_id,
                    "operator": "equals",
                }],
            )

            seen = set()
            matches = []
            for hit in user_results + global_results:
                payload = hit.get("payload", {})
                aid = payload.get("annotation_id")
                if aid in seen:
                    continue
                seen.add(aid)
                matches.append({
                    "annotation_id": aid,
                    "object_name": payload.get("object_name"),
                    "similarity_score": hit.get("score", 0.0),
                    "metadata": payload,
                })
            matches.sort(key=lambda m: m["similarity_score"], reverse=True)
            return matches[:top_k]

        except Exception as e:
            logger.error(f"Object similarity search failed: {e}")
            return []

    async def add_annotation_example(
        self,
        annotation_id: str,
        user_id: str,
        object_name: str,
        combined_embedding: List[float],
        source_document_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add an additional example embedding for an existing annotation.

        Returns:
            Point ID of the new vector.
        """
        return await self.store_object_annotation(
            annotation_id=annotation_id,
            user_id=user_id,
            object_name=object_name,
            combined_embedding=combined_embedding,
            source_document_id=source_document_id,
            metadata={**(metadata or {}), "example": True},
        )

    async def delete_vectors_by_source_document(
        self, document_id: str, user_id: Optional[str] = None
    ) -> int:
        """
        Delete all object embedding vectors that were created from this document (Qdrant only).
        Call when a picture is deleted so IMAGE embeddings are removed.
        """
        try:
            if not self._initialized:
                await self.initialize()
            filters = [{
                "field": "source_document_id",
                "value": document_id,
                "operator": "equals",
            }]
            collections = [self.COLLECTION_NAME]
            if user_id:
                await self._ensure_user_collection_exists(user_id)
                collections.append(self._user_collection_name(user_id))
            total = 0
            for coll_name in collections:
                try:
                    result = await self.vector_service_client.delete_vectors(
                        collection_name=coll_name,
                        filters=filters,
                    )
                    if result.get("success"):
                        total += result.get("points_deleted", 0)
                except Exception as e:
                    logger.debug("Delete by source_document_id in %s: %s", coll_name, e)
            if total > 0:
                logger.info("Deleted %s object vector(s) for source document %s", total, document_id)
            return total
        except Exception as e:
            logger.warning("Failed to delete object vectors by source document %s: %s", document_id, e)
            return 0

    async def delete_annotation(self, annotation_id: str, user_id: Optional[str] = None) -> int:
        """
        Delete all vectors for an annotation (annotation + examples) from global
        and, if user_id is provided, from the user's collection.

        Returns:
            Number of points deleted (sum across collections).
        """
        try:
            if not self._initialized:
                await self.initialize()

            total_deleted = 0
            collections_to_clear = [self.COLLECTION_NAME]
            if user_id:
                collections_to_clear.append(self._user_collection_name(user_id))

            for coll_name in collections_to_clear:
                delete_result = await self.vector_service_client.delete_vectors(
                    collection_name=coll_name,
                    filters=[{
                        "field": "annotation_id",
                        "value": annotation_id,
                        "operator": "equals",
                    }],
                )
                if not delete_result.get("success"):
                    error = delete_result.get("error", "Unknown error")
                    raise Exception(f"Vector Service delete failed ({coll_name}): {error}")
                total_deleted += delete_result.get("points_deleted", 0)

            if total_deleted > 0:
                logger.info("Deleted %s vectors for annotation %s", total_deleted, annotation_id)
            return total_deleted

        except Exception as e:
            logger.error("Failed to delete annotation vectors: %s", e)
            raise


_object_encoding_service: Optional[ObjectEncodingService] = None


async def get_object_encoding_service() -> ObjectEncodingService:
    """Get or create global Object Encoding Service instance."""
    global _object_encoding_service
    if _object_encoding_service is None:
        _object_encoding_service = ObjectEncodingService()
        await _object_encoding_service.initialize()
    return _object_encoding_service

"""
Face Encoding Service - Manages face encodings in Qdrant vector database

Stores multiple face encodings per identity for robust matching across:
- Different angles and perspectives
- Various lighting conditions  
- Different expressions and ages
- Image quality variations

More samples = better matching, not drift!

Routes all operations through Vector Service gRPC for centralized Qdrant access.
"""

import logging
import uuid
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from config import settings
from clients.vector_service_client import get_vector_service_client

logger = logging.getLogger(__name__)


class FaceEncodingService:
    """
    Vector database service for face encoding storage and matching.

    Uses Qdrant to store multiple face encodings per identity. Maintains a global
    collection (face_encodings) and per-user collections (user_{user_id}_face_encodings)
    so searches can query both.
    """

    COLLECTION_NAME = "face_encodings"
    VECTOR_SIZE = 128  # face_recognition encoding size
    DISTANCE = "COSINE"

    def __init__(self):
        self.vector_service_client = None
        self._initialized = False
        self._user_collections_ensured: set = set()

    @staticmethod
    def _user_collection_name(user_id: str) -> str:
        return f"user_{user_id}_face_encodings"

    async def initialize(self):
        """Initialize Vector Service client and ensure global collection exists."""
        if self._initialized:
            return

        logger.info("Initializing Face Encoding Service (via Vector Service)...")

        self.vector_service_client = await get_vector_service_client(required=False)

        await self._ensure_collection_exists(self.COLLECTION_NAME)

        self._initialized = True
        logger.info("Face Encoding Service initialized (routing through Vector Service)")

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
                    logger.info("Created face encodings collection: %s", collection_name)
                else:
                    error = create_result.get("error", "Unknown error")
                    logger.error("Failed to create face encodings collection: %s", error)
                    raise Exception(f"Failed to create collection: {error}")
            else:
                logger.debug("Face encodings collection already exists: %s", collection_name)
        except Exception as e:
            logger.error("Failed to ensure collection exists: %s", e)
            raise

    async def _ensure_user_collection_exists(self, user_id: str) -> None:
        """Ensure per-user face_encodings collection exists (lazy)."""
        if user_id in self._user_collections_ensured:
            return
        name = self._user_collection_name(user_id)
        await self._ensure_collection_exists(name)
        self._user_collections_ensured.add(user_id)
    
    async def add_face_encoding(
        self,
        identity_name: str,
        face_encoding: List[float],
        source_document_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """
        Add a face encoding to the vector database (global and, if user_id given, user collection).

        Args:
            identity_name: Name of the person
            face_encoding: 128-dimensional face encoding vector
            source_document_id: Document ID where this face was detected
            metadata: Additional metadata (optional)
            user_id: If set, also store in per-user collection for this user

        Returns:
            Point ID (UUID string)
        """
        try:
            if not self._initialized:
                await self.initialize()

            point_id = str(uuid.uuid4())

            payload = {
                "identity_name": identity_name,
                "source_document_id": source_document_id,
                "tagged_at": datetime.utcnow().isoformat(),
                **(metadata or {}),
            }

            if user_id:
                await self._ensure_user_collection_exists(user_id)

            collections_to_write = [self.COLLECTION_NAME]
            if user_id:
                collections_to_write.append(self._user_collection_name(user_id))

            for coll_name in collections_to_write:
                result = await self.vector_service_client.upsert_vectors(
                    collection_name=coll_name,
                    points=[{
                        "id": point_id,
                        "vector": face_encoding,
                        "payload": payload,
                    }],
                )
                if not result.get("success"):
                    error = result.get("error", "Unknown error")
                    raise Exception(f"Vector Service upsert failed ({coll_name}): {error}")

            logger.info("Stored face encoding for '%s' (point: %s)", identity_name, point_id)
            return point_id

        except Exception as e:
            logger.error("Failed to add face encoding: %s", e)
            raise
    
    async def match_face(
        self,
        face_encoding: List[float],
        confidence_threshold: float = 0.82,
        limit: int = 5,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Match a face encoding against known identities (global and, if user_id given, user collection).

        Args:
            face_encoding: 128-dimensional face encoding to match
            confidence_threshold: Minimum cosine similarity (0.0-1.0). 0.82 aligns with L2 < 0.6 "same person" rule.
            limit: Number of nearest neighbors to retrieve
            user_id: If set, also search this user's face collection and merge results

        Returns:
            Dict with matched_identity, confidence, and sample_count, or None if no match
        """
        try:
            if not self._initialized:
                await self.initialize()

            collections_to_search = [self.COLLECTION_NAME]
            if user_id:
                await self._ensure_user_collection_exists(user_id)
                collections_to_search.append(self._user_collection_name(user_id))

            all_results = []
            for coll_name in collections_to_search:
                results = await self.vector_service_client.search_vectors(
                    collection_name=coll_name,
                    query_vector=face_encoding,
                    limit=limit,
                    score_threshold=confidence_threshold,
                )
                all_results.extend(results)

            if not all_results:
                logger.debug("No face matches found above threshold")
                return None

            all_results.sort(key=lambda r: r.get("score", 0.0), reverse=True)
            best_match = all_results[0]
            payload = best_match.get("payload", {})
            identity_name = payload.get("identity_name")
            similarity_score = best_match.get("score", 0.0)
            confidence = similarity_score * 100

            count_results = await self.vector_service_client.search_vectors(
                collection_name=self.COLLECTION_NAME,
                query_vector=face_encoding,
                limit=100,
                score_threshold=0.0,
                filters=[{
                    "field": "identity_name",
                    "value": identity_name,
                    "operator": "equals",
                }],
            )
            sample_count = len(count_results) if count_results else 1

            logger.info(
                "Best match: '%s' (%.1f%% confidence, %s samples)",
                identity_name,
                confidence,
                sample_count,
            )

            return {
                "matched_identity": identity_name,
                "confidence": round(confidence, 1),
                "sample_count": sample_count,
                "similarity_score": similarity_score,
            }

        except Exception as e:
            logger.error("Face matching failed: %s", e)
            return None
    
    async def get_identity_sample_count(self, identity_name: str) -> int:
        """Get number of face encoding samples for an identity via Vector Service"""
        try:
            if not self._initialized:
                await self.initialize()
            
            # Use search with identity filter to count samples
            # Use a dummy vector and low threshold to get all matches
            dummy_vector = [0.0] * self.VECTOR_SIZE
            results = await self.vector_service_client.search_vectors(
                collection_name=self.COLLECTION_NAME,
                query_vector=dummy_vector,
                limit=100,
                score_threshold=0.0,  # Low threshold to get all matches
                filters=[{
                    "field": "identity_name",
                    "value": identity_name,
                    "operator": "equals"
                }]
            )
            
            return len(results) if results else 0
            
        except Exception as e:
            logger.error(f"❌ Failed to get sample count: {e}")
            return 0
    
    async def delete_encodings_by_source_document(
        self, document_id: str, user_id: Optional[str] = None
    ) -> int:
        """
        Delete all face encodings that were added from this document (Qdrant only).
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
            total = 0
            result = await self.vector_service_client.delete_vectors(
                collection_name=self.COLLECTION_NAME,
                filters=filters,
            )
            if result.get("success"):
                total += result.get("points_deleted", 0)
            if user_id:
                await self._ensure_user_collection_exists(user_id)
                result_user = await self.vector_service_client.delete_vectors(
                    collection_name=self._user_collection_name(user_id),
                    filters=filters,
                )
                if result_user.get("success"):
                    total += result_user.get("points_deleted", 0)
            if total > 0:
                logger.info("Deleted %s face encoding(s) for source document %s", total, document_id)
            return total
        except Exception as e:
            logger.warning("Failed to delete face encodings by source document %s: %s", document_id, e)
            return 0

    async def delete_identity(self, identity_name: str) -> int:
        """
        Delete all face encodings for an identity
        
        Returns:
            Number of encodings deleted
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            # Get all points for this identity
            results = self.client.scroll(
                collection_name=self.COLLECTION_NAME,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="identity_name",
                            match=MatchValue(value=identity_name)
                        )
                    ]
                ),
                limit=100
            )
            
            if not results or not results[0]:
                return 0
            
            point_ids = [point.id for point in results[0]]
            
            # Delete all points
            self.client.delete(
                collection_name=self.COLLECTION_NAME,
                points_selector=point_ids
            )
            
            logger.info(f"✅ Deleted {len(point_ids)} face encodings for '{identity_name}'")
            return len(point_ids)
            
        except Exception as e:
            logger.error(f"❌ Failed to delete identity: {e}")
            raise
    
    async def clear_all_encodings(self) -> int:
        """Delete all face encodings from Qdrant via Vector Service"""
        try:
            if not self._initialized:
                await self.initialize()
            
            # Get count before deleting
            col_info = await self.vector_service_client.get_collection_info(self.COLLECTION_NAME)
            total_count = col_info.get("collection", {}).get("points_count", 0) if col_info.get("success") else 0
            
            # Delete collection and recreate via Vector Service
            delete_result = await self.vector_service_client.delete_collection(self.COLLECTION_NAME)
            if not delete_result.get("success"):
                error = delete_result.get("error", "Unknown error")
                raise Exception(f"Failed to delete collection: {error}")
            
            await self._ensure_collection_exists(self.COLLECTION_NAME)

            logger.info("Cleared all face encodings: %s vectors deleted", total_count)
            return total_count
            
        except Exception as e:
            logger.error(f"❌ Failed to clear face encodings: {e}")
            raise
    
    async def cleanup_orphaned_vectors(self) -> int:
        """
        Clean up Qdrant vectors for identities that no longer exist in PostgreSQL
        Called periodically or after document deletions
        
        Note: This requires scrolling through all vectors. Since Vector Service doesn't
        have a scroll RPC, we use search with a dummy vector and very low threshold
        to retrieve all vectors, then filter client-side.
        
        Returns:
            Number of orphaned vectors deleted
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            from services.database_manager.database_helpers import fetch_all
            
            # Get all identity names still in PostgreSQL
            valid_identities_rows = await fetch_all(
                "SELECT DISTINCT identity_name FROM known_identities"
            )
            valid_identities = {row["identity_name"] for row in valid_identities_rows}
            
            # Get all vectors from Qdrant via Vector Service
            # Use search with dummy vector and very low threshold to get all vectors
            # This is a workaround until Vector Service has a scroll RPC
            dummy_vector = [0.0] * self.VECTOR_SIZE
            search_results = await self.vector_service_client.search_vectors(
                collection_name=self.COLLECTION_NAME,
                query_vector=dummy_vector,
                limit=10000,  # Max limit
                score_threshold=0.0  # Very low threshold to get all
            )
            
            if not search_results:
                logger.debug("No face encoding vectors in Qdrant")
                return 0
            
            # Find orphaned vectors (identity not in PostgreSQL)
            orphaned_identities = set()
            for result in search_results:
                payload = result.get("payload", {})
                identity_name = payload.get("identity_name")
                if identity_name and identity_name not in valid_identities:
                    orphaned_identities.add(identity_name)
            
            # Delete orphaned vectors (delete by identity_name filter)
            total_deleted = 0
            for identity_name in orphaned_identities:
                filters = [{
                    "field": "identity_name",
                    "value": identity_name,
                    "operator": "equals"
                }]
                delete_result = await self.vector_service_client.delete_vectors(
                    collection_name=self.COLLECTION_NAME,
                    filters=filters
                )
                if delete_result.get("success"):
                    deleted_count = delete_result.get("points_deleted", 0)
                    total_deleted += deleted_count
            
            if total_deleted > 0:
                logger.info(f"🧹 Cleaned up {total_deleted} orphaned face encoding vectors")
                return total_deleted
            else:
                logger.debug("No orphaned vectors found")
                return 0
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup orphaned vectors: {e}")
            return 0


# Global service instance
_face_encoding_service: Optional[FaceEncodingService] = None


async def get_face_encoding_service() -> FaceEncodingService:
    """Get or create global Face Encoding Service instance"""
    global _face_encoding_service
    if _face_encoding_service is None:
        _face_encoding_service = FaceEncodingService()
        await _face_encoding_service.initialize()
    return _face_encoding_service

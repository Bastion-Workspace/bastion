"""
Resilient Embedding Manager - Handles embedding generation with crash recovery and immediate persistence
"""

import asyncio
import logging
import time
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle

from openai import AsyncOpenAI
from qdrant_client.models import PointStruct

from config import settings
from models.api_models import Chunk
from services.embedding_service_wrapper import get_embedding_service
from services.vector_store_service import get_vector_store

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingProgress:
    """Track embedding progress for crash recovery"""
    document_id: str
    total_chunks: int
    processed_chunks: int
    failed_chunks: List[str]
    completed_chunks: List[str]
    start_time: float
    last_update: float
    status: str  # 'processing', 'completed', 'failed', 'paused'
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbeddingProgress':
        return cls(**data)


@dataclass
class ChunkEmbeddingState:
    """State of individual chunk embedding"""
    chunk_id: str
    document_id: str
    content_hash: str
    embedding_generated: bool
    stored_in_db: bool
    retry_count: int
    last_error: Optional[str]
    timestamp: float


class ResilientEmbeddingManager:
    """
    Embedding manager with crash recovery and immediate persistence.
    
    Key improvements:
    1. Immediate persistence: Embed and store chunks one by one to avoid losing work
    2. Progress tracking: Save progress to disk for crash recovery
    3. Resume capability: Can resume from where it left off after a crash
    4. Atomic operations: Each chunk is processed atomically
    5. Better error handling: Isolate failures to individual chunks
    """
    
    def __init__(self):
        self.embedding_service = None
        self.vector_store = None
        self.progress_dir = Path(settings.LOGS_DIR) / "embedding_progress"
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory tracking
        self.active_progress: Dict[str, EmbeddingProgress] = {}
        self.chunk_states: Dict[str, ChunkEmbeddingState] = {}
        
        # Configuration
        self.max_retries_per_chunk = 3
        self.batch_size = 1  # Process one chunk at a time for maximum resilience
        self.save_progress_interval = 5  # Save progress every 5 chunks
        
        logger.info("Resilient Embedding Manager initialized")
    
    async def initialize(self):
        """Initialize the resilient embedding manager"""
        logger.info("Initializing Resilient Embedding Manager...")
        
        # Initialize embedding service (Vector Service)
        self.embedding_service = await get_embedding_service()
        
        # Initialize vector store (for storage)
        self.vector_store = await get_vector_store()
        
        # Load any existing progress from disk
        await self._load_existing_progress()
        
        logger.info("Resilient Embedding Manager initialized")
    
    async def embed_and_store_chunks_resilient(
        self, 
        chunks: List[Chunk], 
        document_id: str = None,
        user_id: str = None,
        resume_existing: bool = True
    ) -> Dict[str, Any]:
        """
        Embed and store chunks with full resilience and crash recovery.
        
        Args:
            chunks: List of chunks to process
            document_id: Document ID (extracted from chunks if not provided)
            user_id: User ID for user-specific collections
            resume_existing: Whether to resume existing progress
            
        Returns:
            Dict with processing results and statistics
        """
        start_time = time.time()
        
        try:
            if not chunks:
                return {"status": "success", "message": "No chunks to process"}
            
            # Extract document ID if not provided
            if not document_id:
                document_id = chunks[0].document_id
            
            logger.info(f"🔄 Starting resilient embedding for document {document_id}: {len(chunks)} chunks")
            
            # Check for existing progress
            existing_progress = None
            if resume_existing:
                existing_progress = await self._load_progress(document_id)
                if existing_progress:
                    logger.info(f"📋 Found existing progress: {existing_progress.processed_chunks}/{existing_progress.total_chunks} chunks completed")
            
            # Initialize or resume progress tracking
            if existing_progress and existing_progress.status == 'processing':
                progress = existing_progress
                progress.last_update = time.time()
            else:
                progress = EmbeddingProgress(
                    document_id=document_id,
                    total_chunks=len(chunks),
                    processed_chunks=0,
                    failed_chunks=[],
                    completed_chunks=[],
                    start_time=start_time,
                    last_update=start_time,
                    status='processing'
                )
            
            self.active_progress[document_id] = progress
            await self._save_progress(progress)
            
            # Process chunks one by one for maximum resilience
            successful_chunks = 0
            failed_chunks = 0
            
            for i, chunk in enumerate(chunks):
                # Skip if already processed (for resume scenarios)
                if chunk.chunk_id in progress.completed_chunks:
                    logger.debug(f"⏭️ Skipping already processed chunk: {chunk.chunk_id}")
                    successful_chunks += 1
                    continue
                
                # Skip if failed too many times
                if chunk.chunk_id in progress.failed_chunks:
                    logger.debug(f"⏭️ Skipping previously failed chunk: {chunk.chunk_id}")
                    failed_chunks += 1
                    continue
                
                logger.info(f"🔄 Processing chunk {i+1}/{len(chunks)}: {chunk.chunk_id}")
                
                # Process single chunk with retries
                success = await self._process_single_chunk_resilient(chunk, user_id, progress)
                
                if success:
                    successful_chunks += 1
                    progress.completed_chunks.append(chunk.chunk_id)
                    logger.info(f"✅ Chunk {chunk.chunk_id} processed successfully")
                else:
                    failed_chunks += 1
                    if chunk.chunk_id not in progress.failed_chunks:
                        progress.failed_chunks.append(chunk.chunk_id)
                    logger.error(f"❌ Chunk {chunk.chunk_id} failed after retries")
                
                # Update progress
                progress.processed_chunks = successful_chunks
                progress.last_update = time.time()
                
                # Save progress periodically
                if (successful_chunks + failed_chunks) % self.save_progress_interval == 0:
                    await self._save_progress(progress)
                    logger.debug(f"💾 Progress saved: {successful_chunks + failed_chunks}/{len(chunks)} chunks processed")
                
                # Small delay to avoid overwhelming the system
                await asyncio.sleep(0.1)
            
            # Final status update
            if failed_chunks == 0:
                progress.status = 'completed'
                logger.info(f"✅ All chunks processed successfully for document {document_id}")
            else:
                progress.status = 'completed_with_errors'
                logger.warning(f"⚠️ Document {document_id} completed with {failed_chunks} failed chunks")
            
            progress.last_update = time.time()
            await self._save_progress(progress)
            
            # Clean up completed progress after a delay (keep for debugging)
            if progress.status == 'completed':
                asyncio.create_task(self._cleanup_progress_after_delay(document_id, delay_hours=24))
            
            processing_time = time.time() - start_time
            
            result = {
                "status": "success" if failed_chunks == 0 else "partial_success",
                "document_id": document_id,
                "total_chunks": len(chunks),
                "successful_chunks": successful_chunks,
                "failed_chunks": failed_chunks,
                "processing_time": processing_time,
                "chunks_per_second": len(chunks) / processing_time if processing_time > 0 else 0,
                "message": f"Processed {successful_chunks}/{len(chunks)} chunks successfully"
            }
            
            logger.info(f"📊 Resilient embedding completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Resilient embedding failed for document {document_id}: {e}")
            
            # Update progress status
            if document_id in self.active_progress:
                self.active_progress[document_id].status = 'failed'
                await self._save_progress(self.active_progress[document_id])
            
            return {
                "status": "error",
                "document_id": document_id,
                "error": str(e),
                "message": f"Embedding failed: {str(e)}"
            }
    
    async def _process_single_chunk_resilient(
        self, 
        chunk: Chunk, 
        user_id: str = None,
        progress: EmbeddingProgress = None
    ) -> bool:
        """
        Process a single chunk with retries and immediate persistence.
        
        Returns True if successful, False if failed after all retries.
        """
        chunk_id = chunk.chunk_id
        retry_count = 0
        
        while retry_count < self.max_retries_per_chunk:
            try:
                logger.debug(f"🔄 Attempt {retry_count + 1}/{self.max_retries_per_chunk} for chunk {chunk_id}")
                
                # Step 1: Generate embedding
                logger.debug(f"Generating embedding for chunk {chunk_id}")
                embeddings = await self.embedding_service.generate_embeddings([chunk.content])
                
                if not embeddings or len(embeddings) != 1:
                    raise Exception("Failed to generate embedding")
                
                embedding = embeddings[0]
                logger.debug(f"✅ Embedding generated for chunk {chunk_id}")
                
                # Step 2: Immediately store in vector database
                logger.debug(f"💾 Storing embedding for chunk {chunk_id}")
                await self._store_single_chunk_embedding(chunk, embedding, user_id)
                logger.debug(f"✅ Embedding stored for chunk {chunk_id}")
                
                # Success!
                return True
                
            except Exception as e:
                retry_count += 1
                error_msg = str(e)
                logger.warning(f"⚠️ Chunk {chunk_id} attempt {retry_count} failed: {error_msg}")
                
                # Update chunk state
                content_hash = hashlib.md5(chunk.content.encode()).hexdigest()
                self.chunk_states[chunk_id] = ChunkEmbeddingState(
                    chunk_id=chunk_id,
                    document_id=chunk.document_id,
                    content_hash=content_hash,
                    embedding_generated=False,
                    stored_in_db=False,
                    retry_count=retry_count,
                    last_error=error_msg,
                    timestamp=time.time()
                )
                
                if retry_count < self.max_retries_per_chunk:
                    # Exponential backoff
                    wait_time = min(30, 2 ** retry_count)
                    logger.debug(f"⏳ Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"❌ Chunk {chunk_id} failed after {self.max_retries_per_chunk} attempts: {error_msg}")
                    return False
        
        return False
    
    async def _store_single_chunk_embedding(
        self, 
        chunk: Chunk, 
        embedding: List[float], 
        user_id: str = None
    ):
        """Store a single chunk embedding immediately"""
        try:
            # Ensure collection exists if needed
            if user_id:
                await self.vector_store.ensure_user_collection_exists(user_id)
                collection_name = self.vector_store._get_user_collection_name(user_id)
            else:
                collection_name = settings.VECTOR_COLLECTION_NAME
            
            # Use content hash for consistent IDs
            content_hash = abs(hash(chunk.content))
            
            # Create point
            point = PointStruct(
                id=content_hash,
                vector=embedding,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "quality_score": chunk.quality_score,
                    "method": chunk.method,
                    "metadata": chunk.metadata,
                    "content_hash": content_hash,
                    "user_id": user_id,
                    "stored_at": datetime.utcnow().isoformat(),
                    "resilient_processing": True
                }
            )
            
            # Store immediately via VectorStoreService with timeout
            await asyncio.wait_for(
                self.vector_store.insert_points([point], collection_name),
                timeout=30.0
            )
            
        except Exception as e:
            logger.error(f"Failed to store chunk embedding: {e}")
            raise
    
    async def _save_progress(self, progress: EmbeddingProgress):
        """Save progress to disk for crash recovery"""
        try:
            progress_file = self.progress_dir / f"{progress.document_id}.json"
            
            # Save as JSON for human readability
            with open(progress_file, 'w') as f:
                json.dump(progress.to_dict(), f, indent=2)
            
            logger.debug(f"💾 Progress saved for document {progress.document_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save progress for {progress.document_id}: {e}")
    
    async def _load_progress(self, document_id: str) -> Optional[EmbeddingProgress]:
        """Load progress from disk"""
        try:
            progress_file = self.progress_dir / f"{document_id}.json"
            
            if not progress_file.exists():
                return None
            
            with open(progress_file, 'r') as f:
                data = json.load(f)
            
            progress = EmbeddingProgress.from_dict(data)
            logger.info(f"📋 Loaded progress for document {document_id}")
            return progress
            
        except Exception as e:
            logger.error(f"❌ Failed to load progress for {document_id}: {e}")
            return None
    
    async def _load_existing_progress(self):
        """Load all existing progress files on startup"""
        try:
            progress_files = list(self.progress_dir.glob("*.json"))
            
            for progress_file in progress_files:
                try:
                    with open(progress_file, 'r') as f:
                        data = json.load(f)
                    
                    progress = EmbeddingProgress.from_dict(data)
                    
                    # Only load if still processing
                    if progress.status == 'processing':
                        self.active_progress[progress.document_id] = progress
                        logger.info(f"📋 Loaded active progress for document {progress.document_id}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load progress file {progress_file}: {e}")
            
            if self.active_progress:
                logger.info(f"📋 Loaded {len(self.active_progress)} active progress records")
            
        except Exception as e:
            logger.error(f"❌ Failed to load existing progress: {e}")
    
    async def _cleanup_progress_after_delay(self, document_id: str, delay_hours: int = 24):
        """Clean up progress files after a delay"""
        try:
            await asyncio.sleep(delay_hours * 3600)  # Convert hours to seconds
            
            progress_file = self.progress_dir / f"{document_id}.json"
            if progress_file.exists():
                progress_file.unlink()
                logger.debug(f"🧹 Cleaned up progress file for document {document_id}")
            
            # Remove from active progress
            if document_id in self.active_progress:
                del self.active_progress[document_id]
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup progress for {document_id}: {e}")
    
    async def resume_document_processing(self, document_id: str, chunks: List[Chunk], user_id: str = None) -> Dict[str, Any]:
        """Resume processing for a specific document"""
        logger.info(f"🔄 Resuming processing for document {document_id}")
        
        return await self.embed_and_store_chunks_resilient(
            chunks=chunks,
            document_id=document_id,
            user_id=user_id,
            resume_existing=True
        )
    
    async def get_processing_status(self, document_id: str) -> Dict[str, Any]:
        """Get current processing status for a document"""
        try:
            # Check active progress first
            if document_id in self.active_progress:
                progress = self.active_progress[document_id]
            else:
                # Try to load from disk
                progress = await self._load_progress(document_id)
                if not progress:
                    return {"status": "not_found", "message": "No processing record found"}
            
            return {
                "status": progress.status,
                "document_id": progress.document_id,
                "total_chunks": progress.total_chunks,
                "processed_chunks": progress.processed_chunks,
                "failed_chunks": len(progress.failed_chunks),
                "completed_chunks": len(progress.completed_chunks),
                "progress_percentage": (progress.processed_chunks / progress.total_chunks * 100) if progress.total_chunks > 0 else 0,
                "start_time": progress.start_time,
                "last_update": progress.last_update,
                "processing_time": progress.last_update - progress.start_time,
                "failed_chunk_ids": progress.failed_chunks
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get processing status for {document_id}: {e}")
            return {"status": "error", "error": str(e)}
    
    async def list_active_processing(self) -> List[Dict[str, Any]]:
        """List all active processing jobs"""
        try:
            active_jobs = []
            
            for document_id, progress in self.active_progress.items():
                status = await self.get_processing_status(document_id)
                active_jobs.append(status)
            
            return active_jobs
            
        except Exception as e:
            logger.error(f"❌ Failed to list active processing: {e}")
            return []
    
    async def cancel_processing(self, document_id: str) -> bool:
        """Cancel processing for a document"""
        try:
            if document_id in self.active_progress:
                progress = self.active_progress[document_id]
                progress.status = 'cancelled'
                progress.last_update = time.time()
                
                await self._save_progress(progress)
                del self.active_progress[document_id]
                
                logger.info(f"🛑 Cancelled processing for document {document_id}")
                return True
            else:
                logger.warning(f"⚠️ No active processing found for document {document_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to cancel processing for {document_id}: {e}")
            return False
    
    async def retry_failed_chunks(self, document_id: str, chunks: List[Chunk], user_id: str = None) -> Dict[str, Any]:
        """Retry processing only the failed chunks for a document"""
        try:
            progress = await self._load_progress(document_id)
            if not progress:
                return {"status": "error", "message": "No progress record found"}
            
            if not progress.failed_chunks:
                return {"status": "success", "message": "No failed chunks to retry"}
            
            # Filter chunks to only failed ones
            failed_chunk_ids = set(progress.failed_chunks)
            failed_chunks = [chunk for chunk in chunks if chunk.chunk_id in failed_chunk_ids]
            
            if not failed_chunks:
                return {"status": "error", "message": "Failed chunks not found in provided chunk list"}
            
            logger.info(f"🔄 Retrying {len(failed_chunks)} failed chunks for document {document_id}")
            
            # Reset failed chunks in progress
            progress.failed_chunks = []
            progress.status = 'processing'
            await self._save_progress(progress)
            
            # Process only failed chunks
            return await self.embed_and_store_chunks_resilient(
                chunks=failed_chunks,
                document_id=document_id,
                user_id=user_id,
                resume_existing=True
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to retry failed chunks for {document_id}: {e}")
            return {"status": "error", "error": str(e)}
    
    # Delegate methods to base manager for compatibility
    async def generate_embeddings(self, texts: List[str], max_retries: int = 5) -> List[List[float]]:
        """Generate embeddings for a list of texts (delegates to embedding service)"""
        return await self.embedding_service.generate_embeddings(texts)
    
    async def search_similar(self, query_text: str, limit: int = 50, score_threshold: float = 0.7, 
                           user_id: str = None) -> List[Dict[str, Any]]:
        """Search with vector store service"""
        # Generate embedding
        embeddings = await self.embedding_service.generate_embeddings([query_text])
        if not embeddings:
            return []
        
        # Search via vector store
        return await self.vector_store.search_similar(
            query_embedding=embeddings[0],
            limit=limit,
            score_threshold=score_threshold,
            user_id=user_id
        )
    
    async def delete_document_chunks(self, document_id: str, user_id: str = None):
        """Delete all chunks for a specific document"""
        return await self.vector_store.delete_points_by_filter(document_id, user_id)
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector collection"""
        return await self.vector_store.get_collection_stats()
    
    async def close(self):
        """Clean up resources"""
        logger.info("Shutting down Resilient Embedding Manager...")
        
        # Save any active progress
        for progress in self.active_progress.values():
            await self._save_progress(progress)
        
        logger.info("Resilient Embedding Manager shut down complete")

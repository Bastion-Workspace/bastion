"""
Parallel Embedding Manager - Handles concurrent embedding generation and vector storage
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams

from config import settings
from models.api_models import Chunk
from utils.embedding_manager import EmbeddingManager

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingJob:
    """Represents an embedding generation job"""
    job_id: str
    chunks: List[Chunk]
    document_id: str
    priority: int = 0
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


@dataclass
class EmbeddingConfig:
    """Configuration for parallel embedding processing"""
    max_concurrent_requests: int = 8
    max_concurrent_storage: int = 4
    embedding_batch_size: int = 20
    storage_batch_size: int = 100
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_delay: float = 0.1
    enable_batch_optimization: bool = True
    enable_concurrent_storage: bool = True


class ParallelEmbeddingManager:
    """Enhanced embedding manager with parallel processing capabilities"""
    
    def __init__(self, config: EmbeddingConfig = None):
        self.config = config or EmbeddingConfig()
        self.base_embedding_manager = None
        
        # OpenAI clients for parallel requests
        self.openai_clients = []
        self.client_semaphore = None
        
        # Qdrant client
        self.qdrant_client = None
        
        # Processing queues
        self.embedding_queue = asyncio.Queue()
        self.storage_queue = asyncio.Queue()
        
        # Active jobs tracking
        self.active_embedding_jobs: Dict[str, EmbeddingJob] = {}
        self.completed_jobs: Dict[str, Dict[str, Any]] = {}
        self.failed_jobs: Dict[str, Exception] = {}
        
        # Concurrency controls
        self.embedding_semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        self.storage_semaphore = asyncio.Semaphore(self.config.max_concurrent_storage)
        
        # Worker tasks
        self.embedding_workers = []
        self.storage_workers = []
        self.is_running = False
        
        # Thread pool for CPU-intensive operations
        self.thread_pool = None
        
        logger.info(f"🔧 Parallel Embedding Manager initialized with config: {self.config}")
    
    async def initialize(self):
        """Initialize the parallel embedding manager"""
        logger.info("🔧 Initializing Parallel Embedding Manager...")
        
        # Initialize base embedding manager
        self.base_embedding_manager = EmbeddingManager()
        await self.base_embedding_manager.initialize()
        
        # Initialize multiple OpenAI clients for parallel requests
        self.openai_clients = []
        for i in range(self.config.max_concurrent_requests):
            client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            self.openai_clients.append(client)
        
        self.client_semaphore = asyncio.Semaphore(len(self.openai_clients))
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(url=settings.QDRANT_URL)
        
        # Initialize thread pool
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Start worker tasks
        await self.start_workers()
        
        logger.info("✅ Parallel Embedding Manager initialized")
    
    async def start_workers(self):
        """Start background worker tasks"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start embedding workers
        for i in range(self.config.max_concurrent_requests):
            worker_task = asyncio.create_task(self._embedding_worker(f"embed-worker-{i}"))
            self.embedding_workers.append(worker_task)
        
        # Start storage workers
        for i in range(self.config.max_concurrent_storage):
            worker_task = asyncio.create_task(self._storage_worker(f"storage-worker-{i}"))
            self.storage_workers.append(worker_task)
        
        logger.info(f"✅ Started {len(self.embedding_workers)} embedding workers and {len(self.storage_workers)} storage workers")
    
    async def stop_workers(self):
        """Stop all worker tasks"""
        self.is_running = False
        
        # Cancel all workers
        all_workers = self.embedding_workers + self.storage_workers
        for worker in all_workers:
            worker.cancel()
        
        # Wait for workers to finish
        if all_workers:
            await asyncio.gather(*all_workers, return_exceptions=True)
        
        self.embedding_workers.clear()
        self.storage_workers.clear()
        logger.info("🔄 All embedding workers stopped")
    
    async def submit_chunks_for_embedding(self, chunks: List[Chunk], document_id: str, priority: int = 0) -> str:
        """Submit chunks for parallel embedding generation and storage"""
        try:
            job_id = f"{document_id}_{int(time.time())}"
            
            job = EmbeddingJob(
                job_id=job_id,
                chunks=chunks,
                document_id=document_id,
                priority=priority
            )
            
            await self.embedding_queue.put(job)
            self.active_embedding_jobs[job_id] = job
            
            logger.info(f"📊 Submitted {len(chunks)} chunks for parallel embedding: {job_id}")
            return job_id
            
        except Exception as e:
            logger.error(f"❌ Failed to submit chunks for embedding: {e}")
            raise
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get the status of an embedding job"""
        if job_id in self.completed_jobs:
            return {
                "status": "completed",
                "result": self.completed_jobs[job_id],
                "progress": 100.0
            }
        elif job_id in self.failed_jobs:
            return {
                "status": "failed",
                "error": str(self.failed_jobs[job_id]),
                "progress": 0.0
            }
        elif job_id in self.active_embedding_jobs:
            return {
                "status": "processing",
                "progress": 50.0,  # TODO: Implement detailed progress tracking
                "chunks_count": len(self.active_embedding_jobs[job_id].chunks)
            }
        else:
            return {
                "status": "not_found",
                "progress": 0.0
            }
    
    async def _embedding_worker(self, worker_name: str):
        """Worker task that processes embedding jobs from the queue"""
        logger.info(f"🔄 Embedding worker {worker_name} started")
        
        while self.is_running:
            try:
                # Wait for a job with timeout
                job = await asyncio.wait_for(self.embedding_queue.get(), timeout=1.0)
                
                logger.info(f"🔄 Worker {worker_name} processing embedding job: {job.job_id}")
                
                # Process the embedding job
                async with self.embedding_semaphore:
                    try:
                        result = await self._process_embedding_job(job, worker_name)
                        
                        # Submit to storage queue
                        await self.storage_queue.put((job, result))
                        
                        logger.info(f"✅ Worker {worker_name} completed embedding job: {job.job_id}")
                        
                    except Exception as e:
                        logger.error(f"❌ Worker {worker_name} failed embedding job {job.job_id}: {e}")
                        self.failed_jobs[job.job_id] = e
                        
                        # Remove from active jobs
                        if job.job_id in self.active_embedding_jobs:
                            del self.active_embedding_jobs[job.job_id]
                
                # Mark embedding task as done
                self.embedding_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"❌ Embedding worker {worker_name} encountered error: {e}")
                continue
        
        logger.info(f"🔄 Embedding worker {worker_name} stopped")
    
    async def _storage_worker(self, worker_name: str):
        """Worker task that stores embeddings in the vector database"""
        logger.info(f"🔄 Storage worker {worker_name} started")
        
        while self.is_running:
            try:
                # Wait for a storage job with timeout
                job_data = await asyncio.wait_for(self.storage_queue.get(), timeout=1.0)
                job, embeddings_result = job_data
                
                logger.info(f"🔄 Worker {worker_name} storing embeddings for job: {job.job_id}")
                
                # Store embeddings
                async with self.storage_semaphore:
                    try:
                        await self._store_embeddings_parallel(job, embeddings_result, worker_name)
                        
                        # Mark as completed
                        self.completed_jobs[job.job_id] = {
                            "chunks_processed": len(job.chunks),
                            "embeddings_stored": len(embeddings_result),
                            "document_id": job.document_id
                        }
                        
                        # Remove from active jobs
                        if job.job_id in self.active_embedding_jobs:
                            del self.active_embedding_jobs[job.job_id]
                        
                        logger.info(f"✅ Worker {worker_name} stored embeddings for job: {job.job_id}")
                        
                    except Exception as e:
                        logger.error(f"❌ Worker {worker_name} failed to store embeddings for job {job.job_id}: {e}")
                        self.failed_jobs[job.job_id] = e
                        
                        # Remove from active jobs
                        if job.job_id in self.active_embedding_jobs:
                            del self.active_embedding_jobs[job.job_id]
                
                # Mark storage task as done
                self.storage_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"❌ Storage worker {worker_name} encountered error: {e}")
                continue
        
        logger.info(f"🔄 Storage worker {worker_name} stopped")
    
    async def _process_embedding_job(self, job: EmbeddingJob, worker_name: str) -> List[List[float]]:
        """Process an embedding job with parallel batch processing"""
        start_time = time.time()
        
        try:
            chunks = job.chunks
            logger.info(f"🔄 {worker_name} generating embeddings for {len(chunks)} chunks")
            
            # Optimize batching based on chunk sizes
            if self.config.enable_batch_optimization:
                batches = self._optimize_batches(chunks)
            else:
                # Simple batching
                batch_size = self.config.embedding_batch_size
                batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
            
            logger.info(f"🔄 {worker_name} processing {len(batches)} optimized batches")
            
            # Process batches in parallel
            all_embeddings = []
            
            # Process batches concurrently with controlled parallelism
            batch_semaphore = asyncio.Semaphore(3)  # Limit concurrent batches per job
            
            async def process_batch_with_semaphore(batch_chunks):
                async with batch_semaphore:
                    return await self._generate_batch_embeddings(batch_chunks, worker_name)
            
            # Process all batches concurrently
            batch_results = await asyncio.gather(
                *[process_batch_with_semaphore(batch) for batch in batches],
                return_exceptions=True
            )
            
            # Collect results
            for batch_result in batch_results:
                if isinstance(batch_result, Exception):
                    logger.error(f"❌ Batch processing failed: {batch_result}")
                    raise batch_result
                all_embeddings.extend(batch_result)
            
            processing_time = time.time() - start_time
            logger.info(f"✅ {worker_name} generated {len(all_embeddings)} embeddings in {processing_time:.2f}s")
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"❌ Embedding job processing failed: {e}")
            raise
    
    def _optimize_batches(self, chunks: List[Chunk]) -> List[List[Chunk]]:
        """Optimize batch sizes based on chunk content lengths"""
        batches = []
        current_batch = []
        current_batch_tokens = 0
        max_batch_tokens = 100000  # Conservative limit for batch
        
        for chunk in chunks:
            # Estimate tokens for this chunk
            chunk_tokens = self._estimate_tokens(chunk.content)
            
            # If adding this chunk would exceed the limit, start a new batch
            if current_batch and (current_batch_tokens + chunk_tokens > max_batch_tokens):
                batches.append(current_batch)
                current_batch = [chunk]
                current_batch_tokens = chunk_tokens
            else:
                current_batch.append(chunk)
                current_batch_tokens += chunk_tokens
            
            # Also limit by count
            if len(current_batch) >= self.config.embedding_batch_size:
                batches.append(current_batch)
                current_batch = []
                current_batch_tokens = 0
        
        # Add final batch
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        # Simple estimation: 1 token per 3 characters
        return len(text) // 3
    
    def _extract_wait_time_from_error(self, error_str: str) -> float:
        """Extract recommended wait time from OpenAI rate limit error message"""
        import re
        
        try:
            # Look for patterns like "Please try again in 2.969s" or "try again in 30s"
            patterns = [
                r"try again in (\d+\.?\d*)s",
                r"try again in (\d+\.?\d*) seconds",
                r"wait (\d+\.?\d*)s",
                r"wait (\d+\.?\d*) seconds"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, error_str, re.IGNORECASE)
                if match:
                    wait_time = float(match.group(1))
                    # Add a small buffer to the recommended wait time
                    return wait_time + 0.5
            
            # If no specific time found, return None to use exponential backoff
            return None
            
        except Exception as e:
            logger.debug(f"Could not extract wait time from error: {e}")
            return None
    
    async def _generate_batch_embeddings(self, chunks: List[Chunk], worker_name: str) -> List[List[float]]:
        """Generate embeddings for a batch of chunks with enhanced rate limit handling"""
        texts = [chunk.content for chunk in chunks]
        
        for attempt in range(self.config.max_retries):
            try:
                # Get an available OpenAI client
                async with self.client_semaphore:
                    client = self.openai_clients[0]  # Simple round-robin could be improved
                    
                    # Add rate limiting delay
                    await asyncio.sleep(self.config.rate_limit_delay)
                    
                    # Generate embeddings with rate limit handling
                    response = await client.embeddings.create(
                        model=settings.EMBEDDING_MODEL,
                        input=texts
                    )
                    
                    embeddings = [data.embedding for data in response.data]
                    logger.debug(f"✅ {worker_name} generated {len(embeddings)} embeddings (batch size: {len(texts)})")
                    
                    return embeddings
                    
            except Exception as e:
                error_str = str(e)
                
                # Enhanced rate limit handling
                if "rate limit" in error_str.lower() or "429" in error_str:
                    # Extract wait time from error message if available
                    wait_time = self._extract_wait_time_from_error(error_str)
                    
                    if wait_time is None:
                        # Use exponential backoff if no specific wait time provided
                        wait_time = min(60, (2 ** attempt) + (attempt * 0.1))  # Cap at 60 seconds
                    
                    # Enforce minimum 5-second backoff for rate limits
                    wait_time = max(5.0, wait_time)
                    
                    if attempt < self.config.max_retries - 1:
                        logger.warning(f"⏳ {worker_name} rate limit hit (attempt {attempt + 1}/{self.config.max_retries}). Waiting {wait_time:.1f}s before retry (minimum 5s enforced)...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"❌ {worker_name} rate limit exceeded after {self.config.max_retries} attempts")
                        raise
                
                # Check for other retryable errors
                elif any(error_type in error_str.lower() for error_type in ["timeout", "connection", "server error", "503", "502", "500"]):
                    if attempt < self.config.max_retries - 1:
                        wait_time = min(30, (2 ** attempt))  # Shorter wait for non-rate-limit errors
                        logger.warning(f"⏳ {worker_name} retryable error (attempt {attempt + 1}/{self.config.max_retries}): {error_str}. Waiting {wait_time:.1f}s...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"❌ {worker_name} max retries exceeded for error: {error_str}")
                        raise
                
                else:
                    # Non-retryable error, fail immediately
                    logger.error(f"❌ {worker_name} non-retryable error: {error_str}")
                    raise
    
    async def _store_embeddings_parallel(self, job: EmbeddingJob, embeddings: List[List[float]], worker_name: str):
        """Store embeddings in parallel batches"""
        try:
            chunks = job.chunks
            
            if len(chunks) != len(embeddings):
                raise ValueError(f"Chunk count ({len(chunks)}) doesn't match embedding count ({len(embeddings)})")
            
            # Prepare points for Qdrant
            points = []
            for chunk, embedding in zip(chunks, embeddings):
                content_hash = abs(hash(chunk.content))
                
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
                        "job_id": job.job_id
                    }
                )
                points.append(point)
            
            # Store in parallel batches
            if self.config.enable_concurrent_storage:
                await self._store_points_concurrent(points, worker_name)
            else:
                await self._store_points_sequential(points, worker_name)
            
            logger.info(f"✅ {worker_name} stored {len(points)} embeddings for job {job.job_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to store embeddings: {e}")
            raise
    
    async def _store_points_concurrent(self, points: List[PointStruct], worker_name: str):
        """Store points in concurrent batches"""
        batch_size = self.config.storage_batch_size
        batches = [points[i:i + batch_size] for i in range(0, len(points), batch_size)]
        
        # Store batches concurrently
        storage_semaphore = asyncio.Semaphore(2)  # Limit concurrent storage operations
        
        async def store_batch_with_semaphore(batch):
            async with storage_semaphore:
                return await self._store_single_batch(batch, worker_name)
        
        # Store all batches concurrently
        await asyncio.gather(
            *[store_batch_with_semaphore(batch) for batch in batches]
        )
        
        logger.debug(f"✅ {worker_name} stored {len(batches)} batches concurrently")
    
    async def _store_points_sequential(self, points: List[PointStruct], worker_name: str):
        """Store points in sequential batches"""
        batch_size = self.config.storage_batch_size
        
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            await self._store_single_batch(batch, worker_name)
        
        logger.debug(f"✅ {worker_name} stored {len(points)} points sequentially")
    
    async def _store_single_batch(self, batch: List[PointStruct], worker_name: str):
        """Store a single batch of points"""
        loop = asyncio.get_event_loop()
        
        # Run Qdrant operation in thread pool to avoid blocking
        await loop.run_in_executor(
            self.thread_pool,
            lambda: self.qdrant_client.upsert(
                collection_name=settings.VECTOR_COLLECTION_NAME,
                points=batch
            )
        )
        
        logger.debug(f"✅ {worker_name} stored batch of {len(batch)} points")
    
    async def embed_and_store_chunks_parallel(self, chunks: List[Chunk], document_id: str = None) -> str:
        """Main interface for parallel embedding and storage"""
        try:
            if not chunks:
                return ""
            
            # Use document_id from chunks if not provided
            if not document_id and chunks:
                document_id = chunks[0].document_id
            
            # Submit for parallel processing
            job_id = await self.submit_chunks_for_embedding(chunks, document_id)
            
            logger.info(f"📊 Started parallel embedding job {job_id} for {len(chunks)} chunks")
            return job_id
            
        except Exception as e:
            logger.error(f"❌ Failed to start parallel embedding: {e}")
            raise
    
    async def wait_for_job_completion(self, job_id: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Wait for a specific job to complete"""
        start_time = time.time()
        
        while True:
            status = await self.get_job_status(job_id)
            
            if status["status"] in ["completed", "failed"]:
                return status
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise asyncio.TimeoutError(f"Job {job_id} did not complete within {timeout} seconds")
            
            # Wait before checking again
            await asyncio.sleep(0.5)
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get statistics about the embedding queues"""
        return {
            "embedding_queue_size": self.embedding_queue.qsize(),
            "storage_queue_size": self.storage_queue.qsize(),
            "active_jobs": len(self.active_embedding_jobs),
            "completed_jobs": len(self.completed_jobs),
            "failed_jobs": len(self.failed_jobs),
            "embedding_workers_running": len([w for w in self.embedding_workers if not w.done()]),
            "storage_workers_running": len([w for w in self.storage_workers if not w.done()]),
            "is_running": self.is_running,
            "config": {
                "max_concurrent_requests": self.config.max_concurrent_requests,
                "max_concurrent_storage": self.config.max_concurrent_storage,
                "embedding_batch_size": self.config.embedding_batch_size,
                "storage_batch_size": self.config.storage_batch_size
            }
        }
    
    # Delegate methods to base embedding manager for compatibility
    async def generate_embeddings(self, texts: List[str], max_retries: int = 5) -> List[List[float]]:
        """Generate embeddings for a list of texts (delegates to base manager)"""
        return await self.base_embedding_manager.generate_embeddings(texts, max_retries)
    
    async def embed_and_store_chunks(
        self, 
        chunks: List[Chunk], 
        user_id: str = None,
        document_category: str = None,
        document_tags: List[str] = None,
        document_title: str = None,
        document_author: str = None,
        document_filename: str = None
    ):
        """Generate embeddings for chunks and store in vector database (delegates to base manager)"""
        return await self.base_embedding_manager.embed_and_store_chunks(
            chunks, 
            user_id=user_id,
            document_category=document_category,
            document_tags=document_tags,
            document_title=document_title,
            document_author=document_author,
            document_filename=document_filename
        )
    
    async def search_similar(self, query_text: str, limit: int = 50, score_threshold: float = 0.7, 
                           use_query_expansion: bool = True, expansion_model: str = None,
                           user_id: str = None, include_adjacent_chunks: bool = False) -> List[Dict[str, Any]]:
        """Enhanced search with LLM query expansion (delegates to base manager)"""
        return await self.base_embedding_manager.search_similar(
            query_text, limit, score_threshold, use_query_expansion, expansion_model, user_id, include_adjacent_chunks
        )
    
    async def get_chunk_by_id(self, chunk_id: str) -> Dict[str, Any]:
        """Retrieve a specific chunk by ID (delegates to base manager)"""
        return await self.base_embedding_manager.get_chunk_by_id(chunk_id)
    
    async def delete_document_chunks(self, document_id: str):
        """Delete all chunks for a specific document (delegates to base manager)"""
        return await self.base_embedding_manager.delete_document_chunks(document_id)
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector collection (delegates to base manager)"""
        return await self.base_embedding_manager.get_collection_stats()
    
    async def get_all_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Retrieve all chunks for a specific document (delegates to base manager)"""
        return await self.base_embedding_manager.get_all_document_chunks(document_id)
    
    async def search_similar_in_documents(self, query_text: str, document_ids: List[str], 
                                        limit: int = 10, score_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Search for similar chunks within specific documents (delegates to base manager)"""
        return await self.base_embedding_manager.search_similar_in_documents(
            query_text, document_ids, limit, score_threshold
        )
    
    async def update_embeddings_with_title(self, document_id: str, title: str):
        """Update existing embeddings to include title information (delegates to base manager)"""
        return await self.base_embedding_manager.update_embeddings_with_title(document_id, title)
    
    async def update_zip_embeddings_with_title(self, parent_document_id: str, title: str):
        """Update embeddings for all sub-documents within a zip file (delegates to base manager)"""
        return await self.base_embedding_manager.update_zip_embeddings_with_title(parent_document_id, title)
    
    async def batch_update_titles_for_existing_documents(self):
        """Batch update all existing documents with their titles (delegates to base manager)"""
        return await self.base_embedding_manager.batch_update_titles_for_existing_documents()
    
    async def store_chunk_embedding(self, chunk_id: str, document_id: str, content: str, metadata: Dict[str, Any] = None):
        """Store a single chunk embedding (delegates to base manager)"""
        return await self.base_embedding_manager.store_chunk_embedding(chunk_id, document_id, content, metadata)
    
    async def delete_document_embeddings(self, document_id: str):
        """Delete all embeddings for a document (delegates to base manager)"""
        return await self.base_embedding_manager.delete_document_embeddings(document_id)
    
    async def update_embeddings_with_metadata(self, document_id: str, metadata: Dict[str, Any]):
        """Update existing embeddings to include metadata (delegates to base manager)"""
        return await self.base_embedding_manager.update_embeddings_with_metadata(document_id, metadata)

    async def _generate_query_expansions(self, query_text: str, expansion_model: str = None) -> List[str]:
        """Generate alternative query formulations using LLM (delegates to base manager)"""
        return await self.base_embedding_manager._generate_query_expansions(query_text, expansion_model)

    async def close(self):
        """Clean up resources"""
        logger.info("🔄 Shutting down Parallel Embedding Manager...")
        
        # Stop workers
        await self.stop_workers()
        
        # Close thread pool
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
            logger.info("🔄 Thread pool shut down")
        
        # Close base embedding manager
        if self.base_embedding_manager:
            await self.base_embedding_manager.close()
        
        logger.info("✅ Parallel Embedding Manager shut down complete")
